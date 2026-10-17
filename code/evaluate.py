"""Metrics for the LLM conditions, computed so they can be compared with the classical baselines.

Mirrors the selective-prediction definitions in baselines.py exactly (the confidence-sorted
cumulative accuracy, AURC_risk = mean(1 - cumulative accuracy), and the abstention rates at 80%
coverage), so a risk-coverage curve from a classical model and one from an LLM are directly
comparable. Adds what the LLM conditions need on top: explicit-abstention operating points,
abstention recall on the "moved from a specific component" subset, cost and latency per bug, and
a paired McNemar test between two conditions.

Usage:
  python evaluate.py --records results/llm_agent_deepseek-flash_2026_n1000_seed0.jsonl
  python evaluate.py --records A.jsonl --compare B.jsonl
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
PRICES = {"deepseek-flash": {"hit": 0.003, "miss": 0.15, "out": 0.60},
          "deepseek-v4-pro": {"hit": 0.022, "miss": 0.66, "out": 1.98}}


def load_records(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _wrong_rank(rec):
    """None when the record has no usable first guess (unparsed or empty top3)."""
    return rec["top3"][0] if rec.get("top3") else None


def correctness(recs):
    return np.array([_wrong_rank(r) == r["true_team"] for r in recs], dtype=bool)


def moved_from_specific(rec):
    initial = rec.get("initial_label") or ""
    return (initial != rec.get("true_fine") and initial.split("::")[-1] not in ("General", "Untriaged"))


def risk_coverage(conf, correct):
    """Same definition as baselines.py: sort by confidence, walk every prefix."""
    order = np.argsort(-np.asarray(conf, dtype=float), kind="stable")
    cum = np.cumsum(np.asarray(correct, dtype=float)[order]) / np.arange(1, len(order) + 1)
    out = {"AURC_risk": float(np.mean(1.0 - cum))}
    for cov in (0.5, 0.7, 0.8, 0.9):
        k = max(1, int(round(cov * len(order))))
        out[f"selective_acc_cov{int(cov * 100)}"] = float(cum[k - 1])
    return out, order


def abstain_rates(recs, order, coverage=0.8):
    kept = set(order[:max(1, int(round(coverage * len(order))))].tolist())
    groups = {"reassigned": [i for i, r in enumerate(recs) if r.get("comp_changed")],
              "never_reassigned": [i for i, r in enumerate(recs) if not r.get("comp_changed")],
              "moved_from_specific_component": [i for i, r in enumerate(recs)
                                                if moved_from_specific(r)]}
    out = {}
    for name, idx in groups.items():
        if not idx:
            out[name] = None
            continue
        out[name] = {"n": len(idx),
                     "abstain_rate_by_confidence": float(np.mean([i not in kept for i in idx]))}
    return out


def summarise(recs, model=None):
    n = len(recs)
    y = [r["true_team"] for r in recs]
    pred = [_wrong_rank(r) for r in recs]
    correct = correctness(recs)
    present = sorted(set(y))
    conf = [r.get("confidence") if r.get("confidence") is not None else 0.0 for r in recs]
    out = {
        "n": n,
        "coverage_parse": float(np.mean([bool(r.get("top3")) for r in recs])),
        "acc_all": float(correct.mean()),
        "macro_f1": float(f1_score(y, [p if p else "__no_answer__" for p in pred], labels=present,
                                   average="macro", zero_division=0)),
        "top3": float(np.mean([r["true_team"] in (r.get("top3") or []) for r in recs])),
        "n_tool_calls_mean": float(np.mean([r.get("n_tool_calls", 0) for r in recs])),
        "latency_s_mean": float(np.mean([r.get("latency_s", 0.0) for r in recs])),
        "prompt_tokens_mean": float(np.mean([r.get("prompt_tokens", 0) for r in recs])),
        "completion_tokens_mean": float(np.mean([r.get("completion_tokens", 0) for r in recs])),
    }
    rc, order = risk_coverage(conf, correct)
    out.update(rc)
    out["abstain_rates_at_cov80"] = abstain_rates(recs, order, 0.8)

    # explicit abstention the model chose itself
    chose = np.array([bool(r.get("abstain")) for r in recs], dtype=bool)
    answered = ~chose
    out["explicit_abstain"] = {
        "abstain_rate": float(chose.mean()),
        "coverage": float(answered.mean()),
        "acc_on_answered": float(correct[answered].mean()) if answered.any() else None,
        "abstain_recall_moved_from_specific": (
            float(chose[[i for i, r in enumerate(recs) if moved_from_specific(r)]].mean())
            if any(moved_from_specific(r) for r in recs) else None),
        "abstain_rate_reassigned": (
            float(chose[[i for i, r in enumerate(recs) if r.get("comp_changed")]].mean())
            if any(r.get("comp_changed") for r in recs) else None),
        "abstain_rate_never_reassigned": (
            float(chose[[i for i, r in enumerate(recs) if not r.get("comp_changed")]].mean())
            if any(not r.get("comp_changed") for r in recs) else None),
    }

    # Where the wrong answers go: the most frequent (true team, predicted team) pairs. Computed by
    # the same evaluator as everything else, so two conditions can be compared confusion by
    # confusion rather than only by their totals.
    pairs = {}
    for rec, ok in zip(recs, correct):
        if ok:
            continue
        top = (rec.get("top3") or [None])[0]
        if top:
            key_pair = (rec["true_team"], top)
            pairs[key_pair] = pairs.get(key_pair, 0) + 1
    out["top_confusions"] = [
        {"true_team": t, "pred_team": p, "n": c}
        for (t, p), c in sorted(pairs.items(), key=lambda kv: -kv[1])[:10]]

    # cost at list price, from the recorded per-call usage
    key = model or recs[0].get("model")
    price = PRICES.get(key)
    if price:
        hit = sum(r.get("prompt_cache_hit_tokens", 0) for r in recs)
        miss = sum(r.get("prompt_tokens", 0) for r in recs) - hit
        out_tok = sum(r.get("completion_tokens", 0) for r in recs)
        total = miss / 1e6 * price["miss"] + hit / 1e6 * price["hit"] + out_tok / 1e6 * price["out"]
        out["cost_usd_offpeak_list"] = round(total, 4)
        out["cost_usd_per_bug"] = round(total / max(1, n), 6)
    return out


def by_category(recs, categories, initial_teams=None, coverage=0.8):
    """Accuracy and abstention per ownership-ambiguity category.

    This is the paper's central analysis. The claim is that a classifier's confidence does not
    identify the bugs whose ownership actually moved, so the numbers that matter are accuracy and
    abstention on `specific_reassign` (moved away from a concrete component) against
    `never_changed` (never moved). `bot_only` is reported but excluded from the analysis, because
    a bot move carries no evidence of human disagreement.
    """
    correct = correctness(recs)
    conf = [r.get("confidence") if r.get("confidence") is not None else 0.0 for r in recs]
    _, order = risk_coverage(conf, correct)
    kept = set(order[:max(1, int(round(coverage * len(order))))].tolist())
    out = {}
    for cat in sorted({c for c in categories.values() if c}):
        idx = [i for i, r in enumerate(recs) if categories.get(r["id"]) == cat]
        if not idx:
            continue
        abstain_rate = float(np.mean([i not in kept for i in idx]))
        entry = {
            "n": len(idx),
            "acc": float(correct[idx].mean()),
            "top3": float(np.mean([recs[i]["true_team"] in (recs[i].get("top3") or [])
                                   for i in idx])),
            "abstain_rate_by_confidence_at_cov80": abstain_rate,
            # 1.0 means the model declines on this category exactly as often as it does overall,
            # so its confidence carries no information about the category. Below 1.0 means it
            # answers *more* often here than average, which is the opposite of what selective
            # prediction should do on the hardest group.
            "abstain_lift_at_cov80": abstain_rate / (1.0 - coverage),
            "explicit_abstain_rate": float(np.mean([bool(recs[i].get("abstain")) for i in idx])),
        }
        if initial_teams:
            sub = [i for i in idx if initial_teams.get(recs[i]["id"])]
            if sub:
                # how often the model reproduces the team of the component the reporter first
                # filed under, i.e. the answer implied by the pre-reassignment label
                entry["pred_eq_initial_team_rate"] = float(np.mean([
                    bool(recs[i].get("top3"))
                    and recs[i]["top3"][0] == initial_teams[recs[i]["id"]] for i in sub]))
                # Among the mistakes on this category, how many land on the reporter's first
                # choice? On a bug that was reassigned, that choice is the superseded one, so a
                # high value means the model follows the reporter into the error instead of
                # recovering from it.
                wrong = [i for i in sub if not correct[i]]
                if wrong:
                    entry["error_is_initial_team_rate"] = float(np.mean([
                        bool(recs[i].get("top3"))
                        and recs[i]["top3"][0] == initial_teams[recs[i]["id"]] for i in wrong]))
        out[cat] = entry
    return out


def mcnemar(recs_a, recs_b):
    """Paired McNemar test on the top-1 correctness of two conditions over shared bug ids."""
    by_a = {r["id"]: _wrong_rank(r) == r["true_team"] for r in recs_a}
    by_b = {r["id"]: _wrong_rank(r) == r["true_team"] for r in recs_b}
    shared = sorted(set(by_a) & set(by_b))
    if not shared:
        return None
    b = sum(1 for i in shared if by_a[i] and not by_b[i])   # only A correct
    c = sum(1 for i in shared if by_b[i] and not by_a[i])   # only B correct
    if b + c == 0:
        return {"n_shared": len(shared), "a_only_correct": b, "b_only_correct": c, "p_value": 1.0}
    p = binomtest(min(b, c), b + c, 0.5).pvalue
    return {"n_shared": len(shared), "a_only_correct": b, "b_only_correct": c, "p_value": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--compare", default=None, help="second records file for a McNemar test")
    ap.add_argument("--labeled", default=os.path.join(ROOT, "data", "dataset_labeled.jsonl"),
                    help="dataset carrying the `category` field, for the ambiguity breakdown")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = load_records(a.records)
    result = {"records": a.records, "condition": recs[0].get("condition") if recs else None,
              "model": recs[0].get("model") if recs else None,
              "seed": recs[0].get("seed") if recs else None,
              "split": recs[0].get("split") if recs else None,
              "metrics": summarise(recs)}
    categories, initial_teams = {}, {}
    if os.path.exists(a.labeled):
        with open(a.labeled, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    categories[row["id"]] = row.get("category")
                    initial_teams[row["id"]] = row.get("initial_team")
    if categories:
        result["labeled_source"] = a.labeled
        result["metrics"]["by_ambiguity_category"] = by_category(recs, categories, initial_teams)
        sub = [r for r in recs if initial_teams.get(r["id"])]
        if sub:
            result["metrics"]["pred_eq_initial_team_overall"] = float(np.mean([
                bool(r.get("top3")) and r["top3"][0] == initial_teams[r["id"]] for r in sub]))
    if a.compare:
        other = load_records(a.compare)
        result["mcnemar_vs"] = a.compare
        result["mcnemar"] = mcnemar(recs, other)
    print(json.dumps(result, indent=1))
    base = os.path.basename(a.records).replace("llm_", "").replace("pred_", "")
    if base.endswith(".jsonl"):
        base = base[:-6]
    result["tag"] = base        # lets the table generator keep subset runs out of the main table
    out_path = a.out or os.path.join(ROOT, "results", "metrics_" + base + ".json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
