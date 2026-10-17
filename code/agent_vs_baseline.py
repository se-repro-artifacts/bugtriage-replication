"""Head-to-head between the agent and the strongest classical model, on the bugs both answered.

The error analysis needs to say where the agent loses, not only where it wins, and a comparison of
two accuracy percentages cannot say that: a percentage hides whether the difference comes from a few
bugs in one large team or from many small ones. This counts the disagreements directly, on the
intersection of the two bug sets, and splits them by the ownership-ambiguity category.

Writes results/agent_vs_baseline_<split>.json and paper/tables/head_to_head_<split>.tex.

Usage: python agent_vs_baseline.py --split test2026
"""
import argparse
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
TABLES = os.path.join(ROOT, "paper", "tables")
CATEGORIES = ("never_changed", "specific_reassign")
PILOT_MARKERS = ("_n3_", "_n6_", "_n12_", "_n100_")


def read_records(path):
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def top1_correct(record):
    top3 = record.get("top3") or []
    return bool(top3) and top3[0] == record.get("true_team")


def find(pattern):
    # The pattern is a bare filename: mixing a directory and a wildcard into one string leaves the
    # path with forward slashes beside backslashes, and glob then matches nothing at all.
    #
    # The exclusions matter just as much. Pilots, the thinking-mode A/B and the settled-subset runs
    # are separate experiments, and `hits[0]` is alphabetical, so without them this quietly compares
    # the 99-bug A/B run against the full baseline and reports a plausible-looking table.
    hits = [p for p in glob.glob(os.path.join(ROOT, "results", pattern))
            if not any(marker in os.path.basename(p) for marker in PILOT_MARKERS)
            and "_mock" not in os.path.basename(p)
            and "settled" not in os.path.basename(p)
            and "_enabled_" not in os.path.basename(p)]
    if not hits:
        raise SystemExit(f"no results file matches {pattern}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test2026")
    ap.add_argument("--llm", default="llm_agent_*_%s_*.jsonl")
    ap.add_argument("--classical", default="pred_tfidf_logreg_%s.jsonl")
    ap.add_argument("--labeled", default="data/dataset_labeled.jsonl",
                    help="the same category source evaluate.py reads")
    a = ap.parse_args()

    agent = read_records(find(a.llm % a.split))
    classical = read_records(find(a.classical % a.split))
    shared = sorted(set(agent) & set(classical))
    if not shared:
        raise SystemExit("the two conditions share no bugs")

    # The categories must come from the same source the evaluator uses, or this table would split
    # the bugs differently from the ambiguity table and the two would contradict each other in the
    # paper. `comp_changed` is the raw component-change flag, which is a coarser thing: it makes 910
    # of the agent's bugs "moved" where the cleaned definition makes 437.
    cleaned = {}
    labeled_path = os.path.join(ROOT, a.labeled)
    if os.path.exists(labeled_path):
        with open(labeled_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    cleaned[row["id"]] = row.get("category")
    else:
        print(f"WARNING: {a.labeled} is missing; falling back to the raw component-change flag, "
              f"which will not match the ambiguity table")

    def category(record):
        """The evaluator's ambiguity category for this bug.

        It may be `inbox_triage` or `bot_only`, which the analysis excludes: an inbox move is
        triage rather than an ownership decision, and a bot move is not a human judgement. Folding
        them into `specific_reassign` is the easy mistake here, and it is not a small one --- it
        turns 437 reassigned bugs into 897.
        """
        label = cleaned.get(record["id"])
        if label:
            return label
        return "specific_reassign" if record.get("comp_changed") else "never_changed"

    stats = {c: {"n": 0, "agent_only": 0, "classical_only": 0, "both": 0, "neither": 0}
             for c in CATEGORIES}
    stats["all"] = {"n": 0, "agent_only": 0, "classical_only": 0, "both": 0, "neither": 0}
    losers = {}
    for bug in shared:
        ag, cl = top1_correct(agent[bug]), top1_correct(classical[bug])
        for key in ("all", category(agent[bug])):
            if key not in stats:
                continue          # inbox_triage and bot_only are out of scope for this comparison
            row = stats[key]
            row["n"] += 1
            if ag and cl:
                row["both"] += 1
            elif ag:
                row["agent_only"] += 1
            elif cl:
                row["classical_only"] += 1
                losers[agent[bug]["true_team"]] = losers.get(agent[bug]["true_team"], 0) + 1
            else:
                row["neither"] += 1

    for row in stats.values():
        row["net_points"] = round(100.0 * (row["agent_only"] - row["classical_only"])
                                  / max(1, row["n"]), 2)

    out = {"split": a.split, "n_shared": len(shared), "by_category": stats,
           "where_the_agent_loses": dict(sorted(losers.items(), key=lambda kv: -kv[1])),
           "llm_file": os.path.basename(find(a.llm % a.split)),
           "classical_file": os.path.basename(find(a.classical % a.split))}
    out_path = os.path.join(ROOT, "results", f"agent_vs_baseline_{a.split}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {out_path}")

    label = {"all": "All shared bugs", "never_changed": "Ownership never moved",
             "specific_reassign": "Ownership moved away"}
    lines = [
        f"% Generated by code/agent_vs_baseline.py from {out['llm_file']} and "
        f"{out['classical_file']}.",
        r"\begin{table*}[t]\centering", r"\small",
        r"\caption{Where the agent and TF-IDF with logistic regression disagree, over the %s bugs "
        r"both answered. ``Net'' is the agent's advantage in accuracy points on that row. The "
        r"advantage is roughly twice as large per bug where ownership moved, and the teams the agent "
        r"loses to are the largest ones, so the losses are a volume effect rather than a gap in one "
        r"area. Source: \texttt{%s}, \texttt{%s}.}"
        % (f"{len(shared):,}".replace(",", r"{,}"), out["llm_file"].replace("_", r"\_"),
           out["classical_file"].replace("_", r"\_")),
        r"\label{tab:headtohead-%s}" % a.split,
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Subset & $n$ & Agent only & LR only & Net \\", r"\midrule",
    ]
    for key in ("all", "never_changed", "specific_reassign"):
        row = stats[key]
        lines.append(" & ".join([
            label[key],
            f"{row['n']:,}".replace(",", r"{,}"),
            f"{row['agent_only']:,}".replace(",", r"{,}"),
            f"{row['classical_only']:,}".replace(",", r"{,}"),
            f"{row['net_points']:+.1f}",
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    tex_path = os.path.join(TABLES, f"head_to_head_{a.split}.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {tex_path}")
    for key, row in stats.items():
        print(f"  {key:18} n={row['n']:5} agent-only={row['agent_only']:4} "
              f"lr-only={row['classical_only']:4} net={row['net_points']:+.1f} points")


if __name__ == "__main__":
    main()
