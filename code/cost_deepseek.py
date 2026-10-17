"""DeepSeek cost model for the paper-01 LLM conditions. NO API calls are made here.

Token counts are MEASURED from the real dataset: results/cost_team_top20.json, produced by
cost_estimate.py using tiktoken o200k_base as a PROXY tokenizer. DeepSeek's own tokenizer will
give different counts; that error has not been measured and is stated as a limitation.

Prices are DeepSeek's published list prices in USD per 1e6 tokens, copied by hand from the
official page https://api-docs.deepseek.com/quick_start/pricing on 2026-10-12. Off-peak is half
of peak; peak hours are 01:00-04:00 and 06:00-10:00 UTC, Mon-Fri, excluding Chinese public
holidays, and every other hour (including all weekend) is off-peak.

Cache model. DeepSeek's context caching is automatic and bills the cached prefix at a much lower
rate. Two bounds are reported:
  no_cache : every input token at the cache-miss rate (upper bound).
  prefix   : the stable prompt prefix (instruction + label block) is a cache hit on every call,
             and inside the agent loop the context re-sent at steps 2..T is a cache hit
             (lower bound). The bug text and the retrieved examples are never cached.

Usage: python cost_deepseek.py [n_bugs]
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")

# USD per 1e6 tokens, off-peak. Copied 2026-10-12 from the official pricing page.
PRICES = {
    "deepseek-flash": {"hit": 0.003, "miss": 0.15, "out": 0.60, "concurrency": 2500},
    "deepseek-v4-pro": {"hit": 0.022, "miss": 0.66, "out": 1.98, "concurrency": 500},
}
PEAK = 2.0  # peak-hour multiplier


def measured():
    with open(os.path.join(ROOT, "results", "cost_team_top20.json")) as fh:
        c = json.load(fh)
    m, a = c["measured"], c["assumed"]
    return {"Q": m["Q_mean"], "E": m["E_mean"], "L": m["L_label_block"], "S": a["S"], "K": a["K"],
            "O": a["O"], "T": a["T"], "R": a["R"], "O_step": a["O_step"], "n_test": c["n_test"]}


def conditions(t):
    """Per-bug (in_total, in_cached, out, calls) for each LLM condition."""
    stable = t["S"] + t["L"]                       # instruction + label block, identical per call
    rag_var = t["Q"] + t["K"] * t["E"]             # bug text + retrieved examples
    zs_in = stable + t["Q"]
    rag_in = stable + rag_var
    # the agent re-sends the whole growing context at every step
    agent_in = sum(stable + t["Q"] + (i - 1) * (t["R"] + t["O_step"]) for i in range(1, t["T"] + 1))
    agent_new = stable + t["Q"] + (t["T"] - 1) * (t["R"] + t["O_step"])   # never-seen tokens
    return {
        "zero_shot": {"in_total": zs_in, "in_cached": stable, "out": t["O"], "calls": 1},
        "rag": {"in_total": rag_in, "in_cached": stable, "out": t["O"], "calls": 1},
        "agent": {"in_total": agent_in, "in_cached": agent_in - agent_new,
                  "out": t["T"] * t["O_step"], "calls": t["T"]},
        "self_consistency_k5": {"in_total": 5 * rag_in, "in_cached": 5 * stable,
                                "out": 5 * t["O"], "calls": 5},
    }


def usd(model, tok_in, tok_cached, tok_out, peak=False, cache=True):
    p = PRICES[model]
    hit = tok_cached if cache else 0
    miss = tok_in - hit
    cost = miss / 1e6 * p["miss"] + hit / 1e6 * p["hit"] + tok_out / 1e6 * p["out"]
    return cost * (PEAK if peak else 1.0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    t = measured()
    conds = conditions(t)

    # scenario = (label, n_bugs, {condition: repeats})
    scenarios = [
        ("pilot: 50 bugs x agent", 50, {"agent": 1}),
        ("pilot: 50 bugs x agent x3 seeds", 50, {"agent": 3}),
        ("main: N x (zero-shot + rag)", n, {"zero_shot": 1, "rag": 1}),
        ("main: N x agent", n, {"agent": 1}),
        ("main: N x agent x3 seeds", n, {"agent": 3}),
        ("main: N x (zs + rag + agent)", n, {"zero_shot": 1, "rag": 1, "agent": 1}),
        ("FULL: N x (zs + rag + agent x3 + sc k5)", n,
         {"zero_shot": 1, "rag": 1, "agent": 3, "self_consistency_k5": 1}),
        ("FULL + second model arm (same sweep, both models counted once)", n,
         {"zero_shot": 1, "rag": 1, "agent": 3, "self_consistency_k5": 1}),
    ]

    print(f"# DeepSeek cost model; prices copied 2026-10-12 from "
          f"https://api-docs.deepseek.com/quick_start/pricing")
    print(f"# measured token counts (tiktoken o200k_base proxy): Q={t['Q']:.1f} E={t['E']:.1f} "
          f"L={t['L']} | assumed: S={t['S']} K={t['K']} O={t['O']} T={t['T']} R={t['R']} "
          f"O_step={t['O_step']}")
    print(f"# per-bug tokens: " + "; ".join(
        f"{k}={v['in_total']:.0f}in({v['in_cached']:.0f}cached)/{v['out']:.0f}outx{v['calls']}calls"
        for k, v in conds.items()))
    print()

    hdr = f"{'scenario':<58}{'model':<18}{'Mtok in':>10}{'Mtok out':>10}{'no-cache':>11}{'prefix':>10}{'peak':>10}"
    print(hdr)
    print("-" * len(hdr))
    out = []
    for label, bugs, mix in scenarios:
        for model in PRICES:
            ti = sum(conds[c]["in_total"] * r for c, r in mix.items()) * bugs
            tc = sum(conds[c]["in_cached"] * r for c, r in mix.items()) * bugs
            to = sum(conds[c]["out"] * r for c, r in mix.items()) * bugs
            row = {"scenario": label, "model": model, "n_bugs": bugs, "mix": mix,
                   "Mtok_in": round(ti / 1e6, 3), "Mtok_out": round(to / 1e6, 3),
                   "usd_no_cache_offpeak": round(usd(model, ti, tc, to, cache=False), 2),
                   "usd_prefix_offpeak": round(usd(model, ti, tc, to, cache=True), 2),
                   "usd_prefix_peak": round(usd(model, ti, tc, to, peak=True), 2)}
            out.append(row)
            print(f"{label:<58}{model:<18}{row['Mtok_in']:>10.1f}{row['Mtok_out']:>10.1f}"
                  f"{row['usd_no_cache_offpeak']:>11.2f}{row['usd_prefix_offpeak']:>10.2f}"
                  f"{row['usd_prefix_peak']:>10.2f}")

    print()
    print("# note: output tokens are capped by DeepSeek's thinking mode, which is ON by default.")
    print("# The assumed O and O_step are NOT measured; run the pilot to calibrate them.")
    with open(os.path.join(ROOT, "results", "cost_deepseek.json"), "w") as fh:
        json.dump({"prices_copied": "2026-10-12",
                   "price_source": "https://api-docs.deepseek.com/quick_start/pricing",
                   "prices_usd_per_Mtok_offpeak": PRICES,
                   "measured": t, "per_bug_tokens": conds, "scenarios": out}, fh, indent=1)
    print("# wrote results/cost_deepseek.json")


if __name__ == "__main__":
    main()
