"""Compare the thinking-mode variants on the shared 100-bug sample.

The question this answers: turning thinking off cuts output tokens by roughly 7x, but does it cost
accuracy? If it does not, every paid run should use it, because output tokens were more than half
the bill.

Cost per file cannot be recovered from the records, because the records do not carry the cache-hit
split; the usage log does but is not tagged per run. So this reports accuracy, parse success and
token counts per variant, and leaves the money to results/usage_<model>.jsonl.

Usage: python ab_thinking.py [--model deepseek-flash] [--n 100]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")

# the mode and optional effort appear in the filename, between the split and the sample size
NAME = re.compile(r"llm_(?P<cond>[a-z]+)_(?P<model>.+?)_(?P<split>\w+?)_"
                  r"(?P<mode>disabled|enabled)(?:-(?P<effort>\w+))?_n(?P<n>\d+)_seed(?P<seed>\d+)\.jsonl$")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "llm_*.jsonl"))):
        m = NAME.search(os.path.basename(path))
        if not m or m.group("model") != a.model or int(m.group("n")) != a.n:
            continue
        recs = load(path)
        if not recs:
            continue
        correct = [bool(r.get("top3")) and r["top3"][0] == r["true_team"] for r in recs]
        faithful = [bool(r.get("top3")) and r["top3"][0] == r.get("initial_team") for r in recs]
        rows.append({
            "condition": m.group("cond"),
            "mode": m.group("mode") + ("-" + m.group("effort") if m.group("effort") else ""),
            "n": len(recs),
            "parse_ok": float(np.mean([bool(r.get("parse_ok")) for r in recs])),
            "top1": float(np.mean(correct)),
            "abstain": float(np.mean([bool(r.get("abstain")) for r in recs])),
            "rounds": float(np.mean([r.get("rounds", 1) for r in recs])),
            "out_tokens": float(np.mean([r.get("completion_tokens", 0) for r in recs])),
            "in_tokens": float(np.mean([r.get("prompt_tokens", 0) for r in recs])),
            "latency_s": float(np.mean([r.get("latency_s", 0.0) for r in recs])),
            "pred_eq_initial": float(np.mean(faithful)),
        })

    if not rows:
        print(f"no run files for model={a.model} n={a.n}")
        return 1

    hdr = ("condition", "mode", "n", "parse_ok", "top1", "abstain", "rounds",
           "in_tok", "out_tok", "lat_s", "pred=init")
    print("%-9s %-12s %5s %8s %7s %8s %7s %9s %8s %7s %9s" % hdr)
    for r in rows:
        print("%-9s %-12s %5d %8.3f %7.3f %8.3f %7.2f %9.0f %8.0f %7.1f %9.3f"
              % (r["condition"], r["mode"], r["n"], r["parse_ok"], r["top1"], r["abstain"],
                 r["rounds"], r["in_tokens"], r["out_tokens"], r["latency_s"],
                 r["pred_eq_initial"]))

    print("\npaired comparison, disabled vs enabled, same condition and same bugs:")
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], {})[r["mode"]] = r
    for cond, modes in sorted(by_cond.items()):
        off, on = modes.get("disabled"), modes.get("enabled")
        if not off or not on:
            continue
        print(f"  {cond:<9} top1 {off['top1']:.3f} -> {on['top1']:.3f} "
              f"({100 * (on['top1'] - off['top1']):+.1f} pp) | "
              f"parse_ok {off['parse_ok']:.3f} -> {on['parse_ok']:.3f} | "
              f"mean output tokens {off['out_tokens']:.0f} -> {on['out_tokens']:.0f} "
              f"({on['out_tokens'] / max(1.0, off['out_tokens']):.1f}x)")

    out = os.path.join(ROOT, "results", f"ab_thinking_{a.model}_n{a.n}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"model": a.model, "n": a.n, "rows": rows}, fh, indent=1)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
