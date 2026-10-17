"""How long after a bug is filed does its last component/product change happen?

The 2026 test period was pulled while it is still young, so a bug filed in, say, September 2026
may be reassigned again after the data freeze, which would make its label a moving target. The
limitations section should state how large that risk actually is rather than guess, so this script
measures the delay between creation and the last recorded reassignment for each period.

It reads the raw files, because only they carry timestamps: data/history.jsonl holds the change
events with their `when`, and data/bugs_meta.jsonl holds `creation_time`.

Usage: python label_stability.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
PERIODS = {"2022-2024": os.path.join(ROOT, "data"),
           "2026": os.path.join(ROOT, "data", "2026")}


def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main():
    report = {}
    for name, folder in PERIODS.items():
        meta = {r["id"]: r.get("creation_time") for r in read_jsonl(os.path.join(folder, "bugs_meta.jsonl"))}
        if not meta:
            continue
        delays, n_moved = [], 0
        for rec in read_jsonl(os.path.join(folder, "history.jsonl")):
            created = meta.get(rec["id"])
            whens = [m["when"] for m in rec.get("moves", []) if m.get("when")]
            if not created or not whens:
                continue
            n_moved += 1
            from datetime import datetime
            try:
                t0 = datetime.fromisoformat(created.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(max(whens).replace("Z", "+00:00"))
            except ValueError:
                continue
            delays.append((t1 - t0).total_seconds() / 86400.0)
        if not delays:
            continue
        d = np.asarray(delays)
        report[name] = {
            "n_bugs_total": len(meta),
            "n_with_moves": n_moved,
            "days_to_last_move": {
                "p50": round(float(np.percentile(d, 50)), 1),
                "p90": round(float(np.percentile(d, 90)), 1),
                "p95": round(float(np.percentile(d, 95)), 1),
                "max": round(float(d.max()), 1),
                "share_within_30d": round(float((d <= 30).mean()), 4),
                "share_within_90d": round(float((d <= 90).mean()), 4),
            },
        }
        print(name, json.dumps(report[name], indent=1))
    out = os.path.join(ROOT, "results", "label_stability.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
