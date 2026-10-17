"""Merge the 2022-2024 dataset and the 2026 test period into one file with three splits.

Why: every driver in this directory takes a single --dataset plus a --train-split and a
--test-split. The label space, the retrieval index and the fitted vocabulary must all come from
the same train split, so the 2026 period has to live next to the 2022-2023 train rows rather
than in its own file.

Split names are preserved exactly:
  train     bugs created 2022-2023 (unchanged from data/dataset.jsonl)
  test      bugs created 2024    (unchanged, so the step-1 numbers stay comparable)
  test2026  bugs created 2026    (the contamination-free period)

Rows are de-duplicated by bug id, later inputs do not overwrite earlier ones.

Usage: python merge_datasets.py
"""
import argparse
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def read(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+",
                    default=[os.path.join(ROOT, "data", "dataset.jsonl"),
                             os.path.join(ROOT, "data", "dataset_2026.jsonl")])
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "dataset_all.jsonl"))
    a = ap.parse_args()

    seen = {}
    order = []
    for path in a.inputs:
        rows = read(path)
        added = 0
        for row in rows:
            if row["id"] not in seen:
                seen[row["id"]] = row
                order.append(row["id"])
                added += 1
        print(f"{path}: {len(rows)} rows, {added} new")
    with open(a.out, "w", encoding="utf-8") as fh:
        for bug_id in order:
            fh.write(json.dumps(seen[bug_id]) + "\n")

    counts = {}
    for bug_id in order:
        split = seen[bug_id]["split"]
        counts[split] = counts.get(split, 0) + 1
    print("wrote", a.out, "total", len(order), "by split", counts)


if __name__ == "__main__":
    main()
