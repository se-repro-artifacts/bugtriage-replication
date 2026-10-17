"""Score the manual review sample once the reviewer has filled it in.

The paper needs the agreement between the automatically derived ownership categories and human
judgement, and that number has to come from a script rather than from counting by eye (rule 1). This
reads results/manual_review_sample.csv, which carries the automatic `category` plus two columns the
reviewer fills (`reviewer_category`, `reviewer_notes`), and reports raw agreement, Cohen's kappa and
the full confusion matrix, so disagreements can be looked at individually rather than only counted.

Usage: python review_agreement.py
"""
import csv
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
SAMPLE = os.path.join(ROOT, "results", "manual_review_sample.csv")
CATEGORIES = ("never_changed", "inbox_triage", "specific_reassign", "bot_only")


def kappa(pairs, labels):
    """Cohen's kappa over the two label sequences."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = 0.0
    for label in labels:
        pa = sum(1 for a, _ in pairs if a == label) / n
        pb = sum(1 for _, b in pairs if b == label) / n
        expected += pa * pb
    return None if expected >= 1.0 else (observed - expected) / (1.0 - expected)


def main():
    if not os.path.exists(SAMPLE):
        raise SystemExit(f"no sample at {SAMPLE}; run code/label_cleaning.py first")
    rows = list(csv.DictReader(open(SAMPLE, encoding="utf-8")))
    reviewed = [r for r in rows if (r.get("reviewer_category") or "").strip()]
    print(f"sample: {len(rows)} bugs, reviewed so far: {len(reviewed)}")
    if not reviewed:
        print("nothing to score yet: fill in the `reviewer_category` column of "
              f"{os.path.relpath(SAMPLE, ROOT)}")
        print("valid entries: " + ", ".join(CATEGORIES))
        return

    unknown = sorted({r["reviewer_category"].strip() for r in reviewed}
                     - set(CATEGORIES))
    if unknown:
        print(f"WARNING: unrecognised reviewer_category values will be counted as disagreements: "
              f"{unknown}")

    pairs = [(r["category"].strip(), r["reviewer_category"].strip()) for r in reviewed]
    agree = sum(1 for a, b in pairs if a == b)
    print(f"raw agreement: {agree}/{len(reviewed)} = {100 * agree / len(reviewed):.1f}%")
    k = kappa(pairs, CATEGORIES)
    print(f"Cohen's kappa: {'--' if k is None else '%.3f' % k}")

    matrix = {a: {b: 0 for b in CATEGORIES} for a in CATEGORIES}
    for a, b in pairs:
        matrix.setdefault(a, {}).setdefault(b, 0)
        matrix[a][b] = matrix[a].get(b, 0) + 1
    print("\nconfusion (rows = automatic, columns = reviewer):")
    print("  " + " " * 18 + "".join(f"{c[:14]:>16}" for c in CATEGORIES))
    for a in CATEGORIES:
        row = matrix.get(a, {})
        print(f"  {a:<18}" + "".join(f"{row.get(c, 0):>16}" for c in CATEGORIES))

    disagree = [(r["bug_id"], r["category"], r["reviewer_category"].strip(),
                 (r.get("reviewer_notes") or "").strip()) for r in reviewed
                if r["category"].strip() != r["reviewer_category"].strip()]
    if disagree:
        print(f"\n{len(disagree)} disagreements, to be looked at individually:")
        for bug, auto, human, note in disagree[:25]:
            print(f"  {bug}  auto={auto:<18} reviewer={human:<18} {note[:60]}")
        if len(disagree) > 25:
            print(f"  ... and {len(disagree) - 25} more")

    out = os.path.join(ROOT, "results", "review_agreement.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"n_sample": len(rows), "n_reviewed": len(reviewed),
                   "raw_agreement": agree / len(reviewed), "cohen_kappa": k,
                   "confusion": matrix,
                   "disagreements": [{"bug_id": b, "automatic": a, "reviewer": h, "note": n}
                                     for b, a, h, n in disagree]}, fh, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
