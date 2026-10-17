"""Clean the reassignment-based ownership-ambiguity weak labels, and emit a manual-review sample.

The weak label for "this bug's ownership was ambiguous" is the reassignment recorded in
Bugzilla's history. Not every reassignment is evidence of ambiguity, so every bug is placed in
exactly one category:

  never_changed      component and product never changed -> the unambiguous negative class
  inbox_triage       moved out of General / Untriaged -> a triage step, weak evidence
  specific_reassign  moved away from a concrete component -> the ambiguity candidates we care about
  bot_only           every recorded move was by an automation account -> excluded, because a bot
                     move carries no evidence of human disagreement

Writes
  data/dataset_labeled.jsonl        input rows plus `category`, `usable` and `moved_from_specific`
  results/label_categories.json     category counts, overall and per split
  results/manual_review_sample.csv  stratified sample for human inspection (UTF-8 with BOM so
                                    Excel on this machine opens it correctly); the final two
                                    columns are left blank for the reviewer
  results/manual_review_sample.md   the same sample as a Markdown table

Usage: python label_cleaning.py --dataset data/dataset.jsonl --n 150
       python label_cleaning.py --dataset data/dataset.jsonl --dataset data/dataset_2026.jsonl
"""
import argparse
import csv
import json
import os
import random

ROOT = os.path.join(os.path.dirname(__file__), "..")
INBOX = ("General", "Untriaged")
CATEGORIES = ["never_changed", "inbox_triage", "specific_reassign", "bot_only"]


def component_of(label):
    return (label or "::").split("::")[-1]


def category_of(row):
    """Which kind of ownership event this bug records.

    The routing target is the team of the *final* component, so a move only matters when the
    component itself changed. The dataset's `comp_changed` flag is true when either the component
    or the product changed, so a bug that only changed product would otherwise be misread as a
    component reassignment (about 2% of the candidates).
    """
    initial = component_of(row.get("initial_label"))
    final = component_of(row.get("label_fine"))
    if initial == final:
        return "never_changed"
    if not row.get("any_human_move") and row.get("any_bot_move"):
        return "bot_only"
    return "inbox_triage" if initial in INBOX else "specific_reassign"


def moved_from_specific(row):
    return (category_of(row) == "specific_reassign"
            and row.get("initial_label") != row.get("label_fine"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--out-labeled", default=os.path.join(ROOT, "data", "dataset_labeled.jsonl"))
    ap.add_argument("--n", type=int, default=150, help="manual-review sample size")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = []
    for path in a.dataset:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    for row in rows:
        row["category"] = category_of(row)
        row["moved_from_specific"] = moved_from_specific(row)
        # bot_only is excluded; the other three are usable, with specific_reassign the positive class
        row["usable"] = row["category"] != "bot_only"

    counts = {}
    for row in rows:
        bucket = counts.setdefault(row["split"], {c: 0 for c in CATEGORIES})
        bucket[row["category"]] += 1
        bucket["total"] = bucket.get("total", 0) + 1
    summary = {"datasets": a.dataset, "n_total": len(rows), "by_split": counts}
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "label_categories.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    with open(a.out_labeled, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print("wrote", a.out_labeled)

    # stratified review sample: emphasise the two categories that carry the ambiguity signal
    rng = random.Random(a.seed)
    quota = {"specific_reassign": 0.45, "inbox_triage": 0.30, "never_changed": 0.15,
             "bot_only": 0.10}
    picked = []
    for cat, frac in quota.items():
        pool = [r for r in rows if r["category"] == cat]
        take = min(len(pool), max(1, round(a.n * frac)))
        picked.extend(rng.sample(pool, take))
    rng.shuffle(picked)
    picked = picked[:a.n]

    csv_path = os.path.join(ROOT, "results", "manual_review_sample.csv")
    header = ["bug_id", "url", "split", "category", "created", "initial_component",
              "final_component", "n_moves", "any_human_move", "any_bot_move", "summary",
              "reviewer_category", "reviewer_notes"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in picked:
            writer.writerow([
                row["id"], f"https://bugzilla.mozilla.org/show_bug.cgi?id={row['id']}",
                row["split"], row["category"], row["created"],
                row.get("initial_label"), row.get("label_fine"), row.get("n_moves"),
                row.get("any_human_move"), row.get("any_bot_move"),
                (row.get("summary") or "").replace("\n", " ")[:300], "", ""])
    print("wrote", csv_path)

    md_path = os.path.join(ROOT, "results", "manual_review_sample.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Manual review sample ({len(picked)} bugs)\n\n")
        fh.write("Fill in `reviewer_category` with one of: "
                 + ", ".join(f"`{c}`" for c in CATEGORIES) + ".\n\n")
        fh.write("| bug | url | split | auto category | initial component | final component | "
                 "n_moves | human move | bot move | summary |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for row in picked:
            summary_text = (row.get("summary") or "").replace("|", "\\|")[:160]
            fh.write(f"| {row['id']} | https://bugzilla.mozilla.org/show_bug.cgi?id={row['id']} | "
                     f"{row['split']} | {row['category']} | {row.get('initial_label')} | "
                     f"{row.get('label_fine')} | {row.get('n_moves')} | "
                     f"{row.get('any_human_move')} | {row.get('any_bot_move')} | {summary_text} |\n")
    print("wrote", md_path)


if __name__ == "__main__":
    main()
