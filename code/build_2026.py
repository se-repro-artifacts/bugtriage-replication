"""Build the contamination-free 2026 test period from the raw data/2026/ JSONL files.

Same schema as data/dataset.jsonl (see code/build_dataset.py), but every record is a test record
with split="test2026", the raw inputs come from data/2026/, and the component -> team mapping is
the one fetched together with that period (data/2026/components.json).

Why a separate period: bug reports created in 2026 may fall after an LLM's knowledge cutoff, so
they make a cleaner test set than 2024; the 2024 split stays available for comparison.

Fields per bug: id, created, product, component, label_fine, label_coarse, summary, description,
split, comp_changed, n_moves, initial_label, initial_product, history_fetched, label_team,
any_human_move, any_bot_move, initial_team. Identical to build_dataset.py, so one evaluator can
read either file.

Usage: python build_2026.py [--raw data/2026] [--out data/dataset_2026.jsonl] [--split test2026]
"""
import argparse
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def read_jsonl(path, key="id"):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    out[rec[key]] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "2026"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "dataset_2026.jsonl"))
    ap.add_argument("--split", default="test2026")
    a = ap.parse_args()

    desc = read_jsonl(os.path.join(a.raw, "first_comment.jsonl"))
    hist = read_jsonl(os.path.join(a.raw, "history.jsonl"))
    with open(os.path.join(a.raw, "changed_ids.json"), encoding="utf-8") as fh:
        ch = json.load(fh)
    changed = set(ch["component"]) | set(ch["product"])
    with open(os.path.join(a.raw, "components.json"), encoding="utf-8") as fh:
        comps = json.load(fh)
    team = {f"{p['name']}::{c['name']}": c.get("team_name") or "?" for p in comps["products"]
            for c in p["components"]}

    n = 0
    no_team = 0
    with open(os.path.join(a.raw, "bugs_meta.jsonl"), encoding="utf-8") as src, \
            open(a.out, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            b = json.loads(line)
            final = f"{b['product']}::{b['component']}"
            moves = hist.get(b["id"], {}).get("moves", [])
            init_p, init_c = b["product"], b["component"]
            for field in ("product", "component"):
                events = [m for m in moves if m["field"] == field]
                if events:
                    first = min(events, key=lambda m: m["when"])
                    if field == "product":
                        init_p = first["removed"]
                    else:
                        init_c = first["removed"]
            if team.get(final, "?") == "?":
                no_team += 1
            rec = {
                "id": b["id"], "created": b["creation_time"], "product": b["product"],
                "component": b["component"], "label_fine": final, "label_coarse": b["product"],
                "summary": b["summary"], "description": desc.get(b["id"], {}).get("description"),
                "split": a.split,
                "comp_changed": b["id"] in changed,
                "n_moves": len({m["when"] for m in moves}),
                "initial_label": f"{init_p}::{init_c}", "initial_product": init_p,
                "history_fetched": b["id"] in hist,
                "label_team": team.get(final, "?"),
                "any_human_move": any(not m.get("by_bot", False) for m in moves),
                "any_bot_move": any(m.get("by_bot", False) for m in moves),
                "initial_team": team.get(f"{init_p}::{init_c}", "?"),
            }
            dst.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} bugs to {a.out}; {no_team} have no team for their final component; "
          f"{len(desc)} with description; {len(hist)} with history")


if __name__ == "__main__":
    main()
