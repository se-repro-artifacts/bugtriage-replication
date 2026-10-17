"""Merge raw JSONL files into one dataset file: data/dataset.jsonl

Fields per bug:
  id, created, product, component, label_fine ("Product::Component"), label_coarse (product),
  summary, description (first comment, truncated), split (train/test by creation date),
  comp_changed (component or product ever changed), n_moves, initial_label (first filed
  Product::Component, recovered from history; equals final label if never changed)
"""
import json, os

D = os.path.join(os.path.dirname(__file__), "..", "data")
TEST_FROM = "2024-01-01"   # train: 2022-01-01..2023-12-31, test: 2024-01-01..2024-12-31


def main():
    desc = {}
    p = os.path.join(D, "first_comment.jsonl")
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            desc[r["id"]] = r["description"]
    ch = json.load(open(os.path.join(D, "changed_ids.json")))
    changed = set(ch["component"]) | set(ch["product"])
    hist = {}
    p = os.path.join(D, "history.jsonl")
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            hist[r["id"]] = r["moves"]
    comps = json.load(open(os.path.join(D, "components.json")))
    # NOTE: component -> team mapping is BMO's *current* mapping (fetched once), not historical
    team = {f"{p['name']}::{c['name']}": c.get("team_name") or "?" for p in comps["products"]
            for c in p["components"]}
    n = 0
    with open(os.path.join(D, "dataset.jsonl"), "w") as f:
        for l in open(os.path.join(D, "bugs_meta.jsonl")):
            b = json.loads(l)
            final = f"{b['product']}::{b['component']}"
            moves = hist.get(b["id"], [])
            # recover initial product / component: the 'removed' value of the earliest change per field
            init_p, init_c = b["product"], b["component"]
            for fld in ("product", "component"):
                ms = [m for m in moves if m["field"] == fld]
                if ms:
                    first = min(ms, key=lambda m: m["when"])
                    if fld == "product":
                        init_p = first["removed"]
                    else:
                        init_c = first["removed"]
            # count "move events" = distinct timestamps where product/component changed
            n_moves = len({m["when"] for m in moves})
            rec = {
                "id": b["id"], "created": b["creation_time"], "product": b["product"],
                "component": b["component"], "label_fine": final, "label_coarse": b["product"],
                "summary": b["summary"], "description": desc.get(b["id"]),
                "split": "test" if b["creation_time"] >= TEST_FROM else "train",
                "comp_changed": b["id"] in changed, "n_moves": n_moves,
                "initial_label": f"{init_p}::{init_c}", "initial_product": init_p,
                "history_fetched": b["id"] in hist,
                "label_team": team.get(final, "?"),
                "any_human_move": any(not m.get("by_bot", False) for m in moves),
                "any_bot_move": any(m.get("by_bot", False) for m in moves),
                "initial_team": team.get(f"{init_p}::{init_c}", "?"),
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    print("wrote", n, "bugs; with description:", sum(1 for v in desc.values() if v), "; history:", len(hist))


if __name__ == "__main__":
    main()
