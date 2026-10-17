"""Descriptive statistics of component / product / team reassignment in data/dataset.jsonl."""
import json, os
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..")


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "dataset.jsonl"))]
    n = len(rows)
    out = {"n_bugs": n,
           "by_split": dict(Counter(r["split"] for r in rows)),
           "by_product": dict(Counter(r["product"] for r in rows)),
           "with_description": sum(1 for r in rows if r.get("description")),
           "n_fine_labels": len({r["label_fine"] for r in rows}),
           "n_team_labels": len({r["label_team"] for r in rows}),
           "team_unknown": sum(r["label_team"] == "?" for r in rows)}
    ch = [r for r in rows if r["comp_changed"]]
    out["ever_changed_component_or_product"] = {"n": len(ch), "pct": pct(len(ch), n)}
    out["history_fetched_for_changed"] = sum(r["history_fetched"] for r in ch)
    fine_moved = [r for r in rows if r["initial_label"] != r["label_fine"]]
    team_moved = [r for r in rows if r["initial_team"] != r["label_team"]
                  and "?" not in (r["initial_team"], r["label_team"])]
    prod_moved = [r for r in rows if r["initial_product"] != r["product"]]
    out["initial_ne_final"] = {
        "component": {"n": len(fine_moved), "pct": pct(len(fine_moved), n)},
        "team (both known)": {"n": len(team_moved), "pct": pct(len(team_moved), n)},
        "initial_team_unknown_among_changed": sum(r["initial_team"] == "?" for r in ch),
        "product": {"n": len(prod_moved), "pct": pct(len(prod_moved), n)}}
    out["changed_by_split"] = {s: pct(sum(r["comp_changed"] for r in rows if r["split"] == s),
                                      sum(r["split"] == s for r in rows)) for s in ("train", "test")}
    out["changed_by_final_product"] = {p: pct(sum(r["comp_changed"] for r in rows if r["product"] == p),
                                              sum(r["product"] == p for r in rows))
                                       for p in out["by_product"]}
    out["n_move_events_among_changed"] = dict(sorted(Counter(min(r["n_moves"], 5) for r in ch).items()))
    out["bot_vs_human_among_changed"] = {
        "only_bot_moves": sum(r["any_bot_move"] and not r["any_human_move"] for r in ch),
        "any_human_move": sum(r["any_human_move"] for r in ch),
        "any_bot_move": sum(r["any_bot_move"] for r in ch)}
    out["top_initial_components_of_moved"] = Counter(r["initial_label"] for r in fine_moved).most_common(12)
    out["top_moves(initial->final)"] = [[a, b, c] for (a, b), c in
                                        Counter((r["initial_label"], r["label_fine"]) for r in fine_moved).most_common(12)]
    generic = [r for r in fine_moved if r["initial_label"].split("::")[1] in ("General", "Untriaged")]
    out["moved_from_General_or_Untriaged"] = {"n": len(generic), "pct_of_moved": pct(len(generic), len(fine_moved))}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "reassign_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
