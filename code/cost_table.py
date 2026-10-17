"""Apply list prices (USD per million tokens, copied from platform.claude.com/docs/en/about-claude/pricing
on 2026-10-10; re-check before use) to the token totals in results/cost_*.json. No API calls."""
import json, os
ROOT = os.path.join(os.path.dirname(__file__), "..")
PRICES = {"Claude Haiku 5.5 (<=100k prompt)": (0.10, 0.50), "Claude Sonnet 5.5": (2.0, 10.0)}
rows = []
for tag in ("team_top20", "fine_top30"):
    c = json.load(open(os.path.join(ROOT, "results", f"cost_{tag}.json")))
    t = c["total_test_set_Mtok"]
    for name, (pi, po) in PRICES.items():
        for setup in ("rag", "agent"):
            full = t[f"{setup}_in"] * pi + t[f"{setup}_out"] * po
            rows.append({"labels": tag, "n_test": c["n_test"], "model": name, "setup": setup,
                         "in_Mtok": round(t[f"{setup}_in"], 2), "out_Mtok": round(t[f"{setup}_out"], 2),
                         "usd_full_test_no_cache": round(full, 2), "usd_batch_50pct": round(full / 2, 2),
                         "usd_per_1000_bugs_no_cache": round(full * 1000 / c["n_test"], 2)})
for r in rows: print(r)
json.dump(rows, open(os.path.join(ROOT, "results", "cost_table.json"), "w"), indent=1)
