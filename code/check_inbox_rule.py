"""Check the `inbox_triage` rule for false positives.

`label_cleaning.category_of` decides a bug is inbox triage when the *name* of its initial component
is `General` or `Untriaged`, regardless of product. That is right for the products where those names
are the triage inbox (`Firefox::Untriaged`, `Firefox::General`), but `Core`, `DevTools`, `Toolkit`
and several others have a concrete component called `General` as well. Those bugs are filed into a
real component and moved out of it by a human, which is exactly what `specific_reassign` means.

The distinction decides which bugs are analysed at all: inbox triage is excluded from the ambiguous
subgroup, so every misrouted bug is a bug missing from the paper's positive class.

Usage: python check_inbox_rule.py
"""
import json
import os
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..")
LABELED = os.path.join(ROOT, "data", "dataset_labeled.jsonl")
# the products whose General/Untriaged really is a triage inbox rather than a component
INBOX_PRODUCTS = ("Firefox",)


def component_name(label):
    return (label or "::").split("::")[-1]


def product_name(label):
    return (label or "::").split("::")[0]


def main():
    rows = [json.loads(line) for line in open(LABELED, encoding="utf-8") if line.strip()]
    per_split = {}
    print(f"{'split':10} {'n':>7} {'specific':>9} {'inbox':>7} {'firefox-inbox':>14} "
          f"{'suspect':>8} {'corrected specific':>19}")
    for split in ("train", "test", "test2026"):
        sub = [r for r in rows if r.get("split") == split]
        cats = Counter(r.get("category") for r in sub)
        inbox = [r for r in sub if r.get("category") == "inbox_triage"]
        firefox = [r for r in inbox if product_name(r.get("initial_label")) in INBOX_PRODUCTS]
        suspect = [r for r in inbox if component_name(r.get("initial_label"))
                   in ("General", "Untriaged")
                   and product_name(r.get("initial_label")) not in INBOX_PRODUCTS]
        corrected = cats["specific_reassign"] + len(suspect)
        per_split[split] = {"n": len(sub), "specific_reassign": cats["specific_reassign"],
                            "inbox_triage": cats["inbox_triage"], "firefox_inbox": len(firefox),
                            "suspect_general": len(suspect), "corrected_specific": corrected}
        print(f"{split:10} {len(sub):>7} {cats['specific_reassign']:>9} {cats['inbox_triage']:>7} "
              f"{len(firefox):>14} {len(suspect):>8} {corrected:>19}")
    print()
    suspect_products = Counter(product_name(r.get("initial_label")) for r in rows
                               if r.get("category") == "inbox_triage"
                               and component_name(r.get("initial_label")) in ("General", "Untriaged")
                               and product_name(r.get("initial_label")) not in INBOX_PRODUCTS)
    print("products contributing the suspects:")
    for product, n in suspect_products.most_common():
        print(f"   {product:24} {n}")
    print()
    print("The author reviewed two such bugs on 2026-10-16 --- 1760880 (GeckoView::General) and")
    print("1790055 (Core::General) --- and kept both as `inbox_triage`, so the conservative reading")
    print("stands and these bugs stay out of the positive class. Recorded below for the paper.")

    out_path = os.path.join(ROOT, "results", "inbox_rule.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"per_split": per_split, "suspect_products": dict(suspect_products),
                   "decision": {"date": "2026-10-16",
                                "reviewed_bugs": [1760880, 1790055],
                                "reviewer_category": "inbox_triage",
                                "reading": "conservative: keep them out of the positive class",
                                "alternative_effect": "would enlarge the 2026 positive class by "
                                                      "about a tenth"}},
                  fh, indent=1, ensure_ascii=False)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
