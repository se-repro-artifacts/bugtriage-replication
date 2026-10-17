"""Generate the paper's figures from the recorded predictions. Nothing is drawn by hand.

Figure 1  risk-coverage curves: accuracy as a function of how much of the test set the model
          answers, ordered by the model's own confidence. This is the comparison the paper is
          built on, and it is drawn from the same definitions evaluate.py scores with.
Figure 2  accuracy and confidence-based abstention by ownership-ambiguity category.

Every input is a prediction file in the shared record schema, so a classical baseline and an LLM
condition land on the same axes without any conversion.

Usage: python make_figures.py --split test2026
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")                      # headless: no display on this machine
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGDIR = os.path.join(ROOT, "paper", "figures")
CATEGORIES = ["never_changed", "inbox_triage", "specific_reassign", "bot_only"]
CAT_LABEL = {"never_changed": "never changed", "inbox_triage": "inbox triage",
             "specific_reassign": "specific reassign", "bot_only": "bot only (excluded)"}


def read_records(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def discover(split):
    """Every prediction file for this split, mock runs excluded."""
    found = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "*.jsonl"))):
        base = os.path.basename(path)
        if "_mock" in base:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                first = json.loads(fh.readline())
        except Exception:
            continue
        if first.get("split") != split:
            continue
        name = first.get("condition") or base
        model = first.get("model") or ""
        label = name if not model or model == "classical" else f"{name} ({model})"
        found.append((label, path))
    return found


def curve(conf, correct, grid):
    order = np.argsort(-np.asarray(conf, dtype=float), kind="stable")
    ordered = np.asarray(correct, dtype=float)[order]
    n = len(ordered)
    cum = np.cumsum(ordered) / np.arange(1, n + 1)
    ks = np.maximum(1, np.round(grid * n).astype(int))
    return cum[ks - 1]


def figure_risk_coverage(split, series, grid):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for label, (coverage, accuracy) in sorted(series.items()):
        aurc = float(np.mean(1.0 - accuracy))
        ax.plot(coverage, accuracy, linewidth=1.6, label=f"{label} (AURC={aurc:.3f})")
    ax.set_xlabel("coverage (fraction of bugs the model answers)")
    ax.set_ylabel("accuracy on the answered bugs")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"Risk-coverage, {split} split")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"risk_coverage_{split}.{ext}"), dpi=160)
    plt.close(fig)
    print(f"wrote risk_coverage_{split}.pdf/.png")


def figure_categories(split, cats_by_label):
    labels = [l for l in cats_by_label]
    present = [c for c in CATEGORIES if any(c in cats_by_label[l] for l in labels)]
    if not present:
        print("no ambiguity categories found; skipping figure 2")
        return
    x = np.arange(len(present))
    width = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, label in enumerate(labels):
        accs = [cats_by_label[label].get(c, {}).get("acc") for c in present]
        vals = [0 if a is None else 100 * a for a in accs]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABEL.get(c, c) for c in present], fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Accuracy by ownership-ambiguity category, {split} split")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"ambiguity_accuracy_{split}.{ext}"), dpi=160)
    plt.close(fig)
    print(f"wrote ambiguity_accuracy_{split}.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test2026")
    a = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)

    grid = np.round(np.arange(0.05, 1.0001, 0.01), 4)
    series, cats_by_label = {}, {}
    for label, path in discover(a.split):
        recs = read_records(path)
        if not recs:
            continue
        correct = np.array([bool(r.get("top3")) and r["top3"][0] == r["true_team"] for r in recs])
        conf = np.array([r.get("confidence") if r.get("confidence") is not None else 0.0
                         for r in recs], dtype=float)
        series[label] = (grid, curve(conf, correct, grid))

        stem = os.path.basename(path).replace("llm_", "").replace("pred_", "")
        if stem.endswith(".jsonl"):
            stem = stem[:-6]
        metrics_path = os.path.join(ROOT, "results", "metrics_" + stem + ".json")
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as fh:
                m = json.load(fh)
            cats = (m.get("metrics") or {}).get("by_ambiguity_category")
            if cats:
                cats_by_label[label] = cats

    if not series:
        print(f"no prediction files for split {a.split}; nothing to draw")
        return
    figure_risk_coverage(a.split, series, grid)
    figure_categories(a.split, cats_by_label)


if __name__ == "__main__":
    main()
