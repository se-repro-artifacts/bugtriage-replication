"""Baseline: trust the component the reporter filed the bug under.

At filing time the reporter picks a component, and the team that owns it is a signal every real
routing system already has. Predicting it directly is correct for every bug that was never
reassigned, which makes it a strong baseline that the paper has to report: the step-1 report put
it at 78.7% on the team task for 2022-2024, well above TF-IDF + logistic regression.

Two caveats the paper should state. The initial component is *not* a human baseline: the reporter
is often a developer or an automation account, not a triager. And it is unavailable for bugs filed
into an inbox component whose team falls outside the label space; those fall back to the majority
class, which is what drags the number down.

Writes results/pred_initial_team_<split>.jsonl in the shared record schema, so evaluate.py scores
it exactly like every other condition.

Usage: python baseline_initial.py --dataset data/dataset_all.jsonl --test-split test2026
"""
import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baselines_ext import emit, load_split, summary_row   # noqa: E402

ROOT = os.path.join(HERE, "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "dataset_all.jsonl"))
    ap.add_argument("--label", default="team")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test2026")
    ap.add_argument("--max-created", default=None)
    a = ap.parse_args()

    classes, train, test, key = load_split(a.dataset, a.label, a.topn, a.train_split, a.test_split,
                                           "sd", a.max_created)
    keep = set(classes)
    prior = Counter(r[key] for r in train)
    order = [c for c, _ in prior.most_common()]
    majority = order[0]
    n_train = len(train)
    print(f"{a.dataset}: train={n_train} test={len(test)} classes={len(classes)}; "
          f"majority={majority}", flush=True)

    ranked, conf, in_space = [], [], []
    for row in test:
        initial = row.get("initial_team")
        hit = initial in keep
        in_space.append(hit)
        pred = initial if hit else majority
        ranked.append([pred] + [c for c in order if c != pred])
        conf.append(prior.get(pred, 0) / n_train)

    metrics = summary_row([r[key] for r in test], ranked, conf, classes)
    metrics["initial_team_in_label_space_rate"] = float(sum(in_space) / max(1, len(in_space)))
    print(json_dumps(metrics), flush=True)

    out = os.path.join(ROOT, "results",
                       f"pred_initial_team_{a.test_split}"
                       + (f"_settled{a.max_created}" if a.max_created else "") + ".jsonl")
    emit(out, test, ranked, conf, "initial_team", classes)
    print("wrote", out)


def json_dumps(obj):
    import json
    return json.dumps(obj, indent=1)


if __name__ == "__main__":
    main()
