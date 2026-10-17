"""Sentence-embedding + kNN baseline, kept separate so it can be retried on its own.

Windows refuses the symlinks that huggingface_hub creates by default, which is why the embedding
baseline is skipped inside baselines_ext.py. Run this script with symlinks disabled:

  HF_HUB_DISABLE_SYMLINKS=1 python baseline_embed.py --dataset data/dataset_all.jsonl \
      --test-split test2026

It encodes the train and test splits with a sentence encoder, lets the k nearest train bugs vote
by cosine similarity, then merges its metrics into the same results/baselines_ext_<split>.json
that baselines_ext.py writes and emits a prediction file in the shared record schema, so
evaluate.py scores it exactly like every other condition.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baselines_ext import emit, load_split, summary_row, text_of   # noqa: E402

ROOT = os.path.join(HERE, "..")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "dataset_all.jsonl"))
    ap.add_argument("--label", default="team")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test2026")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    from sentence_transformers import SentenceTransformer

    classes, train, test, key = load_split(a.dataset, a.label, a.topn, a.train_split, a.test_split,
                                           "sd")
    y_tr = [r[key] for r in train]
    print(f"{a.dataset}: train={len(train)} test={len(test)} classes={len(classes)}", flush=True)

    encoder = SentenceTransformer(MODEL)
    Etr = encoder.encode([text_of(r, "sd") for r in train], batch_size=a.batch,
                         show_progress_bar=False, normalize_embeddings=True)
    Ete = encoder.encode([text_of(r, "sd") for r in test], batch_size=a.batch,
                         show_progress_bar=False, normalize_embeddings=True)
    Etr = normalize(np.asarray(Etr, dtype=np.float32))

    ranked, conf = [], []
    for i in range(Ete.shape[0]):
        sims = Etr @ np.asarray(Ete[i], dtype=np.float32)
        top = np.argsort(-sims)[:a.k]
        votes = {}
        for j in top:
            votes[y_tr[j]] = votes.get(y_tr[j], 0.0) + float(sims[j])
        order = sorted(votes.items(), key=lambda kv: -kv[1])
        ranked.append([c for c, _ in order])
        total = sum(v for _, v in order) or 1.0
        conf.append(order[0][1] / total)

    metrics = summary_row([r[key] for r in test], ranked, conf, classes)
    metrics["embedding_model"] = MODEL
    metrics["k"] = a.k
    print(json.dumps(metrics, indent=1), flush=True)

    pred_path = os.path.join(ROOT, "results", f"pred_embed_knn_{a.test_split}.jsonl")
    emit(pred_path, test, ranked, conf, "embed_knn", classes)
    print("wrote", pred_path)

    out_path = os.path.join(ROOT, "results", f"baselines_ext_{a.test_split}.json")
    data = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            data = json.load(fh)
    data.setdefault("methods", {})["embed_knn"] = metrics
    data["embed_model"] = MODEL
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print("merged embed_knn into", out_path)


if __name__ == "__main__":
    main()
