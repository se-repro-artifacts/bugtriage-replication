"""Classical and strong baselines for bug -> team routing.

Extends code/baselines.py (which stays the untouched record of the step-1 numbers) with:
  * the ability to score any train/test split pair from one dataset file, so the 2024 and the
    2026 test periods are evaluated against the same label space with the same fitted vocabulary;
  * a second strong baseline family: sentence-embedding + kNN and XGBoost;
  * split-conformal prediction sets with empirical coverage and mean set size;
  * selective-prediction numbers in exactly the form evaluate.py reports for the LLM conditions.

Every method is written out as a prediction file in the SAME record schema as run_llm.py, so
evaluate.py scores a classical model and an LLM with one code path and the risk-coverage curves
are directly comparable by construction.

Usage:
  python baselines_ext.py --dataset data/dataset.jsonl --train-split train --test-split test
  python baselines_ext.py --dataset data/dataset_all.jsonl --test-split test2026
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import keyword_model, text_of, tfidf          # reuse the step-1 definitions verbatim

ROOT = os.path.join(os.path.dirname(__file__), "..")
LABEL_KEYS = {"fine": "label_fine", "team": "label_team", "coarse": "label_coarse"}


def load_split(dataset, label, topn, train_split, test_split, text_mode, max_created=None):
    with open(dataset, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    key = LABEL_KEYS[label]
    counts = {}
    for r in rows:
        if r["split"] == train_split and r[key] not in ("?", "Other"):
            counts[r[key]] = counts.get(r[key], 0) + 1
    classes = [c for c, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:topn] if topn else sorted(counts)
    keep = set(classes)
    train = [r for r in rows if r["split"] == train_split and r[key] in keep]
    # max_created keeps only bugs filed long enough before the data freeze that their
    # component/product can be treated as final (see code/label_stability.py)
    test = [r for r in rows if r["split"] == test_split and r[key] in keep
            and (max_created is None or r["created"][:10] <= max_created)]
    return classes, train, test, key


def emit(path, test, ranked, conf, condition, classes, extra=None):
    """Write predictions in run_llm.py's record schema so evaluate.py can score them."""
    with open(path, "w", encoding="utf-8") as fh:
        for row, rank, c in zip(test, ranked, conf):
            rec = {
                "id": row["id"], "split": row["split"], "created": row["created"],
                "true_team": row.get(LABEL_KEYS["team"]), "true_fine": row.get("label_fine"),
                "comp_changed": row.get("comp_changed"), "initial_label": row.get("initial_label"),
                "condition": condition, "model": "classical", "seed": None,
                "top3": list(rank[:3]), "abstain": None, "confidence": float(c),
                "evidence": "", "parse_ok": True, "unknown_labels": [],
                "n_tool_calls": 0, "tool_calls": [], "rounds": 1,
                "prompt_tokens": 0, "completion_tokens": 0, "cached_rounds": 0,
                "latency_s": 0.0, "error": None,
            }
            if extra:
                rec.update(extra(row))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def scores_to_ranked(scores, classes):
    """scores: (n, n_classes) -> (ranked list-of-lists, max-prob confidence)."""
    order = np.argsort(-scores, axis=1)
    ranked = [[classes[j] for j in row] for row in order]
    return ranked, scores.max(axis=1)


def summary_row(y, ranked, conf, classes):
    correct = np.array([rank[0] == t for rank, t in zip(ranked, y)])
    out = {"n": len(y), "acc": float(correct.mean()),
           "macro_f1": float(f1_score(y, [r[0] for r in ranked], labels=sorted(set(y)),
                                      average="macro", zero_division=0)),
           "top3": float(np.mean([t in r[:3] for r, t in zip(ranked, y)]))}
    order = np.argsort(-np.asarray(conf, dtype=float), kind="stable")
    if len(order):                                # AURC_risk and coverage points, as in evaluate.py
        cum = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
        out["AURC_risk"] = float(np.mean(1 - cum))
        for cov in (0.5, 0.7, 0.8, 0.9):
            k = max(1, int(round(cov * len(order))))
            out[f"selective_acc_cov{int(cov * 100)}"] = float(cum[k - 1])
    return out


def conformal(proba, y_val, proba_test, y_test, alphas=(0.05, 0.10, 0.20)):
    """Split-conformal prediction sets from the classifier's probabilities."""
    out = {}
    p_true_val = proba[np.arange(len(y_val)), y_val]
    scores = 1.0 - p_true_val
    for alpha in alphas:
        n = len(scores)
        q = np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
        sets = [np.where(row >= 1.0 - q)[0] for row in proba_test]
        sizes = np.array([len(s) for s in sets])
        covered = np.array([int(y_test[i]) in sets[i] for i in range(len(y_test))])
        out[f"alpha{alpha:.2f}"] = {
            "target_coverage": 1 - alpha,
            "empirical_coverage": float(covered.mean()),
            "mean_set_size": float(sizes.mean()),
            "singleton_rate": float((sizes == 1).mean()),
            "empty_rate": float((sizes == 0).mean()),
            "threshold": float(q),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "dataset.jsonl"))
    ap.add_argument("--label", default="team", choices=["fine", "team", "coarse"])
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--text", default="sd", choices=["s", "sd"])
    ap.add_argument("--skip-optional", action="store_true", help="skip embedding / XGBoost")
    ap.add_argument("--max-created", default=None,
                    help="keep only test bugs created on or before this date (YYYY-MM-DD)")
    a = ap.parse_args()

    classes, train, test, key = load_split(a.dataset, a.label, a.topn, a.train_split, a.test_split,
                                           a.text, a.max_created)
    y_tr = [r[key] for r in train]
    y_te = [r[key] for r in test]
    Xtr_txt = [text_of(r, a.text) for r in train]
    Xte_txt = [text_of(r, a.text) for r in test]
    print(f"{a.dataset}: train={len(train)} test={len(test)} classes={len(classes)}", flush=True)

    tag = a.test_split + (f"_settled{a.max_created}" if a.max_created else "")
    results = {"dataset": a.dataset, "label": a.label, "topn": a.topn,
               "train_split": a.train_split, "test_split": a.test_split, "text": a.text,
               "max_created": a.max_created, "tag": tag,
               "n_train": len(train), "n_test": len(test), "classes": classes, "methods": {}}
    prefix = os.path.join(ROOT, "results", f"pred_{{m}}_{tag}.jsonl")

    # --- majority -----------------------------------------------------------
    prior = {}
    for label in y_tr:
        prior[label] = prior.get(label, 0) + 1
    majority = [c for c, _ in sorted(prior.items(), key=lambda kv: -kv[1])]
    ranked = [majority] * len(test)
    conf = [prior[majority[0]] / len(y_tr)] * len(test)
    results["methods"]["majority"] = summary_row(y_te, ranked, conf, classes)
    emit(prefix.format(m="majority"), test, ranked, conf, "majority", classes)

    # --- keyword rules ------------------------------------------------------
    cv, kw = keyword_model(Xtr_txt, y_tr, classes)
    analyzer = cv.build_analyzer()
    ranked, conf = [], []
    for text in Xte_txt:
        tokens = set(analyzer(text))
        scored = sorted(((sum(w in tokens for w in kw[c]), prior.get(c, 0), c) for c in classes),
                        reverse=True)
        ranked.append([c for _, _, c in scored])
        total = sum(s for s, _, _ in scored) or 1
        conf.append(scored[0][0] / total)
    results["methods"]["keyword_rules"] = summary_row(y_te, ranked, conf, classes)
    emit(prefix.format(m="keyword_rules"), test, ranked, conf, "keyword_rules", classes)

    # --- TF-IDF + logistic regression (C tuned on the last quarter of train) --
    val_mask = np.array([r["created"] >= "2023-10-01" for r in train])
    fit_idx, val_idx = np.where(~val_mask)[0], np.where(val_mask)[0]
    vec = tfidf()
    Xf = vec.fit_transform([Xtr_txt[i] for i in fit_idx])
    Xv = vec.transform([Xtr_txt[i] for i in val_idx])
    y_fit = np.asarray(y_tr)[fit_idx]
    y_val = np.asarray(y_tr)[val_idx]
    best = None
    for C in (1.0, 4.0, 16.0):
        model = LogisticRegression(C=C, max_iter=3000).fit(Xf, y_fit)
        acc = float((model.predict(Xv) == y_val).mean())
        print(f"  C={C} val acc={acc:.4f}", flush=True)
        if best is None or acc > best[1]:
            best = (C, acc)
    results["lr_C"], results["lr_val_acc"] = best
    vec = tfidf()
    Xtr = vec.fit_transform(Xtr_txt)
    Xte = vec.transform(Xte_txt)
    lr = LogisticRegression(C=best[0], max_iter=3000).fit(Xtr, y_tr)
    proba_te = lr.predict_proba(Xte)
    class_index = {c: i for i, c in enumerate(lr.classes_)}
    ranked, conf = scores_to_ranked(proba_te, list(lr.classes_))
    results["methods"]["tfidf_logreg"] = summary_row(y_te, ranked, conf, classes)
    emit(prefix.format(m="tfidf_logreg"), test, ranked, conf, "tfidf_logreg", classes)

    # --- conformal sets from the same probabilities -------------------------
    Xfit = vec.transform([Xtr_txt[i] for i in fit_idx])
    Xval = vec.transform([Xtr_txt[i] for i in val_idx])
    model_val = LogisticRegression(C=best[0], max_iter=3000).fit(Xfit, y_fit)
    proba_val = np.zeros((Xval.shape[0], len(lr.classes_)))
    proba_val[:, [class_index[c] for c in model_val.classes_]] = model_val.predict_proba(Xval)
    y_val_idx = np.array([class_index[v] for v in y_val])
    results["conformal"] = conformal(proba_val, y_val_idx, proba_te,
                                     np.array([class_index[v] for v in y_te]))
    print("  conformal:", json.dumps(results["conformal"]), flush=True)

    # --- TF-IDF + kNN -------------------------------------------------------
    Xn = normalize(Xtr)
    Xq = normalize(Xte)
    K = 20
    ranked, conf = [], []
    for i in range(Xq.shape[0]):
        sims = np.asarray((Xn @ Xq[i].T).todense()).ravel()
        top = np.argsort(-sims)[:K]
        votes = {}
        for j in top:
            votes[y_tr[j]] = votes.get(y_tr[j], 0.0) + float(sims[j])
        order = sorted(votes.items(), key=lambda kv: -kv[1])
        ranked.append([c for c, _ in order])
        total = sum(v for _, v in order) or 1.0
        conf.append(order[0][1] / total)
    results["methods"]["tfidf_knn"] = summary_row(y_te, ranked, conf, classes)
    emit(prefix.format(m="tfidf_knn"), test, ranked, conf, "tfidf_knn", classes)

    # --- optional: sentence embeddings + kNN, and XGBoost -------------------
    if not a.skip_optional:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            Etr = encoder.encode(Xtr_txt, batch_size=64, show_progress_bar=False,
                                 normalize_embeddings=True)
            Ete = encoder.encode(Xte_txt, batch_size=64, show_progress_bar=False,
                                 normalize_embeddings=True)
            ranked, conf = [], []
            for i in range(Ete.shape[0]):
                sims = Etr @ Ete[i]
                top = np.argsort(-sims)[:K]
                votes = {}
                for j in top:
                    votes[y_tr[j]] = votes.get(y_tr[j], 0.0) + float(sims[j])
                order = sorted(votes.items(), key=lambda kv: -kv[1])
                ranked.append([c for c, _ in order])
                total = sum(v for _, v in order) or 1.0
                conf.append(order[0][1] / total)
            results["methods"]["embed_knn"] = summary_row(y_te, ranked, conf, classes)
            results["embed_model"] = "sentence-transformers/all-MiniLM-L6-v2"
            emit(prefix.format(m="embed_knn"), test, ranked, conf, "embed_knn", classes)
        except Exception as exc:  # noqa: BLE001
            print("  embed_knn skipped:", repr(exc), flush=True)
            results["methods"]["embed_knn"] = {"skipped": repr(exc)}

        try:
            from xgboost import XGBClassifier
            t0 = time.time()
            xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                                n_jobs=4, objective="multi:softprob",
                                num_class=len(classes))
            labels_idx = {c: i for i, c in enumerate(classes)}
            xgb.fit(Xtr, np.array([labels_idx[v] for v in y_tr]))
            proba = xgb.predict_proba(Xte)
            ranked, conf = scores_to_ranked(proba, classes)
            results["methods"]["xgboost"] = summary_row(y_te, ranked, conf, classes)
            results["methods"]["xgboost"]["fit_seconds"] = round(time.time() - t0, 1)
            emit(prefix.format(m="xgboost"), test, ranked, conf, "xgboost", classes)
        except Exception as exc:  # noqa: BLE001
            print("  xgboost skipped:", repr(exc), flush=True)
            results["methods"]["xgboost"] = {"skipped": repr(exc)}

    out_path = os.path.join(ROOT, "results", f"baselines_ext_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps(results["methods"], indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
