"""Cheap baselines for bug -> component / product routing on data/dataset.jsonl.

Baselines: majority class, keyword rules mined from train, TF-IDF + logistic regression.
Metrics: accuracy, macro-F1, top-3 accuracy, overall and split by comp_changed (ever reassigned).
Also: selective prediction (risk-coverage) using LR max probability, and an error analysis of
how often LR's wrong guess equals the component the reporter originally filed.

Usage: python baselines.py --label fine --topn 30 --text sd
       python baselines.py --label coarse --text sd
  --text s  = summary only; sd = summary + first comment (description)
"""
import argparse, json, os, re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
DESC_CHARS = 2000  # truncate description for the classical models
URL_RE = re.compile(r"https?://\S+")


def text_of(r, mode):
    t = r["summary"]
    if mode == "sd" and r.get("description"):
        t += "\n" + URL_RE.sub(" URL ", r["description"][:DESC_CHARS])
    return t


def load(label, topn, mode):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "dataset.jsonl"))]
    key = {"fine": "label_fine", "team": "label_team", "coarse": "label_coarse"}[label]
    tr_counts = Counter(r[key] for r in rows if r["split"] == "train" and r[key] not in ("?", "Other"))
    classes = [c for c, _ in tr_counts.most_common(topn)] if topn else sorted(tr_counts)
    cs = set(classes)
    stats = {"n_all": len(rows), "n_train_all": sum(r["split"] == "train" for r in rows),
             "n_test_all": sum(r["split"] == "test" for r in rows)}
    rows = [r for r in rows if r[key] in cs]
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    stats.update(n_train=len(tr), n_test=len(te), n_classes=len(classes))
    out = []
    for part in (tr, te):
        out.append(([text_of(r, mode) for r in part], [r[key] for r in part], part))
    return classes, out[0], out[1], stats, key


def metrics(y, ranked, classes, mask=None):
    """ranked: list of label lists (best first)."""
    y = np.asarray(y)
    pred = np.asarray([r[0] for r in ranked])
    top3 = np.asarray([yy in r[:3] for yy, r in zip(y, ranked)])
    if mask is not None:
        y, pred, top3 = y[mask], pred[mask], top3[mask]
    if len(y) == 0:
        return None
    present = sorted(set(y))  # macro-F1 over classes present in this (sub)set
    return {"n": int(len(y)), "acc": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
            "top3": float(top3.mean())}


def keyword_model(Xtr_txt, ytr, classes, k=10, min_df=3):
    cv = CountVectorizer(binary=True, min_df=min_df, stop_words="english",
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b")
    X = cv.fit_transform(Xtr_txt).tocsc()
    vocab = np.asarray(cv.get_feature_names_out())
    y = np.asarray(ytr)
    N = len(y)
    n_t = np.asarray(X.sum(axis=0)).ravel()
    kw = {}
    for c in classes:
        m = y == c
        Nc = m.sum()
        n_ct = np.asarray(X[m].sum(axis=0)).ravel()
        lo = np.log((n_ct + 1) / (Nc + 2)) - np.log((n_t - n_ct + 1) / (N - Nc + 2))
        lo[n_ct < 3] = -np.inf
        kw[c] = list(vocab[np.argsort(-lo)[:k]])
    return cv, kw


def keyword_predict(cv, kw, texts, classes, prior):
    an = cv.build_analyzer()
    out = []
    for t in texts:
        toks = set(an(t))
        sc = [(sum(w in toks for w in kw[c]), prior[c], c) for c in classes]
        sc.sort(reverse=True)
        out.append([c for _, _, c in sc])
    return out


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w[\w\-\.]+\b", dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="fine", choices=["fine", "team", "coarse"])
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--text", default="sd", choices=["s", "sd"])
    a = ap.parse_args()
    topn = a.topn if a.label != "coarse" else 0
    classes, (Xtr, ytr, rtr), (Xte, yte, rte), stats, key = load(a.label, topn, a.text)
    tag = f"{a.label}{'_top' + str(topn) if topn else ''}_{a.text}"
    print(tag, stats)
    changed = np.asarray([r["comp_changed"] for r in rte])
    ikey0 = {"fine": "initial_label", "team": "initial_team", "coarse": "initial_product"}[a.label]
    moved = np.asarray([r[ikey0] != r[key] for r in rte])
    # "specific" move: initial component was a concrete component, not an inbox (General/Untriaged)
    spec = np.asarray([r["initial_label"] != r["label_fine"] and
                       r["initial_label"].split("::")[-1] not in ("General", "Untriaged") for r in rte])
    res = {"tag": tag, "stats": stats, "classes": classes,
           "test_reassign_rate": float(changed.mean()),
           "test_moved_at_this_granularity_rate": float(moved.mean()),
           "train_reassign_rate": float(np.mean([r["comp_changed"] for r in rtr])), "models": {}}

    def report(name, ranked):
        res["models"][name] = {"all": metrics(yte, ranked, classes),
                               "reassigned": metrics(yte, ranked, classes, changed),
                               "never_reassigned": metrics(yte, ranked, classes, ~changed),
                               "moved_at_this_granularity": metrics(yte, ranked, classes, moved),
                               "moved_from_specific_component": metrics(yte, ranked, classes, spec)}
        print(name, json.dumps(res["models"][name]))

    prior = Counter(ytr)
    maj = [c for c, _ in prior.most_common()]
    report("majority", [maj] * len(yte))

    cv, kw = keyword_model(Xtr, ytr, classes)
    report("keyword_rules", keyword_predict(cv, kw, Xte, classes, prior))
    res["keywords_sample"] = {c: kw[c] for c in classes[:8]}

    # tune C on a time-based validation slice (last quarter of the training period)
    vmask = np.asarray([r["created"] >= "2023-10-01" for r in rtr])
    idx_fit, idx_val = np.where(~vmask)[0], np.where(vmask)[0]
    vec = tfidf()
    Xf = vec.fit_transform([Xtr[i] for i in idx_fit])
    Xv = vec.transform([Xtr[i] for i in idx_val])
    yf, yv = np.asarray(ytr)[idx_fit], np.asarray(ytr)[idx_val]
    best = None
    for C in (1.0, 4.0, 16.0):
        lr = LogisticRegression(C=C, max_iter=3000)
        lr.fit(Xf, yf)
        acc = float((lr.predict(Xv) == yv).mean())
        print(f"  C={C} val acc={acc:.4f}")
        if best is None or acc > best[1]:
            best = (C, acc)
    res["lr_C"], res["lr_val_acc"] = best
    del Xf, Xv
    vec = tfidf()
    Xtr_m = vec.fit_transform(Xtr)
    Xte_m = vec.transform(Xte)
    lr = LogisticRegression(C=best[0], max_iter=3000).fit(Xtr_m, ytr)
    P = lr.predict_proba(Xte_m)
    order = np.argsort(-P, axis=1)
    lab = lr.classes_
    ranked = [[lab[j] for j in row] for row in order]
    report("tfidf_logreg", ranked)

    # selective prediction with max-probability confidence
    conf = P.max(axis=1)
    correct = np.asarray([r[0] == y for r, y in zip(ranked, yte)])
    srt = np.argsort(-conf)
    cum = np.cumsum(correct[srt]) / np.arange(1, len(srt) + 1)
    sel = {}
    for cov in (0.5, 0.7, 0.8, 0.9, 1.0):
        k = max(1, int(round(cov * len(srt))))
        sel[f"cov{int(cov * 100)}"] = {"selective_acc": float(cum[k - 1]),
                                      "frac_reassigned_in_kept": float(changed[srt[:k]].mean())}
    sel["AURC_risk"] = float(np.mean(1 - cum))
    # abstained share by group when keeping 80%
    k80 = int(round(0.8 * len(srt)))
    abst = np.ones(len(srt), bool); abst[srt[:k80]] = False
    sel["abstain_rate_at_cov80"] = {"reassigned": float(abst[changed].mean()),
                                    "moved_from_specific_component": float(abst[spec].mean()) if spec.any() else None,
                                    "never_reassigned": float(abst[~changed].mean())}
    res["selective_tfidf_logreg"] = sel
    print("selective", json.dumps(sel))

    # error analysis: is the wrong guess the component the reporter originally filed?
    ikey = {"fine": "initial_label", "team": "initial_team", "coarse": "initial_product"}[a.label]
    init = np.asarray([r[ikey] for r in rte])
    pred = np.asarray([r[0] for r in ranked])
    wrong = pred != np.asarray(yte)
    ea = {"n_wrong": int(wrong.sum()),
          "wrong_and_pred_eq_initial": int((wrong & (pred == init)).sum()),
          "wrong_among_reassigned": int((wrong & changed).sum()),
          "wrong_among_reassigned_pred_eq_initial": int((wrong & changed & (pred == init)).sum()),
          "reporter_initial_label_acc_on_test": float((init == np.asarray(yte)).mean())}
    pairs = Counter((y, p) for y, p, w in zip(yte, pred, wrong) if w)
    ea["top_confusions(true,pred,count)"] = [[y, p, c] for (y, p), c in pairs.most_common(12)]
    res["error_analysis"] = ea
    print("errors", json.dumps(ea))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", f"baselines_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
