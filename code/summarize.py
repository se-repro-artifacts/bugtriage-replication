"""Print markdown tables from results/baselines_*.json -> results/summary.md"""
import glob, json, os
ROOT = os.path.join(os.path.dirname(__file__), "..")
L = []
order = ["coarse_s", "coarse_sd", "team_top20_s", "team_top20_sd", "fine_top30_s", "fine_top30_sd", "fine_top50_sd"]
R = {json.load(open(f))["tag"]: json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results", "baselines_*.json"))}
L.append("| setting | #cls | train | test | model | acc | macro-F1 | top-3 | acc reassigned | acc never-reassigned | acc moved-from-specific (n) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for t in order:
    r = R[t]; s = r["stats"]
    for m, v in r["models"].items():
        a = v["all"]; f = lambda x: f"{100*x:.1f}"
        sp = v["moved_from_specific_component"]
        L.append(f"| {t} | {s['n_classes']} | {s['n_train']} | {s['n_test']} | {m} | {f(a['acc'])} | {f(a['macro_f1'])} | {f(a['top3'])} | "
                 f"{f(v['reassigned']['acc'])} (n={v['reassigned']['n']}) | {f(v['never_reassigned']['acc'])} (n={v['never_reassigned']['n']}) | "
                 f"{f(sp['acc'])} ({sp['n']}) |")
L.append("")
L.append("| setting | LR C | sel.acc@50% | @70% | @80% | @90% | @100% | AURC(risk) | abstain@80% reassigned | abstain@80% never | abstain@80% specific-move | test reassign rate |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in order:
    r = R[t]; s = r["selective_tfidf_logreg"]; f = lambda x: f"{100*x:.1f}"
    ab = s["abstain_rate_at_cov80"]
    L.append(f"| {t} | {r['lr_C']} | " + " | ".join(f(s[f'cov{c}']['selective_acc']) for c in (50, 70, 80, 90, 100)) +
             f" | {s['AURC_risk']:.3f} | {f(ab['reassigned'])} | {f(ab['never_reassigned'])} | {f(ab['moved_from_specific_component'])} | {f(r['test_reassign_rate'])} |")
L.append("")
L.append("| setting | LR wrong | wrong & pred==reporter's initial label | wrong among reassigned | of those pred==initial | reporter initial-label acc on test |")
L.append("|---|---|---|---|---|---|")
for t in order:
    e = R[t]["error_analysis"]
    L.append(f"| {t} | {e['n_wrong']} | {e['wrong_and_pred_eq_initial']} | {e['wrong_among_reassigned']} | {e['wrong_among_reassigned_pred_eq_initial']} | {100*e['reporter_initial_label_acc_on_test']:.1f} |")
open(os.path.join(ROOT, "results", "summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
