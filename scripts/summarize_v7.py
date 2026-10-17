"""PROTOCOL_v7 B3 analysis: SurvKAN-1000 vs SurvKAN-300, NestKAN vs SurvKAN-1000 (rep 0, paired folds).
Writes results/v7_survkan_budget.csv, results/v7_epochs.csv and results/v7_summary.md."""
import os, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DS = ["rott2", "nwtco", "support2"]
MET = {"rNLL": -1, "C_td": +1, "IBS": -1}          # +1: higher is better
CAP = {"_v6survkan": 300, "_v6survkan1000": 1000}


def load(ds, which):
    if which == "SurvKAN-300":
        d = pd.read_csv(os.path.join(R, f"tuned_v6survkan_{ds}.csv"))
    elif which == "SurvKAN-1000":
        p = os.path.join(R, f"tuned_v6survkan1000_{ds}.csv")
        d = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["rep", "fold"] + list(MET))
    else:
        d = pd.read_csv(os.path.join(R, f"tuned_v5_{ds}.csv")).query("model == 'NestKAN'")
    d = d[d["rep"] == 0]
    assert not d["fold"].duplicated().any(), (ds, which)
    return d.set_index("fold")


rows = []
for ds in DS:
    for first, second, comp in [("SurvKAN-1000", "SurvKAN-300", "a"), ("NestKAN", "SurvKAN-1000", "b")]:
        A, B = load(ds, first), load(ds, second)
        folds = sorted(set(A.index) & set(B.index))
        for m, sgn in MET.items():
            n = len(folds)
            r = dict(comparison=comp, dataset=ds, metric=m, first=first, second=second, n_folds=n,
                     folds=",".join(map(str, folds)))
            if n >= 2:
                d = A.loc[folds, m].to_numpy(float) - B.loc[folds, m].to_numpy(float)
                se = np.sqrt(np.var(d, ddof=1) * (1 / n + 0.2 / 0.64))
                z = d.mean() / se if se > 0 else np.nan
                verdict = "unresolved" if not abs(z) > 1.96 else ("first better" if sgn * z > 0 else "first worse")
                r.update(mean_first=A.loc[folds, m].mean(), mean_second=B.loc[folds, m].mean(),
                         diff=d.mean(), se=se, z=z, verdict=verdict)
            else:
                r.update(verdict="insufficient folds")
            r["provisional"] = n < 5
            rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(R, "v7_survkan_budget.csv"), index=False)

ep = []
for ds in DS:
    for tag, cap in CAP.items():
        p = os.path.join(R, f"trials{tag}_{ds}.csv")
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        t = t[t["rep"] == 0]
        e = t["epochs"].dropna()
        ep.append(dict(dataset=ds, family="SurvKAN-300" if cap == 300 else "SurvKAN-1000", cap=cap,
                       n_trials=len(t), n_failed=int(t["epochs"].isna().sum()), folds=t["fold"].nunique(),
                       mean_epochs=e.mean(), median_epochs=e.median(), max_epochs=e.max(),
                       n_hit_cap=int((e >= cap).sum()), n_over_300=int((e > 300).sum())))
ep = pd.DataFrame(ep)
ep.to_csv(os.path.join(R, "v7_epochs.csv"), index=False)

# decision rule B4
a = res.query("comparison == 'a' and metric == 'rNLL'")
better = a.loc[a["verdict"] == "first better", "dataset"].tolist()
fmt = lambda x: "" if pd.isna(x) else f"{x:.4f}"
L = ["# v7 SurvKAN training-budget sensitivity (PROTOCOL_v7.md)", "",
     "Rep 0, folds 0-4; paired over folds; Nadeau-Bengio SE = sqrt(var(d, ddof=1)(1/n + 0.2/0.64)); "
     "resolved iff |z| > 1.96. diff = first - second (rNLL, IBS lower better; C_td higher better).", ""]
if res["provisional"].any():
    L += ["**PROVISIONAL: some comparisons have fewer than 5 folds (see n_folds).**", ""]
L += ["## Epochs actually used (tuning trials)", "",
      "| dataset | family | cap | trials | mean | median | max | hit cap | >300 |", "|---|---|---|---|---|---|---|---|---|"]
for _, r in ep.iterrows():
    L.append(f"| {r.dataset} | {r.family} | {r.cap} | {r.n_trials} | {r.mean_epochs:.1f} | {r.median_epochs:.0f} | "
             f"{r.max_epochs:.0f} | {r.n_hit_cap} | {r.n_over_300} |")
for comp, title in [("a", "(a) SurvKAN-1000 vs SurvKAN-300"), ("b", "(b) NestKAN vs SurvKAN-1000")]:
    L += ["", f"## {title}", "", "| dataset | metric | n | mean first | mean second | diff | se | z | verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.query("comparison == @comp").iterrows():
        L.append(f"| {r.dataset} | {r.metric} | {r.n_folds} | {fmt(r.get('mean_first'))} | {fmt(r.get('mean_second'))} | "
                 f"{fmt(r.get('diff'))} | {fmt(r.get('se'))} | {'' if pd.isna(r.get('z')) else f'{r.z:.2f}'} | {r.verdict} |")
L += ["", "## Decision rule (B4)", "",
      (f"SurvKAN-1000 is resolved-better than SurvKAN-300 on rNLL on: {', '.join(better)}. "
       "The paper reports SurvKAN-1000 alongside SurvKAN-300 for these datasets.") if better else
      "SurvKAN-1000 is not resolved-better than SurvKAN-300 on rNLL on any dataset; the paper keeps SurvKAN-300 "
      "and cites v7 as a sensitivity check."]
open(os.path.join(R, "v7_summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print(res.to_string()); print(ep.to_string())
