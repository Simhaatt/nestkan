"""PROTOCOL_v8 item 2 analysis (as v7, extended to all 9 datasets):
(a) SurvKAN-1000 vs SurvKAN-300, (b) NestKAN (v5, rep 0) vs SurvKAN-1000; rNLL, C_td, IBS;
paired over rep-0 folds, SE = sqrt(var(d, ddof=1)(1/n + 0.2/0.64)); resolved iff |z| > 1.96.
Writes results/v8_survkan1000.csv, results/v8_survkan1000_epochs.csv, results/v8_survkan1000.md."""
import os, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DS = ["rott2", "smarto", "nwtco", "drsa_clinic", "rhc", "flchain", "support2", "support2_slos", "sparcs_drg302"]
MET = {"rNLL": -1, "C_td": +1, "IBS": -1}
CAP = {"_v6survkan": 300, "_v6survkan1000": 1000}
SRC = {ds: ("v7" if ds in ("rott2", "nwtco", "support2") else "v8") for ds in DS}


def load(ds, which):
    if which == "SurvKAN-300":
        p = os.path.join(R, f"tuned_v6survkan_{ds}.csv")
    elif which == "SurvKAN-1000":
        p = os.path.join(R, f"tuned_v6survkan1000_{ds}.csv")
    else:
        p = os.path.join(R, f"tuned_v5_{ds}.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["rep", "fold"] + list(MET)).set_index("fold")
    d = pd.read_csv(p)
    if which == "NestKAN":
        d = d[d.model == "NestKAN"]
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
                     folds=",".join(map(str, folds)), survkan1000_run=SRC[ds])
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
res.to_csv(os.path.join(R, "v8_survkan1000.csv"), index=False)

ep = []
for ds in DS:
    for tag, cap in CAP.items():
        p = os.path.join(R, f"trials{tag}_{ds}.csv")
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        t = t[t["rep"] == 0]
        e = t["epochs"].dropna() if "epochs" in t else pd.Series(dtype=float)
        ep.append(dict(dataset=ds, family="SurvKAN-300" if cap == 300 else "SurvKAN-1000", cap=cap,
                       run=("v6" if cap == 300 else SRC[ds]), n_trials=len(t),
                       n_failed=int(t["epochs"].isna().sum()) if "epochs" in t else len(t),
                       folds=t["fold"].nunique(), mean_epochs=e.mean(), median_epochs=e.median(),
                       max_epochs=e.max(), n_hit_cap=int((e >= cap).sum()), n_over_300=int((e > 300).sum())))
ep = pd.DataFrame(ep)
ep.to_csv(os.path.join(R, "v8_survkan1000_epochs.csv"), index=False)

fmt = lambda x, k=4: "" if pd.isna(x) else f"{x:.{k}f}"
L = ["# v8 SurvKAN-1000 on all 9 datasets (PROTOCOL_v8 item 2)", "",
     "Rep 0, folds 0-4; paired over folds; SE = sqrt(var(d, ddof=1)(1/n + 0.2/0.64)); resolved iff |z| > 1.96. "
     "diff = first - second. rott2/nwtco/support2 SurvKAN-1000 rows are the v7 run; the rest are v8.", ""]
if res["provisional"].any():
    L += ["**PROVISIONAL: some comparisons have fewer than 5 folds (see n_folds).**", ""]
L += ["## Epochs used (tuning trials)", "", "| dataset | family | cap | trials | mean | median | max | hit cap | >300 |",
      "|---|---|---|---|---|---|---|---|---|"]
for _, r in ep.iterrows():
    L.append(f"| {r.dataset} | {r.family} | {r.cap} | {r.n_trials} | {fmt(r.mean_epochs, 1)} | {fmt(r.median_epochs, 0)} | "
             f"{fmt(r.max_epochs, 0)} | {r.n_hit_cap} | {r.n_over_300} |")
for comp, title in [("a", "(a) SurvKAN-1000 vs SurvKAN-300"), ("b", "(b) NestKAN vs SurvKAN-1000")]:
    L += ["", f"## {title}", "", "| dataset | metric | n | mean first | mean second | diff | se | z | verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.query("comparison == @comp").iterrows():
        L.append(f"| {r.dataset} | {r.metric} | {r.n_folds} | {fmt(r.get('mean_first'))} | {fmt(r.get('mean_second'))} | "
                 f"{fmt(r.get('diff'))} | {fmt(r.get('se'))} | {fmt(r.get('z'), 2)} | {r.verdict} |")
done = sorted(set(res.query("n_folds == 5 and comparison == 'b'").dataset), key=DS.index)
L += ["", "## Decision rule (PROTOCOL_v8 item 2)", "",
      f"SurvKAN-1000 completed (5 folds) on {len(done)}/9 datasets: {', '.join(done)}. The main-text SurvKAN "
      "comparison uses SurvKAN-1000 on these datasets; SurvKAN-300 moves to the supplement."]
open(os.path.join(R, "v8_survkan1000.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print(res[["comparison", "dataset", "metric", "n_folds", "diff", "se", "z", "verdict"]].to_string())
print(ep.to_string())
