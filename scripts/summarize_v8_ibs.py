"""PROTOCOL_v8 item 3 analysis: sensitivity to the tuning metric (validation-IBS selection).
Completed datasets = all 11 v5 families x 5 rep-0 folds present in tuned_v8ibs_<ds>.csv.
Mean ranks (v5 convention: per-dataset mean over units, rank across methods, mean over datasets)
for rNLL, C_td, IBS, IBS_cx for the IBS-selected run, side by side with the rNLL-selected v5 run
restricted to the same rep-0 units and datasets. Paired NestKAN vs each method with the
single-run Nadeau-Bengio SE sqrt(var(d, ddof=1)(1/5 + 0.2/0.64)).
Writes results/v8_ibs_ranks.csv, v8_ibs_counts.csv, v8_ibs_paired.csv, v8_ibs.md."""
import os, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DS = ["rott2", "smarto", "nwtco", "drsa_clinic", "rhc", "flchain", "support2", "support2_slos", "sparcs_drg302"]
FAMS = ["LogisticHazard", "PMF", "MTLR", "DeepHit", "DeepSurv", "NestKAN", "PCHazard", "CoxKAN",
        "NestKAN-PH", "CoxPH", "RSF"]
METRICS = {"rNLL": "min", "C_td": "max", "IBS": "min", "IBS_cx": "min"}
FOCAL, OWN = "NestKAN", {"NestKAN", "NestKAN-PH"}


def load(tag, ds):
    p = os.path.join(R, f"tuned{tag}_{ds}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[(d.rep == 0) & d.model.isin(FAMS)]
    assert not d.duplicated(["fold", "model"]).any(), (tag, ds)
    return d


ibs, v5, done, partial = {}, {}, [], {}
for ds in DS:
    d = load("_v8ibs", ds)
    if d is None:
        continue
    if len(d) == len(FAMS) * 5:
        done.append(ds); ibs[ds] = d; v5[ds] = load("_v5", ds)
        assert len(v5[ds]) == 55, ds
    else:
        partial[ds] = len(d)

rank_tabs, paired = [], []
for sel, src in [("IBS", ibs), ("rNLL (v5)", v5)]:
    if not done:
        break
    d = pd.concat([src[ds] for ds in done])
    summ = d.groupby(["dataset", "model"])[list(METRICS)].mean()
    rk = pd.concat([summ[k].unstack().rank(axis=1, ascending=(dr == "min")).mean(0).rename(k)
                    for k, dr in METRICS.items()], axis=1)
    rk.columns = [f"{c} [{sel}]" for c in rk.columns]
    rank_tabs.append(rk)
    for ds in done:
        g = src[ds]; f = g[g.model == FOCAL].set_index("fold")
        for m in FAMS:
            if m == FOCAL:
                continue
            gm = g[g.model == m].set_index("fold"); idx = sorted(set(f.index) & set(gm.index))
            row = dict(selection=sel, dataset=ds, competitor=m, n_folds=len(idx))
            for k, dr in METRICS.items():
                diff = f.loc[idx, k].to_numpy(float) - gm.loc[idx, k].to_numpy(float)
                se = np.sqrt(diff.var(ddof=1) * (1 / 5 + 0.2 / 0.64))
                row[f"d_{k}"] = diff.mean(); row[f"se_{k}"] = se
                row[f"z_{k}"] = (1 if dr == "max" else -1) * diff.mean() / se   # + = NestKAN better
            paired.append(row)
ranks = pd.concat(rank_tabs, axis=1) if rank_tabs else pd.DataFrame()
if len(ranks):
    ranks = ranks.sort_values("rNLL [IBS]")
    ranks.insert(0, "n_datasets", len(done)); ranks.insert(1, "datasets", ",".join(done))
ranks.index.name = "model"
ranks.to_csv(os.path.join(R, "v8_ibs_ranks.csv"))
paired = pd.DataFrame(paired)
paired.to_csv(os.path.join(R, "v8_ibs_paired.csv"), index=False)
crow = []
for sel in ["IBS", "rNLL (v5)"]:
    if not len(paired):
        break
    p = paired[paired.selection == sel]
    for scope, q in [("external methods (v5 convention, excl. NestKAN-PH)", p[~p.competitor.isin(OWN)]),
                     ("NestKAN-PH only", p[p.competitor == "NestKAN-PH"])] + \
                    [(f"vs {c}", p[p.competitor == c]) for c in FAMS if c != FOCAL]:
        for k in METRICS:
            z = q[f"z_{k}"]
            crow.append(dict(selection=sel, scope=scope, metric=k, n_comparisons=len(q),
                             better=int((z > 1.96).sum()), unresolved=int((z.abs() <= 1.96).sum()),
                             worse=int((z < -1.96).sum())))
counts = pd.DataFrame(crow)
counts.to_csv(os.path.join(R, "v8_ibs_counts.csv"), index=False)

L = ["# v8 IBS-selected sensitivity (PROTOCOL_v8 item 3)", "",
     f"Completed datasets (11 methods x 5 rep-0 folds): {len(done)}/9 - {', '.join(done) if done else 'none'}.",
     f"Partial (not analysed): {partial if partial else 'none'}.", ""]
if len(done) < 9:
    L += ["**PROVISIONAL: not all 9 datasets completed; ranks and counts cover the completed datasets only.**", ""]
if len(ranks):
    L += ["## Mean ranks (lower = better)", "", "```", ranks.drop(columns=["n_datasets", "datasets"]).round(2).to_string(), "```", ""]
    ext = counts[counts.scope.str.startswith("external")]
    L += ["## NestKAN vs external methods: better / unresolved / worse", "",
          "```", ext.pivot(index="metric", columns="selection", values=["better", "unresolved", "worse"]).to_string(), "```", ""]
open(os.path.join(R, "v8_ibs.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print(done, partial); print(ranks.round(2).to_string())
if len(counts):
    print(counts[counts.scope.str.startswith(("external", "NestKAN-PH"))].to_string())
