"""v6 analysis exactly as pre-registered in PROTOCOL_v6.md: NestKAN (v5, rep-0 rows) vs SurvKAN,
KAPLAN-HR (ensembled vs ensembled) and CoxKAN-sym (NestKAN single-seed vs single formula; NestKAN
ensembled also reported). Paired over rep-0 folds, NB-corrected SE var*(1/5 + 0.2/0.64)."""
import os, glob, json
import numpy as np, pandas as pd
R = os.path.join(os.environ.get("DKAN_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))), "results")
DS = ["rott2", "smarto", "nwtco", "drsa_clinic", "rhc", "flchain", "support2", "support2_slos", "sparcs_drg302"]
NEW = {"SurvKAN": "_v6survkan", "KAPLAN-HR": "_v6kaplanhr2", "CoxKAN-sym": "_v6coxkansym"}
MET = ["rNLL", "C_td", "IBS", "IBS_cx"]
LOWER = {"rNLL": True, "C_td": False, "IBS": True, "IBS_cx": True}
SINGLE = {"rNLL": "rNLL_single", "C_td": "C_single", "IBS": "IBS_single"}


def load():
    v5 = pd.concat([pd.read_csv(os.path.join(R, f"tuned_v5_{d}.csv")) for d in DS
                    if os.path.exists(os.path.join(R, f"tuned_v5_{d}.csv"))])
    v5 = v5[v5.rep == 0].drop_duplicates(["dataset", "rep", "fold", "model"], keep="last")
    new = []
    for m, tag in NEW.items():
        for d in DS:
            f = os.path.join(R, f"tuned{tag}_{d}.csv")
            if os.path.exists(f):
                new.append(pd.read_csv(f))
    v6 = pd.concat(new).drop_duplicates(["dataset", "rep", "fold", "model"], keep="last") if new else v5.iloc[:0]
    v6 = v6[v6.rep == 0].copy()
    # PROTOCOL_v6 revision 2 (analysis note): a fit whose test survival curves are non-finite is a
    # failed fit; all its metrics are set to NaN and the unit is dropped from the paired comparison.
    bad = ~np.isfinite(v6["rNLL"].astype(float)) | ~np.isfinite(v6["IBS"].astype(float))
    v6.loc[bad, MET] = np.nan
    v6.loc[bad, ["dataset", "fold", "model"]].to_csv(os.path.join(R, "v6_failed_fits.csv"), index=False)
    return v5, v6


def paired(a, b, metric, ma, mb):
    """a, b: frames indexed by fold; difference NestKAN - comparator (sign: negative = NestKAN better
    for loss metrics; for C_td positive = NestKAN better)."""
    j = a[[ma]].join(b[[mb]], lsuffix="_a", rsuffix="_b", how="inner").dropna()
    dlt = (j.iloc[:, 0] - j.iloc[:, 1]).to_numpy()
    J = len(dlt)
    if J < 2:
        return dict(n=J, diff=np.nan, se=np.nan, z=np.nan, ma=np.nan, mb=np.nan)
    se = np.sqrt(dlt.var(ddof=1) * (1 / J + 0.2 / 0.64))
    return dict(n=J, diff=dlt.mean(), se=se, z=dlt.mean() / se if se > 0 else np.nan,
                ma=j.iloc[:, 0].mean(), mb=j.iloc[:, 1].mean())


def main():
    v5, v6 = load()
    rows = []
    for d in DS:
        nk = v5[(v5.dataset == d) & (v5.model == "NestKAN")].set_index("fold")
        for m in NEW:
            c = v6[(v6.dataset == d) & (v6.model == m)].set_index("fold")
            if len(c) == 0 or len(nk) == 0:
                continue
            modes = [("ens", None)] if m != "CoxKAN-sym" else [("single", "single"), ("ens", None)]
            for mode, _ in modes:
                for met in MET:
                    ma = SINGLE.get(met, met) if mode == "single" else met
                    if ma not in nk.columns:
                        continue
                    r = paired(nk, c, met, ma, met)
                    better = (r["z"] < -1.96) if LOWER[met] else (r["z"] > 1.96)
                    worse = (r["z"] > 1.96) if LOWER[met] else (r["z"] < -1.96)
                    rows.append(dict(dataset=d, comparator=m, nestkan=mode, metric=met, n_folds=r["n"],
                                     nestkan_mean=r["ma"], comp_mean=r["mb"], diff=r["diff"],
                                     se_nb=r["se"], z=r["z"],
                                     verdict="better" if better else ("worse" if worse else "unresolved")))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(R, "v6_paired.csv"), index=False)
    # mean ranks over rep-0 units: per dataset, rank methods by fold-mean metric
    allm = pd.concat([v5, v6])
    rk = []
    for met in MET:
        dm = allm.groupby(["dataset", "model"])[met].mean().unstack()
        rkm = dm.rank(axis=1, ascending=LOWER[met])
        rk.append(rkm.mean().rename(met))
        rkm.assign(metric=met).to_csv(os.path.join(R, f"v6_rank_by_dataset_{met}.csv"))
    RK = pd.concat(rk, axis=1).sort_values("rNLL")
    RK.to_csv(os.path.join(R, "v6_ranks.csv"))
    cnt = P.groupby(["comparator", "nestkan", "metric"]).verdict.value_counts().unstack(fill_value=0)
    cnt.to_csv(os.path.join(R, "v6_counts.csv"))
    print(cnt.to_string()); print(RK.round(2).to_string())
    log = os.path.join(R, "coxkan_sym_log.csv")
    if os.path.exists(log):
        L = pd.read_csv(log)
        print("sym_ok", L.sym_ok.mean().round(3), "finetuned", L.finetuned.mean().round(3), "n", len(L))


if __name__ == "__main__":
    main()
