"""Summarise tuned_v5 per PROTOCOL_v5.md: per-dataset means, NestKAN − competitor paired over
all (rep, fold) units with Nadeau–Bengio SE for repeated k-fold, resolved counts, mean ranks.
Only (dataset, rep, fold) units where every family finished are used.
Writes results/v5_summary.csv, v5_paired.csv, v5_ranks.csv, v5_counts.csv."""
import os, glob
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
METRICS = {"rNLL": "min", "C_td": "max", "IBS": "min", "IBS_cx": "min"}
FOCAL, OWN = "NestKAN", {"NestKAN", "NestKAN-PH"}
K_FOLDS, TEST_TRAIN = 5, 0.2 / 0.64


def main():
    fs = glob.glob(os.path.join(RES, "tuned_v5_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(["dataset", "rep", "fold", "model"])
    fams = sorted(df.model.unique())
    full = df.groupby(["dataset", "rep", "fold"]).model.nunique()
    keep = full[full == len(fams)].index
    d = df.set_index(["dataset", "rep", "fold"]).loc[keep].reset_index()
    units = d.groupby("dataset")[["rep", "fold"]].apply(lambda g: len(g.drop_duplicates())).rename("units")
    summ = d.groupby(["dataset", "model"])[list(METRICS)].mean()
    summ.to_csv(os.path.join(RES, "v5_summary.csv"))
    rows = []
    for ds, g in d.groupby("dataset"):
        f = g[g.model == FOCAL].set_index(["rep", "fold"])
        r_ = g.rep.nunique()
        for m, gm in g.groupby("model"):
            if m == FOCAL:
                continue
            gm = gm.set_index(["rep", "fold"]); idx = f.index.intersection(gm.index)
            row = dict(dataset=ds, competitor=m, units=len(idx), repeats=r_)
            for k, direction in METRICS.items():
                diff = (f.loc[idx, k] - gm.loc[idx, k]).to_numpy()
                se = np.sqrt(diff.var(ddof=1) * (1 / len(diff) + TEST_TRAIN)) if len(diff) > 1 else np.nan
                row[f"d_{k}"] = diff.mean(); row[f"se_{k}"] = se
                row[f"z_{k}"] = (-1 if direction == "max" else 1) * -diff.mean() / se  # + = NestKAN better
            rows.append(row)
    paired = pd.DataFrame(rows); paired.to_csv(os.path.join(RES, "v5_paired.csv"), index=False)
    ext = paired[~paired.competitor.isin(OWN)]
    counts = pd.DataFrame({k: dict(better=int((ext[f"z_{k}"] > 1.96).sum()), worse=int((ext[f"z_{k}"] < -1.96).sum()),
                                   unresolved=int((ext[f"z_{k}"].abs() <= 1.96).sum())) for k in METRICS}).T
    counts.to_csv(os.path.join(RES, "v5_counts.csv"))
    ranks = []
    for k, direction in METRICS.items():
        mean = summ[k].unstack()
        ranks.append(mean.rank(axis=1, ascending=(direction == "min")).mean(0).rename(k))
    ranks = pd.concat(ranks, axis=1).sort_values("rNLL"); ranks.to_csv(os.path.join(RES, "v5_ranks.csv"))
    pd.set_option("display.width", 220)
    print(units.to_string())
    for ds in summ.index.get_level_values(0).unique():
        print(f"\n== {ds} ==\n", summ.loc[ds].sort_values("rNLL").round(4).to_string())
    print("\nmean ranks\n", ranks.round(2).to_string())
    print("\nNestKAN vs external methods (resolved at 1.96 corrected SE)\n", counts.to_string())


if __name__ == "__main__":
    main()
