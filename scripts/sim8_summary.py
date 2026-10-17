"""PROTOCOL_v8 item 1 fixed analysis.
-> results/v8_sim_summary.csv   mean ISE x100 (SE) per (scenario, n, method) + paired NestKAN - method and
                                NestKAN-sel - method (plain paired SE over replicates; resolved = |z| > 1.96)
-> results/v8_sim_selection.csv surface-kept rate of NestKAN-sel per (scenario, n), Wilson 95% CI;
                                plus pooled PH / non-PH rows for the pre-stated expectation (descriptive).
sim_<scen>_n<n>.csv rows are de-duplicated on (scenario, n, rep, model), keeping the first.
"""
import os, sys
import numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SCENS = ["ph_linear", "ph_additive", "nonph", "nonph_inter"]
TARGET = {1000: 20, 4000: 10}


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def load():
    A = []
    for s in SCENS:
        for n in TARGET:
            f = os.path.join(R, f"sim_{s}_n{n}.csv")
            if os.path.exists(f):
                A.append(pd.read_csv(f))
            g = os.path.join(R, f"sim8_sel_{s}_n{n}.csv")
            if os.path.exists(g):
                A.append(pd.read_csv(g))
    d = pd.concat(A, ignore_index=True).drop_duplicates(["scenario", "n", "rep", "model"], keep="first")
    return d[d.rep < d.n.map(TARGET)]


def paired(w, ref, m):
    x = (w[ref] - w[m]).dropna() * 100
    k = len(x)
    if k < 2:
        return k, np.nan, np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(k); z = x.mean() / se if se > 0 else np.nan
    return k, x.mean(), se, z


def main():
    d = load()
    rows = []
    for (s, n), g in d.groupby(["scenario", "n"], sort=False):
        w = g.pivot(index="rep", columns="model", values="ISE")
        for m in w.columns:
            x = w[m].dropna() * 100
            r = dict(scenario=s, n=n, method=m, reps=len(x), ISE_x100=x.mean(), SE_x100=x.std(ddof=1) / np.sqrt(len(x)))
            for ref, tag in (("NestKAN", "NestKAN"), ("NestKAN-sel", "NestKANsel")):
                if ref in w and m != ref:
                    k, mu, se, z = paired(w, ref, m)
                    r.update({f"{tag}_minus_n": k, f"{tag}_minus_x100": mu, f"{tag}_minus_SE": se, f"{tag}_minus_z": z,
                              f"{tag}_minus_resolved": (bool(abs(z) > 1.96) if np.isfinite(z) else False)})
            rows.append(r)
    S = pd.DataFrame(rows)
    S["scenario"] = pd.Categorical(S.scenario, SCENS, ordered=True)
    S = S.sort_values(["n", "scenario", "ISE_x100"])
    S.to_csv(os.path.join(R, "v8_sim_summary.csv"), index=False)
    sel = d[d.model == "NestKAN-sel"].copy()
    sel["kept"] = sel.use_residual.astype(str).str.lower().eq("true")
    rows = []
    for n in TARGET:
        for s in SCENS + ["PH (pooled)", "non-PH (pooled)"]:
            if s == "PH (pooled)":
                g = sel[(sel.n == n) & sel.scenario.isin(SCENS[:2])]
            elif s == "non-PH (pooled)":
                g = sel[(sel.n == n) & sel.scenario.isin(SCENS[2:])]
            else:
                g = sel[(sel.n == n) & (sel.scenario == s)]
            k, m = int(g.kept.sum()), len(g)
            lo, hi = wilson(k, m)
            rows.append(dict(scenario=s, n=n, reps=m, kept=k, kept_rate=(k / m if m else np.nan), wilson_lo=lo, wilson_hi=hi,
                             gate_abs_median_kept=g.loc[g.kept, "gate"].abs().median() if k else np.nan,
                             one_arm_only=int(((g.n_on == 0) | (g.n_off == 0)).sum()) if m else 0))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(R, "v8_sim_selection.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(P.round(3).to_string(index=False))
        print(S.pivot_table(index=["n", "scenario"], columns="method", values="reps", observed=True).to_string())


if __name__ == "__main__":
    main()
