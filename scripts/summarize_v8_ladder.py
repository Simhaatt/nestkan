"""PROTOCOL_v8 item 5 analysis: likelihood ladder NestKAN-lin -> NestKAN-PH -> NestKAN.
step1 = NestKAN-PH - NestKAN-lin (additive block), step2 = NestKAN - NestKAN-PH (surface),
paired over matched (rep, fold) v5 units, Nadeau-Bengio repeated-CV SE
sqrt(var(d, ddof=1) * (1/J + 0.2/0.64)); resolved iff |z| > 1.96.
Writes results/v8_ladder.csv, results/v8_ladder.md and figures/fig_v8_ladder.png."""
import os, json, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "fig_v8_ladder.png")
DS = ["rott2", "smarto", "nwtco", "drsa_clinic", "rhc", "flchain", "support2", "support2_slos", "sparcs_drg302"]
MET = {"rNLL": -1, "C_td": +1}       # +1: higher is better
STEPS = [("step1", "NestKAN-PH", "NestKAN-lin", "additive block"), ("step2", "NestKAN", "NestKAN-PH", "surface")]


def frames(ds):
    v5 = pd.read_csv(os.path.join(R, f"tuned_v5_{ds}.csv"))
    p = os.path.join(R, f"tuned_v8ladder_{ds}.csv")
    lin = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=v5.columns)
    out = {m: v5[v5.model == m] for m in ("NestKAN", "NestKAN-PH")}
    out["NestKAN-lin"] = lin[lin.model == "NestKAN-lin"]
    for m, d in out.items():
        assert not d.duplicated(["rep", "fold"]).any(), (ds, m)
        out[m] = d.set_index(["rep", "fold"])
    return out, v5


rows = []
for ds in DS:
    F, v5 = frames(ds)
    nk = v5[v5.model == "NestKAN"]
    kept = np.mean([bool(json.loads(c).get("use_residual")) for c in nk.best_cfg])
    n_v5_units = len(F["NestKAN"])
    for step, big, small, what in STEPS:
        units = sorted(set(F[big].index) & set(F[small].index) & set(F["NestKAN"].index))
        J = len(units)
        for m, sgn in MET.items():
            r = dict(dataset=ds, step=step, block=what, first=big, second=small, metric=m, J=J,
                     J_v5=n_v5_units, provisional=J < n_v5_units, v5_surface_kept_frac=kept,
                     v5_surface_kept_n=int(round(kept * len(nk))), v5_nestkan_fits=len(nk))
            if J >= 2:
                a = F[big].loc[units, m].to_numpy(float); b = F[small].loc[units, m].to_numpy(float)
                d = a - b
                se = np.sqrt(np.var(d, ddof=1) * (1 / J + 0.2 / 0.64))
                z = d.mean() / se if se > 0 else np.nan
                verdict = "unresolved" if not abs(z) > 1.96 else ("first better" if sgn * z > 0 else "first worse")
                r.update(mean_first=a.mean(), mean_second=b.mean(), diff=d.mean(), se=se, z=z, verdict=verdict)
            else:
                r.update(verdict="insufficient units")
            rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(R, "v8_ladder.csv"), index=False)

fmt = lambda x, k=4: "" if pd.isna(x) else f"{x:.{k}f}"
L = ["# v8 likelihood ladder (PROTOCOL_v8 item 5)", "",
     "step1 = NestKAN-PH - NestKAN-lin (additive block); step2 = NestKAN - NestKAN-PH (surface). "
     "Paired over matched (rep, fold) units; SE = sqrt(var(d, ddof=1)(1/J + 0.2/0.64)); resolved iff |z| > 1.96. "
     "rNLL lower = better, C_td higher = better.", ""]
if res.provisional.any():
    L += ["**PROVISIONAL rows: fewer matched units than the v5 design (see J vs J_v5).**", ""]
L += ["| dataset | step | metric | J | first | second | diff | se | z | verdict | v5 surface kept |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in res.iterrows():
    L.append(f"| {r.dataset} | {r.step} | {r.metric} | {r.J} | {fmt(r.get('mean_first'))} | {fmt(r.get('mean_second'))} | "
             f"{fmt(r.get('diff'))} | {fmt(r.get('se'))} | {fmt(r.get('z'), 2)} | {r.verdict} | "
             f"{r.v5_surface_kept_n}/{r.v5_nestkan_fits} |")
open(os.path.join(R, "v8_ladder.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------- figure ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
COL = {"step1": "#4C72B0", "step2": "#DD8452"}
LAB = {"step1": "additive block (NestKAN-PH − NestKAN-lin)", "step2": "surface (NestKAN − NestKAN-PH)"}
NAME = {"rott2": "ROTT2", "smarto": "SMART", "nwtco": "NWTCO", "drsa_clinic": "DRSA clinic", "rhc": "RHC",
        "flchain": "FLCHAIN", "support2": "SUPPORT2", "support2_slos": "SUPPORT2-LOS", "sparcs_drg302": "SPARCS DRG302"}
ok = res.dropna(subset=["diff"])
dss = [d for d in DS if d in set(ok.dataset)]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.32 * len(dss) + 1.3), sharey=True,
                         gridspec_kw=dict(wspace=0.08))
y0 = np.arange(len(dss))[::-1]
for ax, (m, sgn) in zip(axes, MET.items()):
    for k, (step, *_r) in enumerate(STEPS):
        s = ok[(ok.metric == m) & (ok.step == step)].set_index("dataset").reindex(dss)
        y = y0 + (0.17 if step == "step1" else -0.17)
        ax.errorbar(s["diff"], y, xerr=1.96 * s["se"], fmt="o", ms=3.5, color=COL[step], elinewidth=1,
                    capsize=0, label=LAB[step])
        res_mask = (s.verdict != "unresolved").to_numpy()
        ax.scatter(s["diff"][res_mask], y[res_mask], s=30, facecolors="none", edgecolors=COL[step], lw=0.8)
    ax.axvline(0, color="0.5", lw=0.7, zorder=0)
    ax.margins(x=0.06)
    ax.set_xlabel(("Δ rNLL (negative = step improves)" if m == "rNLL" else "Δ C$_{td}$ (positive = step improves)"))
axes[0].set_yticks(y0); axes[0].set_yticklabels([NAME[d] + ("*" if ok[ok.dataset == d].provisional.any() else "") for d in dss])
axes[0].set_title("Paired test-set gain of each block, mean ± 1.96 SE", loc="left")
h, l = axes[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h.append(Line2D([], [], marker="o", ls="", ms=6, mfc="none", mec="0.3", mew=0.8)); l.append("ring: resolved, |z| > 1.96")
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.01))
fig.subplots_adjust(left=0.16, right=0.98, top=0.92, bottom=0.30 if len(dss) < 5 else 0.17)
os.makedirs(os.path.dirname(FIG), exist_ok=True)
fig.savefig(FIG, dpi=300)
print(res[["dataset", "step", "metric", "J", "diff", "se", "z", "verdict", "v5_surface_kept_frac"]].to_string())
