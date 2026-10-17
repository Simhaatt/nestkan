"""Interpretability analyses for NestKAN (post hoc; no change to any benchmark protocol).

For each dataset, rep 0, folds 0-4: refit NestKAN with the configuration SELECTED in v5 for that fold
(results/tuned_v5_<ds>.csv) and the v5 seeds (fold + {0, 1000, 2000}); then
  A. functional-ANOVA projection of the fitted eta on the TRAINING person-period data (the at-risk
     measure of THEORY Remark 1): hierarchical least squares on {interval dummies} -> + {linear z}
     -> + {per-covariate B-spline basis of u (same knots as the model)}; the remainder is the
     interaction (time-varying / non-additive) block.  Shares = R^2 increments.
  B. per-covariate additive shape g_j(x) (from the projection) on the ORIGINAL covariate scale,
     centred on the training sample; Cox (lifelines, ridge 0.01) coefficient line for comparison.
  C. time-varying partial-dependence log-HR, q90 vs q10 of each numeric covariate, per interval.
  D. Grambsch-Therneau test (lifelines proportional_hazard_test, rank time transform) on fold-0 training data.
Simulation mode: NestKAN on the nonph scenario (n = 4000, 3 replicates) -> recovered shapes and the
time-varying x4 effect against the truth.

usage: python -u scripts/interp.py <ds> [n_seeds=1]      |      python -u scripts/interp.py sim
outputs: results/interp/{anova,shape,tv,cox,gt}_<ds>.csv
"""
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pylib")); sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)
import numpy as np, pandas as pd, torch
torch.set_num_threads(int(os.environ.get("DKAN_THREADS", "3")))
from sklearn.model_selection import StratifiedKFold, train_test_split
from dkan.data import make_preprocessor, Discretiser, load_survset
from dkan.models import MODELS, fit_discrete, KANLayer, predict_grid_survival, interp_survival
from dkan.metrics import evaluate, reference_cuts

ROOT = os.path.join(_HERE, "..")
OUT = os.path.join(ROOT, "results", "interp"); os.makedirs(OUT, exist_ok=True)
DROP = {"support2": ["num_surv2m", "num_surv6m", "fac_sfdm2", "num_hday"], "rhc": ["num_surv2md1"],
        "nwtco": ["fac_in_subcohort"], "flchain": ["fac_chapter"]}
OFFSETS = (0, 1000, 2000)
MAX_PP = 30000


def fit_nest(df, tr, va, time_all, ev_all, cfg, seed):
    cfg = dict(cfg)
    T = cfg.pop("T"); lr = cfg.pop("lr"); wd = cfg.pop("wd"); lam = cfg.pop("lam", 1e-3)
    pp = make_preprocessor("dual").fit(df.iloc[tr])
    Xtr, Xva = pp.transform(df.iloc[tr]), pp.transform(df.iloc[va])
    disc = Discretiser(T, "km").fit(time_all[tr], ev_all[tr])
    itr, etr = disc.transform(time_all[tr], ev_all[tr]); iva, eva = disc.transform(time_all[va], ev_all[va])
    model = MODELS["NestKAN"](Xtr.shape[1], disc.T, v=disc.v, **cfg)
    model.init_baseline(itr, etr)
    model, _, _ = fit_discrete(model, Xtr, itr, etr, Xva, iva, eva, lr=lr, wd=wd, lam=lam, seed=seed)
    return model, pp, disc, Xtr, itr


@torch.no_grad()
def eta_of(m, X, batch=4096):
    out = []
    for b in range(0, len(X), batch):
        x = torch.as_tensor(np.asarray(X[b:b + batch]), dtype=torch.float32)
        N, d = len(x), m.d
        z, u = x[:, :d], x[:, d:]
        g = torch.zeros(N, 1)
        if m.lin is not None: g = g + m.lin(z)
        if m.add is not None: g = g + m.add(u)
        e = m.alpha.view(1, -1) + g
        if m.use_residual:
            hz = torch.tanh(m.lx(u).unsqueeze(1) + m.lt(m.v).unsqueeze(0))
            e = e + m.gate * m.l2(hz.reshape(N * m.T, -1)).view(N, m.T)
        out.append(e.numpy().astype(float))
    return np.vstack(out)


def anova(m, Xtr, itr, grid, rng):
    d = m.d
    rows = np.concatenate([np.c_[np.full(k + 1, i), np.arange(k + 1)] for i, k in enumerate(itr) if k >= 0])
    if len(rows) > MAX_PP:
        rows = rows[rng.choice(len(rows), MAX_PP, replace=False)]
    E = eta_of(m, Xtr)
    y = E[rows[:, 0], rows[:, 1]]
    D1 = np.eye(m.T)[rows[:, 1]]
    z = Xtr[rows[:, 0], :d].astype(float)
    basis = KANLayer(d, 1, grid=grid, base=False)
    with torch.no_grad():
        B = basis.bspline(torch.as_tensor(Xtr[:, d:], dtype=torch.float32)).numpy().astype(float)   # (N, d, G+3)
    Bpp = B[rows[:, 0]].reshape(len(rows), -1)
    sst = ((y - y.mean()) ** 2).sum()
    r2, coefs = [], None
    for D in (D1, np.c_[D1, z], np.c_[D1, z, Bpp]):
        c, *_ = np.linalg.lstsq(D, y, rcond=None)
        r2.append(1 - ((y - D @ c) ** 2).sum() / sst); coefs = c
    T = m.T
    cz, cB = coefs[T:T + d], coefs[T + d:].reshape(d, -1)
    sh = dict(interval=r2[0], linear=r2[1] - r2[0], additive=r2[2] - r2[1], interaction=1 - r2[2])
    cov = 1 - r2[0]
    sh.update({f"{k}_of_cov": (sh[k] / cov if cov > 1e-12 else np.nan) for k in ("linear", "additive", "interaction")})
    sh["var_eta"] = y.var()
    return sh, cz, cB, basis, B


def shapes(pp, df_tr, cz, cB, basis, B, Xtr, d):
    """g_j on the original scale for numeric covariates, centred on training values."""
    names = pp.a.names_
    out = []
    for c in pp.a.num_cols:
        j = names.index(c)
        x = pd.to_numeric(df_tr[c], errors="coerce").to_numpy(float); x = x[np.isfinite(x)]
        if len(np.unique(x)) < 5:
            continue
        xg = np.unique(np.quantile(x, np.linspace(0.02, 0.98, 40)))
        zg = (xg - pp.a.mu[c]) / pp.a.sd[c]
        ug = 2 * np.interp(xg, pp.b.sup[c], pp.b.cdf[c]) - 1
        with torch.no_grad():
            Bg = basis.bspline(torch.as_tensor(np.tile(ug[:, None], (1, d)), dtype=torch.float32)).numpy()[:, j, :]
        g = cz[j] * zg + Bg @ cB[j]
        gtr = cz[j] * Xtr[:, j] + B[:, j, :] @ cB[j]
        out.append(pd.DataFrame(dict(covariate=c, x=xg, g=g - gtr.mean())))
    return pd.concat(out) if out else pd.DataFrame()


def tv_effects(m, pp, disc, df_tr, rng, n_sub=800):
    sub = df_tr.iloc[rng.choice(len(df_tr), min(n_sub, len(df_tr)), replace=False)]
    cuts = disc.cuts; tmid = (cuts[:-1] + cuts[1:]) / 2
    out = []
    for c in pp.a.num_cols:
        x = pd.to_numeric(df_tr[c], errors="coerce").dropna()
        lo, hi = np.quantile(x, [0.1, 0.9])
        if hi <= lo:
            continue
        a, b = sub.copy(), sub.copy(); a[c] = lo; b[c] = hi
        dlt = (eta_of(m, pp.transform(b)) - eta_of(m, pp.transform(a))).mean(0)
        out.append(pd.DataFrame(dict(covariate=c, k=np.arange(m.T), t_mid=tmid[:m.T], logHR=dlt, q10=lo, q90=hi)))
    return pd.concat(out) if out else pd.DataFrame()


def cox_and_gt(pp, df_tr, t, e, do_gt):
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    Z = pp.a.transform(df_tr); names = pp.a.names_
    keep = Z.std(0) > 1e-8
    Zd = pd.DataFrame(Z[:, keep], columns=[f"c{i}" for i in np.where(keep)[0]])
    Zd["T"], Zd["E"] = t, e
    cph = CoxPHFitter(penalizer=0.01).fit(Zd, "T", "E")
    cox = pd.DataFrame(dict(covariate=[names[int(k[1:])] for k in cph.params_.index], beta=cph.params_.values))
    gt = None
    if do_gt:
        r = proportional_hazard_test(cph, Zd, time_transform="rank")
        s = r.summary
        gt = pd.DataFrame(dict(covariate=[names[int(k[1:])] for k in s.index], stat=s["test_statistic"].values,
                               p=s["p"].values))
    return cox, gt


def run_ds(ds, n_seeds=1, folds=range(5)):
    df = load_survset(ds).drop(columns=DROP.get(ds, []), errors="ignore")
    time_all = df["time"].to_numpy(float); ev_all = df["event"].to_numpy(int)
    v5 = pd.read_csv(os.path.join(ROOT, "results", f"tuned_v5_{ds}.csv"))
    v5 = v5[(v5.model == "NestKAN") & (v5.rep == 0)].drop_duplicates("fold", keep="last").set_index("fold")
    skf = list(StratifiedKFold(5, shuffle=True, random_state=0).split(df, ev_all))
    A, S, TV, CX = [], [], [], []
    gt = None
    for fold in folds:
        tr, te = skf[fold]
        tr, va = train_test_split(tr, test_size=0.2, stratify=ev_all[tr], random_state=fold)
        cuts = reference_cuts(time_all[tr], ev_all[tr], 30)
        tq = np.quantile(time_all[tr][ev_all[tr] == 1], np.linspace(0, 1, 301))
        tgrid = np.unique(np.r_[0.0, np.linspace(0, time_all[tr].max(), 200), cuts, tq])
        cfg = json.loads(v5.loc[fold, "best_cfg"])
        for off in OFFSETS[:n_seeds]:
            t0 = time.time()
            m, pp, disc, Xtr, itr = fit_nest(df, tr, va, time_all, ev_all, cfg, fold + off)
            rng = np.random.default_rng([fold, off])
            sh, cz, cB, basis, B = anova(m, Xtr, itr, cfg["grid"], rng)
            gate = float(m.gate.detach()) if m.use_residual else 0.0
            Ste = interp_survival(predict_grid_survival(m, pp.transform(df.iloc[te])), disc.cuts, tgrid)
            sh["rNLL_refit"] = evaluate(Ste, tgrid, time_all[te], ev_all[te], time_all[tr], ev_all[tr], cuts=cuts)["rNLL"]
            sh["rNLL_v5_single"] = float(v5.loc[fold, "rNLL_single"])
            A.append(dict(dataset=ds, fold=fold, seed=off, use_residual=bool(m.use_residual), gate=gate,
                          T=m.T, d=m.d, **sh))
            sp = shapes(pp, df.iloc[tr], cz, cB, basis, B, Xtr, m.d)
            S.append(sp.assign(fold=fold, seed=off))
            TV.append(tv_effects(m, pp, disc, df.iloc[tr], rng).assign(fold=fold, seed=off,
                                                                        use_residual=bool(m.use_residual)))
            if off == 0:
                cx, g = cox_and_gt(pp, df.iloc[tr], time_all[tr], ev_all[tr], do_gt=(fold == 0))
                CX.append(cx.assign(fold=fold))
                if g is not None:
                    gt = g
            print(f"{ds} f{fold} s{off} resid={m.use_residual} shares="
                  f"{ {k: round(v, 3) for k, v in sh.items() if not k.endswith('_cov')} } [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(A).to_csv(os.path.join(OUT, f"anova_{ds}.csv"), index=False)
    pd.concat(S).to_csv(os.path.join(OUT, f"shape_{ds}.csv"), index=False)
    pd.concat(TV).to_csv(os.path.join(OUT, f"tv_{ds}.csv"), index=False)
    pd.concat(CX).to_csv(os.path.join(OUT, f"cox_{ds}.csv"), index=False)
    if gt is not None:
        gt.to_csv(os.path.join(OUT, f"gt_{ds}.csv"), index=False)


def run_sim(n=4000, reps=(0, 1, 2)):
    import sim
    scen = "nonph"
    sm = pd.read_csv(os.path.join(ROOT, "results", f"sim_{scen}_n{n}.csv"))
    sm = sm[sm.model == "NestKAN"].drop_duplicates(["rep", "model"])
    cfg = dict(json.loads(sm.best_cfg.value_counts().index[0]), use_residual=True)
    tfine = np.linspace(1e-4, 4.0, 1001)
    TRUE = {"num_x1": lambda x: np.sin(1.5 * x), "num_x2": lambda x: 0.5 * x ** 2,
            "num_x3": lambda x: 0.5 * np.tanh(2 * x), "num_x5": lambda x: 0.7 * np.log1p(x)}
    S, TV, A = [], [], []
    for rep in reps:
        rng = np.random.default_rng([2026, rep, n, sum(map(ord, scen))])
        df, X, _ = sim.simulate(n, scen, rng, tfine)
        time_all = df["time"].to_numpy(float); ev_all = df["event"].to_numpy(int)
        r2 = np.random.default_rng(rep).permutation(np.arange(n))
        va, tr = np.sort(r2[: n // 5]), np.sort(r2[n // 5:])
        m, pp, disc, Xtr, itr = fit_nest(df, tr, va, time_all, ev_all, cfg, rep)
        rg = np.random.default_rng([rep, 7])
        sh, cz, cB, basis, B = anova(m, Xtr, itr, cfg["grid"], rg)
        A.append(dict(rep=rep, **sh))
        sp = shapes(pp, df.iloc[tr], cz, cB, basis, B, Xtr, m.d)
        for c, f in TRUE.items():
            xs = df.iloc[tr][c].to_numpy(float)
            q = sp.covariate == c
            sp.loc[q, "g_true"] = f(sp.loc[q, "x"].to_numpy()) - f(xs).mean()
        S.append(sp.assign(rep=rep))
        sub = df.iloc[tr].iloc[rg.choice(len(tr), 800, replace=False)]
        a, b = sub.copy(), sub.copy(); a["num_x4"] = -1.0; b["num_x4"] = 1.0
        slope = (eta_of(m, pp.transform(b)) - eta_of(m, pp.transform(a))).mean(0) / 2
        tmid = (disc.cuts[:-1] + disc.cuts[1:]) / 2
        Ft = 1 - np.exp(-(tmid / 0.6) ** sim.SHAPE)
        TV.append(pd.DataFrame(dict(rep=rep, k=np.arange(m.T), t_mid=tmid, slope=slope, slope_true=1 - 2 * Ft)))
        print("sim rep", rep, {k: round(v, 3) for k, v in sh.items() if not k.endswith("_cov")}, flush=True)
    pd.concat(S).to_csv(os.path.join(OUT, "shape_sim.csv"), index=False)
    pd.concat(TV).to_csv(os.path.join(OUT, "tv_sim.csv"), index=False)
    pd.DataFrame(A).to_csv(os.path.join(OUT, "anova_sim.csv"), index=False)
    json.dump(cfg, open(os.path.join(OUT, "sim_cfg.json"), "w"))


if __name__ == "__main__":
    if sys.argv[1] == "sim":
        run_sim()
    else:
        run_ds(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1,
               [int(f) for f in sys.argv[3].split(',')] if len(sys.argv) > 3 else range(5))
