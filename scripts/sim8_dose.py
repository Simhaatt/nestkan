"""PROTOCOL_v8 item 4: dose-response calibration of the PH diagnostic.

Data: log h = log h0(t) + g_additive(x) + delta * (1 - 2 F_t(t)) * x4, with g_additive, h0, covariates and
censoring exactly as scripts/sim.py (delta = 0 is ph_additive, delta = 1 is nonph).
delta in {0, 0.25, 0.5, 0.75, 1.0}; n = 1000; reps 0-19; replicate RNG default_rng([2027, rep, round(100 delta)]).
Split as sim.py (validation = first n/5 of default_rng(rep).permutation).

Per replicate:
  * tune NestKAN-sel (bench3.NEST_SPACE, K = 6, validation rNLL; config sampler and fit seed as sim8.py:
    default_rng([rep, sum(map(ord, "NestKAN-sel"))]), seed = rep) and record use_residual of the selection;
  * the selected configuration's fit (seed = rep; this IS the refit, the tuning fit uses the same seed,
    data and config) -> functional-ANOVA shares (interp.anova, rng default_rng([rep, 7]) as interp.run_sim);
  * x4 partial dependence: per-unit log-HR slope per interval, (eta(x4=+1) - eta(x4=-1)) / 2 averaged over
    the 800 training subjects (as interp.run_sim); its SD over intervals (ddof = 0);
  * Grambsch-Therneau (interp.cox_and_gt: lifelines CoxPHFitter ridge 0.01, proportional_hazard_test with
    rank transform) on the training subjects: p for x4 and min p over covariates.
Output: results/sim8_dose.csv (resumable by (delta, rep)); rep-major order across deltas.

usage: python -u scripts/sim8_dose.py
"""
import os, sys, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np, pandas as pd
import sim, bench2
from bench2 import sample_configs
from bench3 import NEST_SPACE
import interp
from dkan.data import make_preprocessor, Discretiser
from dkan.models import MODELS, fit_discrete, predict_grid_survival, interp_survival
from dkan.metrics import evaluate, reference_cuts

DELTAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N, REPS = 1000, range(20)
K = int(os.environ.get("SIM8_K", "6"))
SEL_FAM = "NestKAN-sel"
OUT = os.environ.get("SIM8_DOSE_OUT", os.path.join(bench2.ROOT, "results", "sim8_dose.csv"))


def log_rel_hazard_dose(X, t, delta):
    base = sim.log_rel_hazard(X, t, "ph_additive")
    return base + delta * (sim.log_rel_hazard(X, t, "nonph") - base)   # = delta * (1 - 2F_t) x4


def simulate_dose(n, delta, rng, tfine):
    """sim.simulate with the dose hazard (identical draws and censoring)."""
    X = sim.make_x(n, rng)
    h0 = sim.SHAPE / sim.SCALE * (tfine / sim.SCALE) ** (sim.SHAPE - 1)
    H = np.cumsum(h0[None, :] * np.exp(log_rel_hazard_dose(X, tfine, delta)), 1) * (tfine[1] - tfine[0])
    E = rng.exponential(size=n)
    idx = (H < E[:, None]).sum(1)
    T = tfine[np.clip(idx, 0, len(tfine) - 1)]
    T[idx >= len(tfine)] = tfine[-1]
    C = rng.uniform(0, 4.0, size=n)
    time_ = np.minimum(T, C); event = (T <= C) & (idx < len(tfine))
    time_ = np.minimum(time_, tfine[-1])
    df = pd.DataFrame(X, columns=[f"num_x{j+1}" for j in range(8)])
    df["time"] = np.maximum(time_, 1e-3); df["event"] = event.astype(int)
    return df


def fit_cfg(cfg, df, tr, va, time_all, ev_all, tgrid, seed):
    cfg = dict(cfg)
    pp = make_preprocessor("dual").fit(df.iloc[tr])
    Xtr, Xva = pp.transform(df.iloc[tr]), pp.transform(df.iloc[va])
    T = cfg.pop("T"); lr = cfg.pop("lr"); wd = cfg.pop("wd"); lam = cfg.pop("lam", 1e-3)
    disc = Discretiser(T, "km").fit(time_all[tr], ev_all[tr])
    itr, etr = disc.transform(time_all[tr], ev_all[tr])
    iva, eva = disc.transform(time_all[va], ev_all[va])
    model = MODELS["NestKAN"](Xtr.shape[1], disc.T, v=disc.v, **cfg)
    model.init_baseline(itr, etr)
    model, vl, ep = fit_discrete(model, Xtr, itr, etr, Xva, iva, eva, lr=lr, wd=wd, lam=lam, seed=seed)
    Sva = interp_survival(predict_grid_survival(model, Xva), disc.cuts, tgrid)
    return Sva, ep, (model, pp, disc, Xtr, itr)


def run_one(delta, rep):
    t0 = time.time()
    tfine = np.linspace(1e-4, 4.0, 1001)
    rng = np.random.default_rng([2027, rep, int(round(100 * delta))])
    df = simulate_dose(N, delta, rng, tfine)
    time_all = df["time"].to_numpy(float); ev_all = df["event"].to_numpy(int)
    r2 = np.random.default_rng(rep).permutation(np.arange(N))
    va, tr = np.sort(r2[: N // 5]), np.sort(r2[N // 5:])
    cuts = reference_cuts(time_all[tr], ev_all[tr], 30)
    tq = np.quantile(time_all[tr][ev_all[tr] == 1], np.linspace(0, 1, 301))
    tgrid = np.unique(np.r_[0.0, np.linspace(0, time_all[tr].max(), sim.TGRID_N), cuts, tq])
    rng_c = np.random.default_rng([rep, sum(map(ord, SEL_FAM))])
    best, arm = None, {True: np.inf, False: np.inf}
    n_arm = {True: 0, False: 0}
    for cfg in sample_configs(NEST_SPACE, K, rng_c):
        try:
            Sva, ep, fit = fit_cfg(cfg, df, tr, va, time_all, ev_all, tgrid, rep)
        except Exception as e:
            print(f"  delta={delta} rep{rep} cfg FAILED {e!r}"[:200], flush=True); continue
        v = evaluate(Sva, tgrid, time_all[va], ev_all[va], time_all[tr], ev_all[tr], cuts=cuts)["rNLL"]
        ur = bool(cfg["use_residual"]); arm[ur] = min(arm[ur], v); n_arm[ur] += 1
        if best is None or v < best[0]:
            best = (v, cfg, fit)
        else:
            del fit
    if best is None:
        return None
    v, cfg, (m, pp, disc, Xtr, itr) = best
    rg = np.random.default_rng([rep, 7])
    sh, *_ = interp.anova(m, Xtr, itr, cfg["grid"], rg)
    sub = df.iloc[tr].iloc[rg.choice(len(tr), min(800, len(tr)), replace=False)]
    a, b = sub.copy(), sub.copy(); a["num_x4"] = -1.0; b["num_x4"] = 1.0
    slope = (interp.eta_of(m, pp.transform(b)) - interp.eta_of(m, pp.transform(a))).mean(0) / 2
    tmid = (disc.cuts[:-1] + disc.cuts[1:]) / 2
    tmid = tmid[: m.T]
    true_slope = delta * (1 - 2 * (1 - np.exp(-(tmid / 0.6) ** sim.SHAPE)))
    _, gt = interp.cox_and_gt(pp, df.iloc[tr], time_all[tr], ev_all[tr], do_gt=True)
    gt_x4 = gt.loc[gt.covariate == "num_x4", "p"]
    row = dict(delta=delta, rep=rep, n=N, event_rate=float(ev_all.mean()), best_cfg=json.dumps(cfg, default=str),
               use_residual=bool(cfg["use_residual"]), gate=(float(m.gate.detach()) if m.use_residual else 0.0),
               T=int(m.T), val_rNLL=v,
               val_rNLL_best_on=(arm[True] if np.isfinite(arm[True]) else np.nan),
               val_rNLL_best_off=(arm[False] if np.isfinite(arm[False]) else np.nan),
               n_on=n_arm[True], n_off=n_arm[False],
               **{f"share_{k}": float(sh[k]) for k in sh},
               x4_slope_sd=float(np.std(slope)), x4_slope_mean=float(np.mean(slope)),
               x4_slope_range=float(slope.max() - slope.min()),
               x4_true_slope_sd=float(np.std(true_slope)),
               gt_p_x4=(float(gt_x4.iloc[0]) if len(gt_x4) else np.nan), gt_min_p=float(gt["p"].min()),
               gt_min_cov=str(gt.loc[gt["p"].idxmin(), "covariate"]),
               secs=round(time.time() - t0, 1))
    return row


if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT); done = set(zip((100 * p["delta"]).round().astype(int), p["rep"]))
    for rep in REPS:
        for delta in DELTAS:
            if (int(round(100 * delta)), rep) in done:
                continue
            row = run_one(delta, rep)
            if row is None:
                continue
            pd.DataFrame([row]).to_csv(OUT, mode="a", index=False, header=not os.path.exists(OUT))
            print(f"dose delta={delta} rep{rep} keep={row['use_residual']} share_int={row['share_interaction_of_cov']:.4f} "
                  f"x4sd={row['x4_slope_sd']:.3f} (true {row['x4_true_slope_sd']:.3f}) gt_p_x4={row['gt_p_x4']:.3g} "
                  f"[{row['secs']:.0f}s]", flush=True)
            if os.environ.get("SIM8_SMOKE"):
                sys.exit(0)
    print("DOSE_ALL_DONE", flush=True)
