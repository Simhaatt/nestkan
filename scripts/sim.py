"""Known-truth simulation: when does the nested KAN hazard help?

Scenarios (continuous-time hazards, d = 8 covariates, three of them log-normal
skewed, x8 pure noise; 20-27% independent uniform censoring):
  S1 ph_linear   : log h = log h0(t) + 0.6 x1 - 0.4 x2 + 0.3 log(x5)
  S2 ph_additive : log h = log h0(t) + sin(1.5 x1) + 0.5 x2^2 - 0.5 + 0.7 log1p(x5) + 0.5 tanh(2 x3)
  S3 nonph       : S2 + (1 - 2 F_t(t)) * 1.0 x4   (effect of x4 reverses over follow-up)
  S4 nonph_inter : S3 + 0.6 x1 x2
h0 = Weibull(shape 1.3).  The true survival S(t|x) is computed on the evaluation
grid, so every model is also scored by its integrated squared error against the
truth (ISE) on a 5,000-subject test set.

usage: python -u scripts/sim.py <scenario> <n> <rep_from>-<rep_to> [K=6]
writes results/sim_<scenario>_n<n>.csv (resumable by (rep, model)).
"""
import os, sys, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np, pandas as pd
import bench2
from bench2 import FAMILIES, sample_configs, fit_one
from dkan.metrics import evaluate, reference_cuts

SIM_FAMILIES = ["LogisticHazard", "PMF", "DeepHit", "DeepSurv", "NestKAN", "NestKAN-PH", "CoxPH", "RSF"]
TGRID_N = 200
SHAPE, SCALE = 1.3, 1.0


def make_x(n, rng):
    X = rng.standard_normal((n, 8))
    X[:, 4] = np.exp(0.8 * X[:, 4])          # skewed (log-normal)
    X[:, 5] = np.exp(1.0 * X[:, 5])          # skewed, noise-free signal not used
    X[:, 6] = np.exp(0.6 * X[:, 6])          # skewed
    return X


def log_rel_hazard(X, t, scen):
    """returns (n, len(t)) log relative hazard at times t (t in model time units)."""
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    if scen == "ph_linear":
        g = 0.6 * x1 - 0.4 * x2 + 0.3 * np.log(x5)
        return np.repeat(g[:, None], len(t), 1)
    g = np.sin(1.5 * x1) + 0.5 * x2 ** 2 - 0.5 + 0.7 * np.log1p(x5) + 0.5 * np.tanh(2 * x3)
    out = np.repeat(g[:, None], len(t), 1)
    if scen in ("nonph", "nonph_inter"):
        Ft = 1 - np.exp(-(t / 0.6) ** SHAPE)    # x4 effect reverses sign near the median time
        out = out + np.outer(x4, 1 - 2 * Ft) * 1.0
    if scen == "nonph_inter":
        out = out + (0.6 * x1 * x2)[:, None]
    return out


def simulate(n, scen, rng, tfine):
    X = make_x(n, rng)
    h0 = SHAPE / SCALE * (tfine / SCALE) ** (SHAPE - 1)
    H = np.cumsum(h0[None, :] * np.exp(log_rel_hazard(X, tfine, scen)), 1) * (tfine[1] - tfine[0])
    E = rng.exponential(size=n)
    idx = (H < E[:, None]).sum(1)                                # first grid index where H >= E
    T = tfine[np.clip(idx, 0, len(tfine) - 1)]
    T[idx >= len(tfine)] = tfine[-1]
    C = rng.uniform(0, 4.0, size=n)
    time_ = np.minimum(T, C); event = (T <= C) & (idx < len(tfine))
    time_ = np.minimum(time_, tfine[-1])
    df = pd.DataFrame(X, columns=[f"num_x{j+1}" for j in range(8)])
    df["time"] = np.maximum(time_, 1e-3); df["event"] = event.astype(int)
    return df, X, H


def run(scen, n, reps, K=6):
    out_path = os.path.join(bench2.ROOT, "results", f"sim_{scen}_n{n}.csv")
    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path); done = set(zip(prev["rep"], prev["model"]))
    tfine = np.linspace(1e-4, 4.0, 1001)
    for rep in reps:
        rng = np.random.default_rng([2026, rep, n, sum(map(ord, scen))])
        df_tr, _, _ = simulate(n, scen, rng, tfine)
        df_te, Xte, Hte = simulate(5000, scen, rng, tfine)
        df = pd.concat([df_tr, df_te], ignore_index=True)
        time_all = df["time"].to_numpy(float); ev_all = df["event"].to_numpy(int)
        idx_tr = np.arange(n); te = np.arange(n, n + 5000)
        r2 = np.random.default_rng(rep).permutation(idx_tr)
        va, tr = np.sort(r2[: n // 5]), np.sort(r2[n // 5:])
        cuts = reference_cuts(time_all[tr], ev_all[tr], 30)
        tq = np.quantile(time_all[tr][ev_all[tr] == 1], np.linspace(0, 1, 301))
        tgrid = np.unique(np.r_[0.0, np.linspace(0, time_all[tr].max(), TGRID_N), cuts, tq])
        S_true = np.exp(-np.stack([np.interp(tgrid, tfine, Hte[i]) for i in range(len(te))]))
        hi = np.quantile(time_all[te], 0.9)
        m = tgrid <= hi
        for fam in SIM_FAMILIES:
            if (rep, fam) in done:
                continue
            kind, mkey, scaling, space, fixed = FAMILIES[fam]
            rng_c = np.random.default_rng([rep, sum(map(ord, fam))])
            best, t0 = None, time.time()
            for cfg in sample_configs(space, K, rng_c):
                try:
                    S, _ = fit_one(kind, mkey, cfg, fixed, df, tr, va, [("te", te)], time_all, ev_all,
                                   tgrid, scaling, rep)
                except Exception as e:
                    print(f"  {fam} FAILED {e!r}"[:200], flush=True); continue
                mv = evaluate(S["va"], tgrid, time_all[va], ev_all[va], time_all[tr], ev_all[tr], cuts=cuts)
                if best is None or mv["rNLL"] < best[0]:
                    best = (mv["rNLL"], cfg, S["te"])
            if best is None:
                continue
            mt = evaluate(best[2], tgrid, time_all[te], ev_all[te], time_all[tr], ev_all[tr], cuts=cuts)
            ise = np.trapezoid(((best[2] - S_true) ** 2)[:, m].mean(0), tgrid[m]) / tgrid[m][-1]
            row = dict(scenario=scen, n=n, rep=rep, model=fam, ISE=ise, **mt,
                       secs=round(time.time() - t0, 1), best_cfg=json.dumps(best[1], default=str))
            pd.DataFrame([row]).to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path))
            print(f"{scen} n={n} rep{rep} {fam:15s} ISE={ise:.5f} C={mt['C_td']:.4f} IBS={mt['IBS']:.4f}", flush=True)


if __name__ == "__main__":
    scen, n = sys.argv[1], int(sys.argv[2])
    a, b = map(int, sys.argv[3].split("-"))
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    run(scen, n, range(a, b + 1), K)
