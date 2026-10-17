"""PROTOCOL_v8 item 1: extend the known-truth simulation and add the NestKAN-sel family.

(a) The 8 existing sim.py families on the new replicates: calls sim.run(scen, n, [rep]) unchanged,
    which appends to results/sim_<scen>_n<n>.csv and skips (rep, model) pairs already present.
(b) NestKAN-sel = NestKAN with the v5 search space (bench3.NEST_SPACE, so use_residual in {True, False}
    is tuned), on ALL replicates, same data RNG / split / K / validation-rNLL selection / fit seed as
    sim.py.  Config sampler seeded exactly as sim.py does for any family:
    default_rng([rep, sum(map(ord, "NestKAN-sel"))]).  Rows -> results/sim8_sel_<scen>_n<n>.csv
    (resumable by rep), all trials -> results/sim8_sel_trials_<scen>_n<n>.csv.
    Extra logging: use_residual of the selected config, its fitted gate s, and the best validation
    rNLL within each arm (surface on / off) of the K sampled configs.

Order (pre-registered): every n = 1000 cell first (rep-major across the 4 scenarios, so partial
completion stays balanced), then n = 4000.

usage: python -u scripts/sim8.py [n=all]
"""
import os, sys, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np, pandas as pd
import sim, bench2
from bench2 import sample_configs
from bench3 import NEST_SPACE
from dkan.data import make_preprocessor, Discretiser
from dkan.models import MODELS, fit_discrete, predict_grid_survival, interp_survival
from dkan.metrics import evaluate, reference_cuts

SCENS = ["ph_linear", "ph_additive", "nonph", "nonph_inter"]
TARGET = {1000: range(0, 20), 4000: range(0, 10)}
SEL_FAM = "NestKAN-sel"
K = 6


def fit_nest_sel(cfg, df, tr, va, te, time_all, ev_all, tgrid, seed):
    """bench2.fit_one for kind='disc', mkey='NestKAN', scaling='dual', fixed={} -- but also returns the model."""
    cfg = dict(cfg)
    pp = make_preprocessor("dual").fit(df.iloc[tr])
    X = {k: pp.transform(df.iloc[idx]) for k, idx in (("tr", tr), ("va", va), ("te", te))}
    T = cfg.pop("T"); lr = cfg.pop("lr"); wd = cfg.pop("wd"); lam = cfg.pop("lam", 1e-3)
    disc = Discretiser(T, "km").fit(time_all[tr], ev_all[tr])
    itr, etr = disc.transform(time_all[tr], ev_all[tr])
    iva, eva = disc.transform(time_all[va], ev_all[va])
    model = MODELS["NestKAN"](X["tr"].shape[1], disc.T, v=disc.v, **cfg)
    model.init_baseline(itr, etr)
    model, vl, ep = fit_discrete(model, X["tr"], itr, etr, X["va"], iva, eva, lr=lr, wd=wd, lam=lam, seed=seed)
    S = {k: interp_survival(predict_grid_survival(model, X[k]), disc.cuts, tgrid) for k in ("va", "te")}
    return S, ep, model


def run_sel(scen, n, rep):
    out_path = os.path.join(bench2.ROOT, "results", f"sim8_sel_{scen}_n{n}.csv")
    tri_path = os.path.join(bench2.ROOT, "results", f"sim8_sel_trials_{scen}_n{n}.csv")
    if os.path.exists(out_path) and rep in set(pd.read_csv(out_path)["rep"]):
        return
    # --- data, split, grids: verbatim from sim.run ---
    tfine = np.linspace(1e-4, 4.0, 1001)
    rng = np.random.default_rng([2026, rep, n, sum(map(ord, scen))])
    df_tr, _, _ = sim.simulate(n, scen, rng, tfine)
    df_te, Xte, Hte = sim.simulate(5000, scen, rng, tfine)
    df = pd.concat([df_tr, df_te], ignore_index=True)
    time_all = df["time"].to_numpy(float); ev_all = df["event"].to_numpy(int)
    idx_tr = np.arange(n); te = np.arange(n, n + 5000)
    r2 = np.random.default_rng(rep).permutation(idx_tr)
    va, tr = np.sort(r2[: n // 5]), np.sort(r2[n // 5:])
    cuts = reference_cuts(time_all[tr], ev_all[tr], 30)
    tq = np.quantile(time_all[tr][ev_all[tr] == 1], np.linspace(0, 1, 301))
    tgrid = np.unique(np.r_[0.0, np.linspace(0, time_all[tr].max(), sim.TGRID_N), cuts, tq])
    S_true = np.exp(-np.stack([np.interp(tgrid, tfine, Hte[i]) for i in range(len(te))]))
    hi = np.quantile(time_all[te], 0.9)
    m = tgrid <= hi
    # --- tuning ---
    rng_c = np.random.default_rng([rep, sum(map(ord, SEL_FAM))])
    best, trials, t0 = None, [], time.time()
    arm_best = {True: np.inf, False: np.inf}
    for ci, cfg in enumerate(sample_configs(NEST_SPACE, K, rng_c)):
        try:
            S, ep, model = fit_nest_sel(cfg, df, tr, va, te, time_all, ev_all, tgrid, rep)
        except Exception as e:
            print(f"  {SEL_FAM} FAILED {e!r}"[:200], flush=True)
            trials.append(dict(scenario=scen, n=n, rep=rep, ci=ci, cfg=json.dumps(cfg, default=str), error=repr(e)[:200]))
            continue
        mv = evaluate(S["va"], tgrid, time_all[va], ev_all[va], time_all[tr], ev_all[tr], cuts=cuts)
        gate = float(model.gate.detach()) if model.use_residual else 0.0
        arm_best[bool(cfg["use_residual"])] = min(arm_best[bool(cfg["use_residual"])], mv["rNLL"])
        trials.append(dict(scenario=scen, n=n, rep=rep, ci=ci, cfg=json.dumps(cfg, default=str), epochs=ep,
                           use_residual=bool(cfg["use_residual"]), gate=gate, val_rNLL=mv["rNLL"]))
        if best is None or mv["rNLL"] < best[0]:
            best = (mv["rNLL"], cfg, S["te"], gate)
        del model
    if best is None:
        return
    mt = evaluate(best[2], tgrid, time_all[te], ev_all[te], time_all[tr], ev_all[tr], cuts=cuts)
    ise = np.trapezoid(((best[2] - S_true) ** 2)[:, m].mean(0), tgrid[m]) / tgrid[m][-1]
    row = dict(scenario=scen, n=n, rep=rep, model=SEL_FAM, ISE=ise, **mt, secs=round(time.time() - t0, 1),
               best_cfg=json.dumps(best[1], default=str), use_residual=bool(best[1]["use_residual"]),
               gate=best[3], val_rNLL=best[0],
               val_rNLL_best_on=(arm_best[True] if np.isfinite(arm_best[True]) else np.nan),
               val_rNLL_best_off=(arm_best[False] if np.isfinite(arm_best[False]) else np.nan),
               n_on=sum(1 for t in trials if t.get("use_residual") is True),
               n_off=sum(1 for t in trials if t.get("use_residual") is False))
    pd.DataFrame(trials).to_csv(tri_path, mode="a", index=False, header=not os.path.exists(tri_path))
    pd.DataFrame([row]).to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path))
    print(f"{scen} n={n} rep{rep} {SEL_FAM:15s} ISE={ise:.5f} C={mt['C_td']:.4f} keep={row['use_residual']} "
          f"gate={best[3]:.3f} [{row['secs']:.0f}s]", flush=True)


if __name__ == "__main__":
    ns = [int(sys.argv[1])] if len(sys.argv) > 1 and sys.argv[1] != "all" else [1000, 4000]
    for n in ns:
        for rep in TARGET[n]:
            for scen in SCENS:
                t0 = time.time()
                sim.run(scen, n, [rep], K)      # existing 8 families; skips (rep, model) present
                run_sel(scen, n, rep)
                print(f"== {scen} n={n} rep{rep} done [{time.time()-t0:.0f}s]", flush=True)
    print("SIM8_ALL_DONE", flush=True)
