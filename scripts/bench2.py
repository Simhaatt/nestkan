"""Tuned benchmark: every model family gets the same random-search budget (K
configurations per fold), selected on validation IBS (model-agnostic and
comparable across grid sizes and likelihoods).  Test metrics of the selected
configuration are written to results/tuned_<ds>.csv, all trials to
results/trials_<ds>.csv.  Resumable: (fold, family) pairs already present are
skipped.

usage: python -u scripts/bench2.py <dataset> [K=8] [families=all] [folds=0-4]
"""
import os, sys, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pylib"))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from dkan.data import make_preprocessor, Discretiser, load_survset
from dkan.models import (MODELS, fit_discrete, predict_grid_survival, interp_survival,
                         DeepSurv, fit_deepsurv, deepsurv_survival)
from dkan.metrics import evaluate, reference_cuts, censoring_model
from dkan.data import Preprocessor

torch.set_num_threads(int(os.environ.get("DKAN_THREADS", "4")))
ROOT = os.path.join(_HERE, "..")
DROP = {"support2": ["num_surv2m", "num_surv6m", "fac_sfdm2", "num_hday"],
        "rhc": ["num_surv2md1"], "nwtco": ["fac_in_subcohort"],
        "flchain": ["fac_chapter"]}  # flchain chapter = cause of death (defined only for deaths)

MLP_SPACE = dict(width=[32, 64, 128], depth=[1, 2, 3], dropout=[0.0, 0.1, 0.3],
                 lr=[1e-3, 3e-3], wd=[1e-5, 1e-4, 1e-3], T=[10, 20, 40])
KAN_SPACE = dict(hidden=[4, 8, 16], grid=[3, 5, 8], lam=[1e-4, 1e-3, 1e-2],
                 lr=[1e-3, 3e-3], wd=[1e-5, 1e-4], lam_alpha=[0.1, 1.0, 10.0], T=[10, 20, 40])
FAMILIES = {
    # name: (kind, model key, scaling, space, fixed kwargs)
    "LogisticHazard": ("disc", "LH-MLP", "standard", MLP_SPACE, {}),
    "PMF": ("disc", "PMF", "standard", MLP_SPACE, {}),
    "MTLR": ("disc", "MTLR", "standard", MLP_SPACE, {}),
    "DeepHit": ("disc", "DeepHit", "standard", MLP_SPACE, {}),
    "DeepSurv": ("deepsurv", None, "standard", {k: v for k, v in MLP_SPACE.items() if k != "T"}, {}),
    "NestKAN": ("disc", "NestKAN", "dual", KAN_SPACE, {}),
    "NestKAN-PH": ("disc", "NestKAN", "dual", {k: v for k, v in KAN_SPACE.items() if k != "hidden"},
                   dict(use_residual=False)),
    "CoxPH": ("cox", None, "standard", dict(alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]), {}),
    "RSF": ("rsf", None, "standard", dict(min_samples_leaf=[5, 10, 20, 40],
                                          max_features=["sqrt", 0.33, 0.5]), {}),
}


def sample_configs(space, K, rng):
    keys = list(space)
    seen, out = set(), []
    total = int(np.prod([len(space[k]) for k in keys]))
    while len(out) < min(K, total):
        c = {k: space[k][rng.integers(len(space[k]))] for k in keys}
        key = json.dumps(c, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key); out.append(c)
    return out


def fit_one(kind, mkey, cfg, fixed, df, tr, va, ev_sets, time_all, ev_all, tgrid, scaling, seed):
    """Returns dict of survival matrices for each requested eval set."""
    cfg = dict(cfg)
    pp = make_preprocessor(scaling).fit(df.iloc[tr])
    X = {k: pp.transform(df.iloc[idx]) for k, idx in [("tr", tr), ("va", va)] + ev_sets}
    if kind == "disc":
        T = cfg.pop("T"); lr = cfg.pop("lr"); wd = cfg.pop("wd"); lam = cfg.pop("lam", 1e-3)
        disc = Discretiser(T, "km").fit(time_all[tr], ev_all[tr])
        itr, etr = disc.transform(time_all[tr], ev_all[tr])
        iva, eva = disc.transform(time_all[va], ev_all[va])
        kw = dict(cfg, **fixed)
        if mkey in ("NestKAN",) or mkey.startswith("Time"):
            kw["v"] = disc.v
        model = MODELS[mkey](X["tr"].shape[1], disc.T, **kw)
        if hasattr(model, "init_baseline"):
            model.init_baseline(itr, etr)
        model, vl, ep = fit_discrete(model, X["tr"], itr, etr, X["va"], iva, eva,
                                     lr=lr, wd=wd, lam=lam, seed=seed)
        return {k: interp_survival(predict_grid_survival(model, X[k]), disc.cuts, tgrid)
                for k in X if k != "tr"}, ep
    if kind == "deepsurv":
        lr = cfg.pop("lr"); wd = cfg.pop("wd")
        model = DeepSurv(X["tr"].shape[1], **cfg)
        model, vl, ep = fit_deepsurv(model, X["tr"], time_all[tr], ev_all[tr], X["va"],
                                     time_all[va], ev_all[va], lr=lr, wd=wd, seed=seed)
        return {k: deepsurv_survival(model, X["tr"], time_all[tr], ev_all[tr], X[k], tgrid)
                for k in X if k != "tr"}, ep
    from sksurv.util import Surv
    y = Surv.from_arrays(ev_all[tr].astype(bool), time_all[tr])
    if kind == "cox":
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        est = CoxPHSurvivalAnalysis(alpha=cfg["alpha"], ties="efron")
    else:
        from sksurv.ensemble import RandomSurvivalForest
        est = RandomSurvivalForest(n_estimators=100, min_samples_leaf=cfg["min_samples_leaf"],
                                   max_features=cfg["max_features"], n_jobs=1,
                                   random_state=seed, low_memory=False)
    est.fit(X["tr"], y)
    out = {}
    for k in X:
        if k == "tr":
            continue
        fns = est.predict_survival_function(X[k])
        lo, hi = fns[0].domain
        S = np.vstack([f(np.clip(tgrid, lo, hi)) for f in fns])
        S[:, tgrid < lo] = 1.0
        out[k] = S
    del est
    return out, 0


def run(ds, K=8, families=None, folds=range(5), seed=0):
    df = load_survset(ds).drop(columns=DROP.get(ds, []), errors="ignore")
    time_all = df["time"].to_numpy(float); ev_all = df["event"].to_numpy(int)
    tag = os.environ.get("DKAN_TAG", "_v4")
    SEL = os.environ.get("DKAN_SELECT", "rNLL")
    res_path = os.path.join(ROOT, "results", f"tuned{tag}_{ds}.csv")
    tri_path = os.path.join(ROOT, "results", f"trials{tag}_{ds}.csv")
    done = set()
    if os.path.exists(res_path):
        prev = pd.read_csv(res_path); done = set(zip(prev["fold"], prev["model"]))
    skf = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(df, ev_all))
    for fold in folds:
        tr, te = skf[fold]
        tr, va = train_test_split(tr, test_size=0.2, stratify=ev_all[tr], random_state=seed + fold)
        cuts = reference_cuts(time_all[tr], ev_all[tr], 30)
        # evaluation grid: uniform + reference cuts + 300 quantiles of training event
        # times, so S(t) is resolved where events occur (C_td ties otherwise)
        tq = np.quantile(time_all[tr][ev_all[tr] == 1], np.linspace(0, 1, 301))
        tgrid = np.unique(np.r_[0.0, np.linspace(0, time_all[tr].max(), 200), cuts, tq])
        ppc = Preprocessor("standard").fit(df.iloc[tr])
        Xc_tr = ppc.transform(df.iloc[tr])
        Gc = {k: censoring_model(Xc_tr, time_all[tr], ev_all[tr], ppc.transform(df.iloc[idx]), time_all[idx], tgrid)
              for k, idx in (("va", va), ("te", te))}
        for fam in (families or FAMILIES):
            if (fold, fam) in done:
                continue
            kind, mkey, scaling, space, fixed = FAMILIES[fam]
            rng = np.random.default_rng([seed, fold, sum(map(ord, fam))])
            best, trials, t0 = None, [], time.time()
            for ci, cfg in enumerate(sample_configs(space, K, rng)):
                try:
                    S, ep = fit_one(kind, mkey, cfg, fixed, df, tr, va, [("te", te)], time_all,
                                    ev_all, tgrid, scaling, seed + fold)
                except Exception as e:  # record, never silently drop
                    trials.append(dict(dataset=ds, fold=fold, model=fam, cfg=json.dumps(cfg, default=str),
                                       error=repr(e)[:200]))
                    print(f"  {fam} cfg{ci} FAILED {e!r}"[:200], flush=True)
                    continue
                mv = evaluate(S["va"], tgrid, time_all[va], ev_all[va], time_all[tr], ev_all[tr], cuts=cuts, Gcond=Gc["va"])
                trials.append(dict(dataset=ds, fold=fold, model=fam, cfg=json.dumps(cfg, default=str),
                                   epochs=ep, **{f"val_{k}": v for k, v in mv.items()}))
                if best is None or mv[SEL] < best[0]:
                    best = (mv[SEL], cfg, S["te"])
            if best is None:
                continue
            mt = evaluate(best[2], tgrid, time_all[te], ev_all[te], time_all[tr], ev_all[tr], cuts=cuts, Gcond=Gc["te"])
            row = dict(dataset=ds, fold=fold, model=fam, n=len(df), K=len(trials),
                       secs=round(time.time() - t0, 1), best_cfg=json.dumps(best[1], default=str),
                       select=SEL, val_sel=best[0], **mt)
            pd.DataFrame([row]).to_csv(res_path, mode="a", index=False, header=not os.path.exists(res_path))
            pd.DataFrame(trials).to_csv(tri_path, mode="a", index=False, header=not os.path.exists(tri_path))
            print(f"{ds} f{fold} {fam:15s} rNLL={mt['rNLL']:.4f} C={mt['C_td']:.4f} IBS={mt['IBS']:.4f} "
                  f"IBScx={mt['IBS_cx']:.4f} [{row['secs']:.0f}s] {best[1]}", flush=True)


if __name__ == "__main__":
    ds = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    fams = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] != "all" else None
    folds = range(5)
    if len(sys.argv) > 4:
        a, b = map(int, sys.argv[4].split("-")); folds = range(a, b + 1)
    run(ds, K, fams, folds)
