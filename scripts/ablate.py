"""Ablations with the selected NestKAN configuration held fixed (per dataset: the
most frequently selected config in tuned_v4), same folds and evaluation as bench2.

 A1 inputs : spline block on ECDF scale vs min-max scale
 A2 grid   : T in {10, 20, 40, 80} for NestKAN vs Logistic-Hazard (its selected cfg)
 A3 blocks : Nest-lin -> Nest-PH -> NestKAN (residual surface)

usage: python -u scripts/ablate.py <dataset>   -> results/ablate_<ds>.csv (resumable)
"""
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "pylib"))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
import bench2
from bench2 import DROP, FAMILIES, fit_one
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from dkan.data import load_survset, Preprocessor
from dkan.metrics import evaluate, reference_cuts, censoring_model


def modal_cfg(ds, fam):
    t = pd.read_csv(os.path.join(bench2.ROOT, "results", f"tuned_v4_{ds}.csv"))
    return json.loads(t[t.model == fam].best_cfg.value_counts().index[0])


def run(ds):
    df = load_survset(ds).drop(columns=DROP.get(ds, []), errors="ignore")
    ta = df["time"].to_numpy(float); ea = df["event"].to_numpy(int)
    out = os.path.join(bench2.ROOT, "results", f"ablate_{ds}.csv")
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out); done = set(zip(prev.fold, prev.arm))
    nk, lh = modal_cfg(ds, "NestKAN"), modal_cfg(ds, "LogisticHazard")
    arms = [("A1 ecdf", "NestKAN", "dual", nk, {}), ("A1 minmax", "NestKAN", "dual_minmax", nk, {})]
    for T in (10, 20, 40, 80):
        arms += [(f"A2 NestKAN T{T}", "NestKAN", "dual", dict(nk, T=T), {}),
                 (f"A2 LH-MLP T{T}", "LH-MLP", "standard", dict(lh, T=T), {})]
    arms += [("A3 Nest-lin", "NestKAN", "dual", nk, dict(use_additive=False, use_residual=False)),
             ("A3 Nest-PH", "NestKAN", "dual", nk, dict(use_residual=False)),
             ("A3 NestKAN", "NestKAN", "dual", nk, {})]
    skf = list(StratifiedKFold(5, shuffle=True, random_state=0).split(df, ea))
    for fold in range(5):
        tr, te = skf[fold]
        tr, va = train_test_split(tr, test_size=0.2, stratify=ea[tr], random_state=fold)
        cuts = reference_cuts(ta[tr], ea[tr], 30)
        tq = np.quantile(ta[tr][ea[tr] == 1], np.linspace(0, 1, 301))
        tgrid = np.unique(np.r_[0.0, np.linspace(0, ta[tr].max(), 200), cuts, tq])
        ppc = Preprocessor("standard").fit(df.iloc[tr])
        Gte = censoring_model(ppc.transform(df.iloc[tr]), ta[tr], ea[tr], ppc.transform(df.iloc[te]), ta[te], tgrid)
        for arm, mkey, scaling, cfg, fixed in arms:
            if (fold, arm) in done:
                continue
            cfg = {k: v for k, v in cfg.items() if not (fixed.get("use_residual") is False and k == "hidden")}
            t0 = time.time()
            S, ep = fit_one("disc", mkey, cfg, fixed, df, tr, va, [("te", te)], ta, ea, tgrid, scaling, fold)
            m = evaluate(S["te"], tgrid, ta[te], ea[te], ta[tr], ea[tr], cuts=cuts, Gcond=Gte)
            pd.DataFrame([dict(dataset=ds, fold=fold, arm=arm, cfg=json.dumps(cfg), epochs=ep,
                               secs=round(time.time() - t0, 1), **m)]).to_csv(
                out, mode="a", index=False, header=not os.path.exists(out))
            print(f"{ds} f{fold} {arm:18s} rNLL={m['rNLL']:.4f} C={m['C_td']:.4f} IBS={m['IBS']:.4f}", flush=True)


if __name__ == "__main__":
    for ds in sys.argv[1].split(","):
        run(ds)
