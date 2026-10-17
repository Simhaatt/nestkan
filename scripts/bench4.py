"""v6 (pre-registered in PROTOCOL_v6.md): adds SurvKAN, KAPLAN-HR (faithful re-implementations)
and the official CoxKAN with its symbolic step to the v5 benchmark, under the identical v5
protocol: same repeated-CV splits (rep seeds), K = 8 random configurations, selection on
validation rNLL, 3-seed ensemble of the selected configuration, same metrics.  Reuses bench3.run.

usage: python -u scripts/bench4.py <dataset> [K=8] [rep=0] [families=SurvKAN,KAPLAN-HR,CoxKAN-sym]
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("DKAN_TAG", "_v6")
import bench3
from bench3 import make_preprocessor
from dkan.ctkan import CTKAN, SurvKANLoss, KaplanLoss, fit_ctkan, ctkan_survival

SURVKAN_SPACE = dict(hidden=[0, 1, 2, 3], grid=[3, 5, 8], base_fn=["identity", "silu"],
                     lam=[0.0, 1e-4, 1e-3, 1e-2], lr=[1e-3, 3e-3, 1e-2])
KAPLAN_SPACE = dict(depth=[1, 2, 3], hidden=[2, 4, 8, 16, 32], k=[2, 3, 4], grid=[3, 5, 8, 10],
                    lam=[0.0, 1e-5, 1e-4, 1e-3], lr=[1e-3, 3e-3])
COXSYM_SPACE = dict(hidden=[0, 2], grid=[3, 5], lr=[1e-3, 1e-2], lamb=[0.0, 1e-3, 1e-2])

bench3.FAMILIES.update({
    "SurvKAN": ("ctkan", "SurvKAN", "minmax", SURVKAN_SPACE, {}),
    "SurvKAN-1000": ("ctkan", "SurvKAN1000", "minmax", SURVKAN_SPACE, {}),   # PROTOCOL_v7: max_epochs 1000
    "KAPLAN-HR": ("ctkan", "KAPLAN", "minmax", KAPLAN_SPACE, {}),
    "CoxKAN-sym": ("coxkan_sym", None, "minmax", COXSYM_SPACE, {}),
})
bench3.RNG_ALIAS["SurvKAN-1000"] = "SurvKAN"   # PROTOCOL_v7: same K=8 configurations as v6 SurvKAN
bench3.NEURAL = tuple(bench3.NEURAL) + ("ctkan",)   # CoxKAN-sym: single formula, no ensemble (PROTOCOL_v6)
_fit_one_v5 = bench3.fit_one
SYM_LOG = os.path.join(bench3.ROOT, "results", "coxkan_sym_log.csv")


def fit_one(kind, mkey, cfg, fixed, df, tr, va, ev_sets, time_all, ev_all, tgrid, scaling, seed):
    if kind not in ("ctkan", "coxkan_sym"):
        return _fit_one_v5(kind, mkey, cfg, fixed, df, tr, va, ev_sets, time_all, ev_all, tgrid, scaling, seed)
    cfg = dict(cfg)
    pp = make_preprocessor(scaling).fit(df.iloc[tr])
    X = {k: pp.transform(df.iloc[idx]) for k, idx in [("tr", tr), ("va", va)] + ev_sets}
    if kind == "ctkan":
        if mkey in ("SurvKAN", "SurvKAN1000"):
            model = CTKAN(X["tr"].shape[1], widths=(cfg["hidden"],), grid=cfg["grid"], k=2, base_fn=cfg["base_fn"])
            loss, batch, max_ep = SurvKANLoss(50), 256, (1000 if mkey == "SurvKAN1000" else 300)
        else:
            widths = (cfg["hidden"],) * (cfg["depth"] - 1)
            model = CTKAN(X["tr"].shape[1], widths=widths, grid=cfg["grid"], k=cfg["k"], base_fn="silu")
            loss, batch, max_ep = KaplanLoss(128), 128, 200
        model, vl, ep = fit_ctkan(model, loss, X["tr"], time_all[tr], ev_all[tr], X["va"], time_all[va],
                                  ev_all[va], lr=cfg["lr"], lam=cfg["lam"], batch=batch, max_epochs=max_ep, seed=seed)
        return {k: ctkan_survival(model, X[k], tgrid) for k in X if k != "tr"}, ep
    from dkan.coxkan_official import fit_coxkan_symbolic
    import pandas as pd, json
    sym = bool(cfg.pop("_symbolic", False))     # tuning on the spline stage; symbolic step post-selection
    S, info = fit_coxkan_symbolic(X["tr"], time_all[tr], ev_all[tr], X["va"], time_all[va], ev_all[va],
                                  {k: X[k] for k in X if k != "tr"}, tgrid, seed=seed, steps=100,
                                  symbolic=sym, **cfg)
    if not sym:
        return S, 0
    pd.DataFrame([dict(seed=seed, cfg=json.dumps(cfg), sym_ok=info.get("sym_ok"), err=info.get("err", ""), finetuned=info.get("finetuned", False),
                       val_spline=info.get("val_spline"), val_sym=info.get("val_sym"),
                       formula=info.get("formula", ""))]).to_csv(SYM_LOG, mode="a", index=False,
                                                                 header=not os.path.exists(SYM_LOG))
    return S, 0


bench3.fit_one = fit_one
bench3.POST_SELECT = {"coxkan_sym"}

if __name__ == "__main__":
    ds = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rep = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    fams = sys.argv[4].split(",") if len(sys.argv) > 4 else ["SurvKAN", "KAPLAN-HR", "CoxKAN-sym"]
    folds = range(5)
    if len(sys.argv) > 5:
        folds = [int(f) for f in sys.argv[5].split(",")]
    bench3.run(ds, K, fams, folds, rep)
