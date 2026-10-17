"""Wrapper around the official CoxKAN package (Knottenbelt et al. 2024; pip `coxkan` 0.0.2 on
pykan 0.2.8): train -> prune edges -> auto_symbolic -> short retrain, then a Breslow baseline
(identical to the one used for DeepSurv / CoxKAN-re-implementation) for survival curves."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", "..", "pylib_ck"))
import numpy as np, pandas as pd, torch


def _breslow(risk_fit, tfit, efit, risk_new, tgrid):
    rf, rt = np.exp(risk_fit - risk_fit.max()), np.exp(np.clip(risk_new - risk_fit.max(), -60, 30))
    ut = np.unique(tfit[efit == 1])
    order = np.argsort(tfit); ts, rs = tfit[order], rf[order]
    rcum = np.cumsum(rs[::-1])[::-1]
    idx = np.searchsorted(ts, ut, side="left")
    d = np.bincount(np.searchsorted(ut, tfit[efit == 1]), minlength=len(ut))
    H0 = np.cumsum(d / rcum[idx])
    k = np.searchsorted(ut, tgrid, side="right") - 1
    H = np.where(k >= 0, H0[np.clip(k, 0, None)], 0.0)
    return np.exp(-np.outer(rt, H))


# library restricted to functions bounded on bounded inputs (no 1/x, log, exp, tan, arctanh),
# as used for tabular survival data in the CoxKAN paper's examples
SAFE_LIB = ["x", "x^2", "x^3", "sqrt", "abs", "sin", "tanh", "sigmoid", "arctan", "gaussian", "0"]


def _allow_ckpt_globals():
    """coxkan saves and reloads its own temporary checkpoints, which pickle pandas Index
    objects; torch>=2.6 loads weights_only by default, so allowlist exactly those classes."""
    import pandas.core.indexes.base as pib
    import numpy.core.multiarray as ma
    allowed = [pib._new_Index, pd.Index, pd.RangeIndex, ma._reconstruct, np.ndarray, np.dtype]
    try:
        allowed += [type(np.dtype("float64")), type(np.dtype("O")), type(np.dtype("int64"))]
    except Exception:
        pass
    torch.serialization.add_safe_globals(allowed)


def fit_coxkan_symbolic(Xtr, ttr, etr, Xva, tva, eva, Xev, tgrid, hidden=0, grid=5, lr=0.01,
                        steps=100, lamb=0.01, prune=0.02, seed=0, symbolic=True):
    from coxkan import CoxKAN
    _allow_ckpt_globals()
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xtr.shape[1]
    cols = [f"x{j}" for j in range(d)]
    mk = lambda X, t, e: pd.DataFrame(np.c_[X, t, e], columns=cols + ["duration", "event"])
    dtr, dva = mk(Xtr, ttr, etr), mk(Xva, tva, eva)
    width = [d, 1] if hidden == 0 else [d, hidden, 1]
    m = CoxKAN(width=width, grid=grid, k=3, seed=seed)
    kw = dict(duration_col="duration", event_col="event", lr=lr, early_stopping=True, lamb=lamb, progress_bar=False, log=10 ** 9)
    m.train(dtr, dva, steps=steps, **kw)
    pred = lambda X: m(torch.tensor(np.asarray(X), dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
    import copy
    from .models import cox_ph_loss
    tv = torch.as_tensor(np.asarray(tva), dtype=torch.float32); ev_ = torch.as_tensor(np.asarray(eva), dtype=torch.float32)
    vloss = lambda mm: float(cox_ph_loss(mm(torch.tensor(np.asarray(Xva), dtype=torch.float32)).reshape(-1), tv, ev_))
    info = {"sym_ok": False, "symbolic": bool(symbolic)}
    if symbolic:
        sub = np.random.default_rng(seed).permutation(len(Xtr))[:2000]   # activations for symbolic fitting
        Xt = torch.tensor(np.asarray(Xtr)[sub], dtype=torch.float32)
        with torch.no_grad():
            m(Xt)          # refresh cached activations on TRAINING data (grad-free -> copyable)
        spline = copy.deepcopy(m)
        try:
            m.prune_edges(threshold=prune, verbose=False)
            with torch.no_grad():
                m(Xt)
            ok = m.auto_symbolic(lib=SAFE_LIB, verbose=0)
            info["sym_ok"] = bool(ok) if ok is not None else True
            v0 = vloss(m); info["val_spline"] = vloss(spline); info["val_sym"] = v0
            # fine-tune affine parameters; kept only if validation partial likelihood improves
            with torch.no_grad():
                m(Xt)
            ft = copy.deepcopy(m)
            try:
                ft.train(dtr, dva, steps=max(steps // 4, 20), **dict(kw, lr=lr / 10, lamb=0.0, early_stopping=False))
                v1 = vloss(ft)
                if np.isfinite(v1) and v1 < v0:
                    m, info["finetuned"], info["val_sym"] = ft, True, v1
            except Exception:
                pass
            if not np.isfinite(info["val_sym"]):
                raise ValueError("non-finite symbolic validation loss")
            try:
                info["formula"] = str(m.symbolic_formula(floating_digit=3)[0][0])[:400]
            except Exception:
                info["formula"] = ""
        except Exception as ex:
            m = spline; info["sym_ok"] = False; info["err"] = repr(ex)[:200]   # fall back, recorded
    rtr = pred(Xtr)
    out = {k: _breslow(rtr, ttr, etr, pred(X), tgrid) for k, X in Xev.items()}
    return out, info
