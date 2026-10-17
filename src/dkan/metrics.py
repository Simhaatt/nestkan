"""Model-agnostic survival metrics on continuous test times.

Every model supplies a survival matrix S (N_test, M) on a common time grid
`tgrid`; metrics interpolate as a right-continuous step function on that grid
(the grid is fine, ~200 points, so the step error is negligible).
"""
import numpy as np
from .data import km, km_eval

_trap = getattr(np, "trapezoid", None) or np.trapz


def _S_at(S, tgrid, t_each):
    """S_j(t_i) for all j at each time t_i -> (len(t_each), N)."""
    k = np.clip(np.searchsorted(tgrid, t_each, side="right") - 1, 0, len(tgrid) - 1)
    return S[:, k].T


def concordance_td(S, tgrid, time, event, chunk=512):
    """Antolini et al. (2005) time-dependent concordance."""
    ev = np.where(event == 1)[0]
    num = den = 0.0
    for b in range(0, len(ev), chunk):
        ii = ev[b:b + chunk]
        Sa = _S_at(S, tgrid, time[ii])                # (c, N): S_j(T_i)
        Si = Sa[np.arange(len(ii)), ii][:, None]       # S_i(T_i)
        comp = (time[None, :] > time[ii][:, None])     # T_j > T_i
        num += ((Si < Sa) & comp).sum() + 0.5 * ((Si == Sa) & comp).sum()
        den += comp.sum()
    return num / den


def _ipcw(time_tr, event_tr):
    ut, g = km(time_tr, 1 - event_tr)
    return lambda t: np.clip(km_eval(ut, g, t), 0.05, 1.0)  # truncate weights


def brier_curve(S, tgrid, time, event, G, eval_times):
    Gt_i = G(time - 1e-10)  # G(T_i-)
    out = []
    for t in eval_times:
        s = S[:, min(np.searchsorted(tgrid, t, side="right") - 1, len(tgrid) - 1)]
        died = (time <= t) & (event == 1)
        alive = time > t
        out.append(np.mean(died * s ** 2 / Gt_i + alive * (1 - s) ** 2 / G(np.array([t]))[0]))
    return np.array(out)


def bll_curve(S, tgrid, time, event, G, eval_times, eps=1e-7):
    Gt_i = G(time - 1e-10)
    out = []
    for t in eval_times:
        s = np.clip(S[:, min(np.searchsorted(tgrid, t, side="right") - 1, len(tgrid) - 1)], eps, 1 - eps)
        died = (time <= t) & (event == 1)
        alive = time > t
        out.append(np.mean(died * np.log(1 - s) / Gt_i + alive * np.log(s) / G(np.array([t]))[0]))
    return np.array(out)


def reference_cuts(time_tr, event_tr, T=30):
    """Common evaluation grid for a fold: equal-KM-step cut points of the training
    data (same rule as the model grid, but fixed T and shared by every model)."""
    from .data import Discretiser
    return Discretiser(T, "km").fit(time_tr, event_tr).cuts


def rnll(S, tgrid, time, event, cuts, eps=1e-8):
    """Held-out interval log-likelihood on a common reference grid (lower = better).
    Event in (a_{k-1}, a_k]: -log[S(a_{k-1}) - S(a_k)];  censored in (a_{k-1}, a_k]:
    -log S(a_{k-1}).  Proper under censoring that is non-informative given x (no
    marginal censoring model needed, unlike IPCW scores), and comparable across
    discrete and continuous models because it only uses S(t)."""
    j = np.clip(np.searchsorted(tgrid, cuts, side="right") - 1, 0, len(tgrid) - 1)
    Sc = S[:, j]                                         # (N, T+1) at cut points
    Sc[:, 0] = 1.0
    k = np.clip(np.searchsorted(cuts, time, side="left"), 1, len(cuts) - 1)
    n = np.arange(len(time))
    s_prev, s_k = Sc[n, k - 1], Sc[n, k]
    ll = np.where(event == 1, np.log(np.clip(s_prev - s_k, eps, 1)), np.log(np.clip(s_prev, eps, 1)))
    # open last cell (a_T, inf): the grid ends at the last training event, so a
    # test time beyond it (event or censored) is scored by log S(a_T)
    beyond = time > cuts[-1]
    ll = np.where(beyond, np.log(np.clip(Sc[:, -1], eps, 1)), ll)
    return -ll.mean()


def brier_curve_cond(S, tgrid, time, event, Gmat, Gown, eval_times):
    """IPCW Brier with a covariate-dependent censoring model: Gmat (N, len(tgrid))
    = G(t | x_i) on tgrid, Gown (N,) = G(T_i- | x_i)."""
    out = []
    for t in eval_times:
        j = min(np.searchsorted(tgrid, t, side="right") - 1, len(tgrid) - 1)
        s = S[:, j]
        died = (time <= t) & (event == 1)
        alive = time > t
        out.append(np.mean(died * s ** 2 / Gown + alive * (1 - s) ** 2 / Gmat[:, j]))
    return np.array(out)


def evaluate(S, tgrid, time, event, time_tr, event_tr, q=(0.05, 0.9), n_eval=100,
             cuts=None, Gcond=None):
    """Returns dict: C_td, IBS, IBLL (negative = log-likelihood; we report -IBLL so
    lower is better for both IBS and INBLL)."""
    G = _ipcw(time_tr, event_tr)
    lo, hi = np.quantile(time, q)
    et = np.linspace(lo, hi, n_eval)
    bs = brier_curve(S, tgrid, time, event, G, et)
    bl = bll_curve(S, tgrid, time, event, G, et)
    out = {"C_td": concordance_td(S, tgrid, time, event),
           "IBS": _trap(bs, et) / (hi - lo),
           "INBLL": -_trap(bl, et) / (hi - lo)}
    if cuts is not None:
        out["rNLL"] = rnll(S, tgrid, time, event, cuts)
    if Gcond is not None:
        Gmat, Gown = Gcond
        bc = brier_curve_cond(S, tgrid, time, event, Gmat, Gown, et)
        out["IBS_cx"] = _trap(bc, et) / (hi - lo)
    return out


def censoring_model(Xtr, time_tr, event_tr, Xev, time_ev, tgrid, floor=0.05):
    """Cox model for the censoring distribution given covariates (ridge 1e-2).
    Returns (Gmat on tgrid, G(T_i-) per subject) for the evaluation subjects."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    est = CoxPHSurvivalAnalysis(alpha=1e-2, ties="efron")
    est.fit(Xtr, Surv.from_arrays((1 - event_tr).astype(bool), time_tr))
    fns = est.predict_survival_function(Xev)
    lo, hi = fns[0].domain
    Gmat = np.vstack([f(np.clip(tgrid, lo, hi)) for f in fns])
    Gmat[:, tgrid < lo] = 1.0
    Gown = np.array([f(np.clip(t - 1e-10, lo, hi)) if t - 1e-10 >= lo else 1.0
                     for f, t in zip(fns, time_ev)])
    return np.clip(Gmat, floor, 1.0), np.clip(Gown, floor, 1.0)
