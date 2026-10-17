"""Within-interval censoring bias (run: python sim_censor_bias.py out.csv [gamma]) of person-period conventions for the grouped (cloglog) Cox model.

Truth: continuous PH, Lambda(t|x) = (t/lam)^k * exp(beta*x), x ~ N(0,1), k = 1.5, beta = 0.7,
S0(1) = 0.3, administrative censoring at tau = 1, plus C ~ Uniform(0, cmax) chosen so that a target
fraction of subjects is censored before tau.  Under grouping the exact discrete model is
cloglog h_k(x) = log(Lambda0(c_k) - Lambda0(c_{k-1})) + beta*x, so beta is the target.

Grid: KM-quantile cuts (equal KM steps, last cut at the last event) as in dkan.data.Discretiser.
Conventions for a subject censored inside interval k:
  drop   : counts as surviving intervals 1..k-1 (current Discretiser.transform)
  keep   : counts as surviving intervals 1..k
  half   : survives 1..k-1 with weight 1 and interval k with weight 1/2 (actuarial)
  oracle : censoring time rounded DOWN to c_{k-1} before observation (events after it unobserved);
           this is the complete-interval scheme for which the KL identity holds.
Events (T <= C) are counted in their interval under every convention except oracle, where an
event is observed only if T <= floor-to-boundary(C).
Fit: cloglog binomial GLM with interval intercepts + x, by Fisher scoring using bincount.
Output: CSV, one row per (rep, censoring, T, convention, interval k): beta bias and signed error of S(c_k|x=0).
"""
import sys
import numpy as np, pandas as pd

K_SHAPE, BETA, S0_TAU, TAU = 1.5, 0.7, 0.3, 1.0
CENS_GAMMA = 0.0   # C ~ Uniform(0, cmax * exp(-CENS_GAMMA * x)): covariate-dependent censoring if > 0
LAM = (-np.log(S0_TAU)) ** (-1 / K_SHAPE)
L0 = lambda t: (np.asarray(t) / LAM) ** K_SHAPE


def gen(n, cmax, rng):
    x = rng.standard_normal(n)
    T = LAM * (rng.exponential(size=n) / np.exp(BETA * x)) ** (1 / K_SHAPE)
    C = np.minimum(rng.uniform(0, 1, n) * cmax * np.exp(-CENS_GAMMA * x) if np.isfinite(cmax) else np.full(n, np.inf), TAU)
    return x, T, C


def km_cuts(time, event, T):
    o = np.argsort(time); t, e = time[o], event[o]
    ut, idx = np.unique(t, return_index=True)
    d = np.add.reduceat(e, idx); nrisk = len(t) - idx
    s = np.cumprod(1 - d / nrisk)
    qs = np.linspace(1.0, s.min(), T + 1)[1:]
    cuts = [ut[min(np.searchsorted(-s, -q, side="left"), len(ut) - 1)] for q in qs]
    cuts = np.unique(np.r_[0.0, cuts]); cuts[-1] = time[event == 1].max()
    return cuts


def person_period(x, time, event, C, cuts, conv):
    Tn = len(cuts) - 1
    if conv == "oracle":
        kc = np.searchsorted(cuts, C, side="right") - 1          # C in [c_j, c_{j+1}) -> boundary c_j
        Cb = cuts[np.clip(kc, 0, Tn)]
        Cb = np.where(C >= cuts[-1], np.inf, Cb)
        event = (time <= Cb).astype(int) & (time <= cuts[-1])
        time = np.where(event == 1, time, np.minimum(Cb, cuts[-1] + 1e-9))
    k = np.clip(np.searchsorted(cuts, time, side="left"), 1, Tn)   # time in (c_{k-1}, c_k] -> k
    beyond = time > cuts[-1]
    if conv in ("drop", "half", "oracle"):
        last = np.where(event == 1, k, k - 1)                    # number of complete rows
    else:                                                        # keep
        last = k.copy()
    last = np.where(beyond & (event == 0), Tn, last)
    if conv == "oracle":                                         # censored exactly at a boundary c_{k-1}
        on_b = (event == 0) & ~beyond
        kb = np.searchsorted(cuts, time, side="left")            # time == c_j -> j
        last = np.where(on_b, np.clip(kb, 0, Tn), last)
    rows_i = np.repeat(np.arange(len(x)), last)
    rows_k = np.concatenate([np.arange(m) for m in last]) if len(last) else np.array([], int)
    y = np.zeros(len(rows_i)); w = np.ones(len(rows_i))
    ev_end = np.cumsum(last) - 1
    y[ev_end[(event == 1) & (last > 0)]] = 1.0
    if conv == "half":
        hi = np.where((event == 0) & ~beyond)[0]
        rows_i = np.r_[rows_i, hi]; rows_k = np.r_[rows_k, k[hi] - 1]
        y = np.r_[y, np.zeros(len(hi))]; w = np.r_[w, np.full(len(hi), 0.5)]
    return x[rows_i], rows_k, y, w


def fit_cloglog(xr, kr, y, w, Tn, iters=100):
    hz = np.clip(np.bincount(kr, w * y, Tn) / np.maximum(np.bincount(kr, w, Tn), 1e-12), 1e-6, 1 - 1e-6)
    a = np.log(-np.log(1 - hz)); b = 0.0
    for _ in range(iters):
        eta = np.clip(a[kr] + b * xr, -30, 5); ee = np.exp(eta)
        mu = np.clip(1 - np.exp(-ee), 1e-12, 1 - 1e-12); dmu = ee * np.exp(-ee)
        s = w * (y - mu) * dmu / (mu * (1 - mu)); W = w * dmu ** 2 / (mu * (1 - mu))
        ga = np.bincount(kr, s, Tn); gb = (s * xr).sum()
        Haa = np.bincount(kr, W, Tn) + 1e-10; Hab = np.bincount(kr, W * xr, Tn); Hbb = (W * xr * xr).sum()
        # Schur complement solve for the (Tn+1)-dim system with diagonal alpha block
        db = (gb - (Hab * ga / Haa).sum()) / (Hbb - (Hab ** 2 / Haa).sum())
        da = (ga - Hab * db) / Haa
        sc = min(1.0, 1.0 / max(abs(db), np.abs(da).max()))      # damped step, max change 1 per iteration
        da, db = sc * da, sc * db
        a += da; b += db
        if max(abs(db), np.abs(da).max()) < 1e-9:
            break
    return a, b


def cmax_for(target, rng, n=400_000):
    x, T, _ = gen(n, np.inf, rng)
    lo, hi = 0.05, 200.0
    for _ in range(60):
        mid = np.sqrt(lo * hi); C = np.minimum(rng.uniform(0, 1, n) * mid * np.exp(-CENS_GAMMA * x), TAU)
        frac = ((C < T) & (C < TAU)).mean()
        lo, hi = (lo, mid) if frac < target else (mid, hi)
    return np.sqrt(lo * hi)


def main(n=50_000, reps=10, targets=(0.10, 0.25, 0.40), Ts=(5, 10, 20, 40)):
    rng0 = np.random.default_rng(2028)
    cm = {p: cmax_for(p, rng0) for p in targets}
    out = []
    for rep in range(reps):
        for p in targets:
            rng = np.random.default_rng([2028, rep, int(100 * p)])
            x, T, C = gen(n, cm[p], rng)
            time = np.minimum(T, C); event = (T <= C).astype(int)
            for Tn in Ts:
                cuts = km_cuts(time, event, Tn); Tg = len(cuts) - 1
                dL = np.diff(L0(cuts)); S_true = np.exp(-np.cumsum(dL))
                k_int = np.clip(np.searchsorted(cuts, time, side="left"), 1, Tg)
                cens_within = ((event == 0) & (time < cuts[-1])).mean()
                for conv in ("drop", "keep", "half", "oracle"):
                    xr, kr, y, w = person_period(x, time, event, C, cuts, conv)
                    a, b = fit_cloglog(xr, kr, y, w, Tg)
                    S_hat = np.exp(-np.cumsum(np.exp(a)))
                    for kk in range(Tg):
                        out.append(dict(cens_gamma=CENS_GAMMA, rep=rep, cens_target=p, cens_within_obs=cens_within, T_req=Tn, T=Tg, conv=conv,
                                        beta_hat=b, beta_bias=b - BETA, k=kk + 1, S_true=S_true[kk],
                                        S_err=S_hat[kk] - S_true[kk], n_rows=len(y)))
        print("rep", rep, "done", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(sys.argv[1] if len(sys.argv) > 1 else "censor_bias.csv", index=False)
    return df


if __name__ == "__main__":
    if len(sys.argv) > 2:
        CENS_GAMMA = float(sys.argv[2])
    main()
