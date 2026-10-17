"""Data loading, preprocessing, ECDF transform and time discretisation."""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Covariate preprocessing
# ----------------------------------------------------------------------------
class Preprocessor:
    """Fit on training rows only.

    numeric columns ('num_*'): median imputation + missing indicator (if any
    missing in train), then one of
        'ecdf'     : mid-rank empirical CDF of the training column, mapped to [-1, 1]
                     (piecewise-linear interpolation between training support points)
        'minmax'   : training min/max mapped to [-1, 1], clipped
        'standard' : z-score with training mean/sd
    factor columns ('fac_*'): one-hot of training levels (missing = own level),
    coded {-1, +1} for ECDF/minmax and {0, 1} for standard.
    """

    def __init__(self, scaling="ecdf"):
        assert scaling in ("ecdf", "minmax", "standard")
        self.scaling = scaling

    def fit(self, df):
        self.num_cols = [c for c in df.columns if c.startswith("num_")]
        self.fac_cols = [c for c in df.columns if c.startswith("fac_")]
        self.med, self.miss_ind, self.sup, self.cdf = {}, [], {}, {}
        self.lo, self.hi, self.mu, self.sd = {}, {}, {}, {}
        for c in self.num_cols:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            self.med[c] = np.nanmedian(x) if np.isfinite(x).any() else 0.0
            if np.isnan(x).any():
                self.miss_ind.append(c)
            x = np.where(np.isnan(x), self.med[c], x)
            # mid-rank ECDF at each unique training value
            u, cnt = np.unique(x, return_counts=True)
            cum = np.cumsum(cnt)
            mid = (cum - cnt / 2.0) / len(x)
            self.sup[c], self.cdf[c] = u, mid
            self.lo[c], self.hi[c] = x.min(), x.max()
            self.mu[c], self.sd[c] = x.mean(), x.std() + 1e-12
        self.levels = {}
        for c in self.fac_cols:
            v = np.where(df[c].isna().to_numpy(), "__NA__", df[c].astype(str).to_numpy())
            self.levels[c] = sorted(set(v.tolist()))
        return self

    def transform(self, df):
        out, names = [], []
        for c in self.num_cols:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            miss = np.isnan(x)
            x = np.where(miss, self.med[c], x)
            if self.scaling == "ecdf":
                if len(self.sup[c]) == 1:
                    z = np.zeros_like(x)
                else:
                    z = 2.0 * np.interp(x, self.sup[c], self.cdf[c]) - 1.0
            elif self.scaling == "minmax":
                rng = self.hi[c] - self.lo[c]
                z = np.zeros_like(x) if rng == 0 else np.clip(2 * (x - self.lo[c]) / rng - 1, -1, 1)
            else:
                z = (x - self.mu[c]) / self.sd[c]
            out.append(z); names.append(c)
            if c in self.miss_ind:
                m = miss.astype(float)
                out.append(m if self.scaling == "standard" else 2 * m - 1); names.append(c + "_miss")
        for c in self.fac_cols:
            v = np.where(df[c].isna().to_numpy(), "__NA__", df[c].astype(str).to_numpy())
            levs = self.levels[c]
            for lev in levs[1:]:  # drop first level (reference)
                m = (v == lev).astype(float)
                out.append(m if self.scaling == "standard" else 2 * m - 1); names.append(f"{c}={lev}")
        self.names_ = names
        return np.column_stack(out).astype(np.float32) if out else np.zeros((len(df), 0), np.float32)


class DualPreprocessor:
    """Output = [z-scored block | ECDF block], identical column order in both.
    The linear part of NestKAN reads the z-scored block, the spline parts the
    ECDF block."""

    def __init__(self, spline_scaling="ecdf"):
        self.a, self.b = Preprocessor("standard"), Preprocessor(spline_scaling)

    def fit(self, df):
        self.a.fit(df); self.b.fit(df)
        return self

    def transform(self, df):
        return np.hstack([self.a.transform(df), self.b.transform(df)])


def make_preprocessor(scaling):
    if scaling == "dual":
        return DualPreprocessor("ecdf")
    if scaling == "dual_minmax":
        return DualPreprocessor("minmax")
    return Preprocessor(scaling)


# ----------------------------------------------------------------------------
# Kaplan-Meier helpers
# ----------------------------------------------------------------------------
def km(time, event):
    """Kaplan-Meier: returns (unique event-or-censor times, S(t) right-continuous)."""
    order = np.argsort(time, kind="mergesort")
    t, d = time[order], event[order]
    ut = np.unique(t)
    n_at = len(t) - np.searchsorted(t, ut, side="left")
    d_at = np.bincount(np.searchsorted(ut, t), weights=d, minlength=len(ut))
    s = np.cumprod(1.0 - d_at / n_at)
    return ut, s


def km_eval(ut, s, t):
    idx = np.searchsorted(ut, t, side="right") - 1
    return np.where(idx < 0, 1.0, s[np.clip(idx, 0, None)])


# ----------------------------------------------------------------------------
# Discretisation
# ----------------------------------------------------------------------------
class Discretiser:
    """Cut points from the training KM of event times (Kvamme & Borgan quantile
    scheme) or equidistant.  Convention (same for every discrete model):
      event in (a_{k-1}, a_k]     -> index k-1 (0-based), event = 1
      censored in (a_{k-1}, a_k]  -> survived through interval k-2 only; stored as
                                     index k-2 with event = 0 (dropped if k = 1 -> idx -1
                                     means contributes nothing).
    Interval positions on the KM scale, v_k in [-1, 1], are exposed for time-as-input
    models (the 'ECDF of time').
    """

    def __init__(self, n_intervals=20, scheme="km"):
        self.T = n_intervals
        self.scheme = scheme

    def fit(self, time, event):
        # The grid ends at the LAST TRAINING EVENT TIME, not at the last follow-up
        # time.  Beyond it the hazard is not identified (no events), survival is
        # held constant (as Kaplan-Meier does), and subjects censored after it count
        # as surviving every interval.  Ending the grid at max follow-up instead
        # creates one long final interval in which almost everyone is censored; with
        # the censored-drop convention that inflates its hazard and drags predicted
        # survival far below KM in long event-free tails (seen on nwtco).
        tlast = time[event == 1].max()
        if self.scheme == "km":
            ut, s = km(time, event)
            smin = s.min()
            qs = np.linspace(1.0, smin, self.T + 1)[1:]  # equal steps in KM
            cuts = []
            for q in qs:
                idx = np.searchsorted(-s, -q, side="left")  # first time S <= q
                cuts.append(ut[min(idx, len(ut) - 1)])
            cuts = np.unique(np.r_[0.0, cuts])
            cuts[-1] = tlast
        else:
            cuts = np.linspace(0.0, tlast, self.T + 1)
        self.cuts = cuts
        self.T = len(cuts) - 1
        # KM-scale midpoint position of each interval, mapped to [-1, 1]
        ut, s = km(time, event)
        F = 1.0 - km_eval(ut, s, cuts)
        Fn = F / max(F[-1], 1e-12)
        self.v = (2.0 * 0.5 * (Fn[:-1] + Fn[1:]) - 1.0).astype(np.float32)
        return self

    def transform(self, time, event):
        k = np.searchsorted(self.cuts, time, side="left")  # time in (a_{k-1}, a_k] -> k
        beyond = time > self.cuts[-1]
        k = np.clip(k, 1, self.T)
        idx = np.where(event == 1, k - 1, k - 2)
        idx = np.where(beyond & (event == 0), self.T - 1, idx)  # survived every interval
        return idx.astype(np.int64), event.astype(np.float32)

    def transform_pch(self, time, event):
        """For piecewise-constant hazard (PC-Hazard): interval containing the time
        (0-based, events and censored alike) and the fraction of it elapsed.
        Times beyond the last cut sit at the end of the last interval."""
        k = np.clip(np.searchsorted(self.cuts, time, side="left"), 1, self.T)
        a0, a1 = self.cuts[k - 1], self.cuts[k]
        frac = np.clip((time - a0) / np.maximum(a1 - a0, 1e-12), 0.0, 1.0)
        frac = np.where(time > self.cuts[-1], 1.0, frac)
        return (k - 1).astype(np.int64), event.astype(np.float32), frac.astype(np.float32)


def load_survset(name):
    import os
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", f"{name}.csv")
    if os.path.exists(local):  # user cohorts prepared by scripts/prep_cohorts.py
        df = pd.read_csv(local)
        for c in df.columns:
            if c.startswith("fac_"):
                df[c] = df[c].astype(str)
        return df[df["time"] > 0].reset_index(drop=True)
    from SurvSet.data import SurvLoader
    loader = SurvLoader()
    df = loader.load_dataset(ds_name=name)["df"]
    df = df.dropna(subset=["time", "event"])
    df = df[df["time"] > 0].reset_index(drop=True)
    return df
