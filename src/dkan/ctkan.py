"""Continuous-time KAN log-hazard models, re-implemented from the papers (no code released):

  SurvKAN   (arXiv 2602.02179): log h(t|x) = KAN([x, t]), width [d+1, m, 1], m in {0..3};
            edge phi = w_b b(x) + spline(x), b in {identity, SiLU}, quadratic B-splines;
            time normalised to [0, 1]; full-likelihood NLL with Lambda(t_i|x_i) by the
            trapezoid rule on K = 50 uniform points in [0, t_i]; log-hazard clamped to
            [-20, 20]; L1-type activation regulariser.
  KAPLAN-HR (arXiv 2605.23082): log lambda(t|x) = KAN(x, t) with D in {1,2,3} layers
            (D = 1 is the additive/PH model with a time function phi_t(t)), hidden width Q,
            spline order m, J knots; Lambda by a right-endpoint rectangle rule on a shared
            time grid of L = 128 points; shrinkage penalty; batch 128, <= 200 epochs.

Implementation choices common to both (stated in the protocol): inputs min-max scaled to
[-1, 1] (the spline domain), time mapped to [-1, 1] by t / t_max(train); hidden activations
passed through tanh to stay on the fixed uniform knot grid (in place of pykan grid updates);
Adam with early stopping on validation NLL (patience 15).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import KANLayer


class CTKANLayer(KANLayer):
    """KANLayer with a selectable base function (identity or SiLU)."""
    def __init__(self, d_in, d_out, grid=5, k=3, base_fn="silu"):
        super().__init__(d_in, d_out, grid=grid, k=k, base=True)
        self.base_fn = base_fn

    def bspline(self, x):
        """Exact closed form for uniform knots (identical to the Cox-de Boor recursion of the
        parent class on [-1, 1)): only k+1 bases are non-zero; their weights are the uniform
        B-spline polynomials of the fractional position u, scattered into the dense basis."""
        k, G = self.k, self.G
        s = (x + 1.0) * (G / 2.0) + k                 # position in knot units, in [k, G+k)
        j = torch.floor(s).clamp(k, G + k - 1)
        u = s - j
        if k == 2:
            w = [(1 - u) ** 2 / 2, (-2 * u ** 2 + 2 * u + 1) / 2, u ** 2 / 2]
        elif k == 3:
            w = [(1 - u) ** 3 / 6, (3 * u ** 3 - 6 * u ** 2 + 4) / 6,
                 (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6, u ** 3 / 6]
        elif k == 4:
            w = [(1 - u) ** 4 / 24, (-4 * u ** 4 + 12 * u ** 3 - 6 * u ** 2 - 12 * u + 11) / 24,
                 (6 * u ** 4 - 12 * u ** 3 - 6 * u ** 2 + 12 * u + 11) / 24,
                 (-4 * u ** 4 + 4 * u ** 3 + 6 * u ** 2 + 4 * u + 1) / 24, u ** 4 / 24]
        else:
            return super().bspline(x)
        idx = (j.long() - k).unsqueeze(-1) + torch.arange(k + 1, device=x.device)
        B = torch.zeros(*x.shape, G + k, dtype=x.dtype, device=x.device)
        return B.scatter(-1, idx, torch.stack(w, -1))

    def forward(self, x):
        y = torch.einsum("nil,oil->no", self.bspline(x.clamp(-1.0, 1.0 - 1e-6)), self.c)
        b = x if self.base_fn == "identity" else F.silu(x)
        return y + b @ self.w_b.T

    def l1(self):
        return self.c.abs().mean() + self.w_b.abs().mean()


class CTKAN(nn.Module):
    """log h(t|x) = KAN([x, t]).  The first layer is additive over inputs, so its x-part
    (N, m) and t-part (Q, m) are computed separately and broadcast-added; deeper layers act
    on the (N, Q, m) hidden tensor."""

    def __init__(self, d, widths=(4,), grid=5, k=3, base_fn="silu"):
        super().__init__()
        dims = [d] + [w for w in widths if w > 0] + [1]
        out1 = dims[1]
        self.lx = CTKANLayer(d, out1, grid, k, base_fn)
        self.lt = CTKANLayer(1, out1, grid, k, base_fn)
        self.rest = nn.ModuleList([CTKANLayer(a, b, grid, k, base_fn) for a, b in zip(dims[1:-1], dims[2:])])

    def _tail(self, h):
        for L in self.rest:
            h = L(torch.tanh(h))
        return h

    def log_h(self, x, tt):
        """x: (N, d); tt: (N, Q) normalised times in [-1, 1] -> (N, Q) log-hazard."""
        N, Q = tt.shape
        a = self.lx(x)                                    # (N, m)
        b = self.lt(tt.reshape(-1, 1)).reshape(N, Q, -1)  # (N, Q, m)
        h = a.unsqueeze(1) + b
        if len(self.rest):
            h = self._tail(h.reshape(N * Q, -1)).reshape(N, Q, -1)
        return h.squeeze(-1).clamp(-20.0, 20.0)

    def log_h_shared(self, x, tg):
        """tg: (Q,) shared time grid -> (N, Q)."""
        a = self.lx(x)
        b = self.lt(tg.reshape(-1, 1))                    # (Q, m)
        h = a.unsqueeze(1) + b.unsqueeze(0)
        N, Q, m = h.shape
        if len(self.rest):
            h = self._tail(h.reshape(N * Q, m)).reshape(N, Q, -1)
        return h.squeeze(-1).clamp(-20.0, 20.0)

    def l1(self):
        return sum(L.l1() for L in [self.lx, self.lt, *self.rest])


def _norm_t(t, tmax):
    return 2.0 * (t / tmax).clamp(0.0, 1.0) - 1.0


class SurvKANLoss:
    """Trapezoid cumulative hazard on K uniform points in [0, t_i]."""
    def __init__(self, K=50):
        self.K = K

    def __call__(self, model, x, t, e, tmax):
        u = torch.linspace(0.0, 1.0, self.K)
        tq = t.view(-1, 1) * u.view(1, -1)                         # (N, K)
        lh = model.log_h(x, _norm_t(tq, tmax))
        hz = torch.exp(lh)
        Lam = t * (hz[:, 1:] + hz[:, :-1]).sum(1) / (2 * (self.K - 1))
        return -(e * lh[:, -1] - Lam).mean()


class KaplanLoss:
    """Right-endpoint rectangle rule on a shared grid of L points over [0, t_max]."""
    def __init__(self, L=128):
        self.L = L

    def __call__(self, model, x, t, e, tmax):
        g = torch.linspace(0.0, float(tmax), self.L + 1)[1:]        # right endpoints
        dt = float(tmax) / self.L
        hz = torch.exp(model.log_h_shared(x, _norm_t(g, tmax)))    # (N, L)
        j = torch.clamp(torch.ceil(t / dt).long() - 1, 0, self.L - 1)   # cell containing t
        ar = torch.arange(self.L).view(1, -1)
        full = (hz * (ar < j.view(-1, 1)).float()).sum(1) * dt
        part = hz.gather(1, j.view(-1, 1)).squeeze(1) * (t - j.float() * dt)
        lh_t = model.log_h(x, _norm_t(t.view(-1, 1), tmax)).squeeze(1)
        return -(e * lh_t - (full + part)).mean()


def fit_ctkan(model, loss, Xtr, ttr, etr, Xva, tva, eva, lr=3e-3, lam=0.0, batch=256,
              max_epochs=300, patience=15, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ttr, etr, Xva, tva, eva = (torch.as_tensor(np.asarray(a), dtype=torch.float32)
                                    for a in (Xtr, ttr, etr, Xva, tva, eva))
    tmax = float(ttr.max())
    model.tmax = tmax
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, state, bad, ep = np.inf, None, 0, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), batch):
            j = perm[b:b + batch]
            opt.zero_grad()
            l = loss(model, Xtr[j], ttr[j], etr[j], tmax)
            if lam > 0:
                l = l + lam * model.l1()
            l.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = sum(loss(model, Xva[b:b + 2048], tva[b:b + 2048], eva[b:b + 2048], tmax).item()
                    * len(Xva[b:b + 2048]) for b in range(0, len(Xva), 2048)) / len(Xva)
        if not np.isfinite(v):
            bad += 1
        elif v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model, best, ep + 1


def ctkan_survival(model, X, tgrid, n_int=400, batch=1024):
    """S(t|x) on tgrid: trapezoid integration of the hazard on n_int uniform points over
    [0, t_max(train)]; beyond t_max the hazard is held at its value at t_max."""
    tmax = model.tmax
    g = torch.linspace(0.0, tmax, n_int + 1)
    out = []
    with torch.no_grad():
        for b in range(0, len(X), batch):
            x = torch.as_tensor(np.asarray(X[b:b + batch]), dtype=torch.float32)
            hz = torch.exp(model.log_h_shared(x, _norm_t(g, tmax))).numpy().astype(float)
            dt = tmax / n_int
            Lam = np.concatenate([np.zeros((len(hz), 1)), np.cumsum((hz[:, 1:] + hz[:, :-1]) * dt / 2, 1)], 1)
            gn = g.numpy().astype(float)
            Lt = np.vstack([np.interp(np.minimum(tgrid, tmax), gn, r) for r in Lam])
            extra = np.maximum(tgrid - tmax, 0.0)[None, :] * hz[:, -1:]
            out.append(np.exp(-(Lt + extra)))
    return np.vstack(out)
