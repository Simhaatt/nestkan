"""KAN layer and discrete-time hazard networks (all output interval logits)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    """B-spline KAN layer (efficient-kan style): y_o = sum_i w_b[o,i] silu(x_i)
    + sum_i sum_l c[o,i,l] B_l(x_i).  Uniform knots on [-1, 1] extended by k on
    each side, so inputs are expected in [-1, 1] (ECDF / min-max / tanh)."""

    def __init__(self, d_in, d_out, grid=5, k=3, base=True):
        super().__init__()
        self.d_in, self.d_out, self.G, self.k = d_in, d_out, grid, k
        h = 2.0 / grid
        knots = torch.arange(-k, grid + k + 1, dtype=torch.float32) * h - 1.0
        self.register_buffer("knots", knots)  # (G + 2k + 1,)
        self.base = base
        if base:
            self.w_b = nn.Parameter(torch.empty(d_out, d_in))
            nn.init.kaiming_uniform_(self.w_b, a=math.sqrt(5))
        self.c = nn.Parameter(torch.randn(d_out, d_in, grid + k) * (0.1 / math.sqrt(d_in)))

    def bspline(self, x):
        # x: (N, d_in) -> (N, d_in, G + k)
        t = self.knots
        x = x.unsqueeze(-1)
        B = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
        for p in range(1, self.k + 1):
            left = (x - t[: -(p + 1)]) / (t[p:-1] - t[: -(p + 1)]) * B[..., :-1]
            right = (t[p + 1:] - x) / (t[p + 1:] - t[1:-p]) * B[..., 1:]
            B = left + right
        return B

    def forward(self, x):
        y = torch.einsum("nil,oil->no", self.bspline(x), self.c)
        if self.base:
            y = y + F.silu(x) @ self.w_b.T
        return y

    def smooth_penalty(self):
        d2 = self.c[..., 2:] - 2 * self.c[..., 1:-1] + self.c[..., :-2]
        return (d2 ** 2).mean()


def _kan_stack(dims, grid):
    return nn.ModuleList([KANLayer(a, b, grid=grid) for a, b in zip(dims[:-1], dims[1:])])


def _kan_forward(layers, x):
    for j, L in enumerate(layers):
        x = L(x)
        if j < len(layers) - 1:
            x = torch.tanh(x)  # keep hidden activations inside the spline grid
    return x


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width=64, depth=2, dropout=0.1):
        super().__init__()
        layers, a = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(a, width), nn.ReLU(), nn.BatchNorm1d(width), nn.Dropout(dropout)]
            a = width
        layers.append(nn.Linear(a, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# Discrete hazard models: forward(x) -> logits (N, T)
# ----------------------------------------------------------------------------
class LHMLP(nn.Module):
    """Logistic-Hazard (Kvamme et al.) with an MLP: one output per interval."""
    def __init__(self, d, T, width=64, depth=2, dropout=0.1, **_):
        super().__init__()
        self.f = MLP(d, T, width, depth, dropout)

    def forward(self, x):
        return self.f(x)

    def penalty(self):
        return torch.zeros(())


class LHKAN(nn.Module):
    """Logistic-Hazard with a KAN [d, H, T]."""
    def __init__(self, d, T, hidden=8, grid=5, **_):
        super().__init__()
        self.layers = _kan_stack([d, hidden, T], grid)

    def forward(self, x):
        return _kan_forward(self.layers, x)

    def penalty(self):
        return sum(L.smooth_penalty() for L in self.layers)


class TimeKAN(nn.Module):
    """Hazard surface logit h(t|x) = KAN([x, v_t]) (+ optional free alpha_t), with
    v_t the KM-scale position of interval t in [-1, 1].  Smooth in t, so the number
    of parameters does not grow with the number of intervals."""
    def __init__(self, d, T, v, hidden=8, grid=5, free_baseline=False, **_):
        super().__init__()
        self.register_buffer("v", torch.as_tensor(v, dtype=torch.float32).view(-1, 1))
        # first KAN layer on [x, v] split into its x-edges and its v-edges: a KAN
        # layer is additive over inputs, so this is the same [d+1, H] layer but the
        # x part is computed once per subject and the v part once per interval.
        self.lx = KANLayer(d, hidden, grid=grid)
        self.lt = KANLayer(1, hidden, grid=grid)
        self.l2 = KANLayer(hidden, 1, grid=grid)
        self.alpha = nn.Parameter(torch.zeros(T)) if free_baseline else None
        self.T = T

    def forward(self, x):
        N = x.shape[0]
        hx = self.lx(x)                      # (N, H)
        ht = self.lt(self.v)                 # (T, H)
        z = torch.tanh(hx.unsqueeze(1) + ht.unsqueeze(0))   # (N, T, H)
        out = self.l2(z.reshape(N * self.T, -1)).view(N, self.T)
        return out + self.alpha if self.alpha is not None else out

    def penalty(self):
        return self.lx.smooth_penalty() + self.lt.smooth_penalty() + self.l2.smooth_penalty()


class TimeMLP(nn.Module):
    """Same hazard-surface parametrisation as TimeKAN but with an MLP."""
    def __init__(self, d, T, v, width=64, depth=2, dropout=0.1, free_baseline=False, **_):
        super().__init__()
        self.register_buffer("v", torch.as_tensor(v, dtype=torch.float32))
        self.f = MLP(d + 1, 1, width, depth, dropout)
        self.alpha = nn.Parameter(torch.zeros(T)) if free_baseline else None
        self.T = T

    def forward(self, x):
        N, d = x.shape
        xt = torch.cat([x.unsqueeze(1).expand(N, self.T, d),
                        self.v.view(1, self.T, 1).expand(N, self.T, 1)], dim=2)
        out = self.f(xt.reshape(N * self.T, d + 1)).view(N, self.T)
        return out + self.alpha if self.alpha is not None else out

    def penalty(self):
        return torch.zeros(())


class PHKAN(nn.Module):
    """Proportional (logit-link) discrete hazard: alpha_t + additive KAN [d, 1]."""
    def __init__(self, d, T, grid=5, **_):
        super().__init__()
        self.g = KANLayer(d, 1, grid=grid)
        self.alpha = nn.Parameter(torch.zeros(T))

    def forward(self, x):
        return self.alpha + self.g(x)

    def penalty(self):
        return self.g.smooth_penalty()


def cloglog_to_logit(eta):
    """logit(h) for h = 1 - exp(-exp(eta)); stable for all eta."""
    z = torch.exp(torch.clamp(eta, max=15.0))
    return z + torch.log((-torch.expm1(-z)).clamp_min(1e-30))


class NestKAN(nn.Module):
    """Nested discrete hazard:  link h(t|x) = alpha_t + beta'z + sum_j phi_j(u_j)
                                              + s * R(u, v_t)
    z = z-scored covariates (first d columns of x), u = ECDF covariates (last d
    columns), v_t = KM-scale interval position.  The three blocks nest the grouped
    Cox model (alpha + beta'z), an additive PH model (+ phi) and a non-PH
    interaction surface (+ R, a [d+1, H, 1] KAN).  beta, phi and the gate s start at
    zero, so training starts from the baseline-only model and the residual surface
    is only used if validation likelihood supports it."""

    def __init__(self, d2, T, v, hidden=8, grid=5, link="cloglog", use_linear=True,
                 use_additive=True, use_residual=True, lam_alpha=1.0, **_):
        super().__init__()
        d = d2 // 2
        self.d, self.T, self.link = d, T, link
        self.register_buffer("v", torch.as_tensor(v, dtype=torch.float32).view(-1, 1))
        self.alpha = nn.Parameter(torch.zeros(T))
        self.lam_alpha = lam_alpha
        self.lin = nn.Linear(d, 1, bias=False) if use_linear else None
        if self.lin is not None:
            nn.init.zeros_(self.lin.weight)
        self.add = KANLayer(d, 1, grid=grid, base=False) if use_additive else None
        if self.add is not None:
            nn.init.zeros_(self.add.c)
        if use_residual:
            self.lx = KANLayer(d, hidden, grid=grid)
            self.lt = KANLayer(1, hidden, grid=grid)
            self.l2 = KANLayer(hidden, 1, grid=grid)
            self.gate = nn.Parameter(torch.zeros(()))
        self.use_residual = use_residual

    def init_baseline(self, idx, event):
        """alpha_t from the empirical discrete hazard d_t / n_t (training data)."""
        idx = np.asarray(idx); event = np.asarray(event)
        at_risk = np.array([(idx >= t).sum() for t in range(self.T)], float)
        d = np.array([((idx == t) & (event == 1)).sum() for t in range(self.T)], float)
        h = np.clip((d + 0.5) / (at_risk + 1.0), 1e-4, 1 - 1e-4)
        a = np.log(-np.log1p(-h)) if self.link == "cloglog" else np.log(h / (1 - h))
        with torch.no_grad():
            self.alpha.copy_(torch.as_tensor(a, dtype=torch.float32))

    def forward(self, x):
        N, d = x.shape[0], self.d
        z, u = x[:, :d], x[:, d:]
        g = torch.zeros(N, 1, device=x.device)
        if self.lin is not None:
            g = g + self.lin(z)
        if self.add is not None:
            g = g + self.add(u)
        eta = self.alpha.view(1, -1) + g
        if self.use_residual:
            hz = torch.tanh(self.lx(u).unsqueeze(1) + self.lt(self.v).unsqueeze(0))
            eta = eta + self.gate * self.l2(hz.reshape(N * self.T, -1)).view(N, self.T)
        return cloglog_to_logit(eta) if self.link == "cloglog" else eta

    def penalty(self):
        p = torch.zeros(())
        if self.T >= 3:
            a = self.alpha
            p = p + self.lam_alpha * ((a[2:] - 2 * a[1:-1] + a[:-2]) ** 2).mean()
        if self.add is not None:
            p = p + self.add.smooth_penalty()
        if self.use_residual:
            p = p + self.lx.smooth_penalty() + self.lt.smooth_penalty() + self.l2.smooth_penalty()
        return p


# ----------------------------------------------------------------------------
# Published discrete-time comparators (re-implemented in the same framework)
# ----------------------------------------------------------------------------
def nll_pmf(phi, idx, event):
    """PMF likelihood (Kvamme et al. 2019): phi (N, T+1) unnormalised log-pmf over
    T intervals + 'beyond'.  Event in idx: log p_idx.  Censored after surviving
    0..idx: log sum_{j>idx} p_j."""
    N, T1 = phi.shape
    logp = torch.log_softmax(phi, dim=1)
    ar = torch.arange(T1).view(1, T1)
    ev_term = logp.gather(1, idx.clamp_min(0).view(-1, 1)).squeeze(1)
    surv_mask = ar > idx.view(-1, 1)
    cens_term = torch.logsumexp(logp.masked_fill(~surv_mask, -1e30), dim=1)
    ll = torch.where(event > 0.5, ev_term, cens_term)
    return -ll.mean()


def _pmf_grid_survival(phi):
    p = torch.softmax(phi, dim=1)[:, :-1]
    S = 1 - torch.cumsum(p, dim=1)
    return torch.cat([torch.ones(len(S), 1), S.clamp_min(0)], 1)


class PMFMLP(nn.Module):
    kind = "pmf"
    def __init__(self, d, T, width=64, depth=2, dropout=0.1, **_):
        super().__init__()
        self.f = MLP(d, T + 1, width, depth, dropout)

    def forward(self, x):
        return self.f(x)

    def penalty(self):
        return torch.zeros(())

    def loss_fn(self, out, idx, event):
        return nll_pmf(out, idx, event)

    def grid_survival(self, out):
        return _pmf_grid_survival(out)


class MTLR(PMFMLP):
    """Multi-task logistic regression (Yu et al. 2011) with an MLP, as in pycox:
    the pmf logits are reverse cumulative sums of the network outputs."""
    def __init__(self, d, T, width=64, depth=2, dropout=0.1, **_):
        super().__init__(d, T, width, depth, dropout)
        self.f = MLP(d, T, width, depth, dropout)

    def forward(self, x):
        phi = self.f(x)
        phi = torch.flip(torch.cumsum(torch.flip(phi, [1]), 1), [1])
        return torch.cat([phi, torch.zeros(len(phi), 1)], 1)


class DeepHit(PMFMLP):
    """DeepHit single-risk (Lee et al. 2018): PMF likelihood + ranking loss
    (alpha = 0.2, sigma = 0.1, the pycox defaults)."""
    def __init__(self, d, T, width=64, depth=2, dropout=0.1, alpha=0.2, sigma=0.1, **_):
        super().__init__(d, T, width, depth, dropout)
        self.a, self.s = alpha, sigma

    def loss_fn(self, out, idx, event):
        nll = nll_pmf(out, idx, event)
        p = torch.softmax(out, dim=1)[:, :-1]
        F_ = torch.cumsum(p, dim=1)                     # F_j(t) at interval index t
        k = idx.clamp_min(0)
        Fi = F_.gather(1, k.view(-1, 1))                # F_i(T_i)
        Fj = F_[:, k].T                                 # Fj[i, j] = F_j(T_i)
        comp = (event.view(-1, 1) > 0.5) & (idx.view(-1, 1) < idx.view(1, -1))
        rank = (torch.exp(-(Fi - Fj) / self.s) * comp).sum() / comp.sum().clamp_min(1)
        return (1 - self.a) * nll + self.a * rank


class DeepSurv(nn.Module):
    """Continuous-time Cox MLP (Katzman et al. 2018), Breslow-tie batch partial
    likelihood; Breslow baseline computed after training (see fit_deepsurv)."""
    def __init__(self, d, width=64, depth=2, dropout=0.1, **_):
        super().__init__()
        self.f = MLP(d, 1, width, depth, dropout)

    def forward(self, x):
        return self.f(x).squeeze(1)


def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    r, e = risk[order], event[order]
    log_cum = torch.logcumsumexp(r, dim=0)
    return -((r - log_cum) * e).sum() / e.sum().clamp_min(1)


def fit_deepsurv(model, Xtr, ttr, etr, Xva, tva, eva, lr=2e-3, wd=1e-4, batch=256,
                 max_epochs=300, patience=15, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ttr, etr, Xva, tva, eva = (torch.as_tensor(a, dtype=torch.float32)
                                    for a in (Xtr, ttr, etr, Xva, tva, eva))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), batch):
            j = perm[b:b + batch]
            if etr[j].sum() < 1 or len(j) < 2:
                continue
            opt.zero_grad()
            cox_ph_loss(model(Xtr[j]), ttr[j], etr[j]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = cox_ph_loss(model(Xva), tva, eva).item()
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model, best, ep + 1


def deepsurv_survival(model, Xfit, tfit, efit, Xte, tgrid):
    with torch.no_grad():
        rf = torch.exp(model(torch.as_tensor(Xfit))).numpy().astype(float)
        rt = torch.exp(model(torch.as_tensor(Xte))).numpy().astype(float)
    ut = np.unique(tfit[efit == 1])
    order = np.argsort(tfit)
    ts, rs = tfit[order], rf[order]
    rcum = np.cumsum(rs[::-1])[::-1]                  # sum of risk over t_j >= t
    idx = np.searchsorted(ts, ut, side="left")
    d = np.array([((tfit == u) & (efit == 1)).sum() for u in ut]) if len(ut) < 3000 else \
        np.bincount(np.searchsorted(ut, tfit[efit == 1]), minlength=len(ut))
    H0 = np.cumsum(d / rcum[idx])
    k = np.searchsorted(ut, tgrid, side="right") - 1
    H = np.where(k >= 0, H0[np.clip(k, 0, None)], 0.0)
    return np.exp(-np.outer(rt, H))


class PCHazard(nn.Module):
    """Piecewise-constant hazard (Kvamme & Borgan 2021) with an MLP.  Output phi_k =
    log of the cumulative hazard over the whole interval k.  Uses the exact position
    of the time inside its interval (needs_frac)."""
    needs_frac = True

    def __init__(self, d, T, width=64, depth=2, dropout=0.1, **_):
        super().__init__()
        self.f = MLP(d, T, width, depth, dropout)

    def forward(self, x):
        return self.f(x)

    def penalty(self):
        return torch.zeros(())

    def loss_fn(self, phi, idx, event, frac):
        N, T = phi.shape
        H = torch.exp(phi.clamp(max=15.0))
        ar = torch.arange(T).view(1, T)
        before = (H * (ar < idx.view(-1, 1)).float()).sum(1)
        within = H.gather(1, idx.view(-1, 1)).squeeze(1) * frac
        logk = phi.gather(1, idx.view(-1, 1)).squeeze(1)
        return -(event * logk - before - within).mean()

    def grid_survival(self, phi):
        H = torch.cumsum(torch.exp(phi.clamp(max=15.0)), 1)
        return torch.cat([torch.ones(len(H), 1), torch.exp(-H)], 1)


class CoxKAN(nn.Module):
    """Cox partial likelihood with a KAN [d, H, 1] risk function (the CoxKAN
    architecture of Knottenbelt et al. 2024, re-implemented without the symbolic step);
    trained and given a Breslow baseline exactly like DeepSurv."""
    def __init__(self, d, hidden=8, grid=5, **_):
        super().__init__()
        self.layers = _kan_stack([d, hidden, 1], grid)

    def forward(self, x):
        return _kan_forward(self.layers, x).squeeze(1)


MODELS = {"LH-MLP": LHMLP, "LH-KAN": LHKAN, "Time-KAN": TimeKAN,
          "Time-MLP": TimeMLP, "PH-KAN": PHKAN, "NestKAN": NestKAN,
          "PMF": PMFMLP, "MTLR": MTLR, "DeepHit": DeepHit, "PCHazard": PCHazard}


# ----------------------------------------------------------------------------
# Loss, training, prediction
# ----------------------------------------------------------------------------
def nll_logistic_hazard(logits, idx, event):
    """Exact discrete-time negative log-likelihood, mean over subjects.
    idx = last interval at risk (-1 => no contribution); event = 1 if the event
    occurred in interval idx."""
    N, T = logits.shape
    ar = torch.arange(T, device=logits.device).view(1, T)
    mask = (ar <= idx.view(N, 1)).float()
    y = ((ar == idx.view(N, 1)).float() * event.view(N, 1))
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (bce * mask).sum() / N


def fit_discrete(model, Xtr, itr, etr, Xva, iva, eva, lr=2e-3, wd=1e-4, lam=1e-3,
                 batch=256, max_epochs=300, patience=15, seed=0, verbose=False,
                 ftr=None, fva=None):
    """ftr / fva: fraction of the interval elapsed (only for models with needs_frac)."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, itr, etr = map(torch.as_tensor, (Xtr, itr, etr))
    Xva, iva, eva = map(torch.as_tensor, (Xva, iva, eva))
    nf = getattr(model, "needs_frac", False)
    if nf:
        ftr, fva = torch.as_tensor(ftr), torch.as_tensor(fva)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = np.inf, None, 0
    n = len(Xtr)
    loss_f = getattr(model, "loss_fn", nll_logistic_hazard)
    # validation criterion: the model's own likelihood (no ranking term for DeepHit)
    val_f = nll_pmf if getattr(model, "kind", "") == "pmf" else nll_logistic_hazard
    if nf:
        val_f = None
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, batch):
            j = perm[b:b + batch]
            if len(j) < 2:
                continue
            opt.zero_grad()
            if nf:
                loss = loss_f(model(Xtr[j]), itr[j], etr[j], ftr[j]) + lam * model.penalty()
            else:
                loss = loss_f(model(Xtr[j]), itr[j], etr[j]) + lam * model.penalty()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = (loss_f(model(Xva), iva, eva, fva) if nf else val_f(model(Xva), iva, eva)).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, ep + 1


def predict_grid_survival(model, X, batch=2048):
    """Survival at the cut points a_0..a_T: (N, T+1), first column = 1."""
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(X), batch):
            o = model(torch.as_tensor(X[b:b + batch]))
            if hasattr(model, "grid_survival"):
                out.append(model.grid_survival(o).numpy())
                continue
            S = torch.cumprod(1 - torch.sigmoid(o), dim=1)
            out.append(torch.cat([torch.ones(len(S), 1), S], 1).numpy())
    return np.vstack(out)


def interp_survival(S_grid, cuts, times):
    """Constant-hazard interpolation of grid survival to arbitrary times: (N, len(times))."""
    times = np.asarray(times, float)
    k = np.clip(np.searchsorted(cuts, times, side="right"), 1, len(cuts) - 1)
    a0, a1 = cuts[k - 1], cuts[k]
    frac = np.clip((times - a0) / np.maximum(a1 - a0, 1e-12), 0, 1)
    S0 = np.clip(S_grid[:, k - 1], 1e-12, 1)
    S1 = np.clip(S_grid[:, k], 1e-12, 1)
    S = S0 * (S1 / S0) ** frac
    S[:, times >= cuts[-1]] = S_grid[:, [-1]]
    return S
