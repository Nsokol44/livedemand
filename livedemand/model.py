"""model.py - concurrent multi-factor funnel model, fit by penalised
maximum likelihood (MAP) with analytic gradients.

  A_is ~ Bin(S_is, sigmoid(r_i + tau_s + gamma * P_is))      add-to-cart
  O_is ~ Bin(A_is, sigmoid(a2 + lam * r_i))                  order | cart
  r = X beta + u,  (I - rho W) u ~ N(0, sigma^2 I)           spatial prior
  tau_s ~ N(0, s_tau^2) ridge                                hype (common, weekly)
  P_is = (W log1p(C_{i,s-1}))                                peer exposure

r is ONE latent receptivity shared by both funnel stages (the "feelings"
factor, identified only through outcomes); gamma is word-of-mouth; tau is
hype. Uncertainty for r is a diagonal Laplace approximation; for gamma a
profile-likelihood interval (fit_profile_ci).
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class Data:
    S: np.ndarray           # N x T fresh sessions (0 where not exposed)
    A: np.ndarray           # N x T add-to-carts
    O: np.ndarray           # N x T orders (as seen, possibly corrected)
    P: np.ndarray           # N x T peer exposure
    X: np.ndarray           # N x p covariates (with intercept)
    W: sparse.csr_matrix
    HH: np.ndarray


def peer_matrix(O, HH, W):
    """P[:, s] = W log1p(cumulative orders per 1000 HH through week s-1)."""
    N, T = O.shape
    C = np.cumsum(O, axis=1) * 1000.0 / HH[:, None]
    C = np.hstack([np.zeros((N, 1)), C[:, :-1]])
    return np.column_stack([W @ np.log1p(C[:, s]) for s in range(T)])


class Fit:
    def __init__(self, d: Data, rho=0.6, sigma=0.6, s_tau=0.6, ridge_beta=1e-2):
        self.d, self.rho, self.sigma, self.s_tau, self.rb = d, rho, sigma, s_tau, ridge_beta
        N, T = d.S.shape
        self.N, self.T, self.p = N, T, d.X.shape[1]
        rows, cols = np.nonzero(d.S > 0)
        self.ri, self.rs = rows, cols
        self.S = d.S[rows, cols]
        self.A = d.A[rows, cols]
        self.O = d.O[rows, cols]
        self.P = d.P[rows, cols]
        I = sparse.identity(N, format="csr")
        self.B = (I - rho * d.W).tocsr()
        self.BtB = (self.B.T @ self.B).tocsr()
        self.theta = None

    def _unpack(self, th):
        N, T, p = self.N, self.T, self.p
        r = th[:N]
        tau = th[N:N + T]
        a2, ll, g = th[N + T], th[N + T + 1], th[N + T + 2]
        beta = th[N + T + 3:N + T + 3 + p]
        return r, tau, a2, ll, g, beta

    def _nll(self, th, g_fixed=None):
        r, tau, a2, ll, g, beta = self._unpack(th)
        if g_fixed is not None:
            g = g_fixed
        lam = np.exp(ll)
        ri, rs = self.ri, self.rs
        eta_a = r[ri] + tau[rs] + g * self.P
        pa = expit(eta_a)
        eta_o = a2 + lam * r[ri]
        po = expit(eta_o)
        eps = 1e-12
        ll_a = self.A * np.log(pa + eps) + (self.S - self.A) * np.log(1 - pa + eps)
        ll_o = self.O * np.log(po + eps) + (self.A - self.O) * np.log(1 - po + eps)
        u = self.B @ (r - self.d.X @ beta)
        pen = 0.5 * (u @ u) / self.sigma ** 2 + 0.5 * (tau @ tau) / self.s_tau ** 2 \
            + 0.5 * self.rb * (beta @ beta)
        nll = -(ll_a.sum() + ll_o.sum()) + pen
        # gradient
        da = self.A - self.S * pa
        do = self.O - self.A * po
        N = self.N
        gr = -(np.bincount(ri, da + lam * do, minlength=N))
        gt = -np.bincount(rs, da, minlength=self.T)
        g_a2 = -do.sum()
        g_ll = -(do * lam * r[ri]).sum()
        g_g = -(da * self.P).sum()
        Bu = self.B.T @ u / self.sigma ** 2
        gr = gr + Bu
        gt = gt + tau / self.s_tau ** 2
        gb = -(self.d.X.T @ Bu) + self.rb * beta
        grad = np.concatenate([gr, gt, [g_a2, g_ll, 0.0 if g_fixed is not None else g_g], gb])
        return nll, grad

    def fit(self, theta0=None, g_fixed=None, maxiter=1500):
        N, T, p = self.N, self.T, self.p
        if theta0 is None:
            th0 = np.zeros(N + T + 3 + p)
            th0[N + T] = -0.6
            th0[N + T + 1] = np.log(0.7)
            th0[N] = 0.0
        else:
            th0 = np.zeros(N + T + 3 + p)
            Tn = min(T, theta0["tau"].size)
            th0[:N] = theta0["r"]
            th0[N:N + Tn] = theta0["tau"][:Tn]
            th0[N + T:N + T + 3] = [theta0["a2"], np.log(theta0["lam"]), theta0["gamma"]]
            th0[N + T + 3:] = theta0["beta"]
        if g_fixed is not None:
            th0[N + T + 2] = g_fixed
        res = minimize(self._nll, th0, args=(g_fixed,), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": 1e-10})
        r, tau, a2, ll, g, beta = self._unpack(res.x)
        if g_fixed is not None:
            g = g_fixed
        self.theta = dict(r=r, tau=tau, a2=a2, lam=float(np.exp(ll)), gamma=float(g), beta=beta,
                          nll=float(res.fun), converged=bool(res.success), nit=int(res.nit))
        self.theta["phi"] = self._dispersion()
        self.theta["sd_r"] = self._sd_r() * np.sqrt(self.theta["phi"])
        return self.theta

    def _dispersion(self):
        """Quasi-likelihood dispersion from Pearson residuals of the cart stage.
        phi > 1 means the binomial likelihood is overconfident (extra noise the
        model does not know about); intervals are widened by sqrt(phi)."""
        th = self.theta
        pa = expit(th["r"][self.ri] + th["tau"][self.rs] + th["gamma"] * self.P)
        v = np.maximum(self.S * pa * (1 - pa), 1e-9)
        pearson = (((self.A - self.S * pa) ** 2) / v).sum()
        df = max(len(self.A) - len(np.unique(self.ri)) - self.T - 3, 1)
        return float(max(1.0, pearson / df))

    def _sd_r(self):
        th = self.theta
        ri, rs = self.ri, self.rs
        pa = expit(th["r"][ri] + th["tau"][rs] + th["gamma"] * self.P)
        po = expit(th["a2"] + th["lam"] * th["r"][ri])
        info = np.bincount(ri, self.S * pa * (1 - pa) + th["lam"] ** 2 * self.A * po * (1 - po),
                           minlength=self.N)
        pen = self.BtB.diagonal() / self.sigma ** 2
        return 1.0 / np.sqrt(info + pen)

    # ------------------------------------------------------ profile CI
    def profile_ci(self, level=0.95, step=0.025, maxsteps=80, legacy=False):
        """Profile-likelihood interval for gamma (the word-of-mouth parameter).

        Shipped procedure (stage-1 correction, 2026-10-25): the profile is
        evaluated on a fixed fine lattice (multiples of 0.025 in gamma,
        the grid of diagnostics_figrid.py) outward from the lattice point
        nearest the estimate, warm-started along the search path, and
        each bound is the last lattice point whose Delta-NLL does not
        exceed the cut - the grid itself, with no interpolation of the
        crossing. A bound is +-inf when the profile does not cross the
        cut within the search range (+-2.0, the extent of the superseded
        procedure).

        legacy=True recovers the superseded procedure (0.25 steps with
        linear interpolation of the crossing), kept for the record; it
        understates interval width several-fold on the profile shapes
        these fits produce (see diagnostics_figrid.py).
        """
        from scipy.stats import chi2
        th = self.theta
        g0, nll0 = th["gamma"], th["nll"]
        cut = chi2.ppf(level, 1) / 2.0 * th.get("phi", 1.0)
        if legacy:
            return self._profile_ci_coarse(g0, nll0, cut)
        k0 = int(round(g0 / step))
        out = {}
        for sign, name in ((-1, "lo"), (1, "hi")):
            bound = sign * np.inf
            warm = th
            last = None
            for k in range(0, maxsteps + 1):
                g = (k0 + sign * k) * step
                f = Fit(self.d, self.rho, self.sigma, self.s_tau, self.rb)
                t = f.fit(theta0=warm, g_fixed=g)
                warm = t
                if t["nll"] - nll0 > cut:
                    if last is not None:
                        bound = last
                    break
                last = g
            out[name] = bound
        return out["lo"], out["hi"]

    def _profile_ci_coarse(self, g0, nll0, cut):
        """Superseded profile interval: 0.25 steps outward from the
        estimate with linear interpolation of the crossing, search
        extent +-2.0. Retained only so the pre-correction result files
        remain reproducible."""
        out = {}
        for sign, name in ((-1, "lo"), (1, "hi")):
            bound = None
            prev_g, prev_d = g0, 0.0
            for k in range(1, 9):
                g = g0 + sign * 0.25 * k
                f = Fit(self.d, self.rho, self.sigma, self.s_tau, self.rb)
                t = f.fit(theta0=self.theta, g_fixed=g)
                dnll = t["nll"] - nll0
                if dnll >= cut:
                    # linear interpolation of the crossing
                    w = (cut - prev_d) / max(dnll - prev_d, 1e-12)
                    bound = prev_g + w * (g - prev_g)
                    break
                prev_g, prev_d = g, dnll
            out[name] = bound if bound is not None else sign * np.inf
        return out["lo"], out["hi"]

    # ----------------------------------------------------- predictions
    def predict_q(self, P_next, tau_next=None, draw=None, rng=None):
        """Expected orders per fresh session next week."""
        th = self.theta
        r = th["r"]
        if draw:
            r = r + th["sd_r"] * rng.standard_normal(self.N)
        tau = 0.0 if tau_next is None else tau_next
        pa = expit(r + tau + th["gamma"] * P_next)
        po = expit(th["a2"] + th["lam"] * r)
        return pa * po
