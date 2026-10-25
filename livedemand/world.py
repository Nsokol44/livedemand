"""world.py - synthetic market with a shared latent receptivity, a funnel,
peer (word-of-mouth) effects, a common hype shock, audience saturation and
data revisions. This is the ground truth the system is graded against.

DGP (per ZIP i, week t):
  r_i            latent receptivity = X_i beta + spatial error
  peer_it        = (W log1p(c_{t-1}))_i,  c = cumulative orders per 1000 HH
  tau_t          hype: AR(1) common shock on the add-to-cart logit
  S_it           fresh sessions delivered = slot * exp(-kappa * cumS_i / HH_i)
  A_it ~ Bin(S_it, sigmoid(a1 + r_i + gamma*peer_it + tau_t))      add-to-cart
  O_it ~ Bin(A_it, sigmoid(a2 + lam * r_i))                        orders
Data: sessions and add-to-carts are observed exactly; orders are first
released at rho0 completeness (thinned) and revised to the truth one week
later.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from scipy.spatial import cKDTree
from scipy.special import expit

from .records import LiveStore


@dataclass
class WorldConfig:
    n: int = 1500
    k: int = 6
    coords_path: str = ""        # real-geometry rerun only: CSV with lat,lon columns
                                 # (one row per unit, in unit order). Empty = original
                                 # unit-square geometry, unchanged.
    weeks: int = 30
    slot: int = 500              # fresh sessions per funded ZIP-week
    budget_frac: float = 0.06    # share of ZIPs exposed per week
    gamma: float = 0.6           # peer (word-of-mouth) strength, 0 = none
    hype_sd: float = 0.25        # sd of the common hype shock
    hype_rho: float = 0.8
    a1: float = -4.2
    a2: float = -0.6
    lam: float = 0.7             # loading of receptivity on order-given-cart
    beta_sd: float = 0.4         # signal strength of covariates
    sigma_u: float = 0.45        # residual spatial receptivity sd
    rho_u: float = 0.6
    kappa: float = 2.0           # audience saturation
    rho0: float = 0.7            # first-release completeness of orders
    ze_sd: float = 0.35          # extra ZIP-week noise on the cart logit (overdispersion)
    peer_form: str = "log1p"     # "log1p" (what the model assumes) or "sqrt" (misspecified)
    p: int = 4                   # covariates
    seed: int = 20261020

    @property
    def K(self):
        return max(1, int(round(self.n * self.budget_frac)))


def build_graph(n, k, rng, xy=None):
    if xy is None:
        xy = rng.random((n, 2))
    tree = cKDTree(xy)
    _, nbr = tree.query(xy, k=k + 1)
    nbr = nbr[:, 1:]
    rows = np.repeat(np.arange(n), k)
    W = sparse.csr_matrix((np.full(n * k, 1.0 / k), (rows, nbr.ravel())), shape=(n, n))
    return xy, W


def load_real_xy(path, n):
    """Load real ZCTA centroids from a CSV with lat, lon columns and return
    planar coordinates for neighbour search: x = lon * cos(mean latitude),
    y = lat (degrees). Distances in this scaling approximate ground distances
    up to the constant 111.32 km per degree, which is all a k-nearest-
    neighbour graph requires."""
    import csv as _csv
    lats, lons = [], []
    with open(path, newline="") as fh:
        for row in _csv.DictReader(fh):
            lats.append(float(row["lat"])); lons.append(float(row["lon"]))
    lat = np.asarray(lats); lon = np.asarray(lons)
    assert len(lat) == n, f"coords file has {len(lat)} rows, config n={n}"
    x = lon * np.cos(np.radians(lat.mean()))
    return np.column_stack([x, lat])


def peer_transform(c, form="log1p"):
    return np.log1p(c) if form == "log1p" else 0.8 * np.sqrt(c)


class World:
    def __init__(self, cfg: WorldConfig, seed=None):
        self.cfg = cfg
        n = cfg.n
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        xy_real = load_real_xy(cfg.coords_path, n) if cfg.coords_path else None
        self.xy, self.W = build_graph(n, cfg.k, rng, xy=xy_real)
        I = sparse.identity(n, format="csc")
        inv = sparse.linalg.splu((I - 0.6 * self.W).tocsc())
        Z = np.column_stack([inv.solve(rng.standard_normal(n)) for _ in range(cfg.p)])
        Z = (Z - Z.mean(0)) / Z.std(0)
        self.X = np.column_stack([np.ones(n), Z])               # with intercept
        beta = np.r_[0.0, rng.normal(0, cfg.beta_sd, cfg.p)]
        invu = sparse.linalg.splu((I - cfg.rho_u * self.W).tocsc())
        u = cfg.sigma_u * invu.solve(rng.standard_normal(n))
        self.r = self.X @ beta + u                               # TRUE receptivity
        self.beta = beta
        self.HH = np.exp(rng.normal(np.log(4000), 0.6, n)).round()
        # hype path (fixed per world, same for every policy)
        h = np.zeros(cfg.weeks + 1)
        for t in range(1, cfg.weeks + 1):
            h[t] = cfg.hype_rho * h[t - 1] + cfg.hype_sd * np.sqrt(1 - cfg.hype_rho ** 2) * rng.standard_normal()
        self.tau = h[1:]
        self.out_seed = int(rng.integers(1 << 31))
        self.reset()

    # ------------------------------------------------------------- state
    def reset(self, outcome_seed=None):
        n = self.cfg.n
        self.rng = np.random.default_rng(self.out_seed if outcome_seed is None else outcome_seed)
        self.cumO = np.zeros(n)         # true cumulative orders
        self.cumS = np.zeros(n)         # cumulative nominal sessions
        self.store = LiveStore()
        self.total_orders = []
        self.sessions_used = []
        self.t = 0

    def peer(self):
        c = self.cumO * 1000.0 / self.HH
        return self.W @ peer_transform(c, self.cfg.peer_form)

    def fresh_sessions(self, idx):
        c = self.cfg
        f = np.exp(-c.kappa * self.cumS[idx] / self.HH[idx])
        return np.maximum(1, np.round(c.slot * f)).astype(int)

    def true_q(self, tau=None):
        """True expected orders per fresh session now (for the oracle)."""
        c = self.cfg
        tau = self.tau[min(self.t, c.weeks - 1)] if tau is None else tau
        pa = expit(c.a1 + self.r + c.gamma * self.peer() + tau)
        po = expit(c.a2 + c.lam * self.r)
        return pa * po

    # -------------------------------------------------------------- step
    def step(self, idx):
        """Fund exposure in ZIPs `idx` for week self.t; write records."""
        c, t = self.cfg, self.t
        idx = np.asarray(idx, dtype=int)
        S = self.fresh_sessions(idx)
        pa = expit(c.a1 + self.r[idx] + c.gamma * self.peer()[idx] + self.tau[t]
                   + c.ze_sd * self.rng.standard_normal(len(idx)))
        A = self.rng.binomial(S, pa)
        po = expit(c.a2 + c.lam * self.r[idx])
        O = self.rng.binomial(A, po)
        O_rel = self.rng.binomial(O, c.rho0)                     # first release
        sup = idx.astype(str)
        st = self.store
        st.append_many("sessions", "count", sup, t, t, S, source="sim")
        st.append_many("atc", "count", sup, t, t, A, source="sim")
        st.append_many("orders", "count", sup, t, t, O_rel, source="sim", revision=0)
        st.append_many("orders", "count", sup, t, t + 1, O, source="sim", revision=1)
        self.cumO[idx] += O
        self.cumS[idx] += S
        self.total_orders.append(int(O.sum()))
        self.sessions_used.append(int(S.sum()))
        self.t += 1
        return S, A, O
