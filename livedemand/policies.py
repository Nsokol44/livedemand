"""policies.py - who gets the next funded exposure slots.

Every policy picks K ZIPs per week from what it can legally see:
  random          uniform
  size            probability proportional to households (the 'big market' habit)
  frozen          regime A: random pilot, fit once, then exploit that ranking forever
  live            regime B: refit every `refit` weeks on as_of data; Thompson
                  sampling on the posterior; a randomized core of `eps` of the slots
                  stays uniform-random (keeps peer effects identifiable)
  live_seed       live + a word-of-mouth seeding bonus using the fitted gamma
  live_naive      live, but trusts first-release orders as final (no vintage care)
  live_leaky      live, but reads revisions that did not exist yet (the backtest cheat)
  oracle          knows the truth (upper bound, not attainable)
Data views: aware = settled orders + completeness-corrected latest week.
"""
import numpy as np

from .model import Data, Fit, peer_matrix



def build_data(world, t_now, mode):
    """Assemble model inputs for decisions at the start of week t_now,
    using only what the store would have released by then."""
    store = world.store
    N, T = world.cfg.n, t_now
    sup = {str(i): i for i in range(N)}
    if mode == "leaky":
        # week t_now-1's revision is released at t_now: not yet available to a real user
        # at the start of the week? It IS, by design, one week late. The cheat is reading
        # the FINAL value of the latest week at the moment it is first released (t_now-1).
        view_t = t_now           # reads release_time<=t_now, i.e. a week ahead of reality
    else:
        view_t = t_now - 1       # real-time: what existed at the end of week t_now-1
    S = store.matrix("sessions", view_t, sup, T)
    A = store.matrix("atc", view_t, sup, T)
    O = store.matrix("orders", view_t, sup, T)
    if mode == "aware" and T >= 1:
        # latest week is a first release; correct it with the completeness learned from history
        rho = store.completeness("orders", view_t)
        O[:, T - 1] = O[:, T - 1] / max(rho, 0.3)
    P = peer_matrix(O, world.HH, world.W)
    return Data(S, A, O, P, world.X, world.W, world.HH)


class Policy:
    name = "base"

    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.last_fit = None

    def select(self, world, t):
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def select(self, world, t):
        return self.rng.choice(self.cfg.n, self.cfg.K, replace=False)


class SizePolicy(Policy):
    name = "size"

    def select(self, world, t):
        p = world.HH / world.HH.sum()
        return self.rng.choice(self.cfg.n, self.cfg.K, replace=False, p=p)


class OraclePolicy(Policy):
    name = "oracle"

    def select(self, world, t):
        q = world.true_q() * world.fresh_sessions(np.arange(self.cfg.n))
        return np.argsort(-q)[: self.cfg.K]


class LivePolicy(Policy):
    def __init__(self, cfg, seed=0, mode="aware", eps=0.2, pilot=3, refit=2,
                 seed_bonus=False, frozen=False, name=None):
        super().__init__(cfg, seed)
        self.mode, self.eps, self.pilot, self.refit = mode, eps, pilot, refit
        self.seed_bonus, self.frozen = seed_bonus, frozen
        self.name = name or ("frozen" if frozen else "live_" + mode)
        self.theta = None
        self.fit_week = -1

    def _refit(self, world, t):
        d = build_data(world, t, self.mode)
        f = Fit(d)
        self.theta = f.fit(theta0=self.theta)
        self.last_fit = f
        self.fit_week = t

    def select(self, world, t):
        cfg, K = self.cfg, self.cfg.K
        if t < self.pilot:
            return self.rng.choice(cfg.n, K, replace=False)
        need_fit = (self.theta is None) if self.frozen else (t - self.fit_week >= self.refit or self.theta is None)
        if need_fit:
            self._refit(world, t)
        f = self.last_fit
        n_core = int(round(self.eps * K))
        core = self.rng.choice(cfg.n, n_core, replace=False) if n_core else np.array([], int)
        if self.frozen:
            q = f.predict_q(np.zeros(cfg.n))                    # static prior ranking
            score = q
        else:
            d = f.d
            c = (d.O.sum(1)) * 1000.0 / world.HH
            P_next = world.W @ np.log1p(c)
            q = f.predict_q(P_next, draw=True, rng=self.rng)
            score = q * world.fresh_sessions(np.arange(cfg.n))
            if self.seed_bonus and self.theta["gamma"] > 0.05:
                score = score + self._seed_value(world, f, q, P_next, c)
        score = score.copy()
        score[core] = -np.inf
        rest = np.argsort(-score)[: K - len(core)]
        return np.concatenate([core, rest])

    def _seed_value(self, world, f, q, P_next, c, horizon=6):
        """Extra value of an order in i: it raises neighbours' cart odds by
        gamma * W_ji / (1 + c_i) per 1000-HH-normalised order. Valued at the
        neighbours' own expected orders over `horizon` weeks if they are exposed."""
        th = f.theta
        g = th["gamma"]
        fresh = world.fresh_sessions(np.arange(self.cfg.n))
        dc = q * fresh * 1000.0 / world.HH                      # extra c_i from exposing i
        dlog = dc / (1.0 + c)
        # neighbours' conversion sensitivity: d q_j / d logit(pa_j) ~ q_j (1 - pa_j) ; use q_j
        qj = q * fresh
        spill = g * (world.W.T @ qj) * dlog * horizon * (self.cfg.K / self.cfg.n)
        return spill


def make_policy(name, cfg, seed):
    if name == "random":
        return RandomPolicy(cfg, seed)
    if name == "size":
        return SizePolicy(cfg, seed)
    if name == "oracle":
        return OraclePolicy(cfg, seed)
    if name == "frozen":
        return LivePolicy(cfg, seed, mode="aware", frozen=True, eps=0.0, name="frozen")
    if name == "live_aware":
        return LivePolicy(cfg, seed, mode="aware")
    if name == "live_naive":
        return LivePolicy(cfg, seed, mode="naive")
    if name == "live_leaky":
        return LivePolicy(cfg, seed, mode="leaky")
    if name == "live_seed":
        return LivePolicy(cfg, seed, mode="aware", seed_bonus=True, name="live_seed")
    raise ValueError(name)


def run(policy_name, cfg, world_seed, policy_seed=None):
    from .world import World
    w = World(cfg, seed=world_seed)
    pol = make_policy(policy_name, cfg, world_seed + 7 if policy_seed is None else policy_seed)
    for t in range(cfg.weeks):
        w.step(pol.select(w, t))
    return w, pol
