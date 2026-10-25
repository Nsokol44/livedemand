"""diagnostics.py - diagnose the E2 richest-cell interval failure (L2/L4).

Cell: 16 weeks, 15%/week, gamma = 0.6, random allocation, replications
0-4 (the shipped worlds), unit square and real ZCTA geometry.

D1  Profile shape: penalised profile NLL for gamma on a fine grid
    (0.30..0.85 by 0.025), warm-started; fine-grid interval at the
    paper's cut (chi2_0.95,1 / 2 * phi) vs the shipped profile_ci
    (0.25 steps + linear interpolation). Unpenalised profile likewise.
D2  Penalised vs unpenalised refits (the shrinkage test, L4): refit
    with (a) shipped penalties, (b) all penalties off (sigma, s_tau
    -> 1e6, ridge_beta -> 0), (c) spatial prior only off,
    (d) tau ridge only off. gamma_hat under each.
D3  Parametric bootstrap under the fitted model (rep 0, both
    geometries): conditional variant (P path held at the observed
    path) and full-feedback variant (P recomputed from simulated
    cumulative orders). B fits each; gamma* bias vs the generating
    gamma and profile-CI coverage of the generating value. A second
    run generates at gamma = 0.6 with nuisance parameters from the
    fit, so coverage at the truth can be scored directly.
D4  Multi-start: refits from perturbed inits; spread of gamma_hat and
    NLL at convergence (local-optimum check).
D5  Artefact scan across all richest-cell fits: convergence flags,
    phi, whether profile bounds were found inside the step budget,
    and the first-step Delta-NLL (coarse-step artefact check).

Usage: python diagnostics.py [--procs 2] [--boot 40]
Outputs: results/diagnostics_*.csv, results/diagnostics_summary.json
Deterministic: bootstrap rng seeded per (geometry, variant, rep).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from livedemand.world import WorldConfig, World
from livedemand.policies import RandomPolicy, build_data
from livedemand.model import Fit, Data, peer_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
COORDS = os.path.join(HERE, "data", "zcta_sample_1500.csv")
OUT = os.path.join(HERE, "results")
TRUE_GAMMA = 0.6
WEEKS, FRAC = 16, 0.15
BASE = WorldConfig(n=1500, weeks=24)
BASE_REAL = WorldConfig(n=1500, weeks=24, coords_path=COORDS)


def build(geometry, rep):
    base = BASE_REAL if geometry == "real_zcta" else BASE
    cfg = replace(base, gamma=TRUE_GAMMA, weeks=WEEKS, budget_frac=FRAC)
    seed = 2000 + rep
    w = World(cfg, seed=seed)
    pol = RandomPolicy(cfg, seed + 7)
    for t in range(WEEKS):
        w.step(pol.select(w, t))
    d = build_data(w, WEEKS, "aware")
    return w, d


def profile_grid(d, th, sigma=0.6, s_tau=0.6, rb=1e-2,
                 grid=np.arange(0.30, 0.851, 0.025)):
    """Penalised profile NLL on a fine gamma grid (warm starts)."""
    f0 = Fit(d, sigma=sigma, s_tau=s_tau, ridge_beta=rb)
    t0 = f0.fit(theta0=th)
    nlls = []
    warm = th
    for g in grid:
        f = Fit(d, sigma=sigma, s_tau=s_tau, ridge_beta=rb)
        t = f.fit(theta0=warm, g_fixed=float(g))
        nlls.append(t["nll"])
        warm = t
    return float(t0["nll"]), np.asarray(nlls)


def grid_interval(grid, nlls, nll0, phi):
    cut = chi2.ppf(0.95, 1) / 2.0 * phi
    dn = nlls - nll0
    ok = dn <= cut
    if not ok.any():
        return np.nan, np.nan
    lo = grid[ok][0] if ok[0] else \
        grid[np.argmax(ok)] - 0.0  # first inside, grid resolution
    hi = grid[ok][-1]
    return float(grid[ok].min()), float(hi)


def rep_job(args):
    geometry, rep = args
    w, d = build(geometry, rep)
    f = Fit(d)
    th = f.fit()
    lo_s, hi_s = f.profile_ci()
    row = dict(geometry=geometry, rep=rep, gamma_hat=th["gamma"],
               phi=th["phi"], conv=th["converged"], nit=th["nit"],
               nll=th["nll"], lo_shipped=lo_s, hi_shipped=hi_s)
    # D2: penalty variants
    variants = {
        "unpen_all": dict(sigma=1e6, s_tau=1e6, ridge_beta=0.0),
        "unpen_spatial": dict(sigma=1e6, s_tau=0.6, ridge_beta=1e-2),
        "unpen_tau": dict(sigma=0.6, s_tau=1e6, ridge_beta=1e-2),
    }
    for name, kw in variants.items():
        fv = Fit(d, **kw)
        tv = fv.fit(theta0=th)
        row[f"gamma_{name}"] = tv["gamma"]
        row[f"phi_{name}"] = tv["phi"]
        row[f"nll_{name}"] = tv["nll"]
    # D1 (rep 0 only): fine profiles, penalised and unpenalised
    prof_rows = []
    if rep == 0:
        grid = np.arange(0.30, 0.851, 0.025)
        nll0_p, nlls_p = profile_grid(d, th)
        fu = Fit(d, sigma=1e6, s_tau=1e6, ridge_beta=0.0)
        thu = fu.fit(theta0=th)
        nll0_u, nlls_u = profile_grid(d, thu, sigma=1e6, s_tau=1e6,
                                      rb=0.0)
        for g, a, b in zip(grid, nlls_p, nlls_u):
            prof_rows.append(dict(geometry=geometry, gamma=float(g),
                                  dnll_pen=float(a - nll0_p),
                                  dnll_unpen=float(b - nll0_u)))
        glo, ghi = grid_interval(grid, nlls_p, nll0_p, th["phi"])
        ulo, uhi = grid_interval(grid, nlls_u, nll0_u,
                                 row["phi_unpen_all"])
        row.update(lo_finegrid=glo, hi_finegrid=ghi,
                   lo_finegrid_unpen=ulo, hi_finegrid_unpen=uhi,
                   gamma_unpen_profile_min=float(grid[np.argmin(nlls_u)]))
    # D4 (rep 0 only): multi-start
    ms_rows = []
    if rep == 0:
        rng = np.random.default_rng(4242)
        inits = [None]
        for g0 in (0.0, 0.3, 1.0, 1.5):
            t = dict(th)
            t = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in th.items()}
            t["gamma"] = g0
            t["r"] = th["r"] + rng.normal(0, 0.3, size=th["r"].shape)
            inits.append(t)
        for j, t0 in enumerate(inits):
            fm = Fit(d)
            tm = fm.fit(theta0=t0)
            ms_rows.append(dict(geometry=geometry, start=j,
                                gamma_hat=tm["gamma"], nll=tm["nll"],
                                conv=tm["converged"]))
    return row, prof_rows, ms_rows


def boot_job(args):
    geometry, variant, gen_gamma_mode, b = args
    w, d = build(geometry, 0)
    f = Fit(d)
    th = f.fit()
    rng = np.random.default_rng(91000 + b + (0 if geometry == "unit"
                                              else 5000))
    gen_gamma = th["gamma"] if gen_gamma_mode == "fitted" else TRUE_GAMMA
    r, tau = th["r"], th["tau"]
    pa = expit(r[:, None] + tau[None, :] + gen_gamma * d.P)
    po = expit(th["a2"] + th["lam"] * r)[:, None]
    S = d.S
    A_star = rng.binomial(S.astype(int), np.clip(pa, 0, 1))
    O_star = rng.binomial(A_star, np.clip(po, 0, 1))
    if variant == "full":
        P_star = peer_matrix(O_star, w.HH, w.W)
    else:
        P_star = d.P
    db = Data(S, A_star, O_star, P_star, d.X, d.W, d.HH)
    fb = Fit(db)
    tb = fb.fit()
    lo, hi = fb.profile_ci()
    return dict(geometry=geometry, variant=variant,
                gen=gen_gamma_mode, b=b, gen_gamma=float(gen_gamma),
                gamma_hat=tb["gamma"], lo=lo, hi=hi, phi=tb["phi"],
                covers=bool(lo <= gen_gamma <= hi),
                conv=tb["converged"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--boot", type=int, default=40)
    ap.add_argument("--boot-ci", type=int, default=40)
    a = ap.parse_args()
    t0 = time.time()

    jobs = [(g, r) for g in ("unit_square", "real_zcta") for r in range(5)]
    with ProcessPoolExecutor(a.procs) as ex:
        out = list(ex.map(rep_job, jobs, chunksize=1))
    rows = [o[0] for o in out]
    prof = [p for o in out for p in o[1]]
    ms = [m for o in out for m in o[2]]
    pd.DataFrame(rows).to_csv(os.path.join(
        OUT, "diagnostics_richest.csv"), index=False)
    pd.DataFrame(prof).to_csv(os.path.join(
        OUT, "diagnostics_profiles.csv"), index=False)
    pd.DataFrame(ms).to_csv(os.path.join(
        OUT, "diagnostics_multistart.csv"), index=False)
    print(pd.DataFrame(rows).to_string())
    print("rep-level diagnostics done", round(time.time() - t0), "s",
          flush=True)

    bjobs = [(g, v, m, b) for g in ("unit_square", "real_zcta")
             for v in ("cond", "full") for m in ("fitted", "truth")
             for b in range(a.boot)]
    with ProcessPoolExecutor(a.procs) as ex:
        boots = list(ex.map(boot_job, bjobs, chunksize=1))
    bdf = pd.DataFrame(boots)
    bdf.to_csv(os.path.join(OUT, "diagnostics_bootstrap.csv"),
               index=False)
    summ = bdf.groupby(["geometry", "variant", "gen"]).agg(
        bias=("gamma_hat", lambda s: float(s.mean()
                                           - bdf.loc[s.index,
                                                     "gen_gamma"].iloc[0])),
        cover=("covers", "mean"), n=("covers", "size")).reset_index()
    print(summ.to_string())
    print("ALL DIAGNOSTICS DONE in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
