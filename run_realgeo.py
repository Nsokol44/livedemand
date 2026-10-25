"""run_realgeo.py - headline design-condition checks on REAL ZCTA geometry.

The world, model, policies and record store are the LiveDemand v0.1 code,
copied into ./livedemand/ with ONE change: WorldConfig accepts `coords_path`,
a CSV of real ZCTA centroids (Census Bureau 2024 Gazetteer; see
prepare_geometry.py). When it is set, the 1,500 units sit at real sampled
ZCTA centroids and the 6-nearest-neighbour peer graph is built on those real
coordinates. Everything else - data-generating process, estimator, seeds -
is identical to the original unit-square experiments, so the two geometries
are directly comparable. All outcomes remain SYNTHETIC.

Subcommands:
  e2     random-design identification grid at the headline cells:
         (2 wk, 6%), (2 wk, 15%), (4 wk, 15%), (8 wk, 6%), (16 wk, 6%),
         (16 wk, 15%); gamma = 0.6, 5 replications, world seeds 2000+rep
         (the same seeds the original E2 used).
  wsens  W-sensitivity on the (8 wk, 6%) datasets: the SAME simulated data
         refitted with alternative neighbourhood definitions (k = 3,
         k = 10, 100-km distance band) in place of the true k = 6 graph,
         i.e. fitted W != true W. Reports gamma recovery under each.
  e1     live (adaptive, aware) vs frozen contrast at gamma in {0, 0.6, 1.2},
         24 weeks, 2 replications, world seeds 1000+rep (as in E1).
         Simulator-internal contrast only; see the manuscript's claims
         taxonomy before quoting any number from it.

Usage:  python run_realgeo.py all [--procs 4]
Outputs: results/realgeo_e2.csv, results/realgeo_wsens.csv,
         results/realgeo_e1.csv, results/geometry_diagnostics.csv
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from livedemand.world import WorldConfig, World, load_real_xy
from livedemand.policies import RandomPolicy, build_data, run as policy_run
from livedemand.model import Fit, Data, peer_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
COORDS = os.path.join(HERE, "data", "zcta_sample_1500.csv")
OUT = os.path.join(HERE, "results")
BASE = WorldConfig(n=1500, weeks=24, coords_path=COORDS)
TRUE_GAMMA = 0.6


# ------------------------------------------------------------------ helpers
def knn_W(xy, k):
    tree = cKDTree(xy)
    _, nbr = tree.query(xy, k=k + 1)
    nbr = nbr[:, 1:]
    n = xy.shape[0]
    rows = np.repeat(np.arange(n), k)
    return sparse.csr_matrix((np.full(n * k, 1.0 / k), (rows, nbr.ravel())),
                             shape=(n, n))


def band_W(latlon, radius_km=100.0):
    """Row-standardised distance-band weights from true lat/lon (haversine)."""
    from sklearn.neighbors import BallTree
    rad = np.radians(latlon)
    tree = BallTree(rad, metric="haversine")
    ind = tree.query_radius(rad, r=radius_km / 6371.0088)
    n = len(latlon)
    rows, cols = [], []
    for i, js in enumerate(ind):
        js = [j for j in js if j != i]
        if not js:  # isolated unit: fall back to its single nearest neighbour
            _, nn = tree.query(rad[i:i + 1], k=2)
            js = [int(nn[0][1])]
        for j in js:
            rows.append(i); cols.append(j)
    deg = np.bincount(rows, minlength=n).astype(float)
    vals = 1.0 / deg[np.asarray(rows)]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n)), deg


def fit_with_W(world, weeks, W_fit):
    """Refit the model to the world's as-of data using W_fit for BOTH the
    peer exposure and the spatial prior (the model's single-W design)."""
    d0 = build_data(world, weeks, "aware")
    P = peer_matrix(d0.O, world.HH, W_fit)
    d = Data(d0.S, d0.A, d0.O, P, world.X, W_fit, world.HH)
    f = Fit(d)
    th = f.fit()
    lo, hi = f.profile_ci()
    return th, lo, hi


# ------------------------------------------------------------------ E2
def e2_job(args):
    weeks, frac, rep = args
    cfg = replace(BASE, gamma=TRUE_GAMMA, weeks=weeks, budget_frac=frac)
    seed = 2000 + rep
    w = World(cfg, seed=seed)
    pol = RandomPolicy(cfg, seed + 7)
    for t in range(weeks):
        w.step(pol.select(w, t))
    d = build_data(w, weeks, "aware")
    f = Fit(d)
    th = f.fit()
    lo, hi = f.profile_ci()
    return dict(geometry="real_zcta", design="random", weeks=weeks, frac=frac,
                rep=rep, gamma_hat=th["gamma"], lo=lo, hi=hi, phi=th["phi"],
                covers=bool(lo <= TRUE_GAMMA <= hi),
                width=(hi - lo) if np.isfinite(hi - lo) else np.nan,
                finite=bool(np.isfinite(lo) and np.isfinite(hi)),
                corr_r=float(np.corrcoef(th["r"], w.r)[0, 1]),
                orders_total=int(sum(w.total_orders)),
                conv=th["converged"])


def run_e2(procs):
    cells = [(2, 0.06), (2, 0.15), (4, 0.15), (8, 0.06), (16, 0.06), (16, 0.15)]
    jobs = [(wk, fr, rep) for (wk, fr) in cells for rep in range(5)]
    with ProcessPoolExecutor(procs) as ex:
        rows = list(ex.map(e2_job, jobs, chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "realgeo_e2.csv"), index=False)
    print(df.groupby(["weeks", "frac"]).agg(
        gamma_mean=("gamma_hat", "mean"), covers=("covers", "sum"),
        finite=("finite", "sum"), width_med=("width", "median")).to_string())
    return df


# ------------------------------------------------------------------ W sens
def wsens_job(rep):
    weeks, frac = 8, 0.06
    cfg = replace(BASE, gamma=TRUE_GAMMA, weeks=weeks, budget_frac=frac)
    seed = 2000 + rep
    w = World(cfg, seed=seed)
    pol = RandomPolicy(cfg, seed + 7)
    for t in range(weeks):
        w.step(pol.select(w, t))
    samp = pd.read_csv(COORDS)
    latlon = samp[["lat", "lon"]].to_numpy()
    W_band, deg = band_W(latlon, 100.0)
    fits = {"k6_true": w.W, "k3": knn_W(w.xy, 3), "k10": knn_W(w.xy, 10),
            "band100km": W_band}
    out = []
    for name, Wf in fits.items():
        th, lo, hi = fit_with_W(w, weeks, Wf)
        out.append(dict(rep=rep, fitted_W=name, gamma_hat=th["gamma"], lo=lo,
                        hi=hi, covers=bool(lo <= TRUE_GAMMA <= hi),
                        mean_degree=float((Wf > 0).sum(1).mean())))
    return out


def run_wsens(procs):
    with ProcessPoolExecutor(procs) as ex:
        rows = [r for sub in ex.map(wsens_job, range(5), chunksize=1) for r in sub]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "realgeo_wsens.csv"), index=False)
    print(df.groupby("fitted_W").agg(gamma_mean=("gamma_hat", "mean"),
                                     covers=("covers", "sum")).to_string())
    return df


# ------------------------------------------------------------------ E1
def e1_job(args):
    gamma, pol_name, rep = args
    cfg = replace(BASE, gamma=gamma)
    w, _ = policy_run(pol_name, cfg, 1000 + rep)
    cum = np.cumsum(w.total_orders)
    return dict(geometry="real_zcta", gamma=gamma, policy=pol_name, rep=rep,
                c24=int(cum[-1]), orders_per_week=float(np.mean(w.total_orders)))


def run_e1(procs):
    jobs = [(g, p, r) for g in (0.0, 0.6, 1.2)
            for p in ("live_aware", "frozen", "random") for r in range(2)]
    with ProcessPoolExecutor(procs) as ex:
        rows = list(ex.map(e1_job, jobs, chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "realgeo_e1.csv"), index=False)
    print(df.groupby(["gamma", "policy"])["c24"].mean().unstack().to_string())
    return df


# ------------------------------------------------------------------ diag
def diagnostics():
    samp = pd.read_csv(COORDS)
    xy = load_real_xy(COORDS, 1500)
    tree = cKDTree(xy)
    dists, _ = tree.query(xy, k=7)
    km = dists * 111.32
    rows = [dict(quantity="sampled ZCTAs", value=len(samp)),
            dict(quantity="median distance to 1st neighbour (km)",
                 value=float(np.median(km[:, 1]))),
            dict(quantity="median distance to 6th neighbour (km)",
                 value=float(np.median(km[:, 6]))),
            dict(quantity="mean distance to 1st neighbour (km)",
                 value=float(km[:, 1].mean())),
            dict(quantity="share of sampled ZCTAs in AK/HI/PR/territories (|lon|>125 or lat<25)",
                 value=float(((samp.lon.abs() > 125) | (samp.lat < 25)).mean())),
            dict(quantity="median ZCTA land area (sq mi)",
                 value=float(samp.aland_sqmi.median()))]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "geometry_diagnostics.csv"), index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["e2", "wsens", "e1", "diag", "all"])
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if a.which in ("diag", "all"):
        diagnostics()
    if a.which in ("e2", "all"):
        run_e2(a.procs)
    if a.which in ("wsens", "all"):
        run_wsens(a.procs)
    if a.which in ("e1", "all"):
        run_e1(a.procs)
    print(a.which, "done in", round(time.time() - t0), "s")
