#!/usr/bin/env python3
"""stage2_fit_demo_fe.py - ZIP fixed-effects variants of the stage-2
demonstration fits (companion to stage2_fit_demo.py).

The pooled demonstration specification has no unit-level receptivity term,
so its peer coefficient absorbs any time-invariant, spatially clustered
propensity to adopt (the homophily channel, stated in the manuscript).
These variants add a full set of ZIP fixed effects, so the peer
coefficient is identified only from within-ZIP variation over time. They
answer "how much of the pooled association survives removing every
time-invariant ZIP propensity" and nothing stronger: time-varying
confounders (local incentives, installer entry, income growth) remain.

Solar: Poisson with ZIP and year dummies (IRLS on a sparse design);
ZIPs with zero installs in the window are dropped (conditional-Poisson
behavior). EV: OLS with ZIP and year dummies on net additions per 1,000
households. Cluster-robust (by ZIP) sandwich standard errors throughout.

Deterministic. One-command: python stage2_fit_demo_fe.py
Outputs: results/stage2_demo_fits_fe.csv, results/stage2_ev_fits_fe.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
from run_realgeo import knn_W, band_W
from stage2_fit_demo import planar, FIT_YEARS, EV_YEARS


def build_design(panel, years, cumcol, W):
    """Rows for (zip, year in years); returns y-index arrays, the peer
    vector, households, zip codes list, year list."""
    zips = sorted(panel["zip"].unique())
    idx = {z: i for i, z in enumerate(zips)}
    meta = panel.groupby("zip").first().reindex(zips)
    hh = meta["households"].to_numpy(float)
    sub = panel[panel["year"].isin(years)].copy()
    peer_map = {}
    for y in years:
        cumprev = sub[sub["year"] == y].set_index("zip")[cumcol] \
            .reindex(zips).fillna(0).to_numpy()
        peer_map[y] = np.asarray(W @ np.log1p(cumprev / hh * 1000.0)).ravel()
    zi = sub["zip"].map(idx).to_numpy()
    yi = sub["year"].map({y: j for j, y in enumerate(years)}).to_numpy()
    peer = np.array([peer_map[y][i] for y, i in zip(sub["year"], zi)])
    return sub, zi, yi, peer, zips


def sparse_design(zi, yi, peer, nz, ny):
    n = len(peer)
    rows = np.arange(n)
    ones = np.ones(n)
    X = sparse.csr_matrix(
        (np.concatenate([ones, ones, peer]),
         (np.concatenate([rows, rows, rows]),
          np.concatenate([zi, nz + yi, nz + ny + np.zeros(n, int)]))),
        shape=(n, nz + ny + 1))
    return X


def cluster_sandwich(X, score_rows, groups, bread):
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    S = score_rows[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    sums = np.add.reduceat(S, starts, axis=0)
    meat = sums.T @ sums
    return bread @ meat @ bread


def poisson_fe(y, X, offset, groups, tol=1e-9, maxit=60):
    beta = np.zeros(X.shape[1])
    for _ in range(maxit):
        mu = np.exp(offset + X @ beta)
        H = (X.T.multiply(mu) @ X).toarray()
        g = X.T @ (y - mu)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    mu = np.exp(offset + X @ beta)
    bread = np.linalg.inv((X.T.multiply(mu) @ X).toarray())
    scores = X.multiply((y - mu)[:, None]).toarray()
    V = cluster_sandwich(X, scores, groups, bread)
    return beta, np.sqrt(np.diag(V))


def ols_fe(y, X, groups):
    XtX = (X.T @ X).toarray()
    bread = np.linalg.inv(XtX)
    beta = bread @ (X.T @ y)
    resid = y - X @ beta
    scores = X.multiply(resid[:, None]).toarray()
    V = cluster_sandwich(X, scores, groups, bread)
    return beta, np.sqrt(np.diag(V))


def main():
    solar = pd.read_csv(os.path.join(OUT, "stage2_solar_panel.csv"),
                        dtype={"zip": str})
    solar["zip"] = solar["zip"].str.zfill(5)
    solar = solar.sort_values(["zip", "year"])
    zips_all = sorted(solar["zip"].unique())
    meta = solar.groupby("zip").first().reindex(zips_all)
    xy = planar(meta["lat"].to_numpy(), meta["lon"].to_numpy())
    latlon = meta[["lat", "lon"]].to_numpy()
    band, _ = band_W(latlon, 100.0)
    graphs = {"k3": knn_W(xy, 3), "k6": knn_W(xy, 6), "k10": knn_W(xy, 10),
              "band100km": band}

    rows = []
    for name, W in graphs.items():
        sub, zi, yi, peer, zips = build_design(solar, FIT_YEARS, "cum_prev", W)
        keep_zip = sub.groupby("zip")["installs"].sum()
        active = set(keep_zip[keep_zip > 0].index)
        m = sub["zip"].isin(active).to_numpy()
        remap = {z: i for i, z in enumerate(sorted(active))}
        zi2 = sub["zip"].map(remap).to_numpy()[m]
        X = sparse_design(zi2, yi[m], peer[m], len(active), len(FIT_YEARS))
        y = sub["installs"].to_numpy(float)[m]
        off = np.log(sub["households"].to_numpy(float)[m])
        beta, se = poisson_fe(y, X, off, zi2)
        j = X.shape[1] - 1
        rows.append(dict(outcome="solar_installs_fe", fitted_W=name,
                         gamma_hat=float(beta[j]), se_cluster=float(se[j]),
                         z=float(beta[j] / se[j]),
                         p=float(2 * norm.sf(abs(beta[j] / se[j]))),
                         n_obs=int(m.sum()), n_zips=len(active),
                         mean_degree=float((W > 0).sum(1).mean())))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "stage2_demo_fits_fe.csv"),
                              index=False)

    ev = pd.read_csv(os.path.join(OUT, "stage2_ev_panel.csv"),
                     dtype={"zip": str})
    ev["zip"] = ev["zip"].str.zfill(5)
    ev = ev.sort_values(["zip", "year"])
    ev["net_per_1000hh"] = ev["net_additions"] / ev["households"] * 1000.0
    rows = []
    for name, W in graphs.items():
        sub, zi, yi, peer, zips = build_design(ev, EV_YEARS, "ev_stock_prev", W)
        X = sparse_design(zi, yi, peer, len(zips), len(EV_YEARS))
        y = sub["net_per_1000hh"].to_numpy(float)
        beta, se = ols_fe(y, X, zi)
        j = X.shape[1] - 1
        rows.append(dict(outcome="ev_net_additions_per_1000hh_fe",
                         fitted_W=name, gamma_hat=float(beta[j]),
                         se_cluster=float(se[j]),
                         z=float(beta[j] / se[j]),
                         p=float(2 * norm.sf(abs(beta[j] / se[j]))),
                         n_obs=len(y), n_zips=len(zips),
                         mean_degree=float((W > 0).sum(1).mean())))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "stage2_ev_fits_fe.csv"),
                              index=False)
    print("FE fits done")


if __name__ == "__main__":
    main()
