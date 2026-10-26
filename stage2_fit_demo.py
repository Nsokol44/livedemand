#!/usr/bin/env python3
"""stage2_fit_demo.py - stage-2 observational demonstration fits.

DEMONSTRATION, NOT VALIDATION. These fits place the paper's peer exposure
mapping, P = W log1p(C), on real California adoption data, where C is
cumulative adoptions per 1,000 households through the previous year, and
estimate the peer coefficient under four neighbourhood definitions (the
paper's k = 6 graph, k = 3, k = 10, and a 100-km distance band). Nothing
here validates gamma: with observational adoption data, homophily and
contagion are generically confounded (Shalizi & Thomas, 2011), adoption is
not experimentally assigned, and the fitted coefficient absorbs any
spatially clustered propensity to adopt. The exercise exists to show how
the estimate moves with W on real data, beside the paper's simulation
W-sensitivity (results/realgeo_wsens_fg.csv).

Solar (primary): Poisson GLM, log link, offset log(households), year fixed
effects (the hype analogue), fit window 2010-2024, panel from
stage2_build_panels.py. Standard errors clustered by ZIP (sandwich).

EV (robustness): OLS of net additions per 1,000 households (differenced
Jan-1 DMV stocks, addition years 2021-2025; the 2020 difference is dropped
because the 2019 snapshot's internal date is 10/1/2018), year fixed
effects, peer regressor built from the lagged EV stock. Standard errors
clustered by ZIP.

Deterministic (no randomness). One-command: python stage2_fit_demo.py
Outputs: results/stage2_demo_fits.csv, results/stage2_ev_fits.csv
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

FIT_YEARS = list(range(2010, 2025))
EV_YEARS = list(range(2021, 2026))


def planar(lat, lon):
    return np.column_stack([lon * np.cos(np.radians(lat.mean())), lat])


def cluster_vcov(X, resid_score_rows, groups, bread):
    """Sandwich variance: bread @ (sum_g s_g s_g') @ bread, where
    resid_score_rows are per-observation score vectors."""
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    S = resid_score_rows[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    sums = np.add.reduceat(S, starts, axis=0)
    meat = sums.T @ sums
    return bread @ meat @ bread


def poisson_irls(y, X, offset, groups, tol=1e-10, maxit=100):
    beta = np.zeros(X.shape[1])
    beta[0] = np.log((y.sum() + 0.5) / np.exp(offset).sum())
    for _ in range(maxit):
        mu = np.exp(offset + X @ beta)
        Wd = mu
        XtW = X.T * Wd
        H = XtW @ X
        g = X.T @ (y - mu)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    mu = np.exp(offset + X @ beta)
    bread = np.linalg.inv((X.T * mu) @ X)
    scores = X * (y - mu)[:, None]
    V = cluster_vcov(X, scores, groups, bread)
    pearson = float((((y - mu) ** 2) / mu).sum() / (len(y) - X.shape[1]))
    return beta, np.sqrt(np.diag(V)), pearson


def ols_cluster(y, X, groups):
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    scores = X * resid[:, None]
    V = cluster_vcov(X, scores, groups, XtX_inv)
    return beta, np.sqrt(np.diag(V)), resid


def design(panel, years, ycol, cumcol, W):
    """Assemble (y, X, groups, peer) for the given outcome/cumulative
    columns and weights matrix W (units in panel zip order)."""
    zips = sorted(panel["zip"].unique())
    idx = {z: i for i, z in enumerate(zips)}
    sub = panel[panel["year"].isin(years)].copy()
    C = np.zeros((len(zips), len(years) + 1))
    piv_c = sub.pivot(index="zip", columns="year", values=cumcol).reindex(zips)
    hh = sub.groupby("zip")["households"].first().reindex(zips).to_numpy()
    # cumulative through previous year: use cum_prev-style column per year
    ys, Xs, gs, peers = [], [], [], []
    year_d = {y: j for j, y in enumerate(years)}
    for _, r in sub.iterrows():
        i = idx[r["zip"]]
        Cvec = None
        ys.append(r[ycol]); gs.append(i)
        peers.append(0.0)  # filled below in vector form
        row = np.zeros(1 + len(years) - 1 + 1)
        row[0] = 1.0
        if year_d[r["year"]] > 0:
            row[year_d[r["year"]]] = 1.0
        Xs.append(row)
    # vector peer computation: for each year, P = W log1p(cumprev / hh * 1000)
    peer_map = {}
    for y in years:
        cumprev = sub[sub["year"] == y].set_index("zip")["cum_for_peer"] \
            .reindex(zips).fillna(0).to_numpy()
        peer_map[y] = np.asarray(W @ np.log1p(cumprev / hh * 1000.0)).ravel()
    k = 0
    for _, r in sub.iterrows():
        peers[k] = peer_map[r["year"]][idx[r["zip"]]]
        Xs[k][-1] = peers[k]
        k += 1
    return (np.asarray(ys, float), np.asarray(Xs, float),
            np.asarray(gs), np.asarray(peers, float), hh, sub)


def main():
    solar = pd.read_csv(os.path.join(OUT, "stage2_solar_panel.csv"),
                        dtype={"zip": str})
    solar["zip"] = solar["zip"].str.zfill(5)
    zips = sorted(solar["zip"].unique())
    meta = solar.groupby("zip").first().reindex(zips)
    xy = planar(meta["lat"].to_numpy(), meta["lon"].to_numpy())
    latlon = meta[["lat", "lon"]].to_numpy()
    band, deg = band_W(latlon, 100.0)
    graphs = {"k3": knn_W(xy, 3), "k6": knn_W(xy, 6), "k10": knn_W(xy, 10),
              "band100km": band}
    # cumulative installs through previous year, per panel row
    solar = solar.sort_values(["zip", "year"])
    solar["cum_for_peer"] = solar["cum_prev"]

    rows = []
    for name, W in graphs.items():
        y, X, g, peer, hh, sub = design(solar, FIT_YEARS, "installs",
                                        "cum_prev", W)
        offset = np.log(sub["households"].to_numpy(float))
        beta, se, phi = poisson_irls(y, X, offset, g)
        j = X.shape[1] - 1
        rows.append(dict(outcome="solar_installs", fitted_W=name,
                         gamma_hat=beta[j], se_cluster=se[j],
                         z=beta[j] / se[j],
                         p=float(2 * norm.sf(abs(beta[j] / se[j]))),
                         dispersion_phi=phi, n_obs=len(y), n_zips=len(zips),
                         mean_degree=float((W > 0).sum(1).mean())))
        print(rows[-1], flush=True)
    demo = pd.DataFrame(rows)
    demo.to_csv(os.path.join(OUT, "stage2_demo_fits.csv"), index=False)

    ev = pd.read_csv(os.path.join(OUT, "stage2_ev_panel.csv"),
                     dtype={"zip": str})
    ev["zip"] = ev["zip"].str.zfill(5)
    ev = ev.sort_values(["zip", "year"])
    ev["cum_for_peer"] = ev["ev_stock_prev"]
    ev["net_per_1000hh"] = ev["net_additions"] / ev["households"] * 1000.0
    rows = []
    for name, W in graphs.items():
        y, X, g, peer, hh, sub = design(ev, EV_YEARS, "net_per_1000hh",
                                        "ev_stock_prev", W)
        beta, se, resid = ols_cluster(y, X, g)
        j = X.shape[1] - 1
        rows.append(dict(outcome="ev_net_additions_per_1000hh",
                         fitted_W=name, gamma_hat=beta[j], se_cluster=se[j],
                         z=beta[j] / se[j],
                         p=float(2 * norm.sf(abs(beta[j] / se[j]))),
                         n_obs=len(y), n_zips=len(zips),
                         mean_degree=float((W > 0).sum(1).mean())))
        print(rows[-1], flush=True)
    evf = pd.DataFrame(rows)
    evf.to_csv(os.path.join(OUT, "stage2_ev_fits.csv"), index=False)
    print("demo fits done")


if __name__ == "__main__":
    main()
