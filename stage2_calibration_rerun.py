#!/usr/bin/env python3
"""stage2_calibration_rerun.py - E2 key cells rerun in worlds calibrated to
observed California solar adoption (stage 2).

The worlds differ from the paper's E2 worlds in exactly three ways, each
drawn from the observed CA residential adoption distribution (Tracking
the Sun panel; see stage2_build_panels.py and
results/stage2_calibration_targets.csv):
  * receptivity spatial structure: rho_u chosen so the Moran's I of the
    simulator's spatial receptivity component on the real-geometry k = 6
    graph matches the Moran's I of the observed receptivity proxy
    (log cumulative installs per 1,000 households) on the CA panel graph;
  * receptivity spread: sigma_u chosen so the total receptivity spread
    matches the observed proxy's standard deviation, with the covariate
    share held at the paper's beta_sd = 0.4;
  * households: HH resampled (with replacement, seeded per world) from the
    empirical ACS household counts of the CA panel ZCTAs instead of the
    paper's lognormal.
Held at the paper's values: gamma = 0.6, the funnel, hype, saturation,
the unit-week noise ze_sd = 0.35 (an annual panel does not identify a
weekly logit-noise scale), and every estimator setting, including the
corrected fine-grid profile interval shipped in stage 1.

Cells: the verdict-critical cells of the paper's grid, ordered so the
threshold and richest cells complete first: (8 wk, 6%) and (4 wk, 15%)
(the two identification thresholds), (16 wk, 15%) (the richest cell),
and (2 wk, 6%) and (2 wk, 15%) (the two-week cells), on both geometries,
25 replications per cell (the calibration pass trades replication count
for coverage of the verdict cells; the paper's grid used 50), world
seeds 2000 + rep and policy seeds seed + 7, exactly as in the paper's E2.

Output: results/stage2_calib_e2.csv (per-replication rows).

Usage: python stage2_calibration_rerun.py [--procs 2] [--reps 25]
Deterministic: fixed seeds throughout.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from livedemand.world import WorldConfig, World
from livedemand.policies import RandomPolicy, build_data
from livedemand.model import Fit

OUT = os.path.join(HERE, "results")
COORDS = os.path.join(HERE, "data", "zcta_sample_1500.csv")
TRUE_GAMMA = 0.6
CELLS = [(8, 0.06), (4, 0.15), (16, 0.15), (2, 0.06), (2, 0.15)]

with open(os.path.join(OUT, "stage2_calibration_params.json")) as fh:
    PARAMS = json.load(fh)
EMP_HH = pd.read_csv(os.path.join(OUT, "stage2_ca_households.csv"))[
    "households"].to_numpy(float)

BASE = WorldConfig(n=1500, weeks=24, rho_u=PARAMS["rho_u"],
                   sigma_u=PARAMS["sigma_u"])
BASE_REAL = WorldConfig(n=1500, weeks=24, coords_path=COORDS,
                        rho_u=PARAMS["rho_u"], sigma_u=PARAMS["sigma_u"])


def _run(base, geometry, weeks, frac, rep):
    cfg = replace(base, gamma=TRUE_GAMMA, weeks=weeks, budget_frac=frac)
    seed = 2000 + rep
    w = World(cfg, seed=seed)
    rng_hh = np.random.default_rng(seed + 11000)
    w.HH = rng_hh.choice(EMP_HH, size=cfg.n, replace=True).astype(float)
    pol = RandomPolicy(cfg, seed + 7)
    for t in range(weeks):
        w.step(pol.select(w, t))
    d = build_data(w, weeks, "aware")
    f = Fit(d)
    th = f.fit()
    lo, hi = f.profile_ci()
    return dict(geometry=geometry, design="random_calibrated", weeks=weeks,
                frac=frac, rep=rep, gamma_hat=th["gamma"], lo=lo, hi=hi,
                phi=th["phi"], covers=bool(lo <= TRUE_GAMMA <= hi),
                width=(hi - lo) if np.isfinite(hi - lo) else np.nan,
                finite=bool(np.isfinite(lo) and np.isfinite(hi)),
                conv=th["converged"])


def unit_job(args):
    return _run(BASE, "unit_square", *args)


def real_job(args):
    return _run(BASE_REAL, "real_zcta", *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--reps", type=int, default=25)
    a = ap.parse_args()
    t0 = time.time()
    out_path = os.path.join(OUT, "stage2_calib_e2.csv")
    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        for (g, w, f), grp in prev.groupby(["geometry", "weeks", "frac"]):
            if len(grp) >= a.reps:
                done.add((g, int(w), float(f)))
    for fn, geom, gname in ((unit_job, "unit", "unit_square"),
                            (real_job, "real", "real_zcta")):
        for (wk, fr) in CELLS:
            if (gname, wk, fr) in done:
                print(f"[skip] {geom} {wk}wk {fr}: already complete",
                      flush=True)
                continue
            jobs = [(wk, fr, r) for r in range(a.reps)]
            with ProcessPoolExecutor(a.procs) as ex:
                df = pd.DataFrame(ex.map(fn, jobs, chunksize=1))
            df.to_csv(out_path, mode="a", index=False,
                      header=not os.path.exists(out_path))
            covers = int(df["covers"].sum())
            finite = int(df["finite"].sum())
            width_med = float(df["width"].median())
            mean_g = float(df["gamma_hat"].mean())
            print(f"[done] {geom} {wk}wk {fr}: covers {covers}/"
                  f"{a.reps}, finite {finite}, width_med "
                  f"{width_med:.3f}, mean_g {mean_g:.3f} "
                  f"({round(time.time() - t0)} s)", flush=True)
    print("calibration rerun DONE in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
