#!/usr/bin/env python3
"""diagnostics_figrid.py - fine-grid profile intervals at the richest cell.

Revision pass 2026-10-22 (L2 follow-up to diagnostics.py D1). D1 showed,
for replication 0 of the richest cell (16 wk, 15%/wk, gamma = 0.6), that
the shipped profile_ci (0.25 steps + linear interpolation) understates
the interval width by about a factor of three relative to a fine-grid
(0.025-step, warm-started) profile at the same chi-square cut widened by
the same phi. This script computes fine-grid intervals for all 50
replications of the richest cell on both geometries, so coverage with
the interpolation artefact removed can be compared with the shipped
procedure's 12/50 (unit square) and 8/50 (real ZCTA).

Method per replication: rebuild the world exactly as in
run_e2_rerun50.py (seeds 2000+rep / +7), fit the shipped penalised
model, then evaluate the penalised profile NLL on the grid
0.30..0.85 by 0.025 with sequential warm starts, and take the
outermost grid points whose Delta-NLL <= chi2(0.95,1)/2 * phi_hat as
the interval (grid resolution 0.025; no interpolation).

Outputs: results/diagnostics_figrid_richest.csv with, per replication,
the shipped interval, the fine-grid interval, and coverage of each.

Usage: python diagnostics_figrid.py [--procs 2]
Deterministic: fixed seeds; grid and cut fixed.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from livedemand.world import WorldConfig, World
from livedemand.policies import RandomPolicy, build_data
from livedemand.model import Fit

HERE = os.path.dirname(os.path.abspath(__file__))
COORDS = os.path.join(HERE, "data", "zcta_sample_1500.csv")
OUT = os.path.join(HERE, "results")
TRUE_GAMMA = 0.6
WEEKS, FRAC = 16, 0.15
BASE = WorldConfig(n=1500, weeks=24)
BASE_REAL = WorldConfig(n=1500, weeks=24, coords_path=COORDS)
GRID = np.arange(0.30, 0.851, 0.025)


def job(args):
    geometry, rep = args
    base = BASE_REAL if geometry == "real_zcta" else BASE
    cfg = replace(base, gamma=TRUE_GAMMA, weeks=WEEKS, budget_frac=FRAC)
    seed = 2000 + rep
    w = World(cfg, seed=seed)
    pol = RandomPolicy(cfg, seed + 7)
    for t in range(WEEKS):
        w.step(pol.select(w, t))
    d = build_data(w, WEEKS, "aware")
    f = Fit(d)
    th = f.fit()
    lo_s, hi_s = f.profile_ci()
    cut = chi2.ppf(0.95, 1) / 2.0 * th["phi"]
    nlls = []
    warm = th
    for g in GRID:
        fg = Fit(d)
        tg = fg.fit(theta0=warm, g_fixed=float(g))
        nlls.append(tg["nll"])
        warm = tg
    nlls = np.asarray(nlls)
    ok = (nlls - th["nll"]) <= cut
    lo_f = float(GRID[ok].min()) if ok.any() else np.nan
    hi_f = float(GRID[ok].max()) if ok.any() else np.nan
    return dict(geometry=geometry, rep=rep, gamma_hat=th["gamma"],
                phi=th["phi"], lo_shipped=lo_s, hi_shipped=hi_s,
                covers_shipped=bool(lo_s <= TRUE_GAMMA <= hi_s),
                lo_fine=lo_f, hi_fine=hi_f,
                covers_fine=bool(lo_f <= TRUE_GAMMA <= hi_f),
                width_shipped=hi_s - lo_s, width_fine=hi_f - lo_f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(g, r) for g in ("unit_square", "real_zcta")
            for r in range(50)]
    with ProcessPoolExecutor(a.procs) as ex:
        rows = list(ex.map(job, jobs, chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "diagnostics_figrid_richest.csv"),
              index=False)
    for g, sub in df.groupby("geometry"):
        print(g, "shipped covers", int(sub.covers_shipped.sum()), "/50;",
              "fine-grid covers", int(sub.covers_fine.sum()), "/50;",
              "median widths", round(sub.width_shipped.median(), 3),
              "vs", round(sub.width_fine.median(), 3))
    print("DONE in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
