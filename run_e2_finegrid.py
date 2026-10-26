#!/usr/bin/env python3
"""run_e2_finegrid.py - E2 grid rerun with the corrected interval procedure.

Stage-1 correction pass 2026-10-25. Identical worlds, estimator, cells
and seed scheme to run_e2_rerun50.py (world seeds 2000+rep, policy seeds
seed+7); the ONLY change is that Fit.profile_ci now ships the fine-grid
construction (0.025 lattice, no interpolation of the crossing) instead of
the superseded coarse procedure (0.25 steps + linear interpolation).
Point estimates, phi and convergence are therefore unchanged, and this
script asserts gamma_hat against the r50 CSVs value for value before
accepting each cell.

Grid (as in run_e2_rerun50.py):
  random design, unit square: weeks {2,4,8,16} x frac {0.02, 0.06, 0.15}
  random design, real ZCTA:   weeks {2,4,8,16} x frac {0.06, 0.15}
  adaptive designs (unit square, 16 wk, 6%): eps in {0.0, 0.2}
  50 replications per cell.

Outputs (new files; shipped and r50 CSVs untouched):
  results/e2_unit_r50_fg.csv, results/e2_real_r50_fg.csv,
  results/e2_adaptive_r50_fg.csv

Sanity gate (printed at the end): at the richest cell (16 wk, 15%) the
fine-grid intervals are compared replication by replication against
results/diagnostics_figrid_richest.csv, whose diagnostic run reported
coverage 41/50 (unit square) and 40/50 (real ZCTA).

Usage: python run_e2_finegrid.py [--procs 2] [--reps 50] [--skip-done]
Deterministic: fixed seeds; no wall-clock or environment dependence in
any reported value.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e2_rerun50 import unit_job, real_job, adaptive_job, run_cells, OUT


def assert_gamma_hat(df, ref_csv, keys, label):
    """gamma_hat (and phi) must reproduce the coarse-procedure r50 run,
    because the correction changes intervals only."""
    ref = pd.read_csv(os.path.join(OUT, ref_csv))
    m = df.merge(ref, on=keys, suffixes=("_fg", "_r50"))
    assert len(m) == len(df), (label, len(m), len(df))
    err = float(np.max(np.abs(m["gamma_hat_fg"] - m["gamma_hat_r50"])))
    err_phi = float(np.max(np.abs(m["phi_fg"] - m["phi_r50"])))
    print(f"[{label}] gamma_hat vs r50: n={len(m)}, max diff = {err:.2e}; "
          f"phi max diff = {err_phi:.2e}", flush=True)
    assert err < 1e-9 and err_phi < 1e-9, label


def summarise(df, label):
    s = df.groupby(["weeks", "frac"]).agg(
        mean_g=("gamma_hat", "mean"), covers=("covers", "sum"),
        finite=("finite", "sum"), n=("covers", "size"),
        width_med=("width", "median"))
    print(f"--- {label} (fine grid) ---")
    print(s.to_string(), flush=True)
    return s


def sanity_figrid(unit, real):
    """Richest-cell comparison against the 2026-10-22 fine-grid diagnostic."""
    ref = pd.read_csv(os.path.join(OUT, "diagnostics_figrid_richest.csv"))
    print("--- sanity gate vs diagnostics_figrid_richest.csv ---", flush=True)
    for label, df, geom in (("unit", unit, "unit_square"),
                            ("real", real, "real_zcta")):
        sub = df[(df.weeks == 16) & (np.isclose(df.frac, 0.15))].sort_values("rep")
        r = ref[ref.geometry == geom].sort_values("rep")
        m = sub.merge(r, on="rep", suffixes=("_fg", "_diag"))
        assert len(m) == 50, (label, len(m))
        dlo = float(np.max(np.abs(m["lo"] - m["lo_fine"])))
        dhi = float(np.max(np.abs(m["hi"] - m["hi_fine"])))
        same = int((np.isclose(m["lo"], m["lo_fine"])
                    & np.isclose(m["hi"], m["hi_fine"])).sum())
        print(f"{label}: coverage fg = {int(m['covers'].sum())}/50 vs "
              f"diagnostic fine = {int(m['covers_fine'].sum())}/50; "
              f"intervals identical in {same}/50 reps; max |bound diff| "
              f"lo {dlo:.4f}, hi {dhi:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    reps = range(a.reps)
    t0 = time.time()

    unit_cells = [(wk, fr) for wk in (2, 4, 8, 16) for fr in (0.02, 0.06, 0.15)]
    real_cells = [(wk, fr) for wk in (2, 4, 8, 16) for fr in (0.06, 0.15)]
    unit = run_cells(
        unit_job,
        [(f"unit_{wk}wk_{fr}", [(wk, fr, r) for r in reps])
         for (wk, fr) in unit_cells],
        os.path.join(OUT, "e2_unit_r50_fg.csv"), a.procs, a.skip_done)
    assert_gamma_hat(unit, "e2_unit_r50.csv",
                     ["geometry", "design", "weeks", "frac", "eps", "rep"],
                     "unit")
    unit.to_csv(os.path.join(OUT, "e2_unit_r50_fg.csv"), index=False)
    print("unit stage done", round(time.time() - t0), "s", flush=True)

    real = run_cells(
        real_job,
        [(f"real_{wk}wk_{fr}", [(wk, fr, r) for r in reps])
         for (wk, fr) in real_cells],
        os.path.join(OUT, "e2_real_r50_fg.csv"), a.procs, a.skip_done)
    assert_gamma_hat(real, "e2_real_r50.csv",
                     ["geometry", "design", "weeks", "frac", "eps", "rep"],
                     "real")
    real.to_csv(os.path.join(OUT, "e2_real_r50_fg.csv"), index=False)
    print("real stage done", round(time.time() - t0), "s", flush=True)

    adap = run_cells(
        adaptive_job,
        [(f"adap_eps{e}", [(e, r) for r in reps]) for e in (0.0, 0.2)],
        os.path.join(OUT, "e2_adaptive_r50_fg.csv"), a.procs, a.skip_done)
    assert_gamma_hat(adap, "e2_adaptive_r50.csv",
                     ["geometry", "design", "weeks", "frac", "eps", "rep"],
                     "adaptive")
    adap.to_csv(os.path.join(OUT, "e2_adaptive_r50_fg.csv"), index=False)
    print("adaptive stage done", round(time.time() - t0), "s", flush=True)

    summarise(unit, "unit square, random")
    summarise(real, "real ZCTA, random")
    summarise(adap, "unit square, adaptive 16wk 6%")
    sanity_figrid(unit, real)
    print("ALL DONE in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
