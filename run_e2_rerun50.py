"""run_e2_rerun50.py - E2 identification grid rerun at 50 replications/cell.

Revision pass 2026-10-22 (external review item L1). Identical worlds,
estimator, and seed scheme to the shipped 5-replication runs: world seeds
2000+rep, policy seeds seed+7, so replications 0-4 reproduce the shipped
CSVs (results/e2_identification.csv, results/realgeo_e2.csv) value for
value; the script asserts that before writing anything. Replications
5-49 extend the same scheme (seeds 2005..2049).

Grid:
  random design, unit square: weeks {2,4,8,16} x frac {0.02, 0.06, 0.15}
  random design, real ZCTA:   weeks {2,4,8,16} x frac {0.06, 0.15}
  adaptive designs (unit square, 16 wk, 6%): eps in {0.0, 0.2}

Outputs (new files; shipped CSVs untouched):
  results/e2_unit_r50.csv, results/e2_real_r50.csv,
  results/e2_adaptive_r50.csv

Usage: python run_e2_rerun50.py [--procs 2] [--reps 50]
Deterministic: fixed seeds; no wall-clock or environment dependence in
any reported value.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from livedemand.world import WorldConfig, World
from livedemand.policies import RandomPolicy, LivePolicy, build_data
from livedemand.model import Fit

HERE = os.path.dirname(os.path.abspath(__file__))
COORDS = os.path.join(HERE, "data", "zcta_sample_1500.csv")
OUT = os.path.join(HERE, "results")
TRUE_GAMMA = 0.6
BASE = WorldConfig(n=1500, weeks=24)
BASE_REAL = WorldConfig(n=1500, weeks=24, coords_path=COORDS)


def _fit_row(w, weeks):
    d = build_data(w, weeks, "aware")
    f = Fit(d)
    th = f.fit()
    lo, hi = f.profile_ci()
    return th, lo, hi, d


def unit_job(args):
    weeks, frac, rep = args
    cfg = replace(BASE, gamma=TRUE_GAMMA, weeks=weeks, budget_frac=frac)
    seed = 2000 + rep
    w = World(cfg, seed=seed)
    pol = RandomPolicy(cfg, seed + 7)
    for t in range(weeks):
        w.step(pol.select(w, t))
    th, lo, hi, d = _fit_row(w, weeks)
    return dict(geometry="unit_square", design="random", weeks=weeks,
                frac=frac, eps=1.0, rep=rep, gamma_hat=th["gamma"], lo=lo,
                hi=hi, phi=th["phi"], covers=bool(lo <= TRUE_GAMMA <= hi),
                width=(hi - lo) if np.isfinite(hi - lo) else np.nan,
                finite=bool(np.isfinite(lo) and np.isfinite(hi)),
                corr_r=float(np.corrcoef(th["r"], w.r)[0, 1]),
                exposed_share=float((d.S.sum(1) > 0).mean()),
                conv=th["converged"])


def real_job(args):
    weeks, frac, rep = args
    cfg = replace(BASE_REAL, gamma=TRUE_GAMMA, weeks=weeks, budget_frac=frac)
    seed = 2000 + rep
    w = World(cfg, seed=seed)
    pol = RandomPolicy(cfg, seed + 7)
    for t in range(weeks):
        w.step(pol.select(w, t))
    th, lo, hi, d = _fit_row(w, weeks)
    return dict(geometry="real_zcta", design="random", weeks=weeks,
                frac=frac, eps=1.0, rep=rep, gamma_hat=th["gamma"], lo=lo,
                hi=hi, phi=th["phi"], covers=bool(lo <= TRUE_GAMMA <= hi),
                width=(hi - lo) if np.isfinite(hi - lo) else np.nan,
                finite=bool(np.isfinite(lo) and np.isfinite(hi)),
                corr_r=float(np.corrcoef(th["r"], w.r)[0, 1]),
                orders_total=int(sum(w.total_orders)),
                conv=th["converged"])


def adaptive_job(args):
    eps, rep = args
    weeks, frac = 16, 0.06
    cfg = replace(BASE, gamma=TRUE_GAMMA, weeks=weeks, budget_frac=frac)
    seed = 2000 + rep
    w = World(cfg, seed=seed)
    pol = LivePolicy(cfg, seed + 7, mode="aware", eps=eps, name="live")
    for t in range(weeks):
        w.step(pol.select(w, t))
    th, lo, hi, d = _fit_row(w, weeks)
    return dict(geometry="unit_square", design="adaptive", weeks=weeks,
                frac=frac, eps=eps, rep=rep, gamma_hat=th["gamma"], lo=lo,
                hi=hi, phi=th["phi"], covers=bool(lo <= TRUE_GAMMA <= hi),
                width=(hi - lo) if np.isfinite(hi - lo) else np.nan,
                finite=bool(np.isfinite(lo) and np.isfinite(hi)),
                corr_r=float(np.corrcoef(th["r"], w.r)[0, 1]),
                exposed_share=float((d.S.sum(1) > 0).mean()),
                conv=th["converged"])


def check_against_shipped(df, shipped_path, keys, label, design=None):
    """Replications 0..4 must reproduce the shipped CSVs (gamma_hat)."""
    old = pd.read_csv(shipped_path)
    old = old[old["rep"] < 5]
    if design is not None:
        old = old[old["design"] == design]
    m = df[df["rep"] < 5].merge(old, on=keys, suffixes=("_new", "_old"))
    assert len(m) == len(old), (label, len(m), len(old))
    err = float(np.max(np.abs(m["gamma_hat_new"] - m["gamma_hat_old"])))
    print(f"[{label}] reps 0-4 vs shipped CSV: n={len(m)}, "
          f"max |gamma_hat diff| = {err:.2e}")
    assert err < 1e-9, label
    return True


def summarise(df, label):
    g = df.groupby(["weeks", "frac"])
    s = g.agg(mean_g=("gamma_hat", "mean"), covers=("covers", "sum"),
              finite=("finite", "sum"), n=("covers", "size"),
              width_med=("width", "median"))
    print(f"--- {label} ---")
    print(s.to_string())
    return s


def run_cells(fn, jobs_by_cell, out_csv, procs, skip_done):
    """Run jobs cell by cell, appending each cell's rows to out_csv as
    it completes (checkpointing); returns the full DataFrame."""
    done = set()
    if skip_done and os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(prev["cell"].unique())
        frames = [prev]
        print(f"[resume] {os.path.basename(out_csv)}: cells done: "
              f"{sorted(done)}", flush=True)
    else:
        frames = []
    for cell, jobs in jobs_by_cell:
        if cell in done:
            continue
        t = time.time()
        with ProcessPoolExecutor(procs) as ex:
            df = pd.DataFrame(ex.map(fn, jobs, chunksize=1))
        df["cell"] = cell
        frames.append(df)
        df.to_csv(out_csv, mode="a", index=False,
                  header=not os.path.exists(out_csv))
        print(f"[done] {cell}: {len(df)} reps in "
              f"{round(time.time() - t)} s", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--skip-done", action="store_true",
                    help="reuse stage CSVs already on disk")
    a = ap.parse_args()
    reps = range(a.reps)
    t0 = time.time()

    unit_cells = [(wk, fr) for wk in (2, 4, 8, 16) for fr in (0.02, 0.06, 0.15)]
    real_cells = [(wk, fr) for wk in (2, 4, 8, 16) for fr in (0.06, 0.15)]
    unit = run_cells(
        unit_job,
        [(f"unit_{wk}wk_{fr}", [(wk, fr, r) for r in reps])
         for (wk, fr) in unit_cells],
        os.path.join(OUT, "e2_unit_r50.csv"), a.procs, a.skip_done)
    print("unit stage done", round(time.time() - t0), "s", flush=True)
    real = run_cells(
        real_job,
        [(f"real_{wk}wk_{fr}", [(wk, fr, r) for r in reps])
         for (wk, fr) in real_cells],
        os.path.join(OUT, "e2_real_r50.csv"), a.procs, a.skip_done)
    print("real stage done", round(time.time() - t0), "s", flush=True)
    adap = run_cells(
        adaptive_job,
        [(f"adap_eps{e}", [(e, r) for r in reps]) for e in (0.0, 0.2)],
        os.path.join(OUT, "e2_adaptive_r50.csv"), a.procs, a.skip_done)
    print("adaptive stage done", round(time.time() - t0), "s", flush=True)

    check_against_shipped(
        unit, os.path.join(OUT, "e2_identification.csv"),
        ["design", "weeks", "frac", "eps", "rep"], "unit", design="random")
    check_against_shipped(
        real, os.path.join(OUT, "realgeo_e2.csv"),
        ["weeks", "frac", "rep"], "real")
    check_against_shipped(
        adap, os.path.join(OUT, "e2_identification.csv"),
        ["design", "weeks", "frac", "eps", "rep"], "adaptive",
        design="adaptive")

    unit.to_csv(os.path.join(OUT, "e2_unit_r50.csv"), index=False)
    real.to_csv(os.path.join(OUT, "e2_real_r50.csv"), index=False)
    adap.to_csv(os.path.join(OUT, "e2_adaptive_r50.csv"), index=False)
    summarise(unit, "unit square, random")
    summarise(real, "real ZCTA, random")
    summarise(adap, "unit square, adaptive 16wk 6%")
    print("ALL DONE in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
