#!/usr/bin/env python3
"""run_wsens_finegrid.py - W-sensitivity rerun with the corrected interval.

Stage-1 correction pass 2026-10-25. Same design as run_realgeo.py wsens:
the five real-geometry (8 wk, 6%) datasets (world seeds 2000+rep, policy
seeds seed+7) refitted under the true k = 6 graph and three wrong
neighbourhood definitions (k = 3, k = 10, 100-km distance band), via
run_realgeo.fit_with_W. The only change is that Fit.profile_ci now ships
the fine-grid construction, so the intervals - and only the intervals -
differ from results/realgeo_wsens.csv; gamma_hat is asserted against
that file value for value.

Output (new file; shipped CSV untouched):
  results/realgeo_wsens_fg.csv

Usage: python run_wsens_finegrid.py [--procs 2]
Deterministic: fixed seeds.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_realgeo import wsens_job

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(a.procs) as ex:
        rows = [r for sub in ex.map(wsens_job, range(5), chunksize=1)
                for r in sub]
    df = pd.DataFrame(rows)
    ref = pd.read_csv(os.path.join(OUT, "realgeo_wsens.csv"))
    m = df.merge(ref, on=["rep", "fitted_W"], suffixes=("_fg", "_old"))
    assert len(m) == len(ref), (len(m), len(ref))
    err = float(np.max(np.abs(m["gamma_hat_fg"] - m["gamma_hat_old"])))
    print(f"gamma_hat vs shipped wsens: n={len(m)}, max diff = {err:.2e}")
    assert err < 1e-9
    df.to_csv(os.path.join(OUT, "realgeo_wsens_fg.csv"), index=False)
    df["width"] = df["hi"] - df["lo"]
    print(df.groupby("fitted_W").agg(
        gamma_mean=("gamma_hat", "mean"), covers=("covers", "sum"),
        width_med=("width", "median")).to_string())
    print("wsens fine-grid DONE in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
