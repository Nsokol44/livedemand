"""Stage-1 companion: D3 bootstrap rerun with the corrected interval
procedure, cell by cell with checkpointing.

The D3 parametric-bootstrap study in diagnostics.py (Step 3) regenerates
worlds from the rep-0 fits and refits them; the interval it records comes
from Fit.profile_ci, so rerunning the identical design (same per-job rng
seeds, same variants and generating modes) yields the corrected-procedure
counterpart of results/diagnostics_bootstrap.csv. D1 baselines are
unaffected (point estimates and the chi-square calibration do not involve
profile_ci). The shipped diagnostics files are left intact as the record
of the diagnosis; this script writes results/diagnostics_bootstrap_fg.csv.

Cells (geometry x variant x generating mode, 40 jobs each) are computed
and appended one at a time; rerunning skips cells already present in the
output file, so an interrupted run resumes where it stopped. gamma_hat
is asserted against the shipped bootstrap file cell by cell: the
correction changes intervals only.

Invocation detail that matters: the shipped run called boot_job with
geometry labels "unit_square"/"real_zcta" (diagnostics.py main), and
boot_job's per-job rng seed branches on the label (the "unit" branch,
offset 0, was never taken in the shipped run; both geometries used the
+5000 offset). This rerun passes the same labels, so the bootstrap
worlds coincide job for job with the shipped ones.

Run:  python3 diagnostics_boot_fg.py [--procs N]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

import diagnostics as D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "diagnostics_bootstrap_fg.csv")
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "diagnostics_bootstrap.csv")
CELLS = [(geom, variant, gen)
         for geom in ("unit_square", "real_zcta")
         for variant in ("cond", "full")
         for gen in ("fitted", "truth")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()

    ref = pd.read_csv(REF)
    frames = []
    done_cells = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        frames.append(prev)
        done_cells = set(zip(prev.geometry, prev.variant, prev.gen))
        print(f"[resume] cells done: {sorted(done_cells)}", flush=True)

    for geom, variant, gen in CELLS:
        if (geom, variant, gen) in done_cells:
            continue
        jobs = [(geom, variant, gen, b) for b in range(args.reps)]
        t0 = time.time()
        with ProcessPoolExecutor(args.procs) as ex:
            rows = list(ex.map(D.boot_job, jobs))
        sub = pd.DataFrame(rows)
        r = ref[(ref.geometry == geom) & (ref.variant == variant)
                & (ref.gen == gen)].sort_values("b")
        s = sub.sort_values("b")
        assert len(r) == len(s) == args.reps
        d = float(np.max(np.abs(s.gamma_hat.values - r.gamma_hat.values)))
        assert d < 1e-9, (geom, variant, gen, d)
        frames.append(sub)
        pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
        print(f"[done] {geom}/{variant}/{gen}: {len(sub)} jobs in "
              f"{time.time() - t0:.0f} s; coverage "
              f"{int(sub.covers.sum())}/{args.reps} (shipped "
              f"{int(r.covers.sum())}/{args.reps}); gamma_hat max diff {d:.2e}",
              flush=True)

    out = pd.read_csv(OUT)
    print("--- bootstrap: corrected (fine grid) coverage ---", flush=True)
    print(out.groupby(["geometry", "variant", "gen"]).covers.agg(["sum", "count"]),
          flush=True)
    print("full-feedback bias by geometry (fg):", flush=True)
    full = out[out.variant == "full"]
    print(full.groupby("geometry").apply(
        lambda g: float((g.gamma_hat - g.gen_gamma).mean()), include_groups=False),
        flush=True)
    print(f"bootstrap fine-grid DONE: {len(out)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
