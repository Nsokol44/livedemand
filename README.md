# livedemand

Simulation code and result files for the manuscript **"Identifying spatial
spillover in geo-experiments"** (Nicholas J. Sokol). The study is a simulation
and design study: every outcome in the result files is synthetic. The only
real data used are U.S. Census Bureau ZCTA centroid coordinates (see
**Data sources** below).

License: MIT (see `LICENSE`).

This folder contains everything needed to reproduce the simulation results
reported in the manuscript.

## Data sources

- **U.S. Census Bureau 2024 Gazetteer, ZCTA national file** (public domain).
  This is the only external dataset. It is **not** committed to this
  repository because of its size; `prepare_geometry.py` downloads it on
  first run from the Census Bureau:
  https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_zcta_national.zip
  and draws the seeded (seed 20261021) sample of 1,500 ZCTA centroids used
  by the real-geometry rerun. The derived sample itself is committed at
  `data/zcta_sample_1500.csv`.
- No real store data are used anywhere in this repository. The real-data
  validation design specified in Section 6 of the manuscript is gated and
  unrun; no pilot data exist.

## Contents

- `livedemand/` — a copy of the LiveDemand v0.1 package (`world.py`,
  `model.py`, `policies.py`, `records.py`) with exactly one modification:
  `WorldConfig` accepts `coords_path`, a CSV of real ZCTA centroids. With
  `coords_path` empty the behavior is byte-identical to the original
  unit-square code. The originals in the project archive are untouched.
- `prepare_geometry.py` — downloads the U.S. Census Bureau 2024 Gazetteer
  ZCTA national file (public domain) if absent and draws the seeded
  (seed 20261021) uniform sample of 1,500 ZCTA centroids used by the rerun:
  `data/zcta_sample_1500.csv`. Only the geometry is real; all outcomes are
  synthetic.
- `run_realgeo.py` — the real-geometry rerun. One command:

  ```
  python run_realgeo.py all --procs 4
  ```

  runs, in order: geometry diagnostics, the E2 identification cells
  (random design; 2 wk @ 6%, 2 wk @ 15%, 4 wk @ 15%, 8 wk @ 6%, 16 wk @ 6%,
  16 wk @ 15%; gamma = 0.6; 5 replications; world seeds 2000+rep), the
  W-sensitivity refits (same 8 wk @ 6% datasets refitted with k = 3, k = 10
  and a 100-km distance-band graph in place of the true k = 6), and the E1
  live-vs-frozen simulator contrast (gamma in {0, 0.6, 1.2}, 24 weeks,
  2 replications, world seeds 1000+rep). Outputs land in `results/`.
- `make_figures.py` — builds the manuscript figures from the result CSVs
  (original results are read from the project archive paths documented at
  the top of the script; copies used for this submission are in `results/`).
- `results/` — result CSVs: `realgeo_e2.csv`, `realgeo_wsens.csv`,
  `realgeo_e1.csv`, `geometry_diagnostics.csv`, plus copies of the original
  `e1_policies.csv` / `e2_identification.csv` the figures draw on.

## Revision-pass additions (2026-10-22)

- `run_e2_rerun50.py` — the E2 identification grid rerun at 50
  replications per cell (review item L1). Same worlds, estimator and seed
  scheme as the original runs (world seeds 2000+rep, policy seeds
  seed+7), so replications 0–4 reproduce the shipped CSVs exactly (the
  script asserts this). Grid: unit square 12 cells, real ZCTA 8 cells,
  adaptive (16 wk @ 6%, eps 0/0.2) 2 cells; outputs
  `results/e2_unit_r50.csv`, `results/e2_real_r50.csv`,
  `results/e2_adaptive_r50.csv`, checkpointed per cell. One command:

  ```
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_e2_rerun50.py --procs 2 --reps 50 --skip-done
  ```

  Measured runtime on the build host (2 cores): 1,580 s for 1,100 fits.
- `diagnostics.py` — the richest-cell interval-failure diagnostics
  (review item L2/L4): fine-grid profiles, penalized vs unpenalized
  refits, parametric bootstrap under the fitted model (conditional and
  full-feedback variants, B = 40), multi-start agreement, artifact scan.
  Outputs `results/diagnostics_*.csv`. One command:
  `python diagnostics.py --procs 2 --boot 40` (576 s on the build host).
- `diagnostics_figrid.py` — fine-grid (0.025-step) profile intervals for
  all 50 replications of the richest cell on both geometries, same cut
  and dispersion factor as the shipped procedure. Output
  `results/diagnostics_figrid_richest.csv` (288 s on the build host).
- `make_fig2_r50.py` — rebuilds Figure 2 from the 50-replication CSVs in
  the `make_figures.py` style (smaller markers for 50 replications).
- `voice_metrics.py` — the before/after voice measurements for the
  revision's voice pass (documented counting rules in its docstring).

## Stage-1 correction (2026-10-25): the interval procedure

The profile-likelihood interval procedure shipped in `livedemand/model.py`
searched outward from the estimate in steps of 0.25 in gamma and linearly
interpolated where the profile crossed its cut, which understates interval
width several-fold at rich designs. `Fit.profile_ci` now computes intervals
on a fine grid (0.025 steps, same cut, same dispersion factor) and takes the
grid points themselves as the bounds, with no interpolation. The superseded
construction remains available as `profile_ci(..., legacy=True)`, for the
record. All other model code and every seed are unchanged.

The E2 grid, the W-sensitivity refits and the parametric bootstrap were
rerun with the corrected procedure, on identical worlds and seeds (the
runners assert the point estimates against the earlier result files, value
for value; only intervals change):

- `run_e2_finegrid.py` — the full E2 grid (unit square 12 cells, real
  ZCTA 8 cells, adaptive 2 cells; 50 replications per cell). Outputs
  `results/e2_unit_r50_fg.csv`, `results/e2_real_r50_fg.csv`,
  `results/e2_adaptive_r50_fg.csv`; run log `results/stage1_fg.log`.
  One command: `python run_e2_finegrid.py --procs 2 --skip-done`.
  Sanity gate: at the richest cell the rerun reproduces the independent
  fine-grid diagnostic (`diagnostics_figrid_richest.csv`) exactly —
  coverage 41/50 (unit square) and 40/50 (real ZCTA), intervals identical
  in 50/50 replications on both geometries.
- `run_wsens_finegrid.py` — the W-sensitivity refits. Output
  `results/realgeo_wsens_fg.csv`.
- `diagnostics_boot_fg.py` — the D3 parametric bootstrap (same per-job
  seeds and geometry labels as the shipped `diagnostics.py` run, so the
  bootstrap worlds coincide job for job). Output
  `results/diagnostics_bootstrap_fg.csv`. Corrected coverage: 22–36 of
  40 per cell (unit square) and 25–33 of 40 (real ZCTA), against 3–15
  of 40 for the superseded procedure.
- `make_fig2_fg.py`, `make_fig3_fg.py` — rebuild Figures 2 and 3 from
  the corrected result files.

Headline consequence (richest cell, 16 weeks at 15%/week): coverage of
the true gamma is 41/50 on the unit square and 40/50 on real ZCTA
geometry, against 12/50 and 8/50 under the superseded procedure. The
residual shortfall against the nominal 95% is attributed in the
manuscript to the estimator's small downward bias through the
accumulated peer state.

## Original experiments

The original unit-square experiments (E1–E3) are reproduced from the
project archive by:

```
python experiments.py e1 --reps 5 --procs 4
python experiments.py e2 --reps 5 --procs 4
python experiments.py e3 --reps 5 --procs 4
```

## Reproducibility notes

- Fixed seeds everywhere: world seeds 2000+rep (E2 family), 1000+rep (E1
  family); geometry sample seed 20261021. Policies see the same world and
  hype path within a replication (common random numbers).
- One seed caveat: with real geometry the unit-square coordinate draw is
  skipped, so the random stream advances differently and a real-geometry
  world with seed s is not the unit-square world with seed s plus new
  coordinates — it is a fresh draw on a real map. Comparisons between
  geometries are therefore distributional (across replications), not
  paired unit-by-unit.
- Dependencies are pinned in `requirements.txt`.
