#!/usr/bin/env python3
"""stage2_summarize.py - assemble every stage-2 number the manuscript
quotes into results/stage2_summary.json, including the calibration
verdict against the paper's corrected (fine-grid) E2 results.

One-command rerun of the whole stage-2 analysis, in order:
  python stage2_build_panels.py && python stage2_fit_demo.py && \
  python stage2_fit_demo_fe.py && python stage2_calibration_rerun.py && \
  python stage2_summarize.py
(This script itself only reads result files.)
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def cell_table(df):
    g = df.groupby(["geometry", "weeks", "frac"]).agg(
        covers=("covers", "sum"), finite=("finite", "sum"),
        n=("covers", "size"), width_med=("width", "median"),
        mean_g=("gamma_hat", "mean"))
    return {f"{i[0]}|{i[1]}|{i[2]}": dict(covers=int(r["covers"]),
                                           finite=int(r["finite"]),
                                           n=int(r["n"]),
                                           width_med=(None if pd.isna(r["width_med"])
                                                      else round(float(r["width_med"]), 3)),
                                           mean_g=round(float(r["mean_g"]), 3))
            for i, r in g.iterrows()}


def main():
    s = {}
    s["targets"] = pd.read_csv(os.path.join(
        OUT, "stage2_calibration_targets.csv")).to_dict("records")
    s["calibration_params"] = json.load(open(os.path.join(
        OUT, "stage2_calibration_params.json")))
    cal = pd.read_csv(os.path.join(OUT, "stage2_calib_e2.csv"))
    s["calibrated_cells"] = cell_table(cal)
    unit = pd.read_csv(os.path.join(OUT, "e2_unit_r50_fg.csv"))
    real = pd.read_csv(os.path.join(OUT, "e2_real_r50_fg.csv"))
    paper = pd.concat([unit, real], ignore_index=True)
    paper = paper[paper["design"] == "random"]
    s["paper_cells"] = cell_table(paper)
    keys = [(g, w, f) for g in ("unit_square", "real_zcta")
            for (w, f) in [(2, 0.06), (2, 0.15), (4, 0.15), (8, 0.06),
                           (8, 0.15), (16, 0.06), (16, 0.15)]]
    verdict = {}
    for g, w, f in keys:
        k = f"{g}|{w}|{f}"
        c, p = s["calibrated_cells"].get(k), s["paper_cells"].get(k)
        if c and p:
            verdict[k] = dict(calibrated_covers=c["covers"],
                              paper_covers=p["covers"],
                              calibrated_width=c["width_med"],
                              paper_width=p["width_med"],
                              calibrated_mean_g=c["mean_g"],
                              paper_mean_g=p["mean_g"])
    s["verdict_cells"] = verdict
    s["demo_fits"] = pd.read_csv(os.path.join(
        OUT, "stage2_demo_fits.csv")).to_dict("records")
    s["demo_fits_fe"] = pd.read_csv(os.path.join(
        OUT, "stage2_demo_fits_fe.csv")).to_dict("records")
    s["ev_fits"] = pd.read_csv(os.path.join(
        OUT, "stage2_ev_fits.csv")).to_dict("records")
    s["ev_fits_fe"] = pd.read_csv(os.path.join(
        OUT, "stage2_ev_fits_fe.csv")).to_dict("records")
    ws = pd.read_csv(os.path.join(OUT, "realgeo_wsens_fg.csv"))
    ws["width"] = ws["hi"] - ws["lo"]
    s["paper_wsens"] = ws.groupby("fitted_W").agg(
        gamma_mean=("gamma_hat", "mean"), covers=("covers", "sum"),
        width_med=("width", "median")).reset_index().to_dict("records")
    with open(os.path.join(OUT, "stage2_summary.json"), "w") as fh:
        json.dump(s, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    print("summary written")


if __name__ == "__main__":
    main()
