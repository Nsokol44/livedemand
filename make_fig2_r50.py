#!/usr/bin/env python3
"""make_fig2_r50.py - Figure 2 rebuilt from the 50-replication E2 rerun.

Revision pass 2026-10-22 (external review item L1). Same design, style,
panels, cells, colours and legend as make_figures.fig2 (the shipped
figure), but reads results/e2_unit_r50.csv and results/e2_real_r50.csv
(50 replications per cell) and draws with smaller markers and lighter
interval strokes so 50 replications per cell remain readable.

Outputs (overwrites the shipped Figure 2 files):
  ../figures/fig2_design_conditions.png (300 dpi) and .pdf (vector)

Usage: python make_fig2_r50.py
Deterministic: fixed jitter seed (7), as in make_figures.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import make_figures as mf

RES = mf.RES
FIG = mf.FIG
CELLS, CELL_LAB = mf.CELLS, mf.CELL_LAB
EST, INT, TRUE = mf.EST, mf.INT, mf.TRUE


def cell_panel50(ax, df, title):
    rng = np.random.default_rng(7)
    for j, (wk, fr) in enumerate(CELLS):
        sub = df[(df.weeks == wk) & (np.isclose(df.frac, fr))]
        assert len(sub) == 50, (title, wk, fr, len(sub))
        for _, r in sub.iterrows():
            x = j + rng.uniform(-0.30, 0.30)
            lo = r.lo if np.isfinite(r.lo) else -2.6
            hi = r.hi if np.isfinite(r.hi) else 2.9
            ax.plot([x, x], [max(lo, -2.6), min(hi, 2.9)], color=INT,
                    lw=0.7, alpha=0.55, zorder=1,
                    solid_capstyle="round")
            if not np.isfinite(r.lo):
                ax.annotate("", xy=(x, -2.55), xytext=(x, -2.1),
                            arrowprops=dict(arrowstyle="->", color=INT,
                                            lw=0.8))
            if not np.isfinite(r.hi):
                ax.annotate("", xy=(x, 2.85), xytext=(x, 2.4),
                            arrowprops=dict(arrowstyle="->", color=INT,
                                            lw=0.8))
            y = float(np.clip(r.gamma_hat, -2.5, 2.8))
            ax.plot(x, y, "o", ms=2.4, color=EST, zorder=3, alpha=0.85)
    ax.axhline(TRUE_G, color=TRUE, ls="--", lw=1.3, zorder=2)
    ax.set_xticks(range(len(CELLS)), CELL_LAB)
    ax.set_ylim(-2.6, 2.9)
    ax.set_title(title, loc="left", fontweight="bold")
    mf._clean(ax)


TRUE_G = mf.TRUE_G


def main():
    unit = pd.read_csv(os.path.join(RES, "e2_unit_r50.csv"))
    real = pd.read_csv(os.path.join(RES, "e2_real_r50.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    cell_panel50(axes[0], unit, "(a) Unit-square geometry (original E2)")
    cell_panel50(axes[1], real, "(b) Real ZCTA geometry (rerun)")
    axes[0].set_ylabel(r"Peer parameter $\gamma$ (true value 0.6)")
    axes[1].tick_params(labelleft=False)
    handles = [Line2D([0], [0], marker="o", ls="", color=EST,
                      label="Replication estimate"),
               Line2D([0], [0], color=INT, lw=1.5,
                      label="95% profile interval (arrow = unbounded)"),
               Line2D([0], [0], color=TRUE, ls="--", lw=1.3,
                      label="True $\\gamma$ = 0.6")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG, "fig2_design_conditions.png"),
                bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig2_design_conditions.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("fig2 (50-rep) written")


if __name__ == "__main__":
    main()
