#!/usr/bin/env python3
"""make_fig3_fg.py - Figure 3 rebuilt with fine-grid W-sensitivity intervals.

Stage-1 correction pass 2026-10-25. Identical to make_figures.fig3
except panel (a) reads results/realgeo_wsens_fg.csv (corrected
fine-grid profile intervals; run_wsens_finegrid.py) instead of
results/realgeo_wsens.csv. Panel (b) is unchanged (E1 simulator
contrast; no intervals involved).

Outputs (overwrites the Figure 3 files):
  ../figures/fig3_wsens_contrast.png (300 dpi) and .pdf (vector)

Usage: python make_fig3_fg.py
Deterministic: fixed jitter seed (11), as in make_figures.py.
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
EST, INT, TRUE, REAL = mf.EST, mf.INT, mf.TRUE, mf.REAL
TRUE_G = mf.TRUE_G


def main():
    ws = pd.read_csv(os.path.join(RES, "realgeo_wsens_fg.csv"))
    e1o = pd.read_csv(os.path.join(RES, "e1_policies.csv"))
    e1r = pd.read_csv(os.path.join(RES, "realgeo_e1.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fig.subplots_adjust(wspace=0.30)
    # (a) W sensitivity
    ax = axes[0]
    order = ["k6_true", "k3", "k10", "band100km"]
    lab = {"k6_true": "k = 6\n(true)", "k3": "k = 3", "k10": "k = 10",
           "band100km": "100-km\nband"}
    rng = np.random.default_rng(11)
    for j, name in enumerate(order):
        sub = ws[ws.fitted_W == name]
        for _, r in sub.iterrows():
            x = j + rng.uniform(-0.14, 0.14)
            ax.plot([x, x], [max(r.lo, -0.6), min(r.hi, 2.2)], color=INT,
                    lw=1.3)
            ax.plot(x, float(np.clip(r.gamma_hat, -0.5, 2.1)), "o",
                    ms=4.5, color=EST, zorder=3)
    ax.axhline(TRUE_G, color=TRUE, ls="--", lw=1.3)
    ax.set_xticks(range(len(order)), [lab[o] for o in order])
    ax.set_ylim(-0.6, 2.2)
    ax.set_ylabel(r"$\gamma$ estimate (8 wk, 6%/wk, real geometry)")
    ax.set_title("(a) Fitted neighbourhood $W$ (true: k = 6)", loc="left",
                 fontweight="bold")
    mf._clean(ax)
    # (b) live vs frozen (simulator-internal)
    ax = axes[1]
    gammas = [0.0, 0.6, 1.2]

    def ratio(df, g, pol_a, pol_b):
        a = df[(df.gamma == g) & (df.policy == pol_a)]["c24"].mean()
        b = df[(df.gamma == g) & (df.policy == pol_b)]["c24"].mean()
        return a / b

    xo = np.arange(3) - 0.19
    xr = np.arange(3) + 0.19
    ro = [ratio(e1o, g, "live_aware", "frozen") for g in gammas]
    rr = [ratio(e1r, g, "live_aware", "frozen") for g in gammas]
    ax.bar(xo, ro, width=0.36, color=EST, label="Unit square (original E1)")
    ax.bar(xr, rr, width=0.36, color=REAL, label="Real ZCTA geometry (rerun)")
    ax.axhline(1.0, color="#555555", lw=1.0)
    ax.set_xticks(range(3), ["$\\gamma$ = 0", "$\\gamma$ = 0.6",
                             "$\\gamma$ = 1.2"])
    ax.set_ylabel("Cumulative orders, live / frozen ratio")
    ax.set_title("(b) Live vs frozen (simulator only)", loc="left",
                 fontweight="bold")
    mf._clean(ax)
    handles = [Line2D([0], [0], marker="o", ls="", color=EST,
                      label="Replication estimate (a)"),
               Line2D([0], [0], color=INT, lw=1.5,
                      label="95% profile interval (a)"),
               Line2D([0], [0], color=TRUE, ls="--", lw=1.3,
                      label="True $\\gamma$ = 0.6 (a)"),
               plt.Rectangle((0, 0), 1, 1, color=EST,
                             label="Unit square (b)"),
               plt.Rectangle((0, 0), 1, 1, color=REAL,
                             label="Real ZCTA geometry (b)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, "fig3_wsens_contrast.png"),
                bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig3_wsens_contrast.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("fig3 (fine grid) written")


if __name__ == "__main__":
    main()
