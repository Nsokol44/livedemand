"""make_figures.py - manuscript figures from result CSVs (real numbers only).

Figure 1: the real-geometry testbed (sampled ZCTA centroids).
Figure 2: E2 design conditions, unit square (a) vs real ZCTA geometry (b):
          per-replication gamma estimates and profile intervals by design
          cell. ONE shared legend above both panels.
Figure 3: (a) W-sensitivity of gamma recovery on real geometry;
          (b) live-vs-frozen simulator contrast by geometry.
          ONE shared legend above both panels.

Run after run_realgeo.py. Outputs PNG (300 dpi) + PDF into ../figures/.

Style (2026-10-22 professional restyle; data handling unchanged from the
original build - same CSVs, same subsets, same jitter seeds):
  Liberation Sans throughout; Okabe-Ito colourblind-safe palette with a
  fixed colour->concept mapping in every figure (estimates / unit square
  = blue #0072B2, intervals = sky blue #56B4E9, true value = vermillion
  dashed #D55E00, real geometry bars = orange #E69F00); panel labels
  (a)/(b) at the top-left of each panel; light y-grid only; top/right
  spines removed except on the map (Figure 1).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
TRUE_G = 0.6
CELLS = [(2, 0.06), (2, 0.15), (4, 0.15), (8, 0.06), (16, 0.06), (16, 0.15)]
CELL_LAB = ["2 wk\n6%/wk", "2 wk\n15%/wk", "4 wk\n15%/wk",
            "8 wk\n6%/wk", "16 wk\n6%/wk", "16 wk\n15%/wk"]

EST = "#0072B2"    # replication estimates; unit-square bars (Okabe-Ito blue)
INT = "#56B4E9"    # profile intervals (Okabe-Ito sky blue)
TRUE = "#D55E00"   # true parameter value (Okabe-Ito vermillion)
REAL = "#E69F00"   # real-geometry bars (Okabe-Ito orange)
GRID = "#E3E7EB"

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def _clean(ax, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def fig1():
    samp = pd.read_csv(os.path.join(HERE, "data", "zcta_sample_1500.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.scatter(samp.lon, samp.lat, s=5, c=EST, alpha=0.7, linewidths=0)
    ax.set_aspect(1.3)  # ~1/cos(39.5 deg): geographically faithful lon/lat
    ax.set_xlabel("Longitude (\u00b0)")
    ax.set_ylabel("Latitude (\u00b0)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_zcta_sample.png"))
    fig.savefig(os.path.join(FIG, "fig1_zcta_sample.pdf"))
    plt.close(fig)


def _cell_panel(ax, df, title):
    rng = np.random.default_rng(7)
    for j, (wk, fr) in enumerate(CELLS):
        sub = df[(df.weeks == wk) & (np.isclose(df.frac, fr))]
        for _, r in sub.iterrows():
            x = j + rng.uniform(-0.16, 0.16)
            lo = r.lo if np.isfinite(r.lo) else -2.6
            hi = r.hi if np.isfinite(r.hi) else 2.9
            ax.plot([x, x], [max(lo, -2.6), min(hi, 2.9)], color=INT,
                    lw=1.3, zorder=1, solid_capstyle="round")
            if not np.isfinite(r.lo):
                ax.annotate("", xy=(x, -2.55), xytext=(x, -2.1),
                            arrowprops=dict(arrowstyle="->", color=INT, lw=1.2))
            if not np.isfinite(r.hi):
                ax.annotate("", xy=(x, 2.85), xytext=(x, 2.4),
                            arrowprops=dict(arrowstyle="->", color=INT, lw=1.2))
            y = float(np.clip(r.gamma_hat, -2.5, 2.8))
            ax.plot(x, y, "o", ms=4.5, color=EST, zorder=3)
    ax.axhline(TRUE_G, color=TRUE, ls="--", lw=1.3, zorder=2)
    ax.set_xticks(range(len(CELLS)), CELL_LAB)
    ax.set_ylim(-2.6, 2.9)
    ax.set_title(title, loc="left", fontweight="bold")
    _clean(ax)


def fig2():
    orig = pd.read_csv(os.path.join(RES, "e2_identification.csv"))
    orig = orig[orig.design == "random"]
    real = pd.read_csv(os.path.join(RES, "realgeo_e2.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    _cell_panel(axes[0], orig, "(a) Unit-square geometry (original E2)")
    _cell_panel(axes[1], real, "(b) Real ZCTA geometry (rerun)")
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


def fig3():
    ws = pd.read_csv(os.path.join(RES, "realgeo_wsens.csv"))
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
            ax.plot([x, x], [max(r.lo, -0.6), min(r.hi, 2.2)], color=INT, lw=1.3)
            ax.plot(x, float(np.clip(r.gamma_hat, -0.5, 2.1)), "o", ms=4.5,
                    color=EST, zorder=3)
    ax.axhline(TRUE_G, color=TRUE, ls="--", lw=1.3)
    ax.set_xticks(range(len(order)), [lab[o] for o in order])
    ax.set_ylim(-0.6, 2.2)
    ax.set_ylabel(r"$\gamma$ estimate (8 wk, 6%/wk, real geometry)")
    ax.set_title("(a) Fitted neighbourhood $W$ (true: k = 6)", loc="left",
                 fontweight="bold")
    _clean(ax)
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
    ax.set_xticks(range(3), ["$\\gamma$ = 0", "$\\gamma$ = 0.6", "$\\gamma$ = 1.2"])
    ax.set_ylabel("Cumulative orders, live / frozen ratio")
    ax.set_title("(b) Live vs frozen (simulator only)", loc="left",
                 fontweight="bold")
    _clean(ax)
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
    fig.savefig(os.path.join(FIG, "fig3_wsens_contrast.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig3_wsens_contrast.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("figures written to", FIG)
