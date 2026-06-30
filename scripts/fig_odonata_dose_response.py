"""
fig_odonata_dose_response.py
============================
Odonata threshold dose-response -- the sharpest single image in the paper.

Two stacked panels sharing the uncertainty-threshold x-axis:
  (top)    energy-distance permutation p-value, with the 0.05 significance line and the
           permutation floor (1/(n_perm+1)) marked; shows the shift is significant under strict
           cuts and non-significant under lenient cuts.
  (bottom) shift magnitude: energy distance and the leading-axis |SMD|, showing the same
           dose-response from the effect-size side.
Subtle shading separates the strict (<=~200 m) and lenient (>~200 m) regimes around the break.

Numbers are the Odonata threshold-sweep results (reports/odonata_threshold_sweep.csv). They are
hard-coded here so the figure is self-contained and reproducible; if the sweep is re-run with a
higher n_perm to push the strict-regime p below the current floor, update FLOOR and the p-values.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
})

# --- sweep results -------------------------------------------------------- #
thr      = np.array([25, 50, 100, 150, 250, 500])      # m
retain   = np.array([53.7, 58.9, 64.4, 65.8, 88.7, 90.1])
energy_p = np.array([0.0083, 0.0083, 0.0083, 0.0083, 0.1570, 0.0661])
energy_d = np.array([0.0322, 0.0261, 0.0257, 0.0310, 0.0076, 0.0097])
top_smd  = np.array([0.200, 0.167, 0.179, 0.174, 0.060, 0.054])   # |SMD| of leading axis
FLOOR    = 1.0 / (120 + 1)        # permutation floor at n_perm=120
BREAK    = 200.0                  # regime break (m)

ALPHA = 0.05
strict = "#2c7fb8"
lenient = "#d95f0e"
shade_strict = "#2c7fb8"
shade_lenient = "#d95f0e"

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(6.4, 6.0), sharex=True,
    gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12})

# regime shading on both panels
for ax in (ax1, ax2):
    ax.axvspan(20, BREAK, color=shade_strict, alpha=0.05, zorder=0)
    ax.axvspan(BREAK, 560, color=shade_lenient, alpha=0.05, zorder=0)

# colour points by regime
colors = np.where(thr <= BREAK, strict, lenient)

# ---- TOP: significance ---------------------------------------------------- #
ax1.set_yscale("log")
ax1.plot(thr, energy_p, "-", color="0.6", lw=1.2, zorder=2)
ax1.scatter(thr, energy_p, c=colors, s=55, zorder=3, edgecolor="white", linewidth=0.8)
ax1.axhline(ALPHA, color="0.30", ls="--", lw=1.0, zorder=1)
ax1.axhline(FLOOR, color="0.55", ls=":", lw=1.0, zorder=1)
ax1.text(545, ALPHA * 1.18, "p = 0.05", ha="right", va="bottom", fontsize=8.5, color="0.30")
ax1.text(545, FLOOR * 0.80, "permutation floor", ha="right", va="top", fontsize=8.0, color="0.55")
ax1.set_ylabel("energy-distance\npermutation $p$")
ax1.set_ylim(FLOOR * 0.55, 0.55)
# significance call-outs
ax1.text(85, 0.0083 * 1.7, "significant", ha="center", va="bottom", fontsize=8.5,
         color=strict, style="italic")
ax1.text(370, 0.155 * 1.25, "n.s.", ha="center", va="bottom", fontsize=9.0,
         color=lenient, style="italic")

# ---- BOTTOM: magnitude ---------------------------------------------------- #
ax2.plot(thr, energy_d, "-o", color="0.35", lw=1.3, ms=5, zorder=3,
         label="energy distance")
ax2.set_ylabel("energy distance", color="0.20")
ax2.set_ylim(0, 0.040)
ax2.tick_params(axis="y", labelcolor="0.20")

ax2b = ax2.twinx()
ax2b.spines["top"].set_visible(False)
ax2b.plot(thr, top_smd, "--s", color="#7b3294", lw=1.3, ms=5, zorder=3,
          label="leading-axis |SMD|")
ax2b.set_ylabel("leading-axis |SMD|", color="#7b3294")
ax2b.set_ylim(0, 0.24)
ax2b.tick_params(axis="y", labelcolor="#7b3294")

ax2.set_xlabel("uncertainty threshold (m)   —   stricter filtering ←   → more retained")
ax2.set_xlim(10, 560)
ax2.set_xticks(thr)
ax2.set_xticklabels([str(t) for t in thr])

# secondary x annotation: retained % (top of panel). 'kept:' label placed above the row.
ax1.text(250, 0.45, "records kept:", ha="center", va="center", fontsize=7.5, color="0.4")
for t, r, c in zip(thr, retain, colors):
    ax1.annotate(f"{r:.0f}%", (t, 0.34), ha="center", va="center", fontsize=7.2, color=c)

# regime legend
leg = [Patch(facecolor=shade_strict, alpha=0.18, label="strict regime (shift significant)"),
       Patch(facecolor=shade_lenient, alpha=0.18, label="lenient regime (shift vanishes)")]
ax1.legend(handles=leg, loc="upper left", fontsize=8.0, frameon=False,
           bbox_to_anchor=(0.0, 0.86), ncol=1)

# combined magnitude legend
l1, lab1 = ax2.get_legend_handles_labels()
l2, lab2 = ax2b.get_legend_handles_labels()
ax2.legend(l1 + l2, lab1 + lab2, loc="upper center", fontsize=8.0, frameon=False, ncol=2)

fig.suptitle("Filtering bias is dose-dependent",
             fontsize=11.5, y=1.0, fontweight="bold")

fig.savefig("Fig_odonata_dose_response.pdf", bbox_inches="tight")
fig.savefig("Fig_odonata_dose_response.png", dpi=300, bbox_inches="tight")
print("wrote Fig_odonata_dose_response.pdf / .png")
