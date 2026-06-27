"""
make_figures.py
Builds the three MEE manuscript figures from the SUMMARY NUMBERS produced in the runs
(simulation 2x2 validation, crayfish diagnostic top-SMD, per-species strategy divergence).
Outputs PDF + PNG at 300 dpi. For the final version, regenerate Fig 1 from a finer beta x rho
sweep and Fig 2 from the full 302-feature SMD table; styling matches (Arial, tab10).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
TAB = plt.cm.tab10.colors


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"/home/claude/pkg/{name}.{ext}")
    plt.close(fig)


# ----------------------------------------------------------------------------- #
# Figure 1 - simulation decision map (2x2 validation): no strategy dominates
# ----------------------------------------------------------------------------- #
regimes = ["no coupling\n(\u03b2=0, \u03c1=0)", "directed error\n(\u03b2=0, \u03c1=1)",
           "coupling,\nisotropic error\n(\u03b2=3, \u03c1=0)", "coupling +\ndirected error\n(\u03b2=3, \u03c1=1)"]
vals = {"ALL": [0.999, 0.857, 0.997, 0.826],
        "FILTER": [0.999, 0.999, 0.864, 0.864],
        "IPW": [0.999, 0.999, 0.970, 0.970]}
colors1 = {"ALL": TAB[7], "FILTER": TAB[3], "IPW": TAB[0]}

fig, ax = plt.subplots(figsize=(6.6, 3.4))
x = np.arange(len(regimes)); w = 0.26
for i, (k, v) in enumerate(vals.items()):
    ax.bar(x + (i - 1) * w, v, w, label=k, color=colors1[k], edgecolor="white", linewidth=0.5)
# mark the best strategy per regime
best = ["tie", "FILTER/IPW", "ALL", "IPW"]
for j in range(len(regimes)):
    top = max(v[j] for v in vals.values())
    ax.text(x[j], top + 0.006, f"best: {best[j]}", ha="center", va="bottom",
            fontsize=7.2, color="#444")
ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=8)
ax.set_ylim(0.78, 1.02); ax.set_ylabel("Spearman with true suitability")
ax.set_title("No strategy dominates: the best treatment flips with the data regime")
ax.legend(frameon=False, ncol=3, loc="lower center", fontsize=8.5)
ax.axhline(1.0, color="#bbb", lw=0.6, ls=":")
save(fig, "Fig1_simulation_decision_map")


# ----------------------------------------------------------------------------- #
# Figure 2 - crayfish diagnostic: per-axis shift + propensity decomposition
# ----------------------------------------------------------------------------- #
smd_feat = ["l_CLI25", "l_CLI27", "l_CLI13", "l_CLI15", "l_CLI26", "l_CLI14",
            "l_SOL47", "l_SOL45", "l_SOL42", "l_SOL46", "l_SOL43", "l_TOP29",
            "l_CLI53", "l_CLI65", "l_SOL41", "l_CLI69", "l_TOP109", "l_CLI37"]
smd_val = [0.161, 0.160, 0.160, 0.160, 0.160, 0.159, 0.155, 0.153, -0.150,
           0.146, -0.142, 0.140, 0.140, 0.137, -0.133, 0.131, 0.130, 0.128]
grp_color = {"l_CLI": TAB[0], "l_SOL": TAB[1], "l_TOP": TAB[2], "l_LAC": TAB[4]}
bar_colors = [grp_color[f[:5]] for f in smd_feat]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.8), gridspec_kw={"width_ratios": [2.1, 1]})
order = np.argsort(smd_val)
ax1.barh(np.arange(len(smd_feat)), [smd_val[i] for i in order],
         color=[bar_colors[i] for i in order], edgecolor="white", linewidth=0.4)
ax1.set_yticks(np.arange(len(smd_feat))); ax1.set_yticklabels([smd_feat[i] for i in order], fontsize=7)
ax1.axvline(0, color="#888", lw=0.6)
ax1.set_xlabel("standardized mean difference  (all \u2192 high-accuracy)")
ax1.set_title("Filtering shifts the niche\nalong climatic axes", fontsize=9.5)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
           [grp_color["l_CLI"], grp_color["l_SOL"], grp_color["l_TOP"]]]
ax1.legend(handles, ["climate", "soil", "topography"], frameon=False, fontsize=7.5, loc="lower right")

auc = {"env": 0.722, "metadata\n(status+year)": 0.688, "env +\nmetadata": 0.763}
ax2.bar(range(3), list(auc.values()), color=[TAB[0], TAB[8], TAB[2]],
        edgecolor="white", linewidth=0.5, width=0.62)
for i, v in enumerate(auc.values()):
    ax2.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
ax2.set_xticks(range(3)); ax2.set_xticklabels(list(auc.keys()), fontsize=7.3)
ax2.set_ylim(0.5, 0.80); ax2.set_ylabel("propensity AUC (basin CV)")
ax2.set_title("Quality is environmentally\nstructured beyond metadata", fontsize=9.5)
ax2.text(0.5, 0.515, "energy-distance shift p = 0.005\nincremental env over metadata +0.075",
         transform=ax2.transData, ha="center", fontsize=6.6, color="#444")
save(fig, "Fig2_crayfish_diagnostic")


# ----------------------------------------------------------------------------- #
# Figure 3 - consequence: modest & status-blind filtering cost; status signal only in IPW reweighting
# ----------------------------------------------------------------------------- #
sp = ["Astacus astacus", "Pontastacus leptodactylus", "Procambarus clarkii",
      "Austropotamobius torrentium", "Pacifastacus leniusculus", "Faxonius limosus",
      "Austropotamobius pallipes"]
sp_lab = ["A. astacus", "P. leptodactylus", "P. clarkii", "A. torrentium",
          "P. leniusculus", "F. limosus", "A. pallipes"]
all_filter = [0.0469, 0.0367, 0.0338, 0.0333, 0.0332, 0.0322, 0.0296]
filter_ipw = [0.0073, 0.0206, 0.0190, 0.0092, 0.0159, 0.0214, 0.0121]
alien = [0.00, 0.55, 0.87, 0.00, 0.98, 0.95, 0.08]
stat_color = [TAB[3] if a >= 0.5 else TAB[0] for a in alien]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.8, 3.6), sharey=False)
yy = np.arange(len(sp))[::-1]
axa.barh(yy, all_filter, color=stat_color, edgecolor="white", linewidth=0.4)
axa.set_yticks(yy); axa.set_yticklabels(sp_lab, fontsize=7.3, style="italic")
axa.set_xlabel("mean |\u0394 suitability|"); axa.set_xlim(0, 0.05)
axa.set_title("Real cost of filtering (ALL \u2192 FILTER):\nmodest and status-blind", fontsize=9)

# order panel b by filter_ipw to show the status sorting
ob = np.argsort(filter_ipw)
axb.barh(np.arange(len(sp)), [filter_ipw[i] for i in ob],
         color=[stat_color[i] for i in ob], edgecolor="white", linewidth=0.4)
axb.set_yticks(np.arange(len(sp))); axb.set_yticklabels([sp_lab[i] for i in ob], fontsize=7.3, style="italic")
axb.set_xlabel("mean |\u0394 suitability|"); axb.set_xlim(0, 0.025)
axb.set_title("IPW reweighting sensitivity (FILTER \u2192 IPW):\nstatus-sorted (a method property, not ecology)", fontsize=9)

h = [plt.Rectangle((0, 0), 1, 1, color=TAB[3]), plt.Rectangle((0, 0), 1, 1, color=TAB[0])]
axa.legend(h, ["alien-dominated", "native-dominated"], frameon=False, fontsize=7.3, loc="lower right")
save(fig, "Fig3_consequence_divergence")

print("figures written")
