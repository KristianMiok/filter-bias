"""
Figure 1 - simulation decision map (replaces the four-corners panel).

Runs a fine beta x rho sweep of the two-knob simulation and draws two heatmaps:
  (A) BEST-STRATEGY map: which practitioner strategy (ALL / FILTER / IPW)
      best reconstructs the true niche in each (beta, rho) regime.
  (B) RECOVERY of the winner: how well the best strategy recovers truth
      (Spearman vs the true suitability surface), showing where even the
      best choice is compromised (the honest "nobody wins" corner).

beta = environment<->quality coupling (Knob 1); rho = directedness of the
coordinate error (Knob 2). D held fixed (a D-sweep goes to the Supplement).
IPW uses the *estimated* propensity (practitioner view), not the oracle.

Self-contained: imports the simulation from qfbias_sim, runs the sweep here,
writes Fig1_simulation_decision_map.pdf / .png at 300 dpi.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from qfbias_sim import SimConfig, run_sweep

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ---- sweep grid -------------------------------------------------------------
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = list(range(8))
STRATS = ["ALL", "FILTER", "IPW_est"]
STRAT_LABEL = {"ALL": "ALL (no filter)", "FILTER": "FILTER (hard cut)", "IPW_est": "IPW (reweight)"}
TAB = plt.get_cmap("tab10").colors
STRAT_COLOR = {"ALL": TAB[7], "FILTER": TAB[3], "IPW_est": TAB[0]}


def build_grids(df):
    metric = "rho_sp_"
    g = df.groupby(["beta", "rho"])[[metric + s for s in STRATS]].mean().reset_index()

    def grid_for(col):
        # rows = beta ascending here; we will flip for display
        return g.pivot(index="beta", columns="rho", values=col)

    grids = {s: grid_for(metric + s) for s in STRATS}
    betas_ax = grids["ALL"].index.values
    rhos_ax = grids["ALL"].columns.values
    stack = np.stack([grids[s].values for s in STRATS], axis=0)   # (S, nbeta, nrho)
    best_idx = np.argmax(stack, axis=0)
    best_rec = np.max(stack, axis=0)
    return betas_ax, rhos_ax, best_idx, best_rec


def main():
    base = SimConfig(grid=160, n_occ=3000, n_bg=8000, corr_len=12.0, D=10.0)
    print(f"Running decision-map sweep: {len(BETAS)} betas x {len(RHOS)} rhos "
          f"x {len(SEEDS)} seeds = {len(BETAS)*len(RHOS)*len(SEEDS)} runs ...")
    df = run_sweep(BETAS, RHOS, [base.D], SEEDS, base)
    df.to_csv("reports/decision_map_sweep.csv", index=False)
    print("  sweep done, wrote reports/decision_map_sweep.csv")

    betas_ax, rhos_ax, best_idx, best_rec = build_grids(df)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 3.9))
    extent = [rhos_ax.min() - 0.125, rhos_ax.max() + 0.125,
              betas_ax.min() - 0.25, betas_ax.max() + 0.5]

    # ---- Panel A: best strategy (categorical) ----
    cmap_cat = ListedColormap([STRAT_COLOR[s] for s in STRATS])
    norm_cat = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_cat.N)
    axA.imshow(best_idx, origin="lower", aspect="auto", extent=extent,
               cmap=cmap_cat, norm=norm_cat, interpolation="nearest")
    axA.set_xlabel("ρ  (directedness of coordinate error)")
    axA.set_ylabel("β  (environment ↔ quality coupling)")
    axA.set_title("Best strategy by regime", fontsize=9.5)
    axA.set_xticks(rhos_ax); axA.set_yticks(betas_ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STRAT_COLOR[s]) for s in STRATS]
    axA.legend(handles, [STRAT_LABEL[s] for s in STRATS], frameon=True,
               framealpha=0.9, fontsize=6.8, loc="upper left")

    # ---- Panel B: recovery of the winner (continuous) ----
    im = axB.imshow(best_rec, origin="lower", aspect="auto", extent=extent,
                    cmap="viridis", vmin=max(0.5, best_rec.min()), vmax=1.0,
                    interpolation="nearest")
    axB.set_xlabel("ρ  (directedness of coordinate error)")
    axB.set_ylabel("β  (environment ↔ quality coupling)")
    axB.set_title("Recovery of the best strategy\n(Spearman vs true suitability)", fontsize=9.5)
    axB.set_xticks(rhos_ax); axB.set_yticks(betas_ax)
    cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    cb.set_label("recovery", fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"Fig1_simulation_decision_map.{ext}", dpi=300, bbox_inches="tight")
    print("wrote Fig1_simulation_decision_map.pdf / .png")

    # text summary for the caption
    print("\nBest-strategy summary (rows beta high->low):")
    for i in range(len(betas_ax) - 1, -1, -1):
        row = [STRATS[best_idx[i, j]].replace("_est", "") for j in range(len(rhos_ax))]
        print(f"  beta={betas_ax[i]:>3}: {row}")


if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)
    main()
