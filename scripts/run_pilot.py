"""Pilot: can intrinsic-dimension geometry detect contamination?  python scripts/run_pilot.py"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from filter_bias.synthetic import Manifold, contaminate_offmanifold, contaminate_onmanifold
from filter_bias.idtools import local_id_mle, id_vs_scale_decimation, knn_dist

D, D_AMBIENT, FRAC, K = 5, 20, 0.10, 50
m = Manifold(d=D, D=D_AMBIENT, seed=1)
X, z = m.sample(4000, seed=1)
print(f"true intrinsic dim d={D}, ambient D={D_AMBIENT}, contamination fraction={FRAC:.0%}\n")

Xc, mask = contaminate_offmanifold(X, FRAC, sigma=1.0, seed=2)
s_cl, id_cl = id_vs_scale_decimation(X, seed=3)
s_co, id_co = id_vs_scale_decimation(Xc, seed=3)

print("OFF-manifold contamination (bad coordinates):")
print("  sigma   local-ID AUC   naive-dist AUC   gap")
sweep = []
for sigma in [0.25, 0.5, 0.75, 1.0, 1.5]:
    Xs, ms = contaminate_offmanifold(X, FRAC, sigma, seed=2)
    a_id = roc_auc_score(ms, local_id_mle(Xs, K)); a_ds = roc_auc_score(ms, knn_dist(Xs, K))
    sweep.append((sigma, a_id, a_ds))
    print(f"  {sigma:4.2f}    {a_id:.3f}          {a_ds:.3f}          {a_id-a_ds:+.3f}")

Xon, mon = contaminate_onmanifold(m, X, FRAC, seed=2)
print(f"\nON-manifold contamination (misID, negative control):")
print(f"  local-ID AUC={roc_auc_score(mon, local_id_mle(Xon, K)):.3f}   "
      f"naive-dist AUC={roc_auc_score(mon, knn_dist(Xon, K)):.3f}   (both ~0.5 = blind)")

# --- separating the two effects: shift (loc) vs concentration (scale) ---
from filter_bias.synthetic import contaminate_onmanifold_local

print("\nTEST A - isolate SHIFT (scale fixed at 1.0, vary loc):")
print("  loc    local-ID AUC   naive-dist AUC")
for loc in [0.0, 1.0, 2.0, 3.0]:
    Xl, ml = contaminate_onmanifold_local(m, X, FRAC, loc=loc, scale=1.0, seed=2)
    print(f"  {loc:4.1f}   {roc_auc_score(ml, local_id_mle(Xl, K)):.3f}          "
          f"{roc_auc_score(ml, knn_dist(Xl, K)):.3f}")

print("\nTEST B - isolate CONCENTRATION (loc fixed at 0.0, vary scale):")
print("  scale   local-ID AUC   naive-dist AUC")
for scale in [1.0, 0.5, 0.2, 0.1]:
    Xl, ml = contaminate_onmanifold_local(m, X, FRAC, loc=0.0, scale=scale, seed=2)
    print(f"  {scale:4.2f}    {roc_auc_score(ml, local_id_mle(Xl, K)):.3f}          "
          f"{roc_auc_score(ml, knn_dist(Xl, K)):.3f}")


fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(s_cl, id_cl, "o-", label="clean"); ax[0].plot(s_co, id_co, "s-", label="off-manifold contaminated")
ax[0].axhline(D, color="k", lw=0.8, label="true ID"); ax[0].set_xscale("log"); ax[0].invert_xaxis()
ax[0].set_xlabel("subsample size (smaller = larger scale)"); ax[0].set_ylabel("ID")
ax[0].set_title("ID vs scale"); ax[0].legend(fontsize=9)
sig = [s[0] for s in sweep]
ax[1].plot(sig, [s[1] for s in sweep], "o-", label="local-ID score")
ax[1].plot(sig, [s[2] for s in sweep], "s-", label="naive distance")
ax[1].axhline(0.5, color="k", lw=0.8); ax[1].set_ylim(0.45, 1.02)
ax[1].set_xlabel("contamination strength (sigma)"); ax[1].set_ylabel("detection ROC-AUC")
ax[1].set_title("detection: geometry vs baseline"); ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig("figures/pilot.png", dpi=150)
print("\nsaved figures/pilot.png")
