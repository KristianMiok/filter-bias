"""
exp0b_error_alignment.py
========================
EXPERIMENT 0b — what actually governs whether IPW helps or harms is not the
"directedness" of coordinate error (rho) but the ALIGNMENT of the displacement
with the quality-coupling axis in environmental space.

Four error geometries, all with |delta| == D, all in the honest build
(fixed_magnitude=True, propensity_on="obs"):

    toward_HQ   displaced along +grad(E1): low-quality records pushed TOWARD the
                environments where high-quality records already sit
                (e.g. snapping to accessible centroids when accessibility drives quality)
    away_HQ     displaced along -grad(E1): pushed AWAY from high-quality environments
    geo_fixed   fixed geographic direction, uncorrelated with E1 -> environmentally
                neutral in expectation
    mixture_HQ  as toward_HQ but each record is either fully directed (prob rho)
                or isotropic (prob 1-rho) -- robustness to the blend definition

Outputs
    reports/exp0b_alignment_sweep.csv    raw rows
    reports/exp0b_alignment_summary.csv  means per geometry x beta x rho
    reports/exp0b_alignment_verdict.txt  printed table

Run from the repo root:   python scripts/exp0b_error_alignment.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfbias_sim import SimConfig, run_sweep  # noqa: E402

BETAS = [0.0, 1.0, 2.0, 3.0, 4.0]
RHOS = [0.0, 0.5, 1.0]
SEEDS = list(range(8))
BASE = dict(grid=160, n_occ=3000, n_bg=8000, corr_len=12.0, D=10.0,
            fixed_magnitude=True, propensity_on="obs")

GEOMS = {
    "toward_HQ":  dict(error_follow_gradient=True,  error_gradient_sign=+1.0, error_mode="blend"),
    "away_HQ":    dict(error_follow_gradient=True,  error_gradient_sign=-1.0, error_mode="blend"),
    "geo_fixed":  dict(error_follow_gradient=False, error_dir=(0.0, 1.0),     error_mode="blend"),
    "mixture_HQ": dict(error_follow_gradient=True,  error_gradient_sign=+1.0, error_mode="mixture"),
}
REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
COLS = ["rho_sp_ALL", "rho_sp_FILTER", "rho_sp_IPW_oracle", "rho_sp_IPW_est",
        "diag_prop_coef_E1", "diag_lowq_dE1_shift", "diag_ess_frac"]


def verdict(summary: pd.DataFrame) -> str:
    L = []; P = L.append
    P("=" * 96)
    P("EXPERIMENT 0b — ERROR ALIGNMENT   (honest build; |delta| = D for all geometries)")
    P("=" * 96)
    P("\nAt rho = 1 (fully directed), by geometry.  coef = fitted propensity coefficient on E1;")
    P("dE1 = mean E1 shift of low-quality records (obs - true); gap = IPW_est - FILTER.\n")
    P("  beta  geometry     coef(E1)   dE1     ALL    FILTER  IPW_est  IPW_orc   gap    winner")
    for beta in BETAS:
        for g in GEOMS:
            r = summary[(summary.beta == beta) & (summary.rho == 1.0) & (summary.geom == g)].iloc[0]
            gap = r.rho_sp_IPW_est - r.rho_sp_FILTER
            vals = {"ALL": r.rho_sp_ALL, "FILTER": r.rho_sp_FILTER, "IPW": r.rho_sp_IPW_est}
            top = sorted(vals.items(), key=lambda kv: -kv[1])
            win = top[0][0] if top[0][1] - top[1][1] >= 0.005 else "TIE"
            P(f"  {beta:4.1f}  {g:11s} {r.diag_prop_coef_E1:+7.2f}  {r.diag_lowq_dE1_shift:+6.2f}  "
              f"{r.rho_sp_ALL:.3f}  {r.rho_sp_FILTER:.3f}   {r.rho_sp_IPW_est:.3f}   {r.rho_sp_IPW_oracle:.3f}  "
              f"{gap:+.3f}  {win}")
        P("")
    P("Reading:")
    P("  toward_HQ : coef attenuated / sign-flipped -> IPW under-corrects, can lose to FILTER")
    P("  away_HQ   : coef exaggerated -> IPW over-corrects but stays near oracle; ALL collapses")
    P("  geo_fixed : coef ~ unbiased (dE1 ~ 0) -> everything behaves as at rho = 0")
    P("  mixture_HQ: same as toward_HQ -> result does not depend on the blend definition")
    P("\nrho = 0 reference (all geometries identical by construction):")
    r0 = summary[(summary.rho == 0.0) & (summary.geom == "toward_HQ")]
    P("  beta   coef(E1)   ALL    FILTER  IPW_est  IPW_orc")
    for _, r in r0.iterrows():
        P(f"  {r.beta:4.1f}  {r.diag_prop_coef_E1:+7.2f}   {r.rho_sp_ALL:.3f}  {r.rho_sp_FILTER:.3f}   "
          f"{r.rho_sp_IPW_est:.3f}   {r.rho_sp_IPW_oracle:.3f}")
    P("")
    return "\n".join(L)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    frames = []
    for name, flags in GEOMS.items():
        t0 = time.time()
        base = SimConfig(**BASE, **flags)
        df = run_sweep(BETAS, RHOS, [base.D], SEEDS, base)
        df.insert(0, "geom", name)
        frames.append(df)
        print(f"  {name:11s} {len(df)} runs  {time.time()-t0:5.1f}s")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(os.path.join(REPORTS, "exp0b_alignment_sweep.csv"), index=False)

    num = [c for c in raw.select_dtypes(include=[np.number]).columns
           if c not in ("beta", "rho", "seed")]
    summary = raw.groupby(["geom", "beta", "rho"])[num].mean().reset_index()
    summary.to_csv(os.path.join(REPORTS, "exp0b_alignment_summary.csv"), index=False)

    text = verdict(summary)
    with open(os.path.join(REPORTS, "exp0b_alignment_verdict.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
