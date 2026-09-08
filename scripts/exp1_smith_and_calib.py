"""
exp1_smith_and_calib.py
=======================
EXPERIMENT 1 — two uses of the same practitioner information (the uncertainty
radius D around each low-quality record), in the honest build:

    SMITH_env / SMITH_geo   Smith et al. 2023 (GEB 32, e13628): impute each
                            imprecise record to the cell within its uncertainty
                            area that is closest to the environmental / geographic
                            centroid of the precise records. Published, packaged
                            (enmSdmX), and the closest prior work to this paper.
    ALL_calib / IPW_calib   regression calibration: replace the point environment
                            of each imprecise record by the MEAN environment over
                            its uncertainty disk (Carroll et al.); use it for the
                            SDM (ALL_calib) and for the propensity (IPW_calib).

Prediction: Smith's imputation pulls imprecise records toward the (already
selection-biased) centroid of the precise records and therefore AMPLIFIES
selection bias when beta > 0. Regression calibration is unbiased for e_true in
expectation under isotropic error and should close the IPW_est -> IPW_oracle gap
at rho = 0; it cannot fix directed error.

Grid: beta x {rho=0} U {rho=1, toward_HQ} U {rho=1, away_HQ}, 8 seeds.

Outputs
    reports/exp1_smith_calib_sweep.csv
    reports/exp1_smith_calib_summary.csv
    reports/exp1_smith_calib_verdict.txt

Run from the repo root:   python scripts/exp1_smith_and_calib.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfbias_sim import SimConfig, run_sweep  # noqa: E402

BETAS = [0.0, 1.0, 2.0, 3.0, 4.0]
SEEDS = list(range(8))
BASE = dict(grid=160, n_occ=3000, n_bg=8000, corr_len=12.0, D=10.0,
            fixed_magnitude=True, propensity_on="obs")

REGIMES = {
    "iso":        dict(rhos=[0.0], error_gradient_sign=+1.0),
    "toward_HQ":  dict(rhos=[1.0], error_gradient_sign=+1.0),
    "away_HQ":    dict(rhos=[1.0], error_gradient_sign=-1.0),
}
STRATS = ["ALL", "FILTER", "IPW_est", "IPW_calib", "ALL_calib", "SMITH_env", "SMITH_geo", "IPW_oracle"]
REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def verdict(summary: pd.DataFrame) -> str:
    L = []; P = L.append
    P("=" * 110)
    P("EXPERIMENT 1 — SMITH 2023 IMPUTATION vs REGRESSION CALIBRATION   (honest build, Spearman vs truth)")
    P("=" * 110)
    for regime in REGIMES:
        P(f"\n[{regime}]")
        P("  beta  " + "".join(f"{s:>11s}" for s in STRATS) + "     coef_obs  coef_calib")
        for beta in BETAS:
            r = summary[(summary.regime == regime) & (summary.beta == beta)].iloc[0]
            vals = "".join(f"{r['rho_sp_' + s]:11.3f}" for s in STRATS)
            P(f"  {beta:4.1f}  {vals}     {r.diag_prop_coef_E1:+8.2f}  {r.diag_prop_coef_E1_calib:+8.2f}")
    P("\nKEY CONTRASTS (honest build)")
    P("  regime      beta   SMITH_env - ALL   ALL_calib - ALL   IPW_calib - IPW_est   IPW_oracle - IPW_calib")
    for regime in REGIMES:
        for beta in BETAS:
            r = summary[(summary.regime == regime) & (summary.beta == beta)].iloc[0]
            P(f"  {regime:10s}  {beta:4.1f}   {r.rho_sp_SMITH_env - r.rho_sp_ALL:+14.3f}   "
              f"{r.rho_sp_ALL_calib - r.rho_sp_ALL:+14.3f}   {r.rho_sp_IPW_calib - r.rho_sp_IPW_est:+18.3f}   "
              f"{r.rho_sp_IPW_oracle - r.rho_sp_IPW_calib:+20.3f}")
    P("")
    return "\n".join(L)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    frames = []
    for name, spec in REGIMES.items():
        t0 = time.time()
        base = SimConfig(**BASE, error_follow_gradient=True,
                         error_gradient_sign=spec["error_gradient_sign"])
        df = run_sweep(BETAS, spec["rhos"], [base.D], SEEDS, base)
        df.insert(0, "regime", name)
        frames.append(df)
        print(f"  {name:10s} {len(df)} runs  {time.time()-t0:5.1f}s")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(os.path.join(REPORTS, "exp1_smith_calib_sweep.csv"), index=False)

    num = [c for c in raw.select_dtypes(include=[np.number]).columns
           if c not in ("beta", "rho", "seed")]
    summary = raw.groupby(["regime", "beta", "rho"])[num].mean().reset_index()
    summary.to_csv(os.path.join(REPORTS, "exp1_smith_calib_summary.csv"), index=False)

    text = verdict(summary)
    with open(os.path.join(REPORTS, "exp1_smith_calib_verdict.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
