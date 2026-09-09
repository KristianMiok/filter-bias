"""
exp4_attenuation_and_envelope.py
================================
EXPERIMENT 4 — two pieces of theory, both testable against the simulation.

(a) ATTENUATION FORMULA. A logistic propensity separates classes by their means.
    If the true separation of high- vs low-quality records on the coupling axis is
    dm = m1 - m0, displacement moves low-quality records by mu on that axis, and
    the pooled within-class variance is inflated from s2_true to s2_obs, then

        coef_obs / coef_true  ~=  (1 - mu / dm) * (s2_true / s2_obs).

    First factor: alignment (collapse when mu = dm; sign flip when mu > dm;
    exaggeration when mu < 0). Second factor: reliability ratio (regression
    dilution). Classical errors-in-variables, adapted to error present in one
    class only. Tested over every beta x regime x seed.

(b) KAPPA-ENVELOPE. Because the corruption is multiplicative on the coefficient,
    the natural sensitivity parameter is the factor kappa on the centred logit of
    the estimated propensity (recalibrated to the observed retention rate). For
    kappa in [1/Lambda, Lambda] we refit IPW and report the envelope of surfaces.
    A one-parameter marginal sensitivity model (Tan 2006) whose parameter is
    mechanistically interpretable: kappa < 1 = attenuated (toward_HQ regime),
    kappa > 1 = exaggerated (away_HQ regime).

Outputs
    reports/exp4_sweep.csv, reports/exp4_summary.csv, reports/exp4_verdict.txt

Run from the repo root:   python scripts/exp4_attenuation_and_envelope.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfbias_sim import SimConfig, run_sweep  # noqa: E402

BETAS = [0.5, 1.0, 2.0, 3.0, 4.0]        # beta=0 excluded from (a): dm = 0 -> ratio undefined
SEEDS = list(range(8))
BASE = dict(grid=160, n_occ=3000, n_bg=8000, corr_len=12.0, D=10.0,
            fixed_magnitude=True, propensity_on="obs")
REGIMES = {
    "iso":        dict(rhos=[0.0], error_gradient_sign=+1.0),
    "toward_HQ":  dict(rhos=[1.0], error_gradient_sign=+1.0),
    "away_HQ":    dict(rhos=[1.0], error_gradient_sign=-1.0),
    "toward_half": dict(rhos=[0.5], error_gradient_sign=+1.0),
}
KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0)
REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def verdict(raw: pd.DataFrame, summary: pd.DataFrame) -> str:
    L = []; P = L.append
    P("=" * 100)
    P("EXPERIMENT 4 — (a) ATTENUATION FORMULA  and  (b) KAPPA-ENVELOPE   (honest build)")
    P("=" * 100)

    # ---------- (a) ----------
    P("\n(a) ATTENUATION FORMULA:  coef_obs/coef_true  vs  predicted")
    d = raw.copy()
    d["ratio_real"] = d.diag_prop_coef_E1 / d.diag_prop_coef_E1_true
    d["ratio_pred"] = d.diag_obs_sep_E1 / d.diag_true_sep_E1
    d["ratio_pred_full"] = d.ratio_pred * (d.diag_pooled_var_E1_true / d.diag_pooled_var_E1_obs)
    ok = d[np.isfinite(d.ratio_real) & np.isfinite(d.ratio_pred) & np.isfinite(d.ratio_pred_full)]
    for lab, col in [("mean-shift only ", "ratio_pred"), ("+ reliability   ", "ratio_pred_full")]:
        r = np.corrcoef(ok[col], ok.ratio_real)[0, 1]
        slope, intercept = np.polyfit(ok[col], ok.ratio_real, 1)
        mae = np.mean(np.abs(ok[col] - ok.ratio_real))
        P(f"    [{lab}]  n = {len(ok)}   Pearson r = {r:.3f}   fit: real = {slope:.3f} * pred + {intercept:+.3f}   MAE = {mae:.3f}")
    P("    per regime x beta (means):")
    P("    regime        beta   true_sep   mu(dE1)   reliab.   pred_mean   pred_full   real_ratio   coef_true  coef_obs")
    for regime in REGIMES:
        for beta in BETAS:
            s = summary[(summary.regime == regime) & (summary.beta == beta)].iloc[0]
            pred = s.diag_obs_sep_E1 / s.diag_true_sep_E1
            rel = s.diag_pooled_var_E1_true / s.diag_pooled_var_E1_obs
            real = s.diag_prop_coef_E1 / s.diag_prop_coef_E1_true
            P(f"    {regime:12s}  {beta:4.1f}    {s.diag_true_sep_E1:6.3f}   {s.diag_lowq_dE1_shift:+6.3f}    {rel:5.3f}    "
              f"{pred:+7.3f}     {pred*rel:+7.3f}     {real:+7.3f}     {s.diag_prop_coef_E1_true:+7.2f}   {s.diag_prop_coef_E1:+7.2f}")
    P("    --> if r is high and slope ~1, a one-line formula predicts when IPW collapses.")

    # ---------- (b) ----------
    P("\n(b) KAPPA-ENVELOPE  (Lambda = 4; kappa in {0.25, 0.5, 1, 2, 4})")
    P("    regime        beta   IPW_est(k=1)  best_in_env  best_kappa  IPW_oracle   env_width(D)   FILTER    ALL")
    for regime in REGIMES:
        for beta in BETAS:
            s = summary[(summary.regime == regime) & (summary.beta == beta)].iloc[0]
            P(f"    {regime:12s}  {beta:4.1f}      {s.rho_sp_IPW_k1:.3f}        {s.env_best_rec:.3f}        "
              f"{s.env_best_kappa:4.1f}       {s.rho_sp_IPW_oracle:.3f}        {s.env_width_D:.3f}       "
              f"{s.rho_sp_FILTER:.3f}    {s.rho_sp_ALL:.3f}")
    P("\n    recovery by kappa (mean over seeds):")
    P("    regime        beta   " + "".join(f"k={k:<7g}" for k in KAPPAS))
    for regime in REGIMES:
        for beta in BETAS:
            s = summary[(summary.regime == regime) & (summary.beta == beta)].iloc[0]
            P(f"    {regime:12s}  {beta:4.1f}   " + "".join(f"{s[f'rho_sp_IPW_k{k:g}']:.3f}    " for k in KAPPAS))
    P("    --> read: does the envelope reach oracle? is best_kappa at the edge (positivity-bound) or interior?")
    P("        env_width is observable on real data; it is the honest replacement for 'the bracket is narrow'.")
    P("")
    return "\n".join(L)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    frames = []
    for name, spec in REGIMES.items():
        t0 = time.time()
        base = SimConfig(**BASE, error_follow_gradient=True,
                         error_gradient_sign=spec["error_gradient_sign"], kappa_grid=KAPPAS)
        df = run_sweep(BETAS, spec["rhos"], [base.D], SEEDS, base)
        df.insert(0, "regime", name)
        frames.append(df)
        print(f"  {name:12s} {len(df)} runs  {time.time()-t0:5.1f}s")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(os.path.join(REPORTS, "exp4_sweep.csv"), index=False)
    num = [c for c in raw.select_dtypes(include=[np.number]).columns if c not in ("beta", "rho", "seed")]
    summary = raw.groupby(["regime", "beta", "rho"])[num].mean().reset_index()
    summary.to_csv(os.path.join(REPORTS, "exp4_summary.csv"), index=False)
    text = verdict(raw, summary)
    with open(os.path.join(REPORTS, "exp4_verdict.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
