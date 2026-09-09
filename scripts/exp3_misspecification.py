"""
exp3_misspecification.py
========================
EXPERIMENT 3 — does the attenuation result survive misspecification? (R1 #4)

The derivation of

    coef_obs / coef_true  ~=  (1 - mu_delta/dm) * (var_true/var_obs)

assumes an LDA-like structure: Gaussian classes on the coupling axis, common
covariance, logit-linear quality model, correctly specified quadratic SDM learner.
R1 asks whether the favourable behaviour depends on that controlled setup. This
script varies each assumption one at a time and asks two questions:

    Q-A  does the FORMULA still predict the realised attenuation?
    Q-B  does the REGIME ORDERING of strategies still hold
         (iso -> ALL, toward_HQ -> SMITH/FILTER, away_HQ -> SMITH/IPW)?

Attenuation is measured learner-agnostically as `eff_coef` = OLS slope of
logit(p_hat) on E1, so a random-forest propensity (which has no coefficient) is
directly comparable to a logistic one.

Variations (each against the "base" cell):
    niche_shape         gaussian | bimodal | skewed      <- learner misspecified
    quality_link        linear | quadratic | threshold   <- nonlinear quality-environment
    env_corr            0.0 | 0.6                        <- correlated predictors
    propensity_learner  logit_quad | logit_linear | rf   <- propensity misspecified
    retain_rate         0.6 | 0.3 | 0.85                 <- filtering fraction
    n_occ               3000 | 600                       <- sample size

Outputs
    reports/exp3_sweep.csv, reports/exp3_summary.csv, reports/exp3_verdict.txt

Run from the repo root:   python scripts/exp3_misspecification.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfbias_sim import SimConfig, run_one  # noqa: E402

SEEDS = list(range(6))
BASE = dict(grid=160, n_occ=3000, n_bg=8000, corr_len=12.0, D=10.0, beta=3.0,
            fixed_magnitude=True, propensity_on="obs", error_follow_gradient=True)
REGIMES = {"iso": (0.0, +1.0), "toward_HQ": (1.0, +1.0), "away_HQ": (1.0, -1.0)}

VARIANTS = {
    "base":              {},
    "niche_bimodal":     dict(niche_shape="bimodal"),
    "niche_skewed":      dict(niche_shape="skewed"),
    "quality_quadratic": dict(quality_link="quadratic"),
    "quality_threshold": dict(quality_link="threshold"),
    "env_corr_0.6":      dict(env_corr=0.6),
    "prop_linear":       dict(propensity_learner="logit_linear"),
    "prop_rf":           dict(propensity_learner="rf"),
    "retain_0.30":       dict(retain_rate=0.30),
    "retain_0.85":       dict(retain_rate=0.85),
    "n_occ_600":         dict(n_occ=600),
}
STRATS = {"ALL": "rho_sp_ALL", "FILTER": "rho_sp_FILTER", "IPW": "rho_sp_IPW_est",
          "IPWc": "rho_sp_IPW_calib", "SMITH": "rho_sp_SMITH_env"}
REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def collect() -> pd.DataFrame:
    rows = []
    for vname, over in VARIANTS.items():
        t0 = time.time()
        base = {**BASE, **over}
        for reg, (rho, sign) in REGIMES.items():
            for sd in SEEDS:
                cfg = SimConfig(**base, rho=rho, error_gradient_sign=sign)
                o = run_one(cfg, sd); o["variant"] = vname; o["regime"] = reg
                rows.append(o)
        print(f"  {vname:19s} {len(REGIMES)*len(SEEDS)} runs  {time.time()-t0:5.1f}s")
    return pd.DataFrame(rows)


def verdict(raw: pd.DataFrame, sm: pd.DataFrame) -> str:
    L = []; P = L.append
    P("=" * 112)
    P("EXPERIMENT 3 — MISSPECIFICATION   (honest build; attenuation via learner-agnostic eff_coef)")
    P("=" * 112)

    # ---- Q-A ----
    P("\n(Q-A) DOES THE FORMULA STILL PREDICT ATTENUATION?")
    P("      ratio_real = eff_coef_obs / eff_coef_true ;  ratio_pred = (1 - mu/dm) * (var_true/var_obs)")
    P("\n      variant              n    Pearson r   slope    MAE      | per-regime real vs pred (iso / toward / away)")
    for vname in VARIANTS:
        d = raw[raw.variant == vname].copy()
        d["ratio_real"] = d.diag_eff_coef_obs / d.diag_eff_coef_true
        d["ratio_pred"] = (d.diag_obs_sep_E1 / d.diag_true_sep_E1) * \
                          (d.diag_pooled_var_E1_true / d.diag_pooled_var_E1_obs)
        ok = d[np.isfinite(d.ratio_real) & np.isfinite(d.ratio_pred)]
        if len(ok) < 4:
            P(f"      {vname:19s}  insufficient"); continue
        r = np.corrcoef(ok.ratio_pred, ok.ratio_real)[0, 1]
        sl = np.polyfit(ok.ratio_pred, ok.ratio_real, 1)[0]
        mae = np.mean(np.abs(ok.ratio_pred - ok.ratio_real))
        cells = []
        for reg in REGIMES:
            g = ok[ok.regime == reg]
            cells.append(f"{g.ratio_real.mean():+.2f}/{g.ratio_pred.mean():+.2f}")
        P(f"      {vname:19s} {len(ok):3d}    {r:+.3f}     {sl:5.2f}   {mae:.3f}    | " + "  ".join(cells))

    P("\n      pooled over ALL variants:")
    d = raw.copy()
    d["ratio_real"] = d.diag_eff_coef_obs / d.diag_eff_coef_true
    d["ratio_pred"] = (d.diag_obs_sep_E1 / d.diag_true_sep_E1) * \
                      (d.diag_pooled_var_E1_true / d.diag_pooled_var_E1_obs)
    ok = d[np.isfinite(d.ratio_real) & np.isfinite(d.ratio_pred)]
    r = np.corrcoef(ok.ratio_pred, ok.ratio_real)[0, 1]
    sl, ic = np.polyfit(ok.ratio_pred, ok.ratio_real, 1)
    P(f"      n = {len(ok)}   Pearson r = {r:.3f}   real = {sl:.3f}*pred {ic:+.3f}   "
      f"MAE = {np.mean(np.abs(ok.ratio_pred-ok.ratio_real)):.3f}")

    # ---- Q-B ----
    P("\n(Q-B) DOES THE REGIME ORDERING HOLD?   (winner per variant x regime, TIE if margin < 0.005)")
    P("      variant              " + "".join(f"{r:>26s}" for r in REGIMES))
    for vname in VARIANTS:
        cells = []
        for reg in REGIMES:
            s = sm[(sm.variant == vname) & (sm.regime == reg)]
            if len(s) == 0:
                cells.append(f"{'--':>26s}"); continue
            s = s.iloc[0]
            v = {k: s[c] for k, c in STRATS.items()}
            top = sorted(v.items(), key=lambda kv: -kv[1])
            win = top[0][0] if top[0][1] - top[1][1] >= 0.005 else "TIE"
            cells.append(f"{win + ' ' + format(top[0][1], '.3f') + ' (2nd ' + top[1][0] + ')':>26s}")
        P(f"      {vname:19s} " + "".join(cells))

    P("\n      full recovery table (mean over seeds):")
    P("      variant              regime        " + "".join(f"{k:>9s}" for k in STRATS) + "   IPW_orc")
    for vname in VARIANTS:
        for reg in REGIMES:
            s = sm[(sm.variant == vname) & (sm.regime == reg)]
            if len(s) == 0:
                continue
            s = s.iloc[0]
            P(f"      {vname:19s} {reg:12s}  " + "".join(f"{s[c]:9.3f}" for c in STRATS.values()) +
              f"   {s.rho_sp_IPW_oracle:.3f}")
    P("")
    return "\n".join(L)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    raw = collect()
    raw.to_csv(os.path.join(REPORTS, "exp3_sweep.csv"), index=False)
    num = [c for c in raw.select_dtypes(include=[np.number]).columns if c not in ("seed",)]
    sm = raw.groupby(["variant", "regime"])[num].mean().reset_index()
    sm.to_csv(os.path.join(REPORTS, "exp3_summary.csv"), index=False)
    text = verdict(raw, sm)
    with open(os.path.join(REPORTS, "exp3_verdict.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
