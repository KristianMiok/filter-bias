"""
exp2_scale_and_grain.py
=======================
EXPERIMENT 2 — positional-error magnitude D and analytical grain (R2 #5, R1).

Not a table of extra axes: a TEST of the prediction implied by the attenuation
formula of Exp 4. For a standardised stationary field with autocorrelation
varrho(h), displacing a record by D gives

    E_obs | E_true = x   ~   N( varrho(D) * x ,  1 - varrho(D)^2 ),

so for the displaced (low-quality) class, with m0 = E[E_true | low],
v0 = Var(E_true | low), v1 = Var(E_true | high), pi1 = P(high):

    mu_delta          =  (varrho(D) - 1) * m0
    pooled_var_obs    =  pi1*v1 + (1-pi1) * ( varrho(D)^2 * v0 + (1 - varrho(D)^2) )
    reliability       =  pooled_var_true / pooled_var_obs

i.e. D and grain enter the problem ONLY through varrho(D), the field
autocorrelation at the displacement lag. Coarsening the analysis grain raises
varrho(D) and therefore mechanically reduces the damage: that is why coarsening
is an alternative to filtering (R2 #5), and the formula says by how much.

Part A  isotropic error, D x corr_len grid: predicted vs realised mu and reliability.
Part B  grain sweep at fixed D and corr_len: does grain move points along the SAME
        varrho(D) curve, and what does it cost in absolute recovery?
Part C  strategy performance across D, all three alignment regimes.

Outputs
    reports/exp2_sweep.csv, reports/exp2_summary.csv, reports/exp2_verdict.txt

Run from the repo root:   python scripts/exp2_scale_and_grain.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfbias_sim import SimConfig, run_one  # noqa: E402

SEEDS = list(range(6))
BETA = 3.0
BASE = dict(grid=160, n_occ=3000, n_bg=8000, fixed_magnitude=True, propensity_on="obs", beta=BETA)

CORR_LENS = [6.0, 12.0, 24.0]
DS = [3.0, 6.0, 10.0, 16.0, 25.0]
GRAINS = [1, 2, 4, 8]
REGIMES = {"iso": (0.0, +1.0), "toward_HQ": (1.0, +1.0), "away_HQ": (1.0, -1.0)}
REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def collect() -> pd.DataFrame:
    rows = []
    # Part A + C: D x corr_len x regime, grain = 1
    t0 = time.time()
    for cl in CORR_LENS:
        for D in DS:
            for reg, (rho, sign) in REGIMES.items():
                for sd in SEEDS:
                    cfg = SimConfig(**BASE, corr_len=cl, D=D, rho=rho,
                                    error_follow_gradient=True, error_gradient_sign=sign, grain=1)
                    o = run_one(cfg, sd); o["part"] = "A"; o["regime"] = reg
                    rows.append(o)
    print(f"  part A/C  {len(rows)} runs  {time.time()-t0:5.1f}s")
    # Part B: grain sweep, isotropic, fixed corr_len
    t0 = time.time(); n0 = len(rows)
    for g in GRAINS:
        for D in DS:
            for sd in SEEDS:
                cfg = SimConfig(**BASE, corr_len=12.0, D=D, rho=0.0,
                                error_follow_gradient=True, error_gradient_sign=+1.0, grain=g)
                o = run_one(cfg, sd); o["part"] = "B"; o["regime"] = "iso"
                rows.append(o)
    print(f"  part B    {len(rows)-n0} runs  {time.time()-t0:5.1f}s")
    return pd.DataFrame(rows)


def verdict(raw: pd.DataFrame, sm: pd.DataFrame) -> str:
    L = []; P = L.append
    P("=" * 104)
    P("EXPERIMENT 2 — POSITIONAL-ERROR MAGNITUDE AND ANALYTICAL GRAIN")
    P("=" * 104)

    # ---- A ----
    P("\n(A) CLOSED-FORM TEST, isotropic error (the regime the derivation covers)")
    a = raw[(raw.part == "A") & (raw.regime == "iso")].copy()
    a["rel_real"] = a.diag_pooled_var_E1_true / a.diag_pooled_var_E1_obs
    for lab, pc, rc in [("mu_delta   ", "pred_mu_delta", "diag_lowq_dE1_shift"),
                        ("reliability", "pred_reliability", "rel_real")]:
        ok = a[np.isfinite(a[pc]) & np.isfinite(a[rc])]
        r = np.corrcoef(ok[pc], ok[rc])[0, 1]
        sl, ic = np.polyfit(ok[pc], ok[rc], 1)
        P(f"    {lab}  n={len(ok)}  Pearson r = {r:.3f}   real = {sl:.3f}*pred {ic:+.3f}   MAE = {np.mean(np.abs(ok[pc]-ok[rc])):.4f}")
    P("\n    corr_len     D   rho(D)   mu_pred   mu_real   rel_pred  rel_real   IPW_est  FILTER    ALL")
    s = sm[(sm.part == "A") & (sm.regime == "iso")]
    for cl in CORR_LENS:
        for D in DS:
            r = s[(s.corr_len == cl) & (s.D == D)].iloc[0]
            P(f"    {cl:7.1f} {D:5.1f}   {r.diag_rho_D:+.3f}   {r.pred_mu_delta:+.4f}  {r.diag_lowq_dE1_shift:+.4f}    "
              f"{r.pred_reliability:.3f}     {r.rel_real:.3f}     {r.rho_sp_IPW_est:.3f}   {r.rho_sp_FILTER:.3f}  {r.rho_sp_ALL:.3f}")
        P("")

    # ---- B ----
    P("(B) GRAIN: coarsening raises rho(D) and reduces the damage (corr_len = 12, isotropic)")
    P("    grain     D   rho(D)   rel_real   coef_obs   IPW_est   FILTER    ALL     ESS")
    s = sm[sm.part == "B"]
    for g in GRAINS:
        for D in DS:
            r = s[(s.grain == g) & (s.D == D)].iloc[0]
            P(f"    {g:5d} {D:5.1f}   {r.diag_rho_D:+.3f}    {r.rel_real:.3f}     {r.diag_prop_coef_E1:+6.2f}    "
              f"{r.rho_sp_IPW_est:.3f}   {r.rho_sp_FILTER:.3f}  {r.rho_sp_ALL:.3f}   {r.diag_ess_frac:.3f}")
        P("")
    P("    Same-rho(D) check: pairs from part A (grain=1, varying corr_len) and part B (varying grain)")
    P("    that share a rho(D) should show the same reliability.")
    both = pd.concat([sm[(sm.part == "A") & (sm.regime == "iso")], sm[sm.part == "B"]])
    both = both.sort_values("diag_rho_D")
    ok = both[np.isfinite(both.rel_real)]
    r = np.corrcoef(ok.diag_rho_D, ok.rel_real)[0, 1]
    P(f"    corr(rho(D), reliability) over ALL {len(ok)} cells (both parts pooled) = {r:.3f}")

    # ---- C ----
    P("\n(C) STRATEGY PERFORMANCE ACROSS D, by alignment regime (corr_len = 12, grain = 1)")
    P("    regime          D   rho(D)     ALL   FILTER  IPW_est  IPW_calib  SMITH_env  IPW_orc   winner")
    s = sm[(sm.part == "A") & (sm.corr_len == 12.0)]
    for reg in REGIMES:
        for D in DS:
            r = s[(s.regime == reg) & (s.D == D)].iloc[0]
            v = {"ALL": r.rho_sp_ALL, "FILTER": r.rho_sp_FILTER, "IPW": r.rho_sp_IPW_est,
                 "IPWc": r.rho_sp_IPW_calib, "SMITH": r.rho_sp_SMITH_env}
            top = sorted(v.items(), key=lambda kv: -kv[1])
            win = top[0][0] if top[0][1] - top[1][1] >= 0.005 else "TIE"
            P(f"    {reg:12s} {D:5.1f}   {r.diag_rho_D:+.3f}   {r.rho_sp_ALL:.3f}   {r.rho_sp_FILTER:.3f}   "
              f"{r.rho_sp_IPW_est:.3f}     {r.rho_sp_IPW_calib:.3f}      {r.rho_sp_SMITH_env:.3f}    {r.rho_sp_IPW_oracle:.3f}   {win}")
        P("")
    return "\n".join(L)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    raw = collect()
    raw["rel_real"] = raw.diag_pooled_var_E1_true / raw.diag_pooled_var_E1_obs
    raw["corr_len"] = raw.diag_corr_len
    raw["grain"] = raw.diag_grain.astype(int)
    raw.to_csv(os.path.join(REPORTS, "exp2_sweep.csv"), index=False)
    num = [c for c in raw.select_dtypes(include=[np.number]).columns
           if c not in ("D", "corr_len", "grain", "seed")]
    sm = raw.groupby(["part", "regime", "corr_len", "D", "grain"])[num].mean().reset_index()
    sm.to_csv(os.path.join(REPORTS, "exp2_summary.csv"), index=False)
    text = verdict(raw, sm)
    with open(os.path.join(REPORTS, "exp2_verdict.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
