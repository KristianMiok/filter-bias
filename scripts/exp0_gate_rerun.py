"""
exp0_gate_rerun.py
==================
EXPERIMENT 0 — the gate for the MEE resubmission (Reviewer 1, point 13).

Re-runs the paper's beta x rho decision-map sweep under four builds:

    A_orig     fixed_magnitude=False  propensity_on="true"   (paper as submitted)
    B_magfix   fixed_magnitude=True   propensity_on="true"   (fix 1 only)
    C_propfix  fixed_magnitude=False  propensity_on="obs"    (fix 2 only)
    D_both     fixed_magnitude=True   propensity_on="obs"    (honest build)

and answers, numerically, the three questions in Lucian's email of 31 Aug:

    Q1  Does IPW still dominate FILTER wherever coupling is present?
    Q2  Does the positivity ceiling still appear?
    Q3  Does the FILTER-to-IPW spread stay narrow?

plus the one the reviewers did not ask but the data will:

    Q4  Does IPW_est now depend on rho at all? (in the original build it could
        not, because the propensity never saw a displaced record)

Outputs
    reports/exp0_gate_sweep.csv     all raw rows, with a `build` column
    reports/exp0_gate_summary.csv   mean over seeds per build x beta x rho
    reports/exp0_gate_winners.csv   best strategy per build x beta x rho, with
                                    ties declared when the margin < TIE_MARGIN
    reports/exp0_gate_verdict.txt   the printed verdict

Run from the repo root:   python scripts/exp0_gate_rerun.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfbias_sim import SimConfig, run_sweep  # noqa: E402

# ---- sweep grid: identical to fig_decision_heatmap.py / the paper ----
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = list(range(8))
BASE = dict(grid=160, n_occ=3000, n_bg=8000, corr_len=12.0, D=10.0)

BUILDS = {
    "A_orig":    dict(fixed_magnitude=False, propensity_on="true"),
    "B_magfix":  dict(fixed_magnitude=True,  propensity_on="true"),
    "C_propfix": dict(fixed_magnitude=False, propensity_on="obs"),
    "D_both":    dict(fixed_magnitude=True,  propensity_on="obs"),
}
STRATS = ["ALL", "FILTER", "IPW_est"]       # practitioner strategies (oracle excluded)
METRIC = "rho_sp_"
TIE_MARGIN = 0.005                          # Spearman units; below this, declare a tie

REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def winners(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (build, beta, rho), g in summary.groupby(["build", "beta", "rho"]):
        vals = {s: float(g[METRIC + s].iloc[0]) for s in STRATS}
        order = sorted(vals.items(), key=lambda kv: -kv[1])
        best, second = order[0], order[1]
        margin = best[1] - second[1]
        rows.append({
            "build": build, "beta": beta, "rho": rho,
            "winner": best[0] if margin >= TIE_MARGIN else "TIE",
            "best": best[0], "margin": margin,
            **{f"rec_{s}": vals[s] for s in STRATS},
            "rec_IPW_oracle": float(g[METRIC + "IPW_oracle"].iloc[0]),
        })
    return pd.DataFrame(rows)


def verdict(summary: pd.DataFrame, win: pd.DataFrame) -> str:
    L = []
    P = L.append
    D = summary[summary.build == "D_both"].set_index(["beta", "rho"])
    A = summary[summary.build == "A_orig"].set_index(["beta", "rho"])

    P("=" * 78)
    P("EXPERIMENT 0 — GATE VERDICT   (build D_both = honest build)")
    P("=" * 78)

    # ---------- Q1 ----------
    P("\nQ1  Does IPW_est still beat FILTER wherever coupling is present (beta > 0)?")
    sub = D[D.index.get_level_values("beta") > 0]
    gap = (sub["rho_sp_IPW_est"] - sub["rho_sp_FILTER"])
    n_win = int((gap > TIE_MARGIN).sum()); n_tie = int((gap.abs() <= TIE_MARGIN).sum()); n_lose = int((gap < -TIE_MARGIN).sum())
    P(f"    cells with beta>0: {len(gap)}   IPW wins: {n_win}   ties: {n_tie}   FILTER wins: {n_lose}")
    P(f"    IPW_est - FILTER gap:  min {gap.min():+.4f}   median {gap.median():+.4f}   max {gap.max():+.4f}")
    gA = (A.loc[sub.index, "rho_sp_IPW_est"] - A.loc[sub.index, "rho_sp_FILTER"])
    P(f"    (same gap in ORIGINAL build:  min {gA.min():+.4f}   median {gA.median():+.4f}   max {gA.max():+.4f})")
    P("    by rho (mean gap over beta>0):")
    for rho in RHOS:
        g_r = gap[gap.index.get_level_values("rho") == rho]
        gA_r = gA[gA.index.get_level_values("rho") == rho]
        P(f"      rho={rho:.2f}   fixed {g_r.mean():+.4f}   original {gA_r.mean():+.4f}")
    q1 = "YES" if n_lose == 0 and n_win >= 0.8 * len(gap) else ("PARTLY" if n_win > n_lose else "NO")
    P(f"    --> Q1: {q1}")

    # ---------- Q2 ----------
    P("\nQ2  Does the positivity ceiling still appear?  (IPW recovery falls with beta at rho=0; ESS falls)")
    r0 = D[D.index.get_level_values("rho") == 0.0]
    P("    beta   IPW_oracle   IPW_est   ESS_frac(est)   prop_AUC   prop_coef_E1")
    for beta in BETAS:
        row = r0.loc[(beta, 0.0)]
        P(f"    {beta:4.1f}   {row['rho_sp_IPW_oracle']:.4f}     {row['rho_sp_IPW_est']:.4f}    {row['diag_ess_frac']:.3f}          {row['diag_propensity_auc']:.3f}      {row['diag_prop_coef_E1']:+.2f}")
    ceiling = r0["rho_sp_IPW_oracle"].iloc[-1] < r0["rho_sp_IPW_oracle"].iloc[0] - 0.01
    P(f"    --> Q2: {'YES' if ceiling else 'NO'} (oracle IPW at beta=4 vs beta=0: "
      f"{r0['rho_sp_IPW_oracle'].iloc[-1]:.4f} vs {r0['rho_sp_IPW_oracle'].iloc[0]:.4f})")

    # ---------- Q3 ----------
    P("\nQ3  Does the FILTER-to-IPW spread stay narrow?  (Schoener D between the two surfaces)")
    P("    Also: how far is practitioner IPW from ORACLE IPW (the cost of not knowing e_true)?")
    P("    beta   D(IPWest,FILTER)  D(IPWest,IPWoracle)   [original: D(IPWest,IPWoracle)]")
    for beta in BETAS:
        d1 = D.loc[beta, "D_IPWest_vs_FILTER"].mean()
        d2 = D.loc[beta, "D_IPWest_vs_IPWoracle"].mean()
        d2A = A.loc[beta, "D_IPWest_vs_IPWoracle"].mean()
        P(f"    {beta:4.1f}      {d1:.3f}             {d2:.3f}                   {d2A:.3f}")
    P("    --> Q3: see table; narrow spread now partly reflects IPW_est collapsing TOWARD FILTER,")
    P("        not the two corrections agreeing on the truth. Check D(IPWest,IPWoracle) at high rho.")

    # ---------- Q4 ----------
    P("\nQ4  Does IPW_est now depend on rho?  (range of IPW_est over rho, at each beta)")
    P("    beta   original(range)   fixed(range)   fixed: IPW_est at rho=0 -> rho=1")
    for beta in BETAS:
        a = A.loc[beta, "rho_sp_IPW_est"]; d = D.loc[beta, "rho_sp_IPW_est"]
        P(f"    {beta:4.1f}     {a.max()-a.min():.4f}          {d.max()-d.min():.4f}         {d.iloc[0]:.4f} -> {d.iloc[-1]:.4f}")
    P("    --> Q4: in the original build the range is ~0 by construction (bug 2).")

    # ---------- mechanism ----------
    P("\nMECHANISM  propensity coefficient on E1 (true generative beta on the standardised axis):")
    P("    beta   rho=0.00   rho=0.25   rho=0.50   rho=0.75   rho=1.00      [original, any rho]")
    for beta in BETAS:
        cs = [D.loc[(beta, r), "diag_prop_coef_E1"] for r in RHOS]
        cA = A.loc[(beta, 0.0), "diag_prop_coef_E1"]
        P(f"    {beta:4.1f}   " + "   ".join(f"{c:+7.2f}" for c in cs) + f"        {cA:+7.2f}")
    P("    Attenuation toward 0 (or sign flip) = displaced records pulling the estimated")
    P("    retention gradient flat. This is the non-identifiability, made visible.")

    # ---------- winners ----------
    P("\nBEST STRATEGY MAP  (TIE = margin < %.3f Spearman)" % TIE_MARGIN)
    for build in BUILDS:
        P(f"\n  build {build}:")
        w = win[win.build == build].pivot(index="beta", columns="rho", values="winner")
        w = w.reindex(index=sorted(BETAS, reverse=True))
        P("    beta \\ rho " + "".join(f"{r:>9.2f}" for r in RHOS))
        for beta in w.index:
            P(f"    {beta:6.1f}     " + "".join(f"{str(w.loc[beta, r]):>9s}" for r in RHOS))
    P("")
    return "\n".join(L)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    n_runs = len(BETAS) * len(RHOS) * len(SEEDS)
    frames = []
    for name, flags in BUILDS.items():
        t0 = time.time()
        base = SimConfig(**BASE, **flags)
        df = run_sweep(BETAS, RHOS, [base.D], SEEDS, base)
        df.insert(0, "build", name)
        frames.append(df)
        print(f"  {name:10s} {n_runs} runs  {time.time()-t0:6.1f}s")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(os.path.join(REPORTS, "exp0_gate_sweep.csv"), index=False)

    num = [c for c in raw.select_dtypes(include=[np.number]).columns
           if c not in ("beta", "rho", "seed")]
    summary = raw.groupby(["build", "beta", "rho"])[num].mean().reset_index()
    summary.to_csv(os.path.join(REPORTS, "exp0_gate_summary.csv"), index=False)

    win = winners(summary)
    win.to_csv(os.path.join(REPORTS, "exp0_gate_winners.csv"), index=False)

    text = verdict(summary, win)
    with open(os.path.join(REPORTS, "exp0_gate_verdict.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
