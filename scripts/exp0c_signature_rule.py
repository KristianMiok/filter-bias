"""
exp0c_signature_rule.py
=======================
EXPERIMENT 0c — can OBSERVABLE diagnostics identify the regime, and can they be
turned into an explicit decision rule? (R1 #12, R2 #6)

Setup: 480 runs with randomised generative parameters
    beta ~ U(0, 4), regime in {iso, toward_HQ, away_HQ} (balanced),
    D in {6, 10, 16}, corr_len in {8, 12, 18}, seeds fresh per run.

Observable features (everything a practitioner can compute):
    F_basic = prop_auc, prop_brier, ess_frac, w_max_over_mean, w_p99, retain_obs
    F_calib = F_basic + coef_gap (|coef_calib - coef_obs|), dess (ess_calib - ess),
              rho_D (field autocorrelation at the uncertainty radius)

Targets:
    T1  regime (3-class)
    T2  "IPW_est beats FILTER by > 0.005"   (the actionable question)
    T3  "propensity collapsed": |eff_coef_obs / eff_coef_true| < 0.5   (the mechanism)

Classifier: depth-3 decision tree (interpretable; its printed rules ARE the
deliverable for R1 #12), 5-fold stratified CV, plus a depth-1 stump to extract
single-threshold rules. Honesty check: confusion matrices, and performance at
beta < 1 where signals are weak.

Outputs
    reports/exp0c_sweep.csv, reports/exp0c_verdict.txt

Run from the repo root:   python scripts/exp0c_signature_rule.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfbias_sim import SimConfig, run_one  # noqa: E402

N_RUNS = 480
REGIMES = {"iso": (0.0, +1.0), "toward_HQ": (1.0, +1.0), "away_HQ": (1.0, -1.0)}
BASE = dict(grid=160, n_occ=3000, n_bg=8000, fixed_magnitude=True, propensity_on="obs",
            error_follow_gradient=True)
REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")

F_BASIC = ["diag_propensity_auc", "diag_prop_brier", "diag_ess_frac",
           "diag_w_max_over_mean", "diag_w_p99", "retain_obs"]
F_CALIB = F_BASIC + ["coef_gap", "dess", "diag_rho_D"]


def collect() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    regs = list(REGIMES.items())
    rows = []
    t0 = time.time()
    for i in range(N_RUNS):
        reg, (rho, sign) = regs[i % 3]
        beta = float(rng.uniform(0.0, 4.0))
        D = float(rng.choice([6.0, 10.0, 16.0]))
        cl = float(rng.choice([8.0, 12.0, 18.0]))
        cfg = SimConfig(**BASE, beta=beta, rho=rho, error_gradient_sign=sign,
                        D=D, corr_len=cl)
        o = run_one(cfg, int(rng.integers(0, 10_000)))
        o["regime"] = reg
        rows.append(o)
        if (i + 1) % 120 == 0:
            print(f"  {i+1}/{N_RUNS}  {time.time()-t0:6.1f}s")
    d = pd.DataFrame(rows)
    d["coef_gap"] = (d.diag_prop_coef_E1_calib - d.diag_prop_coef_E1).abs()
    d["dess"] = d.diag_ess_frac_calib - d.diag_ess_frac
    d["t2_ipw_beats_filter"] = (d.rho_sp_IPW_est - d.rho_sp_FILTER > 0.005).astype(int)
    ratio = (d.diag_eff_coef_obs / d.diag_eff_coef_true).replace([np.inf, -np.inf], np.nan)
    d["t3_collapsed"] = (ratio.abs() < 0.5).astype(int)
    return d


def eval_target(d, feats, y, name, depth=3):
    X = d[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    yv = d[y].values if isinstance(y, str) else y
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=25, random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(clf, X, yv, cv=cv)
    acc = float((pred == yv).mean())
    lines = [f"    [{name}]  depth={depth}  5-fold CV accuracy = {acc:.3f}   (chance = {max(np.bincount(pd.factorize(yv)[0]))/len(yv):.3f})"]
    cm = pd.crosstab(pd.Series(yv, name="true"), pd.Series(pred, name="pred"))
    for ln in cm.to_string().split("\n"):
        lines.append("      " + ln)
    clf.fit(X, yv)
    lines.append("      fitted tree (full data):")
    for ln in export_text(clf, feature_names=feats, max_depth=depth).split("\n"):
        if ln.strip():
            lines.append("        " + ln)
    return acc, "\n".join(lines)


def verdict(d: pd.DataFrame) -> str:
    L = []; P = L.append
    P("=" * 100)
    P("EXPERIMENT 0c — OBSERVABLE SIGNATURE AS AN EXPLICIT DECISION RULE")
    P("=" * 100)
    P(f"\n{len(d)} runs; beta ~ U(0,4); D in {{6,10,16}}; corr_len in {{8,12,18}}; regimes balanced.")

    P("\n(T1) REGIME, 3-class, observable features only")
    for feats, lab in [(F_BASIC, "F_basic"), (F_CALIB, "F_calib = F_basic + coef_gap + dess + rho_D")]:
        acc, txt = eval_target(d, feats, "regime", lab)
        P(txt); P("")

    P("(T2) DOES IPW BEAT FILTER?  (actionable)")
    for feats, lab in [(F_BASIC, "F_basic"), (F_CALIB, "F_calib")]:
        acc, txt = eval_target(d, feats, "t2_ipw_beats_filter", lab)
        P(txt); P("")

    P("(T3) PROPENSITY COLLAPSE  (|eff ratio| < 0.5)")
    for feats, lab in [(F_BASIC, "F_basic"), (F_CALIB, "F_calib")]:
        acc, txt = eval_target(d, feats, "t3_collapsed", lab)
        P(txt); P("")

    P("HONESTY CHECKS")
    lo = d[d.beta < 1.0]
    hi = d[d.beta >= 1.0]
    for lab, sub in [("beta < 1 (weak coupling)", lo), ("beta >= 1", hi)]:
        X = sub[F_CALIB].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        try:
            pred = cross_val_predict(clf, X, sub.regime.values, cv=cv)
            P(f"    regime accuracy, {lab:26s}: {float((pred==sub.regime.values).mean()):.3f}   (n={len(sub)})")
        except Exception:
            P(f"    regime accuracy, {lab:26s}: n too small")
    P("\n    mean observables by regime (for the paper's signature table):")
    g = d.groupby("regime")[["diag_propensity_auc", "diag_ess_frac", "coef_gap", "dess", "diag_rho_D"]].mean()
    for ln in g.round(3).to_string().split("\n"):
        P("      " + ln)
    P("")
    return "\n".join(L)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    d = collect()
    d.to_csv(os.path.join(REPORTS, "exp0c_sweep.csv"), index=False)
    text = verdict(d)
    with open(os.path.join(REPORTS, "exp0c_verdict.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
