"""
sdm_consequence_reconcile.py
============================
Task 1 (reconciliation with the Robustness companion study).

Robustness reports that low-accuracy CONTAMINATION inflates predicted range ~25-30% and
measures niche overlap with Schoener's D / Warren's I. Our draft calls the filtering effect
"modest" on Jaccard / mean-|delta|. Anyone reading both will ask which it is. This script
re-expresses the SAME five-strategy comparison on Robustness's metrics so the correspondence
is explicit: it adds Schoener's D, Warren's I, and predicted-range-area % change (each
strategy vs ALL), alongside the overlap statistics we already report.

Reuses the existing pipeline (no changes to it): run_species() supplies the per-strategy
prediction surfaces; this script only adds metrics on top.

Output: reports/sdm_consequence_reconcile.csv (strategies-vs-ALL in rows; D / I / range% /
Jaccard / mean-|delta| in columns), and prints the ALL->FILTER slice -- the direct analogue
of Robustness's contamination range change.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sdm_consequence_strategies import compute_p_low_oof, run_species, STRATEGIES
from sdm_consequence_multispecies import CSV, SPECIES, get_feature_cols


def _norm(p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    s = p.sum()
    return p / s if s > 0 else p


def schoener_D(a, b):
    pa, pb = _norm(a), _norm(b)
    return float(1.0 - 0.5 * np.abs(pa - pb).sum())


def warren_I(a, b):
    pa, pb = _norm(a), _norm(b)
    return float(1.0 - 0.5 * np.sum((np.sqrt(pa) - np.sqrt(pb)) ** 2))


def range_pct_change(all_vec, strat_vec, top_frac):
    """% change in predicted-range area: fraction of the surface above a threshold fixed on
    ALL (its (1-top_frac) quantile), measured under the strategy vs under ALL."""
    thr = np.quantile(all_vec, 1.0 - top_frac)
    base = np.mean(all_vec >= thr)                 # ~ top_frac by construction
    alt = np.mean(strat_vec >= thr)
    return float(100.0 * (alt - base) / base) if base > 0 else float("nan")


def top_jaccard(a, b, top_frac=0.10):
    ta = a >= np.quantile(a, 1.0 - top_frac)
    tb = b >= np.quantile(b, 1.0 - top_frac)
    union = np.sum(ta | tb)
    return float(np.sum(ta & tb) / union) if union else float("nan")


def metrics(all_vec, s_vec):
    return {
        "schoener_D": schoener_D(all_vec, s_vec),
        "warren_I": warren_I(all_vec, s_vec),
        "range_top30_pct": range_pct_change(all_vec, s_vec, 0.30),
        "range_top10_pct": range_pct_change(all_vec, s_vec, 0.10),
        "top10_jaccard": top_jaccard(all_vec, s_vec, 0.10),
        "mean_abs_diff": float(np.mean(np.abs(np.asarray(all_vec, float) - np.asarray(s_vec, float)))),
    }


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    feature_cols = get_feature_cols(header)
    extra = [c for c in ["Crayfish_scientific_name", "Accuracy", "Status", "strahler", "WoCID"]
             if c in header]
    df = pd.read_csv(CSV, usecols=list(dict.fromkeys(extra + feature_cols)))
    print(f"Loaded {len(df)} records, {len(feature_cols)} features.")

    print("\nFitting OOF accuracy-risk propensity ...")
    df_scored = compute_p_low_oof(df, feature_cols)

    rows = []
    for sp in SPECIES:
        if (df_scored["Crayfish_scientific_name"] == sp).sum() < 20:
            print(f"  skip {sp} (too few records)")
            continue
        preds, _ = run_species(df_scored, sp, feature_cols)
        base = preds["all"]
        for strat in STRATEGIES:
            if strat == "all":
                continue
            m = metrics(base, preds[strat])
            m.update({"species": sp, "strategy_vs_all": strat})
            rows.append(m)
        print(f"  {sp}")

    cols = ["species", "strategy_vs_all", "schoener_D", "warren_I",
            "range_top30_pct", "range_top10_pct", "top10_jaccard", "mean_abs_diff"]
    out = pd.DataFrame(rows)[cols]
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/sdm_consequence_reconcile.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== ALL -> FILTER  (the Robustness reconciliation slice) ===")
    f = out[out["strategy_vs_all"] == "filter"].drop(columns="strategy_vs_all")
    print(f.round(3).to_string(index=False))
    print(f"\nmean ALL->FILTER:  D={f['schoener_D'].mean():.3f}  I={f['warren_I'].mean():.3f}  "
          f"range@top30={f['range_top30_pct'].mean():+.1f}%  "
          f"range@top10={f['range_top10_pct'].mean():+.1f}%")

    print("\nfull table (every strategy vs ALL) -> reports/sdm_consequence_reconcile.csv")
    print("\nReading: Robustness reports contamination INFLATES predicted range ~25-30%. Filtering")
    print("REMOVES those records, so ALL->FILTER range change should be NEGATIVE; if its magnitude")
    print("is ~20-30%, the two papers are numerically consistent (opposite signs, same order). If")
    print("much smaller, they are NOT the same magnitude -- report that honestly rather than letting")
    print("the draft's 'numerically consistent / same order' sentence stand unchecked.")


if __name__ == "__main__":
    main()
