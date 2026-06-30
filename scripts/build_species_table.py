"""
Per-species descriptive table (Table 2) for the quality-filtering-bias paper.

For each crayfish species, on the SAME data the rest of the pipeline uses, computes:
  - n_total, n_high, high-accuracy %
  - per-species propensity env-AUC: how predictable that species' record accuracy is
    from environment alone, P(high | env), under basin GroupKFold (spatial CV, as in the
    main diagnostic). Well-conditioned via PCA on the 302 collinear env features.
  - per-species IPW effective sample size (ESS), from the same per-species propensity:
    fraction of the high-accuracy subset that the inverse-propensity weights effectively use.

Reuses CSV / SPECIES / get_feature_cols from sdm_consequence_multispecies, so the data,
species list and feature set are identical to the consequence analysis. Read-only on the
master CSV; writes reports/species_table.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from sdm_consequence_multispecies import CSV, SPECIES, get_feature_cols, RANDOM_STATE

PCA_K = 15          # matches the well-conditioned propensity used in the main diagnostic
MAX_FOLDS = 5
GROUP_COL = "basin_id"


def species_env_auc_and_ess(sub, feature_cols, k_pca=PCA_K, seed=RANDOM_STATE):
    """Per-species P(high|env) AUC under basin GroupKFold, plus IPW ESS on the high subset."""
    sub = sub[sub["Accuracy"].isin(["High", "Low"])].copy()
    if len(sub) < 20:
        return np.nan, np.nan, "too few records"
    y = (sub["Accuracy"] == "High").astype(int).values
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, "single accuracy class"

    X = sub[feature_cols].values
    # median-impute any NaNs so a species is never silently dropped (headwater upstream NaNs etc.)
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])

    if GROUP_COL in sub.columns:
        groups = sub[GROUP_COL].values
        n_groups = len(np.unique(groups))
        k = min(MAX_FOLDS, n_groups)
        if k < 2:
            return np.nan, np.nan, "too few basins for spatial CV"
        splitter = GroupKFold(n_splits=k).split(X, y, groups)
    else:
        return np.nan, np.nan, "no basin column"

    kp = min(k_pca, X.shape[1])
    oof = np.full(len(y), np.nan)
    for tr, te in splitter:
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = Pipeline([("sc", StandardScaler()),
                         ("pca", PCA(n_components=kp, random_state=seed)),
                         ("lr", LogisticRegression(max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]

    mask = ~np.isnan(oof)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return np.nan, np.nan, "insufficient OOF coverage"
    auc = roc_auc_score(y[mask], oof[mask])

    # IPW ESS on the high-accuracy subset: w = 1 / P(high|env)
    phigh = np.clip(oof[mask][y[mask] == 1], 0.01, 0.999)
    w = 1.0 / phigh
    ess = (w.sum() ** 2) / (np.sum(w ** 2) * len(w))
    return auc, ess, "ok"


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    feature_cols = get_feature_cols(header)
    extra = [c for c in ["Crayfish_scientific_name", "Accuracy", "Status", GROUP_COL]
             if c in header]
    df = pd.read_csv(CSV, usecols=list(dict.fromkeys(extra + feature_cols)), low_memory=False)
    df = df[df["Accuracy"].isin(["High", "Low"])].copy()
    print(f"Loaded {len(df)} records, {len(feature_cols)} env features, "
          f"basin column {'present' if GROUP_COL in df.columns else 'MISSING'}.\n")

    recs = []
    for sp in SPECIES:
        sub = df[df["Crayfish_scientific_name"] == sp]
        if len(sub) == 0:
            continue
        n_total = len(sub)
        n_high = int((sub["Accuracy"] == "High").sum())
        pct_high = 100.0 * n_high / n_total
        status = sub["Status"].mode().iat[0] if "Status" in sub.columns and len(sub) else ""
        auc, ess, note = species_env_auc_and_ess(sub, feature_cols)
        recs.append({
            "species": sp,
            "status": status,
            "n_total": n_total,
            "n_high": n_high,
            "pct_high": round(pct_high, 1),
            "env_auc": round(auc, 3) if auc == auc else np.nan,
            "ipw_ess_pct": round(100 * ess, 1) if ess == ess else np.nan,
            "note": note,
        })
        print(f"  {sp:32s} n={n_total:6d}  high={pct_high:4.1f}%  "
              f"env-AUC={auc:.3f}  ESS={100*ess:4.1f}%" if auc == auc
              else f"  {sp:32s} n={n_total:6d}  ({note})")

    tab = pd.DataFrame(recs)
    # sort by sample size descending for the table
    tab = tab.sort_values("n_total", ascending=False).reset_index(drop=True)

    Path("reports").mkdir(exist_ok=True)
    out = "reports/species_table.csv"
    tab.to_csv(out, index=False)
    print("\n" + "=" * 70)
    print("TABLE 2  (per-species descriptives)")
    print("=" * 70)
    print(tab.drop(columns=["note"]).to_string(index=False))
    print(f"\nwrote {out}")
    # quick totals line for the text
    print(f"\ntotals: {tab['n_total'].sum()} records across {len(tab)} species; "
          f"overall high-accuracy {100*tab['n_high'].sum()/tab['n_total'].sum():.1f}%")


if __name__ == "__main__":
    main()
