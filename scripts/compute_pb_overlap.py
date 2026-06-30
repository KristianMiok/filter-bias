"""
compute_pb_overlap.py
=====================
Re-test of the manuscript claim that FILTER->IPW reweighting sensitivity is "a property of
niche breadth / background overlap, not ecology." The first breadth measure (absolute presence
variance) showed NO correlation with FILTER->IPW sensitivity (Pearson -0.18). But the claim is
specifically about presence-BACKGROUND overlap (weak presence/background contrast), which is a
different quantity than absolute presence variance. This computes that contrast directly and
correlates it with FILTER->IPW sensitivity, so we can either support the claim properly or
flag it for correction before submission.

Two contrast measures per species (lower contrast = more overlap = hypothesised higher IPW
sensitivity):
  1. presence-vs-background separability AUC (fast logistic on standardized env);
     0.5 = indistinguishable from background (max overlap), 1.0 = fully separable (specialist).
  2. Mahalanobis distance between presence and background centroids (env), pooled covariance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sdm_consequence_multispecies import (
    CSV, SPECIES, RANDOM_STATE, get_feature_cols, prepare_presence_background,
)

FILTER_IPW = {
    "Pacifastacus leniusculus": 0.0073, "Astacus astacus": 0.0206,
    "Faxonius limosus": 0.0190, "Pontastacus leptodactylus": 0.0092,
    "Procambarus clarkii": 0.0159, "Austropotamobius pallipes": 0.0214,
    "Austropotamobius torrentium": 0.0121,
}


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    feature_cols = get_feature_cols(header)
    extra = [c for c in ["Crayfish_scientific_name", "Status"] if c in header]
    df = pd.read_csv(CSV, usecols=list(dict.fromkeys(extra + feature_cols)))
    print(f"Loaded {len(df)} records, {len(feature_cols)} features.")

    rows = []
    for sp in SPECIES:
        dat, pres, bg = prepare_presence_background(df, sp)
        if len(pres) < 20:
            continue
        X = dat[feature_cols].astype(float).values
        y = dat["y"].values

        # 1. separability AUC via 3-fold OOF logistic on imputed + PCA(15) standardized env
        clf = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("pca", PCA(n_components=15, random_state=RANDOM_STATE)),
            ("lr", LogisticRegression(max_iter=2000, C=1.0)),
        ])
        oof = cross_val_predict(clf, X, y, cv=3, method="predict_proba",
                                n_jobs=-1)[:, 1]
        auc = roc_auc_score(y, oof)

        # 2. Mahalanobis distance between presence/background centroids (imputed env, pooled cov)
        imp = SimpleImputer(strategy="median")
        Xi = imp.fit_transform(X)
        mp = Xi[y == 1].mean(axis=0); mb = Xi[y == 0].mean(axis=0)
        cov = np.cov(Xi.T) + np.eye(Xi.shape[1]) * 1e-6
        try:
            inv = np.linalg.pinv(cov)
            d = mp - mb
            maha = float(np.sqrt(d @ inv @ d))
        except Exception:
            maha = np.nan

        status = df.loc[df["Crayfish_scientific_name"] == sp, "Status"].mode().iloc[0]
        rows.append({"species": sp, "status": status, "n_presence": int(len(pres)),
                     "pb_separability_auc": round(auc, 4), "centroid_mahalanobis": round(maha, 3),
                     "filter_ipw": FILTER_IPW.get(sp, np.nan)})
        print(f"  {sp}: AUC={auc:.3f}  maha={maha:.2f}")

    out = pd.DataFrame(rows).sort_values("pb_separability_auc").reset_index(drop=True)
    pd.set_option("display.width", 170)
    print("\nPer-species presence-background contrast (sorted by separability):\n")
    print(out.to_string(index=False))

    sub = out.dropna(subset=["filter_ipw"])
    def corr(a, b):
        r_p = np.corrcoef(a, b)[0, 1]
        r_s = np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1]
        return r_p, r_s
    if len(sub) >= 3:
        # hypothesis: LOWER separability (more overlap) -> HIGHER filter_ipw, so expect NEGATIVE r
        rp_a, rs_a = corr(sub["pb_separability_auc"], sub["filter_ipw"])
        rp_m, rs_m = corr(sub["centroid_mahalanobis"], sub["filter_ipw"])
        print(f"\nseparability AUC  <-> FILTER->IPW :  Pearson {rp_a:+.3f}  Spearman {rs_a:+.3f}  (expect negative)")
        print(f"centroid Mahalanobis <-> FILTER->IPW: Pearson {rp_m:+.3f}  Spearman {rs_m:+.3f}  (expect negative)")
        print("\nIf neither is clearly negative, the 'sensitivity tracks niche breadth/overlap'")
        print("claim is NOT supported and must be softened/removed from the manuscript.")

    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/pb_overlap.csv", index=False)
    print("\nwrote reports/pb_overlap.csv")


if __name__ == "__main__":
    main()
