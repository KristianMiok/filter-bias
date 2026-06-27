"""
sdm_consequence_strategies.py
=============================
Adds the quality-filtering DECISION-FRAMEWORK strategies on top of the existing
multispecies consequence pipeline, and measures the BETWEEN-strategy divergence of
SDM predictions on the real data.

Strategies (per species; presence = target species, background = other species):
  all      all presence, no weights                                    [baseline]
  filter   high-accuracy presence only                                 [hard filter; FILTER axis]
  trust    all presence, w = 1 - p_low (clip 0.05-1)  == existing "weighted" strategy
           suppresses likely-mislocated records; SAME direction as filter (soft filter)
  ipw      high-accuracy presence, w = 1 / P(high|env), trimmed, mass-normalized
           undoes the filtering selection; OPPOSITE direction to filter/trust  [beta axis]
  ipw_str  ipw with weights trimmed WITHIN subgroups (Mondrian)         [positivity-robust]

WHY BOTH trust AND ipw: trust/filter address coordinate error in low-accuracy records
(the rho axis -- suppress them); ipw addresses selection bias from filtering (the beta
axis -- recover the niche from the trustworthy subset). They move predictions in OPPOSITE
directions in environmental space. Which is correct depends on the (unidentifiable from
data alone) coordinate-error structure, so the SPREAD between filter/trust and ipw is the
irreducible cost of not knowing it. This script measures that spread; it does NOT decide
which strategy is "right" (that needs the simulation / ground truth in qfbias_sim.py).

Self-contained: computes p_low(env) OOF on the in-memory frame (no cross-file join),
reuses model / background / shift from sdm_consequence_multispecies.py.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split

from sdm_consequence_multispecies import (
    CSV, SPECIES, N_BACKGROUND, RANDOM_STATE,
    get_feature_cols, make_model, prepare_presence_background,
    summarize_prediction_shift,
)

STRATEGIES = ["all", "filter", "trust", "ipw", "ipw_str"]
SUBGROUP_COL = "Status"        # Mondrian stratum (alt: headwater via strahler == 1)
TRIM_PCT = 99.0
TEST_SIZE = 0.25               # match your existing split if it differs


# --------------------------------------------------------------------------- #
def compute_p_low_oof(df, feature_cols, seed=RANDOM_STATE):
    """OOF P(Accuracy == Low | env) on the High/Low rows, attached row-aligned (no join)."""
    sub = df[df["Accuracy"].isin(["High", "Low"])].copy().reset_index(drop=True)
    y = (sub["Accuracy"] == "Low").astype(int).values
    X = sub[feature_cols]
    prop = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=10,
                                      class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=-1)),
    ])
    oof = np.full(len(sub), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        prop.fit(X.iloc[tr], y[tr])
        oof[te] = prop.predict_proba(X.iloc[te])[:, 1]
    sub["p_low"] = oof
    print(f"  p_low OOF: mean={np.nanmean(oof):.3f}  "
          f"high-accuracy rate={(sub['Accuracy'] == 'High').mean():.1%}")
    return sub


def build_weights(train_df, strategy, subgroup_col=SUBGROUP_COL):
    """Return (keep_mask, sample_weight) for a strategy. Background always kept, weight 1."""
    y = train_df["y"].values
    pres = y == 1
    acc = train_df["Accuracy"].values
    p_low = train_df["p_low"].values
    p_high = np.clip(1.0 - p_low, 1e-3, None)

    keep = np.ones(len(train_df), dtype=bool)
    w = np.ones(len(train_df), dtype=float)

    if strategy == "all":
        return keep, None

    if strategy == "filter":
        keep = (~pres) | (acc == "High")           # drop low-accuracy presence
        return keep, None

    if strategy == "trust":                         # existing "weighted": soft filter
        w[pres] = np.clip(1.0 - p_low[pres], 0.05, 1.0)
        return keep, w

    if strategy in ("ipw", "ipw_str"):
        hi = pres & (acc == "High")
        keep = (~pres) | (acc == "High")            # IPW uses the trustworthy subset only
        raw = np.zeros(len(train_df))
        raw[hi] = 1.0 / p_high[hi]
        if strategy == "ipw":
            raw[hi] = np.clip(raw[hi], None, np.percentile(raw[hi], TRIM_PCT))
        else:                                       # Mondrian: trim within each subgroup
            grp = train_df[subgroup_col].values
            for g in pd.unique(grp[hi]):
                gm = hi & (grp == g)
                if gm.sum() > 5:
                    raw[gm] = np.clip(raw[gm], None, np.percentile(raw[gm], TRIM_PCT))
        # mass-normalize kept presence so total presence weight == #kept presence
        kept_pres = hi
        raw[kept_pres] *= kept_pres.sum() / raw[kept_pres].sum()
        w = np.where(pres, raw, 1.0)
        return keep, w

    raise ValueError(strategy)


def ess_frac(w, mask):
    ww = w[mask]
    return (ww.sum() ** 2) / (np.sum(ww ** 2) * len(ww))


def run_species(df_scored, sp, feature_cols, seed=RANDOM_STATE):
    dat, pres, bg = prepare_presence_background(df_scored, sp)
    train_df, _ = train_test_split(dat, test_size=TEST_SIZE,
                                   stratify=dat["y"], random_state=seed)
    predict_df = bg.copy().reset_index(drop=True)           # surface = background env
    preds, diag = {}, {}
    for strat in STRATEGIES:
        keep, w = build_weights(train_df, strat)
        tr = train_df.loc[keep].reset_index(drop=True)
        sw = None if w is None else w[keep]
        model = make_model(seed)
        model.fit(tr[feature_cols], tr["y"], rf__sample_weight=sw)
        preds[strat] = model.predict_proba(predict_df[feature_cols])[:, 1]
        if strat in ("ipw", "ipw_str") and sw is not None:
            diag[strat + "_ess"] = ess_frac(sw, tr["y"].values == 1)
    return preds, diag


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    feature_cols = get_feature_cols(header)
    extra = [c for c in ["Crayfish_scientific_name", "Accuracy", "Status", "strahler", "WoCID"]
             if c in header]
    df = pd.read_csv(CSV, usecols=list(dict.fromkeys(extra + feature_cols)))
    print(f"Loaded {len(df)} records, {len(feature_cols)} local env features.")

    print("\nFitting OOF accuracy-risk propensity ...")
    df_scored = compute_p_low_oof(df, feature_cols)

    pairs = [("all", "filter"), ("all", "trust"), ("all", "ipw"), ("all", "ipw_str"),
             ("filter", "ipw"), ("trust", "ipw"), ("ipw", "ipw_str")]
    rows = []
    for sp in SPECIES:
        if (df_scored["Crayfish_scientific_name"] == sp).sum() < 20:
            print(f"  skip {sp} (too few records)")
            continue
        preds, diag = run_species(df_scored, sp, feature_cols)
        for a, b in pairs:
            sh = dict(summarize_prediction_shift(preds[a], preds[b]))
            sh.update({"species": sp, "base": a, "alt": b, **diag})
            rows.append(sh)
        ess = diag.get("ipw_ess", float("nan"))
        print(f"  {sp:32s} ipw ESS={ess:.1%}")

    out = pd.DataFrame(rows)
    out.to_csv("figures/sdm_consequence_strategies_shifts.csv", index=False)

    print("\n=== Cost of not knowing: FILTER vs IPW divergence per species ===")
    key = out[(out.base == "filter") & (out.alt == "ipw")]
    show = [c for c in ["species", "top10_jaccard", "mean_abs_diff", "p95_abs_diff"] if c in key]
    print(key[show].to_string(index=False))
    print("\nWrote figures/sdm_consequence_strategies_shifts.csv")


if __name__ == "__main__":
    main()
