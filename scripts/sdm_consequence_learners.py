"""
sdm_consequence_learners.py
===========================
Task 2 (learner sensitivity), realistic version.

Lucian asked for Maxent + XGBoost "wired up from Robustness" -- but Maxent was not used
for this paper, and this repo's make_model is RandomForest only. What is both feasible and
on-point is XGBoost: gradient boosting reweights presence records far more aggressively than
a tree-AVERAGING random forest, so it directly tests whether the "modest consequence" we see
is a property of RF rather than of the data. If the FILTER<->IPW spread widens under XGBoost,
that materially strengthens the paper; if it holds, "modest" is robust to the learner.

Reuses the validated pipeline (compute_p_low_oof, build_weights, prepare_presence_background,
STRATEGIES) unchanged; only the learner in the fit loop is swapped. Preprocessing is identical
across learners (median imputer) so any difference is attributable to the learner alone.

Output: reports/sdm_consequence_learners.csv (learner x species x strategy-vs-ALL), and prints
two reads: ALL->FILTER (reconciliation robustness) and FILTER<->IPW (the cost-of-not-knowing
spread) under each learner.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sdm_consequence_multispecies import (
    CSV, SPECIES, RANDOM_STATE, get_feature_cols, make_model, prepare_presence_background,
)
from sdm_consequence_strategies import compute_p_low_oof, build_weights, STRATEGIES, TEST_SIZE


# ----- metrics (same definitions as the reconcile script) ------------------ #
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
    thr = np.quantile(all_vec, 1.0 - top_frac)
    base = np.mean(all_vec >= thr)
    alt = np.mean(strat_vec >= thr)
    return float(100.0 * (alt - base) / base) if base > 0 else float("nan")


def top_jaccard(a, b, top_frac=0.10):
    ta = a >= np.quantile(a, 1.0 - top_frac)
    tb = b >= np.quantile(b, 1.0 - top_frac)
    union = np.sum(ta | tb)
    return float(np.sum(ta & tb) / union) if union else float("nan")


def metrics(base_vec, alt_vec):
    return {
        "schoener_D": schoener_D(base_vec, alt_vec),
        "warren_I": warren_I(base_vec, alt_vec),
        "range_top30_pct": range_pct_change(base_vec, alt_vec, 0.30),
        "range_top10_pct": range_pct_change(base_vec, alt_vec, 0.10),
        "top10_jaccard": top_jaccard(base_vec, alt_vec, 0.10),
        "mean_abs_diff": float(np.mean(np.abs(np.asarray(base_vec, float) - np.asarray(alt_vec, float)))),
    }


# ----- XGBoost learner in the same Pipeline shape as RF -------------------- #
def make_xgb(seed, scale_pos_weight=1.0):
    from xgboost import XGBClassifier
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            scale_pos_weight=scale_pos_weight,   # presence/background balance (RF uses balanced_subsample)
            random_state=seed, n_jobs=-1, eval_metric="logloss", tree_method="hist",
        )),
    ])


def run_species_learner(df_scored, sp, feature_cols, learner, seed=RANDOM_STATE):
    """Same logic as strategies.run_species, but the learner (and its sample_weight step name)
    is parameterised so we can swap RF <-> XGB without touching the original."""
    dat, _, bg = prepare_presence_background(df_scored, sp)
    train_df, _ = train_test_split(dat, test_size=TEST_SIZE, stratify=dat["y"], random_state=seed)
    predict_df = bg.copy().reset_index(drop=True)

    preds = {}
    for strat in STRATEGIES:
        keep, w = build_weights(train_df, strat)
        tr = train_df.loc[keep].reset_index(drop=True)
        sw = None if w is None else w[keep]
        if learner == "rf":
            model, step = make_model(seed), "rf"
        else:
            npos = int((tr["y"] == 1).sum()); nneg = int((tr["y"] == 0).sum())
            model, step = make_xgb(seed, scale_pos_weight=nneg / max(npos, 1)), "xgb"
        fit_kw = {f"{step}__sample_weight": sw} if sw is not None else {}
        model.fit(tr[feature_cols], tr["y"], **fit_kw)
        preds[strat] = model.predict_proba(predict_df[feature_cols])[:, 1]
    return preds


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
    for learner in ["rf", "xgb"]:
        print(f"\n--- learner: {learner.upper()} ---")
        for sp in SPECIES:
            if (df_scored["Crayfish_scientific_name"] == sp).sum() < 20:
                continue
            preds = run_species_learner(df_scored, sp, feature_cols, learner)
            base = preds["all"]
            for strat in STRATEGIES:
                if strat == "all":
                    continue
                m = metrics(base, preds[strat]); m.update(
                    {"learner": learner, "species": sp, "strategy_vs_all": strat})
                rows.append(m)
            # also the filter<->ipw spread (the cost of not knowing)
            sp_spread = metrics(preds["filter"], preds["ipw"]); sp_spread.update(
                {"learner": learner, "species": sp, "strategy_vs_all": "filter->ipw"})
            rows.append(sp_spread)
            print(f"  {sp}")

    out = pd.DataFrame(rows)[["learner", "species", "strategy_vs_all", "schoener_D", "warren_I",
                              "range_top30_pct", "range_top10_pct", "top10_jaccard", "mean_abs_diff"]]
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/sdm_consequence_learners.csv", index=False)

    pd.set_option("display.width", 200)
    for tag, sub in [("ALL -> FILTER  (reconciliation: does it hold under XGB?)", "filter"),
                     ("FILTER <-> IPW  (cost of not knowing: does the spread widen under XGB?)", "filter->ipw")]:
        print(f"\n=== {tag} ===")
        t = out[out["strategy_vs_all"] == sub]
        g = t.groupby("learner")[["schoener_D", "warren_I", "range_top30_pct",
                                  "range_top10_pct", "mean_abs_diff"]].mean()
        print(g.round(3).to_string())

    print("\nfull table -> reports/sdm_consequence_learners.csv")
    print("\nReading: compare RF vs XGB. If ALL->FILTER range/D are similar under both, the")
    print("reconciliation is learner-robust. If the FILTER<->IPW spread is LARGER under XGB")
    print("(lower D, bigger range gap), then 'modest consequence' was partly an RF property and")
    print("the more sensitive learner shows filtering matters more -- a point in the paper's favour.")


if __name__ == "__main__":
    main()
