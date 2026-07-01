import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr


CSV = os.environ.get("WOC_CSV", "data/crayfish_master.csv")
SCORES = "figures/accuracy_risk_scores_oof.csv"

RANDOM_STATE = 42
N_BACKGROUND = 20000

SPECIES = [
    "Pacifastacus leniusculus",
    "Astacus astacus",
]


def get_feature_cols(header):
    prefixes = ("l_CLI", "l_TOP", "l_SOL", "l_LAC")
    return [c for c in header if c.startswith(prefixes)]


def make_model(seed=RANDOM_STATE):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=14,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def prepare_presence_background(df, species_name, feature_cols):
    pres = df[df["Crayfish_scientific_name"] == species_name].copy()
    bg = df[df["Crayfish_scientific_name"] != species_name].copy()

    if len(bg) > N_BACKGROUND:
        bg = bg.sample(n=N_BACKGROUND, random_state=RANDOM_STATE)

    pres["y"] = 1
    bg["y"] = 0

    dat = pd.concat([pres, bg], ignore_index=True)
    dat = dat.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    return dat, pres, bg


def train_and_predict(train_df, test_df, predict_df, feature_cols, strategy, seed):
    model = make_model(seed)

    X_train = train_df[feature_cols]
    y_train = train_df["y"]

    sample_weight = None

    if strategy == "weighted":
        # Background weight = 1. Presence weight = environmental confidence score.
        # Confidence is high when p_low_accuracy is low.
        sample_weight = np.ones(len(train_df), dtype=float)
        pres_mask = train_df["y"].values == 1
        sample_weight[pres_mask] = 1.0 - train_df.loc[pres_mask, "p_low_accuracy"].values

        # Avoid zero-weight presences.
        sample_weight[pres_mask] = np.clip(sample_weight[pres_mask], 0.05, 1.0)

    model.fit(X_train, y_train, rf__sample_weight=sample_weight)

    test_pred = model.predict_proba(test_df[feature_cols])[:, 1]
    predict_pred = model.predict_proba(predict_df[feature_cols])[:, 1]

    auc = roc_auc_score(test_df["y"], test_pred)
    ap = average_precision_score(test_df["y"], test_pred)

    return {
        "model": model,
        "test_pred": test_pred,
        "predict_pred": predict_pred,
        "auc": auc,
        "ap": ap,
    }


def summarize_prediction_shift(base_pred, alt_pred, name):
    rho = spearmanr(base_pred, alt_pred).correlation
    pearson = np.corrcoef(base_pred, alt_pred)[0, 1]

    abs_diff = np.abs(base_pred - alt_pred)
    mean_abs_diff = abs_diff.mean()
    p95_abs_diff = np.quantile(abs_diff, 0.95)

    base_top = base_pred >= np.quantile(base_pred, 0.90)
    alt_top = alt_pred >= np.quantile(alt_pred, 0.90)

    intersection = np.logical_and(base_top, alt_top).sum()
    union = np.logical_or(base_top, alt_top).sum()
    jaccard_top10 = intersection / union if union > 0 else np.nan

    return {
        "comparison": name,
        "spearman": rho,
        "pearson": pearson,
        "mean_abs_diff": mean_abs_diff,
        "p95_abs_diff": p95_abs_diff,
        "top10_jaccard": jaccard_top10,
    }


def feature_importance_table(model, feature_cols, label):
    rf = model.named_steps["rf"]
    imp = pd.DataFrame(
        {
            "feature": feature_cols,
            f"importance_{label}": rf.feature_importances_,
        }
    )
    return imp


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    feature_cols = get_feature_cols(header)

    score_cols = [
        "WoCID",
        "p_low_accuracy",
        "risk_tertile",
        "risk_decile",
    ]

    usecols = [
        "WoCID",
        "Crayfish_scientific_name",
        "Accuracy",
        "Status",
        "Year_of_record",
        "distance_m",
        "strahler",
    ] + feature_cols

    usecols = [c for c in usecols if c in header]

    print("Loading full environmental data...")
    df = pd.read_csv(CSV, usecols=usecols)

    print("Loading OOF accuracy-risk scores...")
    scores = pd.read_csv(SCORES, usecols=[c for c in score_cols if c in pd.read_csv(SCORES, nrows=0).columns])

    if "WoCID" not in df.columns or "WoCID" not in scores.columns:
        raise ValueError("WoCID must be present in both full data and risk score file.")

    df = df.merge(scores, on="WoCID", how="inner")
    df = df[df["Accuracy"].isin(["High", "Low"])].copy().reset_index(drop=True)

    print(f"Rows after merge: {len(df)}")
    print(f"Local environmental features: {len(feature_cols)}")

    all_rows = []
    all_shift_rows = []

    for species_name in SPECIES:
        print(f"\n=== Species: {species_name} ===")

        dat, pres, bg = prepare_presence_background(df, species_name, feature_cols)

        print(f"Presences: {len(pres)}")
        print(f"Background: {len(bg)}")
        print("Presence risk tertiles:")
        print(pres["risk_tertile"].value_counts(dropna=False).to_string())
        print("Presence observed accuracy:")
        print(pres["Accuracy"].value_counts(dropna=False).to_string())

        train_df, test_df = train_test_split(
            dat,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=dat["y"],
        )

        pres_train = train_df[train_df["y"] == 1]
        bg_train = train_df[train_df["y"] == 0]

        train_all = train_df.copy()

        train_remove_high_risk = pd.concat(
            [
                pres_train[pres_train["risk_tertile"] != "high risk"],
                bg_train,
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=RANDOM_STATE)

        train_weighted = train_df.copy()

        predict_df = bg.copy().reset_index(drop=True)

        strategies = {
            "all": train_all,
            "remove_high_risk": train_remove_high_risk,
            "weighted": train_weighted,
        }

        results = {}
        importance_tables = []

        for strategy, train_strategy_df in strategies.items():
            n_pres_train = int((train_strategy_df["y"] == 1).sum())
            n_bg_train = int((train_strategy_df["y"] == 0).sum())

            print(f"\nTraining strategy: {strategy}")
            print(f"  train presences={n_pres_train}, train background={n_bg_train}")

            res = train_and_predict(
                train_strategy_df,
                test_df,
                predict_df,
                feature_cols,
                strategy=strategy,
                seed=RANDOM_STATE,
            )

            results[strategy] = res

            print(f"  test AUC={res['auc']:.3f}, AP={res['ap']:.3f}")

            all_rows.append(
                {
                    "species": species_name,
                    "strategy": strategy,
                    "train_presences": n_pres_train,
                    "train_background": n_bg_train,
                    "test_auc": res["auc"],
                    "test_ap": res["ap"],
                    "mean_prediction_background": float(np.mean(res["predict_pred"])),
                    "p90_prediction_background": float(np.quantile(res["predict_pred"], 0.90)),
                    "p95_prediction_background": float(np.quantile(res["predict_pred"], 0.95)),
                }
            )

            importance_tables.append(feature_importance_table(res["model"], feature_cols, strategy))

        base = results["all"]["predict_pred"]

        for alt in ["remove_high_risk", "weighted"]:
            shift = summarize_prediction_shift(
                base,
                results[alt]["predict_pred"],
                f"all_vs_{alt}",
            )
            shift["species"] = species_name
            all_shift_rows.append(shift)

            print(f"\nPrediction shift: all vs {alt}")
            print(f"  Spearman={shift['spearman']:.3f}")
            print(f"  Pearson={shift['pearson']:.3f}")
            print(f"  mean abs diff={shift['mean_abs_diff']:.3f}")
            print(f"  p95 abs diff={shift['p95_abs_diff']:.3f}")
            print(f"  top10 Jaccard={shift['top10_jaccard']:.3f}")

        # Save per-record background predictions for later plots.
        pred_out = predict_df[
            [
                c
                for c in [
                    "WoCID",
                    "Crayfish_scientific_name",
                    "Accuracy",
                    "Status",
                    "Year_of_record",
                    "distance_m",
                    "strahler",
                    "p_low_accuracy",
                    "risk_tertile",
                    "risk_decile",
                ]
                if c in predict_df.columns
            ]
        ].copy()

        for strategy, res in results.items():
            pred_out[f"pred_{strategy}"] = res["predict_pred"]

        safe_species = species_name.replace(" ", "_")
        pred_path = f"figures/sdm_consequence_predictions_{safe_species}.csv"
        pred_out.to_csv(pred_path, index=False)
        print(f"\nSaved background predictions to {pred_path}")

        # Save feature importances side-by-side.
        imp = importance_tables[0]
        for tab in importance_tables[1:]:
            imp = imp.merge(tab, on="feature", how="outer")

        imp_path = f"figures/sdm_consequence_importances_{safe_species}.csv"
        imp.to_csv(imp_path, index=False)
        print(f"Saved feature importances to {imp_path}")

    summary = pd.DataFrame(all_rows)
    shifts = pd.DataFrame(all_shift_rows)

    summary.to_csv("figures/sdm_consequence_pilot_summary.csv", index=False)
    shifts.to_csv("figures/sdm_consequence_prediction_shifts.csv", index=False)

    print("\n=== SDM consequence summary ===")
    print(summary.round(3).to_string(index=False))

    print("\n=== Prediction shifts ===")
    print(shifts.round(3).to_string(index=False))

    print("\nSaved:")
    print("  figures/sdm_consequence_pilot_summary.csv")
    print("  figures/sdm_consequence_prediction_shifts.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
