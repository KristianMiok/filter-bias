import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


CSV = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv"
RANDOM_STATE = 42
N_SPLITS = 5


def get_feature_cols(header):
    prefixes = ("l_CLI", "l_TOP", "l_SOL", "l_LAC")
    return [c for c in header if c.startswith(prefixes)]


def make_model(seed):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def summarize_bin(df, bin_col):
    tab = (
        df.groupby(bin_col, observed=True)
        .agg(
            n=("is_low", "size"),
            actual_low_rate=("is_low", "mean"),
            mean_predicted_risk=("p_low_accuracy", "mean"),
            median_predicted_risk=("p_low_accuracy", "median"),
            alien_rate=("is_alien", "mean"),
            native_rate=("is_native", "mean"),
            median_year=("Year_of_record", "median"),
            median_distance_m=("distance_m", "median"),
            median_strahler=("strahler", "median"),
        )
        .reset_index()
    )
    return tab


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()

    # Deliberately local environmental only.
    # Previous tests showed local environmental features contain almost all signal.
    feature_cols = get_feature_cols(header)

    metadata_cols = [
        "WoCID",
        "Accuracy",
        "Status",
        "Year_of_record",
        "distance_m",
        "strahler",
        "basin_id",
        "Crayfish_scientific_name",
        "lat_or",
        "long_or",
        "lat_snap",
        "long_snap",
    ]

    usecols = [c for c in metadata_cols + feature_cols if c in header]

    print("Loading data...")
    df = pd.read_csv(CSV, usecols=usecols)
    df = df[df["Accuracy"].isin(["High", "Low"])].copy().reset_index(drop=True)

    y = (df["Accuracy"] == "Low").astype(int)
    X = df[feature_cols]

    print(f"Rows: {len(df)}")
    print(f"Local environmental features: {len(feature_cols)}")
    print(df["Accuracy"].value_counts())
    print(f"Overall Low rate: {y.mean():.3f}")

    oof = np.zeros(len(df), dtype=float)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print(f"\nFitting {N_SPLITS}-fold out-of-fold accuracy-risk model...")
    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        model = make_model(RANDOM_STATE + fold)
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p

        auc = roc_auc_score(y.iloc[te], p)
        ap = average_precision_score(y.iloc[te], p)

        print(
            f"fold {fold}: AUC={auc:.3f} AP={ap:.3f} "
            f"test_low={y.iloc[te].mean():.3f} n_test={len(te)}"
        )

    df["is_low"] = y
    df["p_low_accuracy"] = oof

    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)

    print("\n=== Overall out-of-fold performance ===")
    print(f"OOF ROC-AUC={auc:.3f}")
    print(f"OOF Average precision={ap:.3f}")

    df["is_alien"] = (df["Status"] == "Alien").astype(int)
    df["is_native"] = (df["Status"] == "Native").astype(int)

    # Risk bins by quantile. These are model-derived environmental risk strata.
    df["risk_decile"] = pd.qcut(df["p_low_accuracy"], q=10, labels=False, duplicates="drop") + 1
    df["risk_tertile"] = pd.qcut(
        df["p_low_accuracy"],
        q=3,
        labels=["low risk", "medium risk", "high risk"],
        duplicates="drop",
    )

    print("\n=== Accuracy-risk tertiles ===")
    tertile_tab = summarize_bin(df, "risk_tertile")
    print(tertile_tab.to_string(index=False))

    print("\n=== Accuracy-risk deciles ===")
    decile_tab = summarize_bin(df, "risk_decile")
    print(decile_tab.to_string(index=False))

    print("\n=== Low rate by status within risk tertile ===")
    status_risk = (
        df.groupby(["Status", "risk_tertile"], observed=True)
        .agg(
            n=("is_low", "size"),
            actual_low_rate=("is_low", "mean"),
            mean_predicted_risk=("p_low_accuracy", "mean"),
        )
        .reset_index()
        .sort_values(["Status", "risk_tertile"])
    )
    print(status_risk.to_string(index=False))

    # Save compact score table.
    score_cols = [
        c
        for c in [
            "WoCID",
            "Accuracy",
            "is_low",
            "p_low_accuracy",
            "risk_tertile",
            "risk_decile",
            "Status",
            "Year_of_record",
            "distance_m",
            "strahler",
            "basin_id",
            "Crayfish_scientific_name",
            "lat_or",
            "long_or",
            "lat_snap",
            "long_snap",
        ]
        if c in df.columns
    ]

    df[score_cols].to_csv("figures/accuracy_risk_scores_oof.csv", index=False)
    tertile_tab.to_csv("figures/accuracy_risk_tertiles_summary.csv", index=False)
    decile_tab.to_csv("figures/accuracy_risk_deciles_summary.csv", index=False)
    status_risk.to_csv("figures/accuracy_risk_by_status.csv", index=False)

    print("\nSaved:")
    print("  figures/accuracy_risk_scores_oof.csv")
    print("  figures/accuracy_risk_tertiles_summary.csv")
    print("  figures/accuracy_risk_deciles_summary.csv")
    print("  figures/accuracy_risk_by_status.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
