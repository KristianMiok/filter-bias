import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline


CSV = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv"
RANDOM_STATE = 42


def get_feature_cols(header):
    prefixes = ("l_CLI", "l_TOP", "l_SOL", "l_LAC", "u_CLI", "u_TOP", "u_SOL", "u_LAC")
    return [c for c in header if c.startswith(prefixes)]


def evaluate_split(name, X_train, X_test, y_train, y_test):
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print(f"\n=== {name} ===")
    print(f"Train n={len(y_train)} | Test n={len(y_test)}")
    print(f"Train Low rate={y_train.mean():.3f} | Test Low rate={y_test.mean():.3f}")
    print(f"ROC-AUC={auc:.3f}")
    print(f"Average precision={ap:.3f}")

    return auc, ap


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    feature_cols = get_feature_cols(header)

    usecols = [
        "Accuracy",
        "basin_id",
        "Crayfish_scientific_name",
        "Status",
        "strahler",
        "distance_m",
    ] + feature_cols

    usecols = [c for c in usecols if c in header]

    print("Loading data...")
    df = pd.read_csv(CSV, usecols=usecols)
    df = df[df["Accuracy"].isin(["High", "Low"])].copy()

    y = (df["Accuracy"] == "Low").astype(int)
    X = df[feature_cols]

    print(f"Rows: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print(df["Accuracy"].value_counts())
    print(f"Overall Low rate: {y.mean():.3f}")

    # 1. Ordinary random split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    evaluate_split("Random row split", X_train, X_test, y_train, y_test)

    # 2. Basin holdout
    if "basin_id" in df.columns:
        basin_groups = df["basin_id"].fillna("missing").astype(str)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=RANDOM_STATE,
        )

        train_idx, test_idx = next(splitter.split(X, y, groups=basin_groups))

        evaluate_split(
            "Basin holdout split",
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
        )

        print(f"Train basins: {basin_groups.iloc[train_idx].nunique()}")
        print(f"Test basins: {basin_groups.iloc[test_idx].nunique()}")

    # 3. Species holdout
    if "Crayfish_scientific_name" in df.columns:
        species_groups = df["Crayfish_scientific_name"].fillna("missing").astype(str)

        counts = species_groups.value_counts()
        keep_species = counts[counts >= 100].index
        keep = species_groups.isin(keep_species)

        Xs = X[keep]
        ys = y[keep]
        gs = species_groups[keep]

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=RANDOM_STATE,
        )

        train_idx, test_idx = next(splitter.split(Xs, ys, groups=gs))

        evaluate_split(
            "Species holdout split",
            Xs.iloc[train_idx],
            Xs.iloc[test_idx],
            ys.iloc[train_idx],
            ys.iloc[test_idx],
        )

        print(f"Train species: {gs.iloc[train_idx].nunique()}")
        print(f"Test species: {gs.iloc[test_idx].nunique()}")

    # 4. Local-only features, because these were strongest and have less missingness
    local_cols = [c for c in feature_cols if c.startswith(("l_CLI", "l_TOP", "l_SOL", "l_LAC"))]
    X_local = df[local_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X_local,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    evaluate_split("Random row split, local features only", X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()