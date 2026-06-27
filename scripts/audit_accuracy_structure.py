import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


CSV = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv"
RANDOM_STATE = 42


def get_feature_groups(columns):
    groups = {
        "l_CLI": [c for c in columns if c.startswith("l_CLI")],
        "l_TOP": [c for c in columns if c.startswith("l_TOP")],
        "l_SOL": [c for c in columns if c.startswith("l_SOL")],
        "l_LAC": [c for c in columns if c.startswith("l_LAC")],
        "u_CLI": [c for c in columns if c.startswith("u_CLI")],
        "u_TOP": [c for c in columns if c.startswith("u_TOP")],
        "u_SOL": [c for c in columns if c.startswith("u_SOL")],
        "u_LAC": [c for c in columns if c.startswith("u_LAC")],
    }
    return {k: v for k, v in groups.items() if len(v) > 0}


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Average precision: {ap:.3f}")
    print(classification_report(y_test, pred, target_names=["High", "Low"]))

    return auc, ap, model


def main():
    print("Loading header...")
    header = pd.read_csv(CSV, nrows=0).columns.tolist()

    metadata_cols = [
        "WoCID",
        "lat_or",
        "long_or",
        "lat_snap",
        "long_snap",
        "Accuracy",
        "Crayfish_scientific_name",
        "Status",
        "Year_of_record",
        "basin_id",
        "subc_id",
        "reg_id",
        "hylak_id",
    ]

    feature_groups = get_feature_groups(header)
    feature_cols = [c for group in feature_groups.values() for c in group]

    extra_numeric = [
        c for c in ["distance_m", "strahler", "area_sqm", "sum_area_sqm", "ab_1000m", "ab_500m", "ab_200m", "is_coastal"]
        if c in header
    ]

    print("\nFeature groups:")
    for group, cols in feature_groups.items():
        print(f"  {group}: {len(cols)}")

    print(f"\nExtra numeric columns: {extra_numeric}")
    print(f"Total environmental features: {len(feature_cols)}")

    usecols = ["Accuracy", "strahler", "distance_m"] + feature_cols
    usecols = list(dict.fromkeys([c for c in usecols if c in header]))

    print("\nLoading selected columns...")
    df = pd.read_csv(CSV, usecols=usecols)
    df = df[df["Accuracy"].isin(["High", "Low"])].copy()

    print(f"Rows: {len(df)}")
    print(df["Accuracy"].value_counts())

    y = (df["Accuracy"] == "Low").astype(int)
    X = df[feature_cols].copy()

    print("\nMissingness by feature group:")
    for group, cols in feature_groups.items():
        miss = X[cols].isna().mean().mean()
        print(f"  {group}: {miss:.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, feature_cols),
        ],
        remainder="drop",
    )

    logreg = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="saga",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rf = Pipeline(
        steps=[
            (
                "pre",
                ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="median"), feature_cols),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "clf",
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

    evaluate_model("Logistic regression: all environmental features", logreg, X_train, X_test, y_train, y_test)
    _, _, fitted_rf = evaluate_model("Random forest: all environmental features", rf, X_train, X_test, y_train, y_test)

    print("\n=== Group-only AUCs, logistic regression ===")
    group_results = []
    for group, cols in feature_groups.items():
        Xg = df[cols].copy()
        yg = y.copy()

        Xg_train, Xg_test, yg_train, yg_test = train_test_split(
            Xg,
            yg,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=yg,
        )

        pre_g = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    cols,
                )
            ],
            remainder="drop",
        )

        model_g = Pipeline(
            steps=[
                ("pre", pre_g),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="saga",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model_g.fit(Xg_train, yg_train)
        proba_g = model_g.predict_proba(Xg_test)[:, 1]
        auc_g = roc_auc_score(yg_test, proba_g)
        ap_g = average_precision_score(yg_test, proba_g)
        group_results.append((group, len(cols), auc_g, ap_g))

    group_results = sorted(group_results, key=lambda x: x[2], reverse=True)
    for group, ncols, auc_g, ap_g in group_results:
        print(f"{group:6s}  n={ncols:3d}  ROC-AUC={auc_g:.3f}  AP={ap_g:.3f}")

    print("\n=== Random forest top 30 features ===")
    rf_clf = fitted_rf.named_steps["clf"]
    importances = rf_clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:30]

    for rank, idx in enumerate(top_idx, start=1):
        print(f"{rank:02d}. {feature_cols[idx]:30s} importance={importances[idx]:.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()