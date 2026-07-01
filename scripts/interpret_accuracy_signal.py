import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CSV = os.environ.get("WOC_CSV", "data/crayfish_master.csv")
RANDOM_STATE = 42


def get_feature_cols(header):
    prefixes = ("l_CLI", "l_TOP", "l_SOL", "l_LAC", "u_CLI", "u_TOP", "u_SOL", "u_LAC")
    return [c for c in header if c.startswith(prefixes)]


def feature_family(name):
    if name.startswith("Year_of_record"):
        return "metadata_year"
    if name.startswith("strahler"):
        return "metadata_strahler"
    if name.startswith("distance_m"):
        return "metadata_distance"
    if name.startswith("Status"):
        return "metadata_status"

    for prefix in ["l_CLI", "l_TOP", "l_SOL", "l_LAC", "u_CLI", "u_TOP", "u_SOL", "u_LAC"]:
        if name.startswith(prefix):
            return prefix

    return "other"


def low_rate_table(df, group_col, min_n=50):
    tab = (
        df.groupby(group_col, dropna=False)
        .agg(
            n=("Accuracy", "size"),
            low_rate=("is_low", "mean"),
            low_n=("is_low", "sum"),
        )
        .reset_index()
    )
    tab = tab[tab["n"] >= min_n].copy()
    tab = tab.sort_values("low_rate", ascending=False)
    return tab


def make_model(numeric_cols, categorical_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return model


def get_transformed_feature_names(model, numeric_cols, categorical_cols):
    pre = model.named_steps["pre"]
    names = []

    if numeric_cols:
        names.extend(numeric_cols)

    if categorical_cols:
        cat_pipe = pre.named_transformers_["cat"]
        onehot = cat_pipe.named_steps["onehot"]
        cat_names = onehot.get_feature_names_out(categorical_cols).tolist()
        names.extend(cat_names)

    return names


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    environmental_cols = get_feature_cols(header)

    metadata_numeric = [c for c in ["Year_of_record", "strahler", "distance_m"] if c in header]
    metadata_categorical = [c for c in ["Status"] if c in header]

    usecols = [
        "Accuracy",
        "Year_of_record",
        "Status",
        "strahler",
        "distance_m",
        "Crayfish_scientific_name",
        "basin_id",
    ] + environmental_cols

    usecols = [c for c in usecols if c in header]

    print("Loading data...")
    df = pd.read_csv(CSV, usecols=usecols)
    df = df[df["Accuracy"].isin(["High", "Low"])].copy()
    df["is_low"] = (df["Accuracy"] == "Low").astype(int)

    print(f"Rows: {len(df)}")
    print(df["Accuracy"].value_counts())
    print(f"Overall Low rate: {df['is_low'].mean():.3f}")

    print("\n=== Low rate by Status ===")
    status_tab = low_rate_table(df, "Status", min_n=1)
    print(status_tab.to_string(index=False))

    print("\n=== Low rate by Strahler ===")
    strahler_tab = low_rate_table(df, "strahler", min_n=50)
    print(strahler_tab.sort_values("strahler").to_string(index=False))

    print("\n=== Low rate by distance_m quartile ===")
    df["distance_bin"] = pd.qcut(df["distance_m"], q=4, duplicates="drop")
    distance_tab = low_rate_table(df, "distance_bin", min_n=50)
    print(distance_tab.sort_values("distance_bin").to_string(index=False))

    print("\n=== Low rate by year period ===")
    year_bins = [-np.inf, 1950, 1970, 1990, 2000, 2010, 2020, np.inf]
    year_labels = ["<=1950", "1951-1970", "1971-1990", "1991-2000", "2001-2010", "2011-2020", ">2020"]
    df["year_period"] = pd.cut(df["Year_of_record"], bins=year_bins, labels=year_labels)
    year_tab = low_rate_table(df, "year_period", min_n=50)
    print(year_tab.sort_values("year_period").to_string(index=False))

    print("\n=== Fit environmental + metadata model for interpretation ===")
    numeric_cols = environmental_cols + metadata_numeric
    categorical_cols = metadata_categorical

    y = df["is_low"]
    X = df[numeric_cols + categorical_cols]

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_model(numeric_cols, categorical_cols)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    proba = model.predict_proba(X.iloc[test_idx])[:, 1]

    auc = roc_auc_score(y.iloc[test_idx], proba)
    ap = average_precision_score(y.iloc[test_idx], proba)

    print(f"Random split AUC={auc:.3f}")
    print(f"Random split AP={ap:.3f}")

    names = get_transformed_feature_names(model, numeric_cols, categorical_cols)
    importances = model.named_steps["rf"].feature_importances_

    imp = pd.DataFrame(
        {
            "feature": names,
            "importance": importances,
        }
    )
    imp["family"] = imp["feature"].map(feature_family)
    imp = imp.sort_values("importance", ascending=False)

    print("\n=== Top 40 features ===")
    print(imp.head(40).to_string(index=False))

    family_imp = (
        imp.groupby("family")
        .agg(
            total_importance=("importance", "sum"),
            mean_importance=("importance", "mean"),
            n_features=("feature", "size"),
        )
        .sort_values("total_importance", ascending=False)
        .reset_index()
    )

    print("\n=== Importance by feature family ===")
    print(family_imp.to_string(index=False))

    # Save outputs
    status_tab.to_csv("figures/low_rate_by_status.csv", index=False)
    strahler_tab.to_csv("figures/low_rate_by_strahler.csv", index=False)
    distance_tab.to_csv("figures/low_rate_by_distance_bin.csv", index=False)
    year_tab.to_csv("figures/low_rate_by_year_period.csv", index=False)
    imp.to_csv("figures/accuracy_feature_importances.csv", index=False)
    family_imp.to_csv("figures/accuracy_family_importances.csv", index=False)

    print("\nSaved interpretation tables to figures/")
    print("Done.")


if __name__ == "__main__":
    main()
