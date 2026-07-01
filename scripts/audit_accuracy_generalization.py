import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CSV = os.environ.get("WOC_CSV", "data/crayfish_master.csv")

RANDOM_STATE = 42
N_REPEATED_BASIN_SPLITS = 20


def get_feature_cols(header):
    prefixes = ("l_CLI", "l_TOP", "l_SOL", "l_LAC", "u_CLI", "u_TOP", "u_SOL", "u_LAC")
    return [c for c in header if c.startswith(prefixes)]


def get_local_feature_cols(feature_cols):
    prefixes = ("l_CLI", "l_TOP", "l_SOL", "l_LAC")
    return [c for c in feature_cols if c.startswith(prefixes)]


def make_rf_model(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(
        steps=[
            ("pre", pre),
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


def evaluate_split(name, df, y, numeric_cols, categorical_cols, train_idx, test_idx):
    X_train = df.iloc[train_idx][numeric_cols + categorical_cols]
    X_test = df.iloc[test_idx][numeric_cols + categorical_cols]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = make_rf_model(numeric_cols, categorical_cols)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print(f"{name:42s}  AUC={auc:.3f}  AP={ap:.3f}  "
          f"train_low={y_train.mean():.3f}  test_low={y_test.mean():.3f}  "
          f"n_test={len(y_test)}")

    return auc, ap


def make_random_split(df, y):
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return train_idx, test_idx


def make_group_split(df, y, group_col, random_state=RANDOM_STATE):
    groups = df[group_col].fillna("missing").astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, y, groups=groups))
    return train_idx, test_idx, groups.iloc[train_idx].nunique(), groups.iloc[test_idx].nunique()
def repeated_basin_holdout(df, y, designs):
    if "basin_id" not in df.columns:
        return

    print(f"\n=== Repeated basin holdout ({N_REPEATED_BASIN_SPLITS} splits) ===")
    rows = []

    for i in range(N_REPEATED_BASIN_SPLITS):
        seed = RANDOM_STATE + i
        train_idx, test_idx, n_train_groups, n_test_groups = make_group_split(
            df,
            y,
            "basin_id",
            random_state=seed,
        )

        for design_name, numeric_cols, categorical_cols in designs:
            auc, ap = evaluate_split(
                f"split {i + 1:02d} | {design_name}",
                df,
                y,
                numeric_cols,
                categorical_cols,
                train_idx,
                test_idx,
            )
            rows.append(
                {
                    "split": i + 1,
                    "design": design_name,
                    "auc": auc,
                    "ap": ap,
                    "train_basins": n_train_groups,
                    "test_basins": n_test_groups,
                    "test_low_rate": y.iloc[test_idx].mean(),
                    "n_test": len(test_idx),
                }
            )

    res = pd.DataFrame(rows)

    print("\n--- Repeated basin holdout summary ---")
    summary = (
        res.groupby("design")
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auc_min=("auc", "min"),
            auc_max=("auc", "max"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
        )
        .sort_values("auc_mean", ascending=False)
    )
    print(summary.round(3))

    pivot = res.pivot(index="split", columns="design", values="auc")
    if {"metadata only", "environmental all", "environmental + metadata"}.issubset(pivot.columns):
        pivot["env_minus_metadata"] = pivot["environmental all"] - pivot["metadata only"]
        pivot["envplus_minus_metadata"] = pivot["environmental + metadata"] - pivot["metadata only"]
        pivot["envplus_minus_env"] = pivot["environmental + metadata"] - pivot["environmental all"]

        print("\n--- Incremental AUC over repeated basin holdouts ---")
        for col in ["env_minus_metadata", "envplus_minus_metadata", "envplus_minus_env"]:
            vals = pivot[col].dropna()
            print(
                f"{col:26s} mean={vals.mean():+.3f}  std={vals.std():.3f}  "
                f"min={vals.min():+.3f}  max={vals.max():+.3f}  "
                f"positive_splits={(vals > 0).sum()}/{len(vals)}"
            )

    out = "figures/repeated_basin_holdout_results.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved detailed repeated-basin results to {out}")


def evaluate_designs_for_split(split_name, df, y, train_idx, test_idx, designs):
    print(f"\n=== {split_name} ===")
    for design_name, numeric_cols, categorical_cols in designs:
        evaluate_split(
            design_name,
            df,
            y,
            numeric_cols,
            categorical_cols,
            train_idx,
            test_idx,
        )


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    feature_cols = get_feature_cols(header)
    local_cols = get_local_feature_cols(feature_cols)

    metadata_numeric = [c for c in ["Year_of_record", "strahler", "distance_m"] if c in header]
    metadata_categorical = [c for c in ["Status"] if c in header]

    usecols = [
        "Accuracy",
        "basin_id",
        "Crayfish_scientific_name",
    ] + feature_cols + metadata_numeric + metadata_categorical
    usecols = [c for c in usecols if c in header]

    print("Loading data...")
    df = pd.read_csv(CSV, usecols=usecols)
    df = df[df["Accuracy"].isin(["High", "Low"])].copy().reset_index(drop=True)

    y = (df["Accuracy"] == "Low").astype(int)

    print(f"Rows: {len(df)}")
    print(f"Environmental features: {len(feature_cols)}")
    print(f"Local environmental features: {len(local_cols)}")
    print(f"Metadata numeric: {metadata_numeric}")
    print(f"Metadata categorical: {metadata_categorical}")
    print(df["Accuracy"].value_counts())
    print(f"Overall Low rate: {y.mean():.3f}")

    designs = [
        ("metadata only", metadata_numeric, metadata_categorical),
        ("environmental all", feature_cols, []),
        ("local environmental only", local_cols, []),
        ("environmental + metadata", feature_cols + metadata_numeric, metadata_categorical),
    ]

    # 1. Ordinary random row split
    train_idx, test_idx = make_random_split(df, y)
    evaluate_designs_for_split("Random row split", df, y, train_idx, test_idx, designs)

    # 2. Basin holdout split
    if "basin_id" in df.columns:
        train_idx, test_idx, n_train_groups, n_test_groups = make_group_split(df, y, "basin_id")
        evaluate_designs_for_split("Basin holdout split", df, y, train_idx, test_idx, designs)
        print(f"Basin groups: train={n_train_groups}, test={n_test_groups}")

        repeated_basin_holdout(df, y, designs)

    # 3. Species holdout split
    if "Crayfish_scientific_name" in df.columns:
        species = df["Crayfish_scientific_name"].fillna("missing").astype(str)
        counts = species.value_counts()
        keep_species = counts[counts >= 100].index
        keep = species.isin(keep_species)

        dfs = df[keep].copy().reset_index(drop=True)
        ys = (dfs["Accuracy"] == "Low").astype(int)

        train_idx, test_idx, n_train_groups, n_test_groups = make_group_split(
            dfs,
            ys,
            "Crayfish_scientific_name",
        )
        evaluate_designs_for_split("Species holdout split", dfs, ys, train_idx, test_idx, designs)
        print(f"Species groups: train={n_train_groups}, test={n_test_groups}")

    print("\nDone.")


if __name__ == "__main__":
    main()