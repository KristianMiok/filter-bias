import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline


CSV = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv"
RANDOM_STATE = 42
N_BASIN_SPLITS = 10


def get_feature_cols(header):
    prefixes = ("l_CLI", "l_TOP", "l_SOL", "l_LAC", "u_CLI", "u_TOP", "u_SOL", "u_LAC")
    return [c for c in header if c.startswith(prefixes)]


def get_local_cols(feature_cols):
    prefixes = ("l_CLI", "l_TOP", "l_SOL", "l_LAC")
    return [c for c in feature_cols if c.startswith(prefixes)]


def make_model():
    return Pipeline(
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


def eval_random_split(df, y, cols, label):
    idx = np.arange(len(df))
    tr, te = train_test_split(
        idx,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_model()
    model.fit(df.iloc[tr][cols], y.iloc[tr])
    p = model.predict_proba(df.iloc[te][cols])[:, 1]

    auc = roc_auc_score(y.iloc[te], p)
    ap = average_precision_score(y.iloc[te], p)

    print(
        f"{label:35s} random      AUC={auc:.3f}  AP={ap:.3f}  "
        f"train_low={y.iloc[tr].mean():.3f}  test_low={y.iloc[te].mean():.3f}  n_test={len(te)}"
    )

    return auc, ap


def eval_repeated_basin_split(df, y, cols, label):
    if "basin_id" not in df.columns:
        return None

    groups = df["basin_id"].fillna("missing").astype(str)
    rows = []

    for i in range(N_BASIN_SPLITS):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=RANDOM_STATE + i,
        )
        tr, te = next(splitter.split(df, y, groups=groups))

        # Need both classes in train and test
        if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
            continue

        model = make_model()
        model.fit(df.iloc[tr][cols], y.iloc[tr])
        p = model.predict_proba(df.iloc[te][cols])[:, 1]

        rows.append(
            {
                "split": i + 1,
                "auc": roc_auc_score(y.iloc[te], p),
                "ap": average_precision_score(y.iloc[te], p),
                "train_low": y.iloc[tr].mean(),
                "test_low": y.iloc[te].mean(),
                "n_test": len(te),
                "train_basins": groups.iloc[tr].nunique(),
                "test_basins": groups.iloc[te].nunique(),
            }
        )

    if not rows:
        print(f"{label:35s} basin       not enough valid basin splits")
        return None

    res = pd.DataFrame(rows)

    print(
        f"{label:35s} basin mean  AUC={res['auc'].mean():.3f}±{res['auc'].std():.3f}  "
        f"AP={res['ap'].mean():.3f}±{res['ap'].std():.3f}  "
        f"valid_splits={len(res)}/{N_BASIN_SPLITS}"
    )

    return res


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    feature_cols = get_feature_cols(header)
    local_cols = get_local_cols(feature_cols)

    usecols = [
        "Accuracy",
        "Status",
        "basin_id",
    ] + feature_cols

    usecols = [c for c in usecols if c in header]

    print("Loading data...")
    df = pd.read_csv(CSV, usecols=usecols)
    df = df[df["Accuracy"].isin(["High", "Low"])].copy()
    df["is_low"] = (df["Accuracy"] == "Low").astype(int)

    print(f"Rows: {len(df)}")
    print(f"Environmental features: {len(feature_cols)}")
    print(f"Local environmental features: {len(local_cols)}")

    statuses = ["Alien", "Native", "Introduced"]

    all_basin_results = []

    for status in statuses:
        sub = df[df["Status"] == status].copy().reset_index(drop=True)

        if len(sub) < 500:
            print(f"\n=== Status: {status} skipped, too few rows: {len(sub)} ===")
            continue

        y = sub["is_low"]

        print(f"\n=== Status: {status} ===")
        print(f"n={len(sub)}")
        print(f"Low rate={y.mean():.3f}")
        print(sub["Accuracy"].value_counts())

        if y.nunique() < 2:
            print("Only one class present; skipping.")
            continue

        eval_random_split(sub, y, feature_cols, "all environmental")
        eval_random_split(sub, y, local_cols, "local environmental")

        basin_all = eval_repeated_basin_split(sub, y, feature_cols, "all environmental")
        if basin_all is not None:
            basin_all["status"] = status
            basin_all["design"] = "all environmental"
            all_basin_results.append(basin_all)

        basin_local = eval_repeated_basin_split(sub, y, local_cols, "local environmental")
        if basin_local is not None:
            basin_local["status"] = status
            basin_local["design"] = "local environmental"
            all_basin_results.append(basin_local)

    if all_basin_results:
        out = pd.concat(all_basin_results, ignore_index=True)
        out.to_csv("figures/within_status_basin_holdout_results.csv", index=False)

        print("\n=== Within-status repeated basin summary ===")
        summary = (
            out.groupby(["status", "design"])
            .agg(
                auc_mean=("auc", "mean"),
                auc_std=("auc", "std"),
                ap_mean=("ap", "mean"),
                ap_std=("ap", "std"),
                valid_splits=("split", "count"),
                mean_test_low=("test_low", "mean"),
                mean_n_test=("n_test", "mean"),
            )
            .reset_index()
            .sort_values(["status", "auc_mean"], ascending=[True, False])
        )

        print(summary.round(3).to_string(index=False))
        summary.to_csv("figures/within_status_basin_holdout_summary.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
