import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr


CSV = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv"
SCORES = "figures/accuracy_risk_scores_oof.csv"

RANDOM_STATE = 42
N_BACKGROUND = 20000

SPECIES = [
    "Pacifastacus leniusculus",
    "Astacus astacus",
    "Faxonius limosus",
    "Pontastacus leptodactylus",
    "Procambarus clarkii",
    "Austropotamobius pallipes",
    "Austropotamobius torrentium",
]


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
                    max_depth=14,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def prepare_presence_background(df, species_name):
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
        sample_weight = np.ones(len(train_df), dtype=float)
        pres_mask = train_df["y"].values == 1
        sample_weight[pres_mask] = 1.0 - train_df.loc[pres_mask, "p_low_accuracy"].values
        sample_weight[pres_mask] = np.clip(sample_weight[pres_mask], 0.05, 1.0)

    model.fit(X_train, y_train, rf__sample_weight=sample_weight)

    test_pred = model.predict_proba(test_df[feature_cols])[:, 1]
    predict_pred = model.predict_proba(predict_df[feature_cols])[:, 1]

    return {
        "model": model,
        "test_pred": test_pred,
        "predict_pred": predict_pred,
        "auc": roc_auc_score(test_df["y"], test_pred),
        "ap": average_precision_score(test_df["y"], test_pred),
    }


def summarize_prediction_shift(base_pred, alt_pred):
    rho = spearmanr(base_pred, alt_pred).correlation
    pearson = np.corrcoef(base_pred, alt_pred)[0, 1]

    abs_diff = np.abs(base_pred - alt_pred)

    base_top10 = base_pred >= np.quantile(base_pred, 0.90)
    alt_top10 = alt_pred >= np.quantile(alt_pred, 0.90)

    base_top05 = base_pred >= np.quantile(base_pred, 0.95)
    alt_top05 = alt_pred >= np.quantile(alt_pred, 0.95)

    top10_intersection = np.logical_and(base_top10, alt_top10).sum()
    top10_union = np.logical_or(base_top10, alt_top10).sum()

    top05_intersection = np.logical_and(base_top05, alt_top05).sum()
    top05_union = np.logical_or(base_top05, alt_top05).sum()

    return {
        "spearman": rho,
        "pearson": pearson,
        "mean_abs_diff": abs_diff.mean(),
        "median_abs_diff": np.median(abs_diff),
        "p90_abs_diff": np.quantile(abs_diff, 0.90),
        "p95_abs_diff": np.quantile(abs_diff, 0.95),
        "top10_jaccard": top10_intersection / top10_union if top10_union > 0 else np.nan,
        "top05_jaccard": top05_intersection / top05_union if top05_union > 0 else np.nan,
    }


def species_short(name):
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[1]}"
    return name


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    feature_cols = get_feature_cols(header)

    score_header = pd.read_csv(SCORES, nrows=0).columns.tolist()
    score_cols = [c for c in ["WoCID", "p_low_accuracy", "risk_tertile", "risk_decile"] if c in score_header]

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
    scores = pd.read_csv(SCORES, usecols=score_cols)

    df = df.merge(scores, on="WoCID", how="inner")
    df = df[df["Accuracy"].isin(["High", "Low"])].copy().reset_index(drop=True)

    print(f"Rows after merge: {len(df)}")
    print(f"Local environmental features: {len(feature_cols)}")

    summary_rows = []
    shift_rows = []

    for sp_i, sp in enumerate(SPECIES, start=1):
        print(f"\n=== Species {sp_i}/{len(SPECIES)}: {sp} ===")

        dat, pres, bg = prepare_presence_background(df, sp)

        if len(pres) < 500:
            print(f"Skipping {sp}: too few presences ({len(pres)})")
            continue

        print(f"Presences: {len(pres)}")
        print(f"Background: {len(bg)}")
        print(f"Presence Low rate: {(pres['Accuracy'] == 'Low').mean():.3f}")
        print("Presence risk tertiles:")
        print(pres["risk_tertile"].value_counts(dropna=False).to_string())

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

        strategies = {
            "all": train_all,
            "remove_high_risk": train_remove_high_risk,
            "weighted": train_weighted,
        }

        predict_df = bg.copy().reset_index(drop=True)
        results = {}

        for strategy, train_strategy_df in strategies.items():
            n_pres_train = int((train_strategy_df["y"] == 1).sum())
            n_bg_train = int((train_strategy_df["y"] == 0).sum())

            if n_pres_train < 100:
                print(f"Skipping strategy {strategy}: too few train presences after filtering")
                continue

            res = train_and_predict(
                train_strategy_df,
                test_df,
                predict_df,
                feature_cols,
                strategy=strategy,
                seed=RANDOM_STATE + sp_i,
            )
            results[strategy] = res

            print(
                f"{strategy:18s} train_pres={n_pres_train:5d} "
                f"AUC={res['auc']:.3f} AP={res['ap']:.3f}"
            )

            summary_rows.append(
                {
                    "species": sp,
                    "species_short": species_short(sp),
                    "strategy": strategy,
                    "n_presences_total": len(pres),
                    "presence_low_rate": (pres["Accuracy"] == "Low").mean(),
                    "presence_low_risk_n": int((pres["risk_tertile"] == "low risk").sum()),
                    "presence_medium_risk_n": int((pres["risk_tertile"] == "medium risk").sum()),
                    "presence_high_risk_n": int((pres["risk_tertile"] == "high risk").sum()),
                    "train_presences": n_pres_train,
                    "train_background": n_bg_train,
                    "test_auc": res["auc"],
                    "test_ap": res["ap"],
                    "mean_prediction_background": float(np.mean(res["predict_pred"])),
                    "p90_prediction_background": float(np.quantile(res["predict_pred"], 0.90)),
                    "p95_prediction_background": float(np.quantile(res["predict_pred"], 0.95)),
                }
            )

        if "all" in results:
            base = results["all"]["predict_pred"]

            for alt in ["remove_high_risk", "weighted"]:
                if alt not in results:
                    continue

                shift = summarize_prediction_shift(base, results[alt]["predict_pred"])
                shift.update(
                    {
                        "species": sp,
                        "species_short": species_short(sp),
                        "comparison": f"all_vs_{alt}",
                    }
                )
                shift_rows.append(shift)

                print(
                    f"Shift all vs {alt:16s} "
                    f"top10_jaccard={shift['top10_jaccard']:.3f} "
                    f"top05_jaccard={shift['top05_jaccard']:.3f} "
                    f"mean_abs_diff={shift['mean_abs_diff']:.3f} "
                    f"p95_abs_diff={shift['p95_abs_diff']:.3f}"
                )

    summary = pd.DataFrame(summary_rows)
    shifts = pd.DataFrame(shift_rows)

    summary.to_csv("figures/sdm_consequence_multispecies_summary.csv", index=False)
    shifts.to_csv("figures/sdm_consequence_multispecies_shifts.csv", index=False)

    print("\n=== Multispecies SDM consequence summary ===")
    print(summary.round(3).to_string(index=False))

    print("\n=== Multispecies prediction shifts ===")
    print(shifts.round(3).to_string(index=False))

    print("\n=== Shift summary by comparison ===")
    shift_summary = (
        shifts.groupby("comparison")
        .agg(
            n_species=("species", "nunique"),
            top10_jaccard_mean=("top10_jaccard", "mean"),
            top10_jaccard_min=("top10_jaccard", "min"),
            top10_jaccard_max=("top10_jaccard", "max"),
            top05_jaccard_mean=("top05_jaccard", "mean"),
            mean_abs_diff_mean=("mean_abs_diff", "mean"),
            p95_abs_diff_mean=("p95_abs_diff", "mean"),
        )
        .reset_index()
    )
    print(shift_summary.round(3).to_string(index=False))
    shift_summary.to_csv("figures/sdm_consequence_multispecies_shift_summary.csv", index=False)

    print("\nSaved:")
    print("  figures/sdm_consequence_multispecies_summary.csv")
    print("  figures/sdm_consequence_multispecies_shifts.csv")
    print("  figures/sdm_consequence_multispecies_shift_summary.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
