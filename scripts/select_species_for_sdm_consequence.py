import pandas as pd


SCORES = "figures/accuracy_risk_scores_oof.csv"


def main():
    print("Loading OOF accuracy-risk scores...")
    df = pd.read_csv(SCORES)

    needed = [
        "Crayfish_scientific_name",
        "Accuracy",
        "is_low",
        "p_low_accuracy",
        "risk_tertile",
        "risk_decile",
        "Status",
        "Year_of_record",
        "distance_m",
        "strahler",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {SCORES}: {missing}")

    species = df["Crayfish_scientific_name"].fillna("missing")

    rows = []
    for sp, sub in df.groupby(species):
        n = len(sub)
        if n < 100:
            continue

        low_n = int(sub["is_low"].sum())
        high_n = int(n - low_n)

        risk_counts = sub["risk_tertile"].value_counts(dropna=False).to_dict()

        low_risk_n = int(risk_counts.get("low risk", 0))
        med_risk_n = int(risk_counts.get("medium risk", 0))
        high_risk_n = int(risk_counts.get("high risk", 0))

        rows.append(
            {
                "species": sp,
                "n": n,
                "high_n": high_n,
                "low_n": low_n,
                "low_rate": low_n / n,
                "low_risk_n": low_risk_n,
                "medium_risk_n": med_risk_n,
                "high_risk_n": high_risk_n,
                "high_risk_rate": high_risk_n / n,
                "mean_p_low_accuracy": sub["p_low_accuracy"].mean(),
                "median_p_low_accuracy": sub["p_low_accuracy"].median(),
                "status_mode": sub["Status"].mode().iloc[0] if not sub["Status"].mode().empty else "missing",
                "median_year": sub["Year_of_record"].median(),
                "median_distance_m": sub["distance_m"].median(),
                "median_strahler": sub["strahler"].median(),
            }
        )

    out = pd.DataFrame(rows)

    # Candidate criteria:
    # enough total records for a simple SDM-like test,
    # enough low-risk and high-risk records to compare,
    # not all records concentrated in one risk group.
    out["candidate_score"] = (
        (out["n"] >= 500).astype(int)
        + (out["low_risk_n"] >= 100).astype(int)
        + (out["high_risk_n"] >= 100).astype(int)
        + (out["low_n"] >= 50).astype(int)
        + (out["high_n"] >= 50).astype(int)
    )

    out = out.sort_values(
        ["candidate_score", "n", "high_risk_n"],
        ascending=[False, False, False],
    )

    print("\n=== Top candidate species for SDM consequence test ===")
    cols = [
        "species",
        "candidate_score",
        "n",
        "high_n",
        "low_n",
        "low_rate",
        "low_risk_n",
        "medium_risk_n",
        "high_risk_n",
        "high_risk_rate",
        "mean_p_low_accuracy",
        "status_mode",
        "median_year",
        "median_distance_m",
        "median_strahler",
    ]
    print(out[cols].head(30).round(3).to_string(index=False))

    print("\n=== Species with strongest high-risk concentration ===")
    strong = out[out["n"] >= 300].sort_values("high_risk_rate", ascending=False)
    print(strong[cols].head(20).round(3).to_string(index=False))

    print("\n=== Species with balanced low-risk and high-risk records ===")
    balanced = out[
        (out["n"] >= 500)
        & (out["low_risk_n"] >= 100)
        & (out["high_risk_n"] >= 100)
    ].copy()
    balanced["risk_balance"] = balanced[["low_risk_n", "high_risk_n"]].min(axis=1) / balanced[
        ["low_risk_n", "high_risk_n"]
    ].max(axis=1)
    balanced = balanced.sort_values(["risk_balance", "n"], ascending=[False, False])
    print(balanced[cols + ["risk_balance"]].head(20).round(3).to_string(index=False))

    out.to_csv("figures/species_sdm_consequence_candidates.csv", index=False)
    balanced.to_csv("figures/species_sdm_consequence_balanced_candidates.csv", index=False)

    print("\nSaved:")
    print("  figures/species_sdm_consequence_candidates.csv")
    print("  figures/species_sdm_consequence_balanced_candidates.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
