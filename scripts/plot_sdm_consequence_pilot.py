import pandas as pd
import matplotlib.pyplot as plt


SUMMARY = "figures/sdm_consequence_pilot_summary.csv"
SHIFTS = "figures/sdm_consequence_prediction_shifts.csv"

SPECIES_FILES = {
    "Pacifastacus leniusculus": "figures/sdm_consequence_predictions_Pacifastacus_leniusculus.csv",
    "Astacus astacus": "figures/sdm_consequence_predictions_Astacus_astacus.csv",
}


def short_species(name):
    if name == "Pacifastacus leniusculus":
        return "P. leniusculus"
    if name == "Astacus astacus":
        return "A. astacus"
    return name


def strategy_label(s):
    return {
        "all": "All records",
        "remove_high_risk": "Remove high-risk",
        "weighted": "Risk-weighted",
    }.get(s, s)


def comparison_label(s):
    return {
        "all_vs_remove_high_risk": "All vs remove high-risk",
        "all_vs_weighted": "All vs risk-weighted",
    }.get(s, s)


def plot_performance(summary):
    summary = summary.copy()
    summary["species_short"] = summary["species"].map(short_species)
    summary["strategy_label"] = summary["strategy"].map(strategy_label)

    for metric, ylabel, out in [
        ("test_auc", "Test ROC-AUC", "figures/sdm_consequence_test_auc.png"),
        ("test_ap", "Test average precision", "figures/sdm_consequence_test_ap.png"),
    ]:
        pivot = summary.pivot(index="species_short", columns="strategy_label", values=metric)
        pivot = pivot[["All records", "Remove high-risk", "Risk-weighted"]]

        ax = pivot.plot(kind="bar", figsize=(7, 4))
        ax.set_xlabel("Species")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.legend(title="Training strategy")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()


def plot_prediction_shift(shifts):
    shifts = shifts.copy()
    shifts["species_short"] = shifts["species"].map(short_species)
    shifts["comparison_label"] = shifts["comparison"].map(comparison_label)

    for metric, ylabel, out in [
        ("top10_jaccard", "Top 10% suitability overlap\n(Jaccard index)", "figures/sdm_consequence_top10_jaccard.png"),
        ("mean_abs_diff", "Mean absolute prediction difference", "figures/sdm_consequence_mean_abs_diff.png"),
        ("p95_abs_diff", "95th percentile absolute difference", "figures/sdm_consequence_p95_abs_diff.png"),
    ]:
        pivot = shifts.pivot(index="species_short", columns="comparison_label", values=metric)
        pivot = pivot[["All vs remove high-risk", "All vs risk-weighted"]]

        ax = pivot.plot(kind="bar", figsize=(7, 4))
        ax.set_xlabel("Species")
        ax.set_ylabel(ylabel)
        if metric == "top10_jaccard":
            ax.set_ylim(0, 1)
        ax.legend(title="Comparison")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()


def plot_scatter_predictions():
    for species, path in SPECIES_FILES.items():
        df = pd.read_csv(path)
        sp = short_species(species).replace(" ", "_").replace(".", "")

        comparisons = [
            ("pred_remove_high_risk", "Remove high-risk", f"figures/sdm_scatter_all_vs_remove_{sp}.png"),
            ("pred_weighted", "Risk-weighted", f"figures/sdm_scatter_all_vs_weighted_{sp}.png"),
        ]

        for col, label, out in comparisons:
            plt.figure(figsize=(5, 5))
            plt.scatter(df["pred_all"], df[col], s=4, alpha=0.25)
            plt.xlabel("Prediction: all records")
            plt.ylabel(f"Prediction: {label}")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(out, dpi=300)
            plt.close()


def plot_prediction_difference_histograms():
    for species, path in SPECIES_FILES.items():
        df = pd.read_csv(path)
        sp = short_species(species).replace(" ", "_").replace(".", "")

        diffs = [
            ("remove_high_risk", df["pred_all"] - df["pred_remove_high_risk"]),
            ("weighted", df["pred_all"] - df["pred_weighted"]),
        ]

        for label, diff in diffs:
            plt.figure(figsize=(7, 4))
            plt.hist(diff, bins=50)
            plt.xlabel(f"Prediction difference: all - {label}")
            plt.ylabel("Number of background records")
            plt.tight_layout()
            plt.savefig(f"figures/sdm_prediction_difference_{label}_{sp}.png", dpi=300)
            plt.close()


def main():
    summary = pd.read_csv(SUMMARY)
    shifts = pd.read_csv(SHIFTS)

    plot_performance(summary)
    plot_prediction_shift(shifts)
    plot_scatter_predictions()
    plot_prediction_difference_histograms()

    print("Saved figures:")
    print("  figures/sdm_consequence_test_auc.png")
    print("  figures/sdm_consequence_test_ap.png")
    print("  figures/sdm_consequence_top10_jaccard.png")
    print("  figures/sdm_consequence_mean_abs_diff.png")
    print("  figures/sdm_consequence_p95_abs_diff.png")
    print("  figures/sdm_scatter_all_vs_remove_P_leniusculus.png")
    print("  figures/sdm_scatter_all_vs_weighted_P_leniusculus.png")
    print("  figures/sdm_scatter_all_vs_remove_A_astacus.png")
    print("  figures/sdm_scatter_all_vs_weighted_A_astacus.png")
    print("  figures/sdm_prediction_difference_remove_high_risk_P_leniusculus.png")
    print("  figures/sdm_prediction_difference_weighted_P_leniusculus.png")
    print("  figures/sdm_prediction_difference_remove_high_risk_A_astacus.png")
    print("  figures/sdm_prediction_difference_weighted_A_astacus.png")


if __name__ == "__main__":
    main()
