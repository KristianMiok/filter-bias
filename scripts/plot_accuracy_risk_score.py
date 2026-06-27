import pandas as pd
import matplotlib.pyplot as plt


SCORES = "figures/accuracy_risk_scores_oof.csv"
DECILES = "figures/accuracy_risk_deciles_summary.csv"
STATUS = "figures/accuracy_risk_by_status.csv"


def main():
    scores = pd.read_csv(SCORES)
    deciles = pd.read_csv(DECILES)
    status = pd.read_csv(STATUS)

    # 1. Actual low rate by environmental risk decile
    plt.figure(figsize=(7, 4))
    plt.plot(deciles["risk_decile"], deciles["actual_low_rate"], marker="o", label="Observed low-accuracy rate")
    plt.plot(deciles["risk_decile"], deciles["mean_predicted_risk"], marker="o", label="Mean predicted risk")
    plt.xlabel("Environmental accuracy-risk decile")
    plt.ylabel("Low-accuracy rate / predicted risk")
    plt.ylim(0, 1)
    plt.xticks(deciles["risk_decile"])
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/accuracy_risk_decile_calibration.png", dpi=300)
    plt.close()

    # 2. Low rate by risk tertile within status
    keep_status = ["Alien", "Native", "Introduced"]
    status = status[status["Status"].isin(keep_status)].copy()

    tertile_order = ["low risk", "medium risk", "high risk"]
    status["risk_tertile"] = pd.Categorical(status["risk_tertile"], categories=tertile_order, ordered=True)
    status = status.sort_values(["Status", "risk_tertile"])

    plt.figure(figsize=(7, 4))
    for s in keep_status:
        sub = status[status["Status"] == s]
        plt.plot(
            sub["risk_tertile"].astype(str),
            sub["actual_low_rate"],
            marker="o",
            label=s,
        )

    plt.xlabel("Environmental accuracy-risk tertile")
    plt.ylabel("Observed low-accuracy rate")
    plt.ylim(0, 1)
    plt.legend(title="Status")
    plt.tight_layout()
    plt.savefig("figures/accuracy_risk_by_status.png", dpi=300)
    plt.close()

    # 3. Distribution of predicted risk for High vs Low records
    plt.figure(figsize=(7, 4))
    for acc in ["High", "Low"]:
        sub = scores[scores["Accuracy"] == acc]
        plt.hist(
            sub["p_low_accuracy"],
            bins=40,
            alpha=0.5,
            density=True,
            label=acc,
        )

    plt.xlabel("Out-of-fold predicted probability of low spatial accuracy")
    plt.ylabel("Density")
    plt.legend(title="Observed accuracy")
    plt.tight_layout()
    plt.savefig("figures/accuracy_risk_score_distribution.png", dpi=300)
    plt.close()

    print("Saved figures:")
    print("  figures/accuracy_risk_decile_calibration.png")
    print("  figures/accuracy_risk_by_status.png")
    print("  figures/accuracy_risk_score_distribution.png")


if __name__ == "__main__":
    main()
