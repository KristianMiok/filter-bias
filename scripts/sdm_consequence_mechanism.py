"""
sdm_consequence_mechanism.py
============================
Tests whether the invasive>native SDM-shift pattern is a MECHANISM or noise.

Hypothesis: invasive species show larger filtering-induced prediction shifts because
their record quality is more strongly environmentally structured (stronger coupling),
which in turn tracks the Alien<->Low-accuracy signal. If so, then across the 7 species
the SDM shift (FILTER vs IPW divergence) should be predicted by per-species coupling,
and invasives should cluster high on both. If the SDM shift is uncorrelated with coupling,
the invasive>native pattern is probably an artifact and should not anchor the paper.

Per species it computes:
  alien_fraction   from Status (the Alien<->Low link)
  low_acc_rate     fraction of Low-accuracy records
  mean_abs_smd     mean |standardized mean difference| of env between {all} and {high}
                   presence  ==  how much filtering shifts THIS species' niche in env space
and joins them to the FILTER<->IPW SDM shift produced by sdm_consequence_strategies.py.

Reads figures/sdm_consequence_strategies_shifts.csv (run that first).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sdm_consequence_multispecies import CSV, SPECIES, get_feature_cols

SHIFTS = "figures/sdm_consequence_strategies_shifts.csv"


def per_species_coupling(df, feature_cols):
    rows = []
    for sp in SPECIES:
        sub = df[(df["Crayfish_scientific_name"] == sp) &
                 (df["Accuracy"].isin(["High", "Low"]))]
        if len(sub) < 20:
            continue
        P = sub[feature_cols]
        H = sub.loc[sub["Accuracy"] == "High", feature_cols]
        sd = P.std().replace(0, np.nan)
        smd = (H.mean() - P.mean()) / sd                    # all -> high, per feature
        rows.append({
            "species": sp,
            "n_presence": len(sub),
            "alien_fraction": float((sub["Status"] != "Native").mean()),
            "low_acc_rate": float((sub["Accuracy"] == "Low").mean()),
            "mean_abs_smd": float(smd.abs().mean()),
        })
    return pd.DataFrame(rows)


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    feature_cols = get_feature_cols(header)
    df = pd.read_csv(CSV, usecols=list(dict.fromkeys(
        ["Crayfish_scientific_name", "Accuracy", "Status"] + feature_cols)))

    print("Status labels present:", sorted(df["Status"].dropna().unique())[:10])
    print("(alien_fraction = fraction with Status != 'Native'; adjust if your label differs)\n")

    coup = per_species_coupling(df, feature_cols)

    shifts = pd.read_csv(SHIFTS)
    key = shifts[(shifts["base"] == "filter") & (shifts["alt"] == "ipw")]
    key = key[["species", "mean_abs_diff", "top10_jaccard"]].rename(
        columns={"mean_abs_diff": "sdm_shift"})

    tab = coup.merge(key, on="species").sort_values("sdm_shift", ascending=False)
    pd.set_option("display.width", 150)
    print(tab.to_string(index=False))

    print("\n=== Does coupling predict the SDM shift? (Spearman across species) ===")
    for col in ["mean_abs_smd", "low_acc_rate", "alien_fraction"]:
        r = spearmanr(tab[col], tab["sdm_shift"]).statistic
        print(f"  {col:16s} vs sdm_shift   rho = {r:+.3f}")

    hi_alien = tab[tab["alien_fraction"] >= 0.5]
    lo_alien = tab[tab["alien_fraction"] < 0.5]
    if len(hi_alien) and len(lo_alien):
        print(f"\nmean SDM shift  alien-dominated species = {hi_alien['sdm_shift'].mean():.4f}  "
              f"(n={len(hi_alien)})")
        print(f"mean SDM shift  native-dominated species = {lo_alien['sdm_shift'].mean():.4f}  "
              f"(n={len(lo_alien)})")

    print("\nReading: if mean_abs_smd (env-space coupling) tracks sdm_shift with strong "
          "positive rho AND alien-dominated species sit high on both, the mechanism is "
          "established and the invasive finding is robust to learner choice. If rho is near "
          "zero, the invasive>native pattern is likely noise -- keep the methods-first framing.")


if __name__ == "__main__":
    main()
