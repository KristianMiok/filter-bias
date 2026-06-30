"""
compute_niche_breadth.py
========================
Fig 3 panel B mechanism. The FILTER->IPW reweighting sensitivity is a property of niche
breadth, not biogeographic status: reweighting has more leverage where the presence-background
contrast is weak (broad, background-overlapping niches) and less for environmental specialists.
This computes a model-free breadth measure per species and correlates it with the FILTER->IPW
divergence, so panel B can be sorted by the real mechanism (breadth) with status as a secondary
annotation -- and the caption can state the breadth<->sensitivity correlation as a result.

Breadth measure: each environmental axis is standardized (z-score over all records); for each
species, breadth = mean over axes of the variance of that species' presence points. Higher =
broader niche relative to the full environmental envelope.

Output: reports/niche_breadth.csv (species, status, breadth, filter_ipw, ...).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sdm_consequence_multispecies import (
    CSV, SPECIES, get_feature_cols, prepare_presence_background,
)

# FILTER->IPW per-species divergence used in Fig 3 panel B (mean_abs_diff), same order as SPECIES.
# (From make_figures.py; kept here so the correlation is computed against the plotted values.)
FILTER_IPW = {
    "Pacifastacus leniusculus": 0.0073,
    "Astacus astacus": 0.0206,
    "Faxonius limosus": 0.0190,
    "Pontastacus leptodactylus": 0.0092,
    "Procambarus clarkii": 0.0159,
    "Austropotamobius pallipes": 0.0214,
    "Austropotamobius torrentium": 0.0121,
}


def main():
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    feature_cols = get_feature_cols(header)
    extra = [c for c in ["Crayfish_scientific_name", "Status"] if c in header]
    df = pd.read_csv(CSV, usecols=list(dict.fromkeys(extra + feature_cols)))
    print(f"Loaded {len(df)} records, {len(feature_cols)} features.")

    # standardize every env axis over all records (z-score); robust to NaN via nan-aware stats
    X = df[feature_cols].astype(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0).replace(0, np.nan)
    Xz = (X - mu) / sd

    rows = []
    for sp in SPECIES:
        mask = df["Crayfish_scientific_name"] == sp
        if mask.sum() < 20:
            continue
        pres_z = Xz[mask.values]
        # breadth = mean over axes of presence-point variance (ignore all-NaN axes)
        per_axis_var = pres_z.var(axis=0, ddof=1)
        breadth = float(np.nanmean(per_axis_var.values))
        status = df.loc[mask, "Status"].mode().iloc[0] if "Status" in df.columns else "NA"
        rows.append({
            "species": sp,
            "status": status,
            "n_presence": int(mask.sum()),
            "niche_breadth": breadth,
            "filter_ipw": FILTER_IPW.get(sp, np.nan),
        })

    out = pd.DataFrame(rows).sort_values("niche_breadth").reset_index(drop=True)
    pd.set_option("display.width", 160)
    print("\nPer-species niche breadth (sorted):\n")
    print(out.round(4).to_string(index=False))

    # correlation breadth <-> FILTER->IPW sensitivity (the mechanism)
    sub = out.dropna(subset=["filter_ipw"])
    if len(sub) >= 3:
        r_p = np.corrcoef(sub["niche_breadth"], sub["filter_ipw"])[0, 1]
        # Spearman (rank) too, robust to the small n
        rp = pd.Series(sub["niche_breadth"]).rank()
        rq = pd.Series(sub["filter_ipw"]).rank()
        r_s = np.corrcoef(rp, rq)[0, 1]
        print(f"\nbreadth <-> FILTER->IPW sensitivity:  Pearson r = {r_p:.3f}   Spearman rho = {r_s:.3f}")
        print("(positive = broader niches show more reweighting sensitivity -- the mechanism,")
        print(" not status; for the panel B caption.)")

    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/niche_breadth.csv", index=False)
    print("\nwrote reports/niche_breadth.csv")


if __name__ == "__main__":
    main()
