"""
check_strategy_pairs.py
=======================
Read-only. Reads figures/sdm_consequence_strategies_shifts.csv (already produced by
sdm_consequence_strategies.py) and lays out every strategy pair per species against
alien fraction.

Decisive question: does the invasive>native pattern hold for ALL -> FILTER (the genuine
cost of filtering), or only for FILTER -> IPW (the effect of IPW reweighting within the
already-filtered set)? ALL -> FILTER strong with alien fraction => real ecological hook.
ALL -> FILTER flat => the invasive signal is about reweighting, keep methods-first framing.
"""

import pandas as pd
from scipy.stats import spearmanr

SHIFTS = "figures/sdm_consequence_strategies_shifts.csv"

# alien_fraction per species (from sdm_consequence_mechanism.py)
ALIEN = {
    "Pacifastacus leniusculus": 0.98,
    "Faxonius limosus": 0.95,
    "Procambarus clarkii": 0.87,
    "Pontastacus leptodactylus": 0.55,
    "Austropotamobius pallipes": 0.08,
    "Austropotamobius torrentium": 0.00,
    "Astacus astacus": 0.00,
}

PAIRS = [("all", "filter"), ("all", "trust"), ("all", "ipw"), ("filter", "ipw")]


def main():
    s = pd.read_csv(SHIFTS)

    tab = None
    for b, a in PAIRS:
        col = (s[(s["base"] == b) & (s["alt"] == a)][["species", "mean_abs_diff"]]
               .rename(columns={"mean_abs_diff": f"{b}->{a}"}))
        tab = col if tab is None else tab.merge(col, on="species")
    tab["alien_frac"] = tab["species"].map(ALIEN)
    tab = tab.sort_values("all->filter", ascending=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(tab.to_string(index=False))

    print("\nSpearman of each pair's shift vs alien_fraction (across 7 species):")
    for b, a in PAIRS:
        rho = spearmanr(tab["alien_frac"], tab[f"{b}->{a}"]).statistic
        print(f"  {b}->{a:8s}  rho = {rho:+.3f}")

    hi = tab[tab["alien_frac"] >= 0.5]
    lo = tab[tab["alien_frac"] < 0.5]
    print(f"\nmean ALL->FILTER shift   alien-dominated = {hi['all->filter'].mean():.4f} (n={len(hi)})  "
          f"native-dominated = {lo['all->filter'].mean():.4f} (n={len(lo)})")
    print("\nReading: if ALL->FILTER tracks alien_frac (strong +rho, alien mean > native "
          "mean), filtering genuinely distorts invasive SDMs more -> ecological hook is real. "
          "If ALL->FILTER is flat, the invasive signal lives only in IPW reweighting "
          "(FILTER->IPW) -> keep methods-first framing.")


if __name__ == "__main__":
    main()
