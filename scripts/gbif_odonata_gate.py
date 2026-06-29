"""
gbif_odonata_gate.py
====================
Cheap go/no-go check BEFORE any heavy work on the second taxon.

Decision (made for this run): taxon = Odonata, region = Germany (DE), source = GBIF,
quality field = coordinateUncertaintyInMeters (continuous).

It answers three questions that decide whether the data supports the analysis:
  1. How many georeferenced Odonata records exist for the region?
  2. Of those, how many actually carry coordinateUncertaintyInMeters (the quality field),
     after removing the known fake geocoding values (301, 3036, 999, 9999)?
  3. Does a sensible accuracy threshold give a non-trivial Low fraction (as the crayfish
     Accuracy label did), or is the field degenerate?

Uses only search/count endpoints -> NO GBIF account needed. A later full download will.
"""

import time
import pandas as pd
from pygbif import occurrences as occ, species

COUNTRY = "DE"
FAKE_UNCERTAINTY = {301.0, 3036.0, 999.0, 9999.0}     # geocoding artefacts, not real values
THRESHOLDS_M = [30, 100, 250]                           # candidate High/Low cutoffs
SAMPLE_TARGET = 6000                                    # records to inspect, spread across the set


def resolve_odonata_key():
    key = 789  # GBIF backbone taxonKey for the order Odonata (stable)
    print(f"using Odonata order taxonKey {key}")
    return key


def windowed_sample(key, georef):
    """Pull ~SAMPLE_TARGET records spread across the dataset (less order-biased than first-N)."""
    ceiling = min(georef, 90000)                        # GBIF search offset hard limit ~100k
    n_windows = max(1, SAMPLE_TARGET // 300)
    step = max(300, ceiling // n_windows)
    offsets = list(range(0, ceiling, step))[:n_windows]
    rows = []
    for off in offsets:
        try:
            r = occ.search(taxonKey=key, country=COUNTRY, hasCoordinate=True,
                           limit=300, offset=off)
        except Exception as e:
            print(f"  (window at offset {off} failed: {e})")
            continue
        for x in r.get("results", []):
            rows.append({
                "uncertainty": x.get("coordinateUncertaintyInMeters"),
                "basisOfRecord": x.get("basisOfRecord"),
                "year": x.get("year"),
                "species": x.get("species"),
                "dataset": x.get("datasetName"),
            })
        time.sleep(0.2)
    return pd.DataFrame(rows)


def main():
    key = resolve_odonata_key()

    total = occ.count(taxonKey=key, country=COUNTRY)
    georef = occ.count(taxonKey=key, country=COUNTRY, isGeoreferenced=True)
    print(f"\nOdonata in {COUNTRY}: {total:,} total records, {georef:,} with coordinates")
    if georef < 5000:
        print("  ** low georeferenced count; a larger region (e.g. Europe) may be needed **")

    print(f"\npulling a representative sample (~{SAMPLE_TARGET}) spread across the set ...")
    df = windowed_sample(key, georef)
    print(f"sample size: {len(df)}")
    if df.empty:
        raise SystemExit("No records returned; stop and check the query.")

    # --- quality field population ---
    has_unc = df["uncertainty"].notna()
    print(f"\ncoordinateUncertaintyInMeters populated: {has_unc.mean():.1%} of sample")

    unc = df.loc[has_unc, "uncertainty"].astype(float)
    fake = unc.isin(FAKE_UNCERTAINTY)
    print(f"  fake geocoding values (301/3036/999/9999): {fake.mean():.1%} of populated")
    clean = unc[~fake]
    usable_frac = len(clean) / len(df)
    print(f"  usable (non-fake) uncertainty: {len(clean)} records = {usable_frac:.1%} of sample")
    if len(clean):
        print(f"  uncertainty (m): median={clean.median():.0f}  "
              f"p25={clean.quantile(.25):.0f}  p75={clean.quantile(.75):.0f}  "
              f"p95={clean.quantile(.95):.0f}  max={clean.max():.0f}")

    # --- High/Low balance at candidate thresholds ---
    print("\nHigh/Low balance among records with usable uncertainty:")
    balances = {}
    for t in THRESHOLDS_M:
        low = float((clean > t).mean()) if len(clean) else float("nan")
        balances[t] = low
        print(f"  threshold {t:>4}m :  High={1 - low:5.1%}   Low={low:5.1%}")

    # --- record-type mix (museum vs observation drives the accuracy spread) ---
    print("\nbasisOfRecord mix (sample):")
    print(df["basisOfRecord"].value_counts(normalize=True).head(6).to_string())
    print("\ntop datasets (sample):")
    print(df["dataset"].value_counts(normalize=True).head(5).to_string())
    yr = df["year"].dropna()
    if len(yr):
        print(f"\nyear range: {int(yr.min())}\u2013{int(yr.max())}  (median {int(yr.median())})")

    # --- verdict ---
    print("\n" + "=" * 60)
    low100 = balances.get(100, float("nan"))
    ok_pop = usable_frac >= 0.20
    ok_bal = 0.05 <= low100 <= 0.80
    if ok_pop and ok_bal:
        print("VERDICT: GO. Enough records carry usable uncertainty, and the 100m")
        print("threshold gives a non-trivial Low fraction. Proceed to full pull + env join.")
    elif not ok_pop:
        print("VERDICT: MARGINAL on field coverage \u2014 too few records carry usable")
        print("uncertainty. Consider widening to Europe, or accept a smaller clean subset.")
    else:
        print("VERDICT: MARGINAL on balance \u2014 the threshold is degenerate (almost all")
        print("High or all Low). Try a different threshold or report across thresholds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
