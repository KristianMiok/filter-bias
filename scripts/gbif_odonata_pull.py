"""
gbif_odonata_pull.py
====================
Full pull of German Odonata WITHOUT a GBIF account, via the search API.

Why stratified (not just page 0->100k): the gate showed GBIF's default sort front-loads
recent iNaturalist records, so a naive pull would be 2025-2026 citizen-science only and the
museum/atlas records (the coarse-coordinate, Low-accuracy end) would be invisible. We pull in
year windows + a dedicated museum-specimen stratum so the full temporal and source spread is
captured. Pre-2008 windows are guaranteed non-iNaturalist (iNaturalist started 2008).

Limitation vs the download API: the search endpoint caps at offset ~100k of the 640k records.
That is fine here -- representativeness matters more than volume, and a stratified ~50-70k is
ample for the diagnostic. Whether the older records actually carry the uncertainty field is an
empirical question the output answers (and reports honestly either way).

Output: reports/odonata_de_occurrences.csv, with year_bin / source_class / basisOfRecord as
metadata covariates, ready for qfbias_diagnose.
"""

import os
import time
import pandas as pd
from pygbif import occurrences as occ

COUNTRY = "DE"
TAXONKEY = 789                                  # GBIF order Odonata (stable)
FAKE = {301.0, 3036.0, 999.0, 9999.0}
THRESHOLD_M = 100                               # High <= 100 m, Low > 100 m (matches crayfish ~27% Low)
CAP_PER_STRATUM = 10000                         # keep runtime sane; dedup handles overlap
OUT = "reports/odonata_de_occurrences.csv"

FIELDS = ["key", "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
          "basisOfRecord", "datasetName", "year", "species", "countryCode"]

STRATA = [
    ("year 1800-1999",          {"year": "1800,1999"}),
    ("year 2000-2009",          {"year": "2000,2009"}),
    ("year 2010-2017",          {"year": "2010,2017"}),
    ("year 2018-2021",          {"year": "2018,2021"}),
    ("year 2022-2023",          {"year": "2022,2023"}),
    ("year 2024-2026",          {"year": "2024,2026"}),
    ("museum specimens (all y)", {"basisOfRecord": "PRESERVED_SPECIMEN"}),
]


def pull(label, cap, **filters):
    rows, offset = [], 0
    count_filters = {k: v for k, v in filters.items() if k in ("year", "basisOfRecord")}
    try:
        n = occ.count(taxonKey=TAXONKEY, country=COUNTRY, isGeoreferenced=True, **count_filters)
    except Exception:
        n = None
    target = cap if n is None else min(n, cap)
    print(f"  {label:<26}: count={n if n is not None else '?':>7}  -> pulling up to {target}")
    while len(rows) < target:
        try:
            r = occ.search(taxonKey=TAXONKEY, country=COUNTRY, hasCoordinate=True,
                           limit=300, offset=offset, **filters)
        except Exception as e:
            print(f"      (offset {offset} failed: {e})")
            break
        res = r.get("results", [])
        if not res:
            break
        for x in res:
            rows.append({k: x.get(k) for k in FIELDS})
        offset += 300
        if r.get("endOfRecords") or offset >= 99000:
            break
        time.sleep(0.15)
    return rows


def year_bin(y):
    if pd.isna(y):
        return "unknown"
    y = int(y)
    if y < 2008:
        return "pre2008_museum_era"
    if y < 2016:
        return "2008-2015"
    if y < 2021:
        return "2016-2020"
    return "2021-2026"


def source_class(basis, dataset):
    d = (dataset if isinstance(dataset, str) else "").lower()
    b = basis if isinstance(basis, str) else ""
    if "inaturalist" in d:
        return "iNaturalist"
    if b == "PRESERVED_SPECIMEN":
        return "museum_specimen"
    return b or "other"


def main():
    all_rows = []
    print("pulling German Odonata, stratified to capture museum + recent records ...")
    for label, filt in STRATA:
        all_rows += pull(label, CAP_PER_STRATUM, **filt)

    df = pd.DataFrame(all_rows)
    print(f"\nraw pulled (with overlap): {len(df)}")
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    before = len(df)
    df = df.drop_duplicates(subset=["species", "decimalLatitude", "decimalLongitude", "year"])
    print(f"after dedup: {len(df)}  (removed {before - len(df)} cross-stratum / cross-dataset dupes)")

    # clean uncertainty: drop fake geocoding values -> NaN
    u = df["coordinateUncertaintyInMeters"].astype(float)
    df["uncertainty_clean"] = u.where(~u.isin(FAKE))
    has = df["uncertainty_clean"].notna()
    print(f"\nusable (non-fake) uncertainty: {has.mean():.1%}  ({int(has.sum())} records)")

    # accuracy label only where uncertainty is present
    df["accuracy_class"] = pd.NA
    df.loc[has, "accuracy_class"] = (df.loc[has, "uncertainty_clean"] <= THRESHOLD_M).map(
        {True: "High", False: "Low"})

    df["year_bin"] = df["year"].map(year_bin)
    df["source_class"] = [source_class(b, d) for b, d in zip(df["basisOfRecord"], df["datasetName"])]

    # --- what the analyzable data looks like ---
    print("\nbasisOfRecord mix (full pull):")
    print(df["basisOfRecord"].value_counts(normalize=True).head(6).to_string())
    print("\nyear_bin mix:")
    print(df["year_bin"].value_counts(normalize=True).to_string())
    if has.any():
        print("\nHigh/Low among records with usable uncertainty:")
        print(df.loc[has, "accuracy_class"].value_counts(normalize=True).to_string())
        # THE confounder check: does accuracy track source/era?
        print("\naccuracy_class x source_class (usable-uncertainty records):")
        print(pd.crosstab(df.loc[has, "source_class"], df.loc[has, "accuracy_class"]).to_string())
        print("\naccuracy_class x year_bin (usable-uncertainty records):")
        print(pd.crosstab(df.loc[has, "year_bin"], df.loc[has, "accuracy_class"]).to_string())

    os.makedirs("reports", exist_ok=True)
    cols = ["key", "species", "decimalLatitude", "decimalLongitude",
            "coordinateUncertaintyInMeters", "uncertainty_clean", "accuracy_class",
            "basisOfRecord", "datasetName", "source_class", "year", "year_bin", "countryCode"]
    df[cols].to_csv(OUT, index=False)
    print(f"\nwrote {len(df)} records -> {OUT}")
    print(f"  of which {int(has.sum())} have usable uncertainty and an accuracy label "
          f"(these are what the diagnostic uses)")
    if has.sum() < 3000:
        print("  ** note: few labelled records; if the museum strata lack uncertainty, the")
        print("     analyzable subset is citizen-science-heavy -- still a valid second taxon, ")
        print("     just a different flavour (continuous citizen-science vs curated binary). **")


if __name__ == "__main__":
    main()
