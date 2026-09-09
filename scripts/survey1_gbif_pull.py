"""
survey1_gbif_pull.py
====================
SURVEY, step 1 — stratified GBIF pull for the multi-taxon diagnostic survey
(Germany). Generalizes the proven gbif_odonata_pull.py mechanics to N taxon
groups; everything downstream (CHELSA annotation, diagnostic battery) is
group-agnostic.

Pre-registered design (fixed BEFORE looking at any group's diagnostics):
    region DE; strata = six year bands + museum specimens, cap 10k/stratum;
    fake uncertainty values {301, 999, 3036, 9999} -> NaN; High <= 100 m;
    a group enters the battery only if >= 1000 records carry usable
    uncertainty AND both classes hold >= 15% (applied in survey3, not here).

Provenance: per group, datasetKey counts are written to
reports/survey_datasetkeys_<slug>.csv — the input for minting a GBIF
derived-dataset DOI at submission time. Raw occurrence tables go to
data/survey/ (git-ignored).

Run:   python scripts/survey1_gbif_pull.py                    (all groups)
       SURVEY_GROUPS="Aves,Coleoptera" python scripts/...     (subset)
Re-runs skip groups whose output already exists (delete the CSV to force).
Requires: pip install pygbif   and network access to api.gbif.org.
"""
import json
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "survey")
REPORTS = os.path.join(ROOT, "reports")

COUNTRY = "DE"
FAKE = {301.0, 999.0, 3036.0, 9999.0}
THRESHOLD_M = 100
CAP_PER_STRATUM = 10000

GROUPS = [
    ("Odonata",      "ORDER"),
    ("Aves",         "CLASS"),
    ("Amphibia",     "CLASS"),
    ("Lepidoptera",  "ORDER"),
    ("Coleoptera",   "ORDER"),
    ("Orchidaceae",  "FAMILY"),
    ("Asteraceae",   "FAMILY"),
]

FIELDS = ["key", "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
          "basisOfRecord", "datasetName", "datasetKey", "year", "species", "countryCode"]

STRATA = [
    ("year 1800-1999",           {"year": "1800,1999"}),
    ("year 2000-2009",           {"year": "2000,2009"}),
    ("year 2010-2017",           {"year": "2010,2017"}),
    ("year 2018-2021",           {"year": "2018,2021"}),
    ("year 2022-2023",           {"year": "2022,2023"}),
    ("year 2024-2026",           {"year": "2024,2026"}),
    ("museum specimens (all y)", {"basisOfRecord": "PRESERVED_SPECIMEN"}),
]


def resolve_taxonkey(name, rank):
    import requests
    r = requests.get("https://api.gbif.org/v1/species/match",
                     params={"name": name, "rank": rank}, timeout=30)
    m = r.json() if r.ok else {}
    if m.get("matchType") == "NONE" or "usageKey" not in m:
        raise RuntimeError(f"could not resolve {name} ({rank}): {m}")
    print(f"  {name} ({rank}) -> taxonKey {m['usageKey']}  [{m.get('matchType')}]")
    return int(m["usageKey"])


def pull(taxonkey, label, cap, **filters):
    from pygbif import occurrences as occ
    rows, offset = [], 0
    count_filters = {k: v for k, v in filters.items() if k in ("year", "basisOfRecord")}
    try:
        n = occ.count(taxonKey=taxonkey, country=COUNTRY, isGeoreferenced=True, **count_filters)
    except Exception:
        n = None
    target = cap if n is None else min(n, cap)
    print(f"    {label:<26}: count={n if n is not None else '?':>9}  -> pulling up to {target}")
    while len(rows) < target:
        try:
            r = occ.search(taxonKey=taxonkey, country=COUNTRY, hasCoordinate=True,
                           limit=300, offset=offset, **filters)
        except Exception as e:
            print(f"        (offset {offset} failed: {e})"); break
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


def process(df, slug):
    print(f"  raw pulled (with overlap): {len(df)}")
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    before = len(df)
    df = df.drop_duplicates(subset=["species", "decimalLatitude", "decimalLongitude", "year"])
    print(f"  after dedup: {len(df)}  (removed {before - len(df)})")

    u = df["coordinateUncertaintyInMeters"].astype(float)
    df = df.copy()
    df["uncertainty_clean"] = u.where(~u.isin(FAKE))
    has = df["uncertainty_clean"].notna()
    print(f"  usable (non-fake) uncertainty: {has.mean():.1%}  ({int(has.sum())} records)")

    df["accuracy_class"] = pd.NA
    df.loc[has, "accuracy_class"] = (df.loc[has, "uncertainty_clean"] <= THRESHOLD_M).map(
        {True: "High", False: "Low"})
    df["year_bin"] = df["year"].map(year_bin)
    df["source_class"] = [source_class(b, d) for b, d in zip(df["basisOfRecord"], df["datasetName"])]

    if has.any():
        vc = df.loc[has, "accuracy_class"].value_counts(normalize=True)
        print("  High/Low among usable: " + "  ".join(f"{k}={v:.2f}" for k, v in vc.items()))

    # provenance for the derived-dataset DOI
    prov = df.groupby("datasetKey", dropna=False).size().reset_index(name="n_records")
    prov.to_csv(os.path.join(REPORTS, f"survey_datasetkeys_{slug}.csv"), index=False)
    return df


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    only = os.environ.get("SURVEY_GROUPS")
    only = {s.strip() for s in only.split(",")} if only else None

    meta = {}
    for name, rank in GROUPS:
        if only and name not in only:
            continue
        slug = name.lower()
        out = os.path.join(RAW_DIR, f"{slug}_occurrences.csv")
        if os.path.exists(out):
            print(f"[{name}] exists, skipping ({out})"); continue
        print(f"[{name}]")
        tk = resolve_taxonkey(name, rank)
        rows = []
        for label, filt in STRATA:
            rows += pull(tk, label, CAP_PER_STRATUM, **filt)
        if not rows:
            print(f"  !! nothing pulled for {name}; check network / API"); continue
        df = process(pd.DataFrame(rows), slug)
        df.to_csv(out, index=False)
        meta[name] = dict(taxonKey=tk, n=len(df),
                          n_usable=int(df.uncertainty_clean.notna().sum()))
        print(f"  -> {out}  ({len(df)} records)\n")

    if meta:
        with open(os.path.join(REPORTS, "survey_pull_meta.json"), "w") as fh:
            json.dump(dict(country=COUNTRY, threshold_m=THRESHOLD_M,
                           cap_per_stratum=CAP_PER_STRATUM, groups=meta,
                           pulled_at=time.strftime("%Y-%m-%d")), fh, indent=2)
        print("meta -> reports/survey_pull_meta.json")


if __name__ == "__main__":
    main()
