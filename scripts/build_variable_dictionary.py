"""
build_variable_dictionary.py
============================
Task 4: name the environmental axes. Joins the diagnostic's per-axis shift table
(reports/qfbias_per_axis_shift.csv -- the crayfish run) to the GeoFRESH variable
dictionary (var_climate_ver4.xlsx), so the top axes get their real BIOCLIM / SoilGrids /
Hydrography90m meanings, units, and direction (which way high-quality records shift).

Output: reports/variable_dictionary_top_axes.csv -- the mapping table for Fig 2 and the
ecological-interpretation paragraph (the `[describe the over-represented bioclim band]`
marker in Lucian's Discussion).
"""

import pandas as pd
from pathlib import Path

DICT_XLSX = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/var_climate_ver4.xlsx"
PER_AXIS = "reports/qfbias_per_axis_shift.csv"     # crayfish diagnostic output
TOP_N = 25
OUT = "reports/variable_dictionary_top_axes.csv"

# (sheet name, local-code column, definition column, unit column)
SHEETS = [
    ("CLIMATE",    "Local Climate",    "Definition", "Unit"),
    ("SOIL",       "Local Soil",       "Definition", "Unit"),
    ("LAND COVER", "Local Climate",    "Definition", "Unit"),   # header mislabels the col
    ("TOPOGRAPHY", "Local Topography", "Definition (Hydrography90m)", "Unit"),
]


def load_dictionary():
    rows = []
    xl = pd.ExcelFile(DICT_XLSX)
    for sheet, code_col, def_col, unit_col in SHEETS:
        df = xl.parse(sheet)
        # tolerate minor header variations
        cc = next((c for c in df.columns if str(c).strip() == code_col), None)
        dc = next((c for c in df.columns if str(c).strip() == def_col), None)
        uc = next((c for c in df.columns if str(c).strip() == unit_col), None)
        if cc is None or dc is None:
            print(f"  ! sheet '{sheet}': could not find columns {code_col!r}/{def_col!r}; skipped")
            continue
        sub = df[[cc, dc] + ([uc] if uc else [])].copy()
        sub.columns = ["code", "definition"] + (["unit"] if uc else [])
        if not uc:
            sub["unit"] = ""
        sub["group"] = sheet.title()
        sub = sub.dropna(subset=["code"])
        sub["code"] = sub["code"].astype(str).str.strip()
        rows.append(sub[["code", "definition", "unit", "group"]])
    d = pd.concat(rows, ignore_index=True)
    d = d.drop_duplicates(subset="code")
    print(f"dictionary loaded: {len(d)} local variables across {d['group'].nunique()} groups")
    return d


def main():
    per = pd.read_csv(PER_AXIS)
    # the diagnostic writes env_var + SMD_all_to_high (+ KS); keep the strongest axes
    per = per.reindex(per["SMD_all_to_high"].abs().sort_values(ascending=False).index)
    top = per.head(TOP_N).copy()

    dct = load_dictionary()
    merged = top.merge(dct, left_on="env_var", right_on="code", how="left")

    merged["direction_in_high_quality"] = merged["SMD_all_to_high"].apply(
        lambda s: "higher" if s > 0 else "lower")
    out = merged[["env_var", "group", "definition", "unit",
                  "SMD_all_to_high", "direction_in_high_quality"]].rename(
        columns={"SMD_all_to_high": "SMD"})

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    print(f"\nTop {TOP_N} axes by |SMD| (high-quality vs all), named:\n")
    print(out.round(3).to_string(index=False))

    miss = out["definition"].isna().sum()
    if miss:
        print(f"\n  ! {miss} axes did not match the dictionary "
              f"(codes: {out.loc[out['definition'].isna(), 'env_var'].tolist()})")

    Path("reports").mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
