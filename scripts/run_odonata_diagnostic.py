"""
run_odonata_diagnostic.py
=========================
Runs the quality-filtering-bias diagnostic on the German Odonata second taxon.

Three taxon-specific steps before the shared diagnostic:
  1. Drop gross coordinate outliers. The GBIF country=DE filter is on the country FIELD,
     which can disagree with the coordinates (wrong-country / null-island records). We keep
     only the Germany bounding box. (These are coarse georeferencing ERRORS, distinct from
     the positional-uncertainty quality we are studying.)
  2. Build a spatial grid cell id for GroupKFold CV -- Odonata has no basin_id, so a ~50 km
     grid is the basin_id analogue. This keeps the propensity AUC spatially honest (the
     crayfish lesson: random CV inflates it under spatial autocorrelation).
  3. Use the CONTINUOUS coordinateUncertaintyInMeters (fakes already -> NaN) thresholded at
     100 m, with year_bin + source_class as metadata -- the confounder audit that quantifies
     how much of the niche shift is genuinely environmental vs an era/source proxy (the
     cross-taxon parallel to the crayfish status/year question).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from qfbias_diagnose import diagnose_quality_shift

ANNOTATED = "reports/odonata_de_annotated.csv"
ENV_COLS = [f"bio_{i}" for i in range(1, 20)]
LON_MIN, LON_MAX = 5.5, 15.5            # Germany bounding box (generous)
LAT_MIN, LAT_MAX = 47.0, 55.5
GRID_DEG = 0.5                           # ~50 km spatial CV cells


def main():
    df = pd.read_csv(ANNOTATED)
    print(f"loaded {len(df)} annotated records")

    inbox = (df["decimalLongitude"].between(LON_MIN, LON_MAX) &
             df["decimalLatitude"].between(LAT_MIN, LAT_MAX))
    n_out = int((~inbox).sum())
    print(f"dropping {n_out} coordinate outliers outside Germany bbox "
          f"(lon[{LON_MIN},{LON_MAX}] lat[{LAT_MIN},{LAT_MAX}])")
    df = df[inbox].reset_index(drop=True)

    df["grid_id"] = (np.floor(df["decimalLongitude"] / GRID_DEG).astype(int).astype(str)
                     + "_" + np.floor(df["decimalLatitude"] / GRID_DEG).astype(int).astype(str))
    print(f"{df['grid_id'].nunique()} spatial grid cells ({GRID_DEG} deg) for GroupKFold")

    res = diagnose_quality_shift(
        df,
        env_cols=ENV_COLS,
        quality_col="uncertainty_clean",          # fakes already NaN; missing dropped inside
        quality_threshold=100,
        smaller_is_better=True,                    # smaller uncertainty = higher quality
        metadata_cols=["year_bin", "source_class"],
        group_col="grid_id",
    )
    print("\n" + res.summary())
    Path("reports").mkdir(exist_ok=True)
    res.per_axis.to_csv("reports/odonata_per_axis_shift.csv", index=False)
    print("\nwrote reports/odonata_per_axis_shift.csv")


if __name__ == "__main__":
    main()
