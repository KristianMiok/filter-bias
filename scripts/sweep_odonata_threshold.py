"""
sweep_odonata_threshold.py
==========================
Sweeps the High/Low uncertainty threshold for the German Odonata data and reports how the
filtering bias behaves as the cut moves. The continuous coordinateUncertaintyInMeters is the
advantage Odonata has over the crayfish binary label, and this uses it directly: if the
energy-distance significance and the environmental residual are stable across thresholds,
then the bias is induced WHEREVER the line is drawn -- i.e. the threshold is a consequential
choice, which is the argument for weighting over a hard filter.

Runs on the already-prepared reports/odonata_de_annotated.csv (no new download).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from qfbias_diagnose import diagnose_quality_shift

ANNOTATED = "reports/odonata_de_annotated.csv"
ENV_COLS = [f"bio_{i}" for i in range(1, 20)]
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 5.5, 15.5, 47.0, 55.5
GRID_DEG = 0.5
THRESHOLDS_M = [25, 50, 100, 150, 250, 500]     # add more here if wanted


def main():
    df = pd.read_csv(ANNOTATED)
    inbox = (df["decimalLongitude"].between(LON_MIN, LON_MAX) &
             df["decimalLatitude"].between(LAT_MIN, LAT_MAX))
    df = df[inbox].reset_index(drop=True)
    df["grid_id"] = (np.floor(df["decimalLongitude"] / GRID_DEG).astype(int).astype(str)
                     + "_" + np.floor(df["decimalLatitude"] / GRID_DEG).astype(int).astype(str))
    print(f"{len(df)} records, {df['grid_id'].nunique()} grid cells; "
          f"sweeping {len(THRESHOLDS_M)} thresholds ...\n")

    rows = []
    for T in THRESHOLDS_M:
        res = diagnose_quality_shift(
            df, env_cols=ENV_COLS, quality_col="uncertainty_clean",
            quality_threshold=T, smaller_is_better=True,
            metadata_cols=["year_bin", "source_class"], group_col="grid_id",
            n_perm=120,
        )
        top = res.per_axis.iloc[0]
        rows.append({
            "thresh_m": T,
            "retain%": round(100 * res.retain_rate, 1),
            "n_high": res.n_high, "n_low": res.n_low,
            "energy_d": round(res.energy_distance, 4),
            "energy_p": round(res.energy_p, 4),
            "auc_env": round(res.auc_env, 3),
            "auc_meta": round(res.auc_meta, 3),
            "auc_e+m": round(res.auc_env_meta, 3),
            "incr_env": round(res.incr_env_over_meta, 3),
            "ESS%": round(100 * res.ess_frac, 1),
            "top_axis": top["env_var"],
            "top_SMD": round(top["SMD_all_to_high"], 3),
        })
        print(f"  threshold {T:>4} m : retain {100 * res.retain_rate:4.1f}%  "
              f"energy_p={res.energy_p:.4f}  incr_env={res.incr_env_over_meta:+.3f}  "
              f"ESS={res.ess_frac:.1%}  top={top['env_var']}({top['SMD_all_to_high']:+.2f})")

    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 175)
    print("\n=== Threshold sweep: is the filtering bias stable across the cut? ===")
    print(tab.to_string(index=False))
    Path("reports").mkdir(exist_ok=True)
    tab.to_csv("reports/odonata_threshold_sweep.csv", index=False)
    print("\nwrote reports/odonata_threshold_sweep.csv")
    print("\nReading: if energy_p stays < 0.05 and incr_env stays positive across thresholds,")
    print("the filtering bias holds WHEREVER the line is drawn -> the threshold is a")
    print("consequential choice, which is the case for weighting over a hard cut.")


if __name__ == "__main__":
    main()
