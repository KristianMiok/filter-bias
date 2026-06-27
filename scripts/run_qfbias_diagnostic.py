# scripts/run_qfbias_diagnostic.py
import numpy as np
import pandas as pd
from pathlib import Path
from qfbias_diagnose import diagnose_quality_shift
from build_accuracy_risk_score import CSV

ENV_PREFIXES = ("l_CLI", "l_TOP", "l_SOL", "l_LAC")
header = pd.read_csv(CSV, nrows=0).columns.tolist()
env_cols = [c for c in header if c.startswith(ENV_PREFIXES)]

usecols = ["Accuracy", "Status", "Year_of_record", "basin_id"] + env_cols
df = pd.read_csv(CSV, usecols=usecols)
df = df[df["Accuracy"].isin(["High", "Low"])].copy()
df["high_quality"] = (df["Accuracy"] == "High").astype(int)
df["year_bin"] = pd.cut(df["Year_of_record"],
                        bins=[-np.inf, 2000, 2010, 2020, np.inf],
                        labels=["<=2000", "2001-2010", "2011-2020", ">2020"])

res = diagnose_quality_shift(
    df,
    env_cols=env_cols,
    quality_col="high_quality",
    metadata_cols=["Status", "year_bin"],   # confounder audit: status + the year/batch effect
    group_col="basin_id",                    # spatial CV = your basin-AUC / LOBO mentality
    propensity_pca=15,                       # well-conditioned, convergent propensity + sane weights
)
print("\n" + res.summary())
Path("reports").mkdir(exist_ok=True)
res.per_axis.to_csv("reports/qfbias_per_axis_shift.csv", index=False)