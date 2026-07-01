import pandas as pd
import numpy as np

CSV = "combined_data_true_master.csv"  # pokreni iz Data foldera, ili stavi punu putanju

# 1. učitaj zaglavlje da nađemo feature kolone (l_ i u_), bez učitavanja celog fajla
header = pd.read_csv(CSV, nrows=0).columns.tolist()
local_cols = [c for c in header if c.startswith("l_")]
up_cols    = [c for c in header if c.startswith("u_")]
print(f"local feature kolona: {len(local_cols)}, upstream feature kolona: {len(up_cols)}")

# 2. učitaj samo Accuracy + feature kolone (ne svih 419), i uzorkuj radi brzine
use = ["Accuracy"] + local_cols + up_cols
df = pd.read_csv(CSV, usecols=use)
print(f"ukupno redova: {len(df)}")
df = df.dropna(subset=["Accuracy"])

# 3. standardizovana razlika High vs Low za svaku varijablu
def std_diff(df, cols):
    hi = df[df.Accuracy == "High"]
    lo = df[df.Accuracy == "Low"]
    diffs = []
    for c in cols:
        sd = df[c].std()
        if sd and not np.isnan(sd):
            diffs.append(abs(lo[c].mean() - hi[c].mean()) / sd)
    return np.nanmean(diffs)

m_local = std_diff(df, local_cols)
m_up    = std_diff(df, up_cols)
print(f"\nLOCAL    varijable: prosečna standardizovana High-vs-Low razlika = {m_local:.4f}")
print(f"UPSTREAM varijable: prosečna standardizovana High-vs-Low razlika = {m_up:.4f}")
print(f"odnos upstream/local = {m_up/m_local:.2f}x")

import pandas as pd
import numpy as np

CSV = "combined_data_true_master.csv"  # pokreni iz Data foldera

header = pd.read_csv(CSV, nrows=0).columns.tolist()
local_cols = [c for c in header if c.startswith("l_")]
up_cols    = [c for c in header if c.startswith("u_")]

use = ["Accuracy", "distance_m"] + local_cols + up_cols
df = pd.read_csv(CSV, usecols=use).dropna(subset=["Accuracy"])
print(f"redova: {len(df)}, local: {len(local_cols)}, upstream: {len(up_cols)}")

# headwater = upstream varijable nedostaju (NaN), kao u Paper 9
head_mask = df[up_cols].isna().all(axis=1)
print(f"headwatera (svi u_ NaN): {head_mask.sum()} ({head_mask.mean():.0%})")

def mean_sdiff(sub, cols):
    hi, lo = sub[sub.Accuracy == "High"], sub[sub.Accuracy == "Low"]
    diffs = []
    for c in cols:
        sd = sub[c].std()
        if sd and not np.isnan(sd):
            diffs.append(abs(lo[c].mean() - hi[c].mean()) / sd)
    return np.nanmean(diffs) if diffs else np.nan

# kolone koje merimo: za headwatere upstream je NaN, pa tamo merimo local;
# za ostale merimo i jedno i drugo
hi_dist = df[df.distance_m > df.distance_m.quantile(0.80)]

print("\n--- LOCAL varijable, High-vs-Low razlika po podskupu ---")
print(f"  svi zapisi:                 {mean_sdiff(df, local_cols):.4f}")
print(f"  velika snap-distanca (20%): {mean_sdiff(hi_dist, local_cols):.4f}")
print(f"  headwateri:                 {mean_sdiff(df[head_mask], local_cols):.4f}")
print(f"  ne-headwateri:              {mean_sdiff(df[~head_mask], local_cols):.4f}")

print("\n--- UPSTREAM varijable (samo ne-headwateri, jer headwateri imaju NaN) ---")
print(f"  svi ne-headwateri:          {mean_sdiff(df[~head_mask], up_cols):.4f}")
hi_dist_nonhead = df[(df.distance_m > df.distance_m.quantile(0.80)) & (~head_mask)]
print(f"  velika snap-distanca:       {mean_sdiff(hi_dist_nonhead, up_cols):.4f}")