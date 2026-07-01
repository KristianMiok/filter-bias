import pandas as pd
import numpy as np

CSV = "combined_data_true_master.csv"  # pokreni iz Data foldera

header = pd.read_csv(CSV, nrows=0).columns.tolist()
local_cols = [c for c in header if c.startswith("l_")]
up_cols    = [c for c in header if c.startswith("u_")]

use = ["Accuracy", "strahler"] + local_cols + up_cols
df = pd.read_csv(CSV, usecols=use).dropna(subset=["Accuracy", "strahler"])
print(f"redova: {len(df)}")

def mean_sdiff(sub, cols):
    hi, lo = sub[sub.Accuracy == "High"], sub[sub.Accuracy == "Low"]
    if len(hi) < 30 or len(lo) < 30:
        return np.nan
    diffs = []
    for c in cols:
        sd = sub[c].std()
        if sd and not np.isnan(sd):
            diffs.append(abs(lo[c].mean() - hi[c].mean()) / sd)
    return np.nanmean(diffs) if diffs else np.nan

print("\n--- LOCAL varijable, High-vs-Low razlika po Strahler redu ---")
for s in range(1, 9):
    sub = df[df.strahler == s]
    print(f"  strahler {s}: {mean_sdiff(sub, local_cols):.4f}  (n={len(sub)})")

print("\n--- UPSTREAM varijable, High-vs-Low razlika po Strahler redu ---")
for s in range(1, 9):
    sub = df[df.strahler == s]
    print(f"  strahler {s}: {mean_sdiff(sub, up_cols):.4f}  (n={len(sub)})")

# direktno poređenje: headwateri (s==1) vs ostali
head = df[df.strahler == 1]
rest = df[df.strahler > 1]
print("\n--- Headwateri (strahler 1) vs ostali ---")
print(f"  LOCAL    headwateri: {mean_sdiff(head, local_cols):.4f}   ostali: {mean_sdiff(rest, local_cols):.4f}")
print(f"  UPSTREAM headwateri: {mean_sdiff(head, up_cols):.4f}   ostali: {mean_sdiff(rest, up_cols):.4f}")