"""
fig6_quality_map.py
===================
Fig 6 — the geography of coordinate quality. Hexbin maps over Germany built
purely from the survey occurrence tables (no basemap dependencies; the
country's shape emerges from record density itself).

  (a) all seven groups pooled: spatial fraction of imprecise records (>100 m)
  (b) Orchidaceae only: the herbarium-era geography that drives the dangerous
      signature (spatial confounding made visible)

Reads data/survey/*_annotated.csv. Outputs figures/fig6_quality_map.pdf/.png
Run from the repo root:   python scripts/fig6_quality_map.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "survey")
FIG = os.path.join(ROOT, "figures")
THRESH = 100.0

plt.rcParams.update({"font.size": 9, "figure.dpi": 120, "savefig.bbox": "tight"})


def load_all():
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW, "*_annotated.csv"))):
        slug = os.path.basename(path).replace("_annotated.csv", "")
        d = pd.read_csv(path, usecols=["decimalLatitude", "decimalLongitude", "uncertainty_clean"],
                        low_memory=False)
        d["group"] = slug
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.uncertainty_clean.notna()]
    d = d[(d.decimalLongitude.between(4.5, 16.2)) & (d.decimalLatitude.between(46.5, 55.8))]
    d["imprecise"] = (d.uncertainty_clean > THRESH).astype(float)
    return d


def panel(ax, d, title):
    hb = ax.hexbin(d.decimalLongitude, d.decimalLatitude, C=d.imprecise,
                   reduce_C_function=np.mean, gridsize=52, mincnt=8,
                   cmap="magma_r", vmin=0.0, vmax=1.0, linewidths=0.1)
    ax.set_aspect(1.0 / np.cos(np.radians(51.0)))
    ax.set_xlabel("longitude"); ax.set_title(title, loc="left")
    ax.set_xlim(5.0, 15.6); ax.set_ylim(47.0, 55.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return hb


def main():
    os.makedirs(FIG, exist_ok=True)
    d = load_all()
    print(f"records with usable uncertainty in window: {len(d)}")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4), sharey=True)
    hb = panel(axes[0], d, f"(a) all groups pooled (n = {len(d):,})")
    axes[0].set_ylabel("latitude")
    do = d[d.group == "orchidaceae"]
    panel(axes[1], do, f"(b) Orchidaceae (n = {len(do):,})")
    cb = fig.colorbar(hb, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(f"fraction of records with uncertainty > {THRESH:.0f} m")
    fig.savefig(os.path.join(FIG, "fig6_quality_map.pdf"))
    fig.savefig(os.path.join(FIG, "fig6_quality_map.png"), dpi=300)
    print("  fig6_quality_map")


if __name__ == "__main__":
    main()
