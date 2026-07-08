#!/usr/bin/env python
"""
run_odonata_overpred.py -- compact cross-taxon replicate of the directional
miscalibration finding, on German Odonata (JAE second taxon).

Mirrors the crayfish worked case exactly in mechanism and axis so it is a true
replicate, not the MEE threshold-sweep analysis:

  * benchmark  = ensemble trained on HIGH-accuracy target presences only
                 (coordinate uncertainty <= 100 m), + target-group background.
  * contaminated(L) = benchmark training set + LOW-accuracy target presences,
                 injected so LOW is L% of the training presences (L = 3/10/20,
                 matching the crayfish L3/L10/L20 levels), 30 replicates each.
  * divergence = contaminated - benchmark (>0 = over-prediction), summarised by
                 BENCHMARK-suitability band with a 2.5-97.5% interval across the
                 30 replicates -- identical to run_overpred_ci.py.

Target-group background (all OTHER Odonata species as background) is the standard,
observer-bias-correcting choice for citizen-science GBIF data.

HONEST SCOPE (state this in the supplement): this is an ENV-SPACE / point-based
demonstration -- the prediction domain is the Odonata record point-cloud, not a
geographic raster. It shows the mechanism generalises to a second taxon, biome,
and quality-metric type (continuous uncertainty vs the crayfish binary flag). It
is deliberately compact (Lucian: "compact, not fully instrumented"); there is no
map, which is fine because the second-taxon claim is about generality, not a
regional management case.

Run from the filter_bias repo root:
    python run_odonata_overpred.py
Writes reports/odonata_overpred_band_ci.csv and
       figures/fig_odonata_jae_dose_response.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

ANNOT = Path("reports/odonata_de_annotated.csv")
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True)
FIGS = Path("figures"); FIGS.mkdir(exist_ok=True)

ENV = [f"bio_{i}" for i in range(1, 20)]
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 5.5, 15.5, 47.0, 55.5   # Germany bbox (as the existing pipeline)
BANDS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0001]
EDGE_BAND, CORE_BAND = "[0.1, 0.3)", "[0.7, 1.0)"

TARGET = "Pyrrhosoma nymphula"
LEVELS = (3, 10, 20)          # % of training presences that are low-accuracy
REPS = 30
NBG = 10000                   # target-group background sample size (fixed across all fits)
N_TREES = 300


def load() -> pd.DataFrame:
    df = pd.read_csv(ANNOT)
    inbox = (df["decimalLongitude"].between(LON_MIN, LON_MAX) &
             df["decimalLatitude"].between(LAT_MIN, LAT_MAX))
    df = df[inbox].dropna(subset=ENV).reset_index(drop=True)
    print(f"loaded {len(df)} records in Germany bbox, env-complete")
    return df


def rf_predict(X_tr, y_tr, X_dom, seed):
    rf = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1,
                                random_state=seed, min_samples_leaf=2)
    rf.fit(X_tr, y_tr)
    return rf.predict_proba(X_dom)[:, 1]


def _ci(vals):
    a = np.asarray(vals, float)
    return {"mean": round(float(a.mean()), 4),
            "lo2.5": round(float(np.percentile(a, 2.5)), 4),
            "hi97.5": round(float(np.percentile(a, 97.5)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()

    df = load()
    tgt = df["species"] == args.target
    X_high = df.loc[tgt & (df["accuracy_class"] == "High"), ENV].to_numpy()
    X_low = df.loc[tgt & (df["accuracy_class"] == "Low"), ENV].to_numpy()
    bg_pool = df.loc[~tgt, ENV]
    X_dom = df[ENV].to_numpy()
    n_high = len(X_high)
    print(f"target {args.target!r}: {n_high} High, {len(X_low)} Low presences; "
          f"background pool {len(bg_pool)}; domain {len(X_dom)} points")

    # fixed target-group background (so divergence is attributable to the Low injection)
    X_bg = bg_pool.sample(n=min(NBG, len(bg_pool)), random_state=0).to_numpy()

    # benchmark: High presences + background (deterministic clean reference)
    X_tr = np.vstack([X_high, X_bg])
    y_tr = np.r_[np.ones(n_high), np.zeros(len(X_bg))]
    p_bench = rf_predict(X_tr, y_tr, X_dom, seed=0)
    band = pd.cut(pd.Series(p_bench), BANDS, right=False, include_lowest=True)
    cats = list(band.cat.categories)
    print("benchmark suitability by band (n):",
          {str(b): int((band == b).sum()) for b in cats})

    report = {}
    for L in LEVELS:
        f = L / 100.0
        n_low = int(round(f / (1 - f) * n_high))
        n_low = min(n_low, len(X_low))
        rep_means = {str(b): [] for b in cats}
        rng = np.random.default_rng(L)
        for rep in range(args.reps):
            sel = rng.choice(len(X_low), size=n_low, replace=False)
            X_tr = np.vstack([X_high, X_low[sel], X_bg])
            y_tr = np.r_[np.ones(n_high + n_low), np.zeros(len(X_bg))]
            p_cont = rf_predict(X_tr, y_tr, X_dom, seed=1000 + rep)
            div = pd.Series(p_cont - p_bench)
            gm = div.groupby(band, observed=False).mean()
            for b in cats:
                rep_means[str(b)].append(float(gm.loc[b]))
        report[f"L{L}"] = {"n_low_injected": n_low,
                           "by_benchmark_band": {str(b): {"n": int((band == b).sum()),
                                                          "cellmean_div": _ci(rep_means[str(b)])}
                                                 for b in cats}}
        print(f"  L{L}: injected {n_low} Low; edge {EDGE_BAND} mean div "
              f"{_ci(rep_means[EDGE_BAND])['mean']:+.4f}, core {CORE_BAND} "
              f"{_ci(rep_means[CORE_BAND])['mean']:+.4f}")

    # save tidy CSV
    rows = []
    for L in LEVELS:
        for b, v in report[f"L{L}"]["by_benchmark_band"].items():
            c = v["cellmean_div"]
            rows.append({"level": L, "band": b, "n": v["n"],
                         "mean_div": c["mean"], "lo2.5": c["lo2.5"], "hi97.5": c["hi97.5"]})
    pd.DataFrame(rows).to_csv(REPORTS / "odonata_overpred_band_ci.csv", index=False)

    print("\n" + "=" * 66)
    pprint(report, sort_dicts=False)
    print("=" * 66)
    print(f"\nHEADLINE -- edge {EDGE_BAND} over-prediction, mean [2.5-97.5% / 30 reps]:")
    for L in LEVELS:
        c = report[f"L{L}"]["by_benchmark_band"][EDGE_BAND]["cellmean_div"]
        print(f"  L{L:>2}: {c['mean']:+.4f}  [{c['lo2.5']:+.4f}, {c['hi97.5']:+.4f}]")

    # figure: band-wise dose-response (same style as crayfish panel b)
    mids = {f"[{lo}, {hi})": (lo + min(hi, 1.0)) / 2.0 for lo, hi in zip(BANDS[:-1], BANDS[1:])}
    cmap = matplotlib.colormaps["RdBu"]
    x = np.arange(len(LEVELS))
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(0, color="0.5", lw=0.9, ls="--")
    for b in [str(c) for c in cats]:
        m = np.array([report[f"L{L}"]["by_benchmark_band"][b]["cellmean_div"]["mean"] for L in LEVELS])
        lo = np.array([report[f"L{L}"]["by_benchmark_band"][b]["cellmean_div"]["lo2.5"] for L in LEVELS])
        hi = np.array([report[f"L{L}"]["by_benchmark_band"][b]["cellmean_div"]["hi97.5"] for L in LEVELS])
        ax.errorbar(x, m, yerr=np.vstack([m - lo, hi - m]), marker="o", ms=4, lw=1.6,
                    capsize=3, color=cmap(1.0 - mids.get(b, 0.5)), label=b)
    ax.set_xticks(x); ax.set_xticklabels([f"L{L}\n({L}%)" for L in LEVELS])
    ax.set_xlabel("Low-accuracy contamination (% of training presences)")
    ax.set_ylabel("Over-prediction  (contaminated \u2212 clean benchmark)")
    ax.set_title(f"Odonata cross-taxon check: directional miscalibration\n"
                 f"{args.target}, German GBIF (30-replicate 2.5\u201397.5% intervals)", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, title="benchmark band", title_fontsize=8)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_odonata_jae_dose_response.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / "fig_odonata_jae_dose_response.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {FIGS}/fig_odonata_jae_dose_response.pdf and "
          f"{REPORTS}/odonata_overpred_band_ci.csv")


if __name__ == "__main__":
    main()
