"""
survey3_battery.py
==================
SURVEY, step 3 — the diagnostic battery across taxon groups, plus the two
survey-level tests the framework makes possible WITHOUT ground truth:

  (i)  SIGNATURE PLACEMENT: where do real datasets sit in the observable
       (AUC, ESS, Brier, dess) space, against the simulation's regime table?
       -> prevalence of the dangerous signature "in the wild".
  (ii) FORMULA TEST ACROSS DATASETS: the contamination scale of each dataset is
       1 - mean varrho(u) (layer autocorrelation at each Low record's own
       uncertainty radius, from the cached CHELSA stack). Prediction: the
       obs-vs-calib effective-coefficient gap grows with 1 - varrho_bar(u);
       datasets whose uncertainty sits below the layer correlation length
       cannot show contamination, whatever their coupling.

Units: each group pooled (primary, pre-registered) + species with >= 1500
usable records inside groups (secondary). Selection: >= 1000 usable records
and both classes >= 15%. Diagnostics only — no SDMs.

Outputs
    reports/survey_battery.csv      one row per unit
    reports/survey_rho_curve.csv    layer variogram varrho(h)
    reports/survey_verdict.txt

Run:   python scripts/survey3_battery.py     (after survey1 + survey2)
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "survey")
CACHE = os.path.join(ROOT, "data", "chelsa_de_window.npz")
REPORTS = os.path.join(ROOT, "reports")

BIOS = [f"bio_{i}" for i in range(1, 20)]
MIN_USABLE, MIN_CLASS_FRAC, MIN_SP = 1000, 0.15, 1500
POOL_CAP_KM, K_DISK = 25.0, 24
BLOCK_KM = 50.0
RNG = np.random.default_rng(2026)

SIM_REFERENCE = [  # from exp0c (480 randomized runs): mean observables by regime
    ("away_HQ (sim)", 0.933, 0.225, None, 0.017),
    ("iso (sim)", 0.778, 0.540, None, 0.099),
    ("toward_HQ (sim)", 0.695, 0.706, None, 0.209),
]


def project_xy(lat, lon):
    lat0 = np.nanmean(lat)
    return np.stack([lon * 111.320 * np.cos(np.radians(lat0)), lat * 110.540], axis=1)


def blocks_of(xy, size_km=BLOCK_KM):
    b = np.floor(xy / size_km).astype(int)
    return b[:, 0] * 100000 + b[:, 1]


def oof_propensity(X, r, groups):
    p = np.full(len(r), np.nan)
    gk = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gk.split(X, r, groups):
        if len(np.unique(r[tr])) < 2:
            p[te] = r[tr].mean(); continue
        m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000)
        m.fit(X[tr], r[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p


def ess_frac(w):
    return float((w.sum() ** 2) / ((w ** 2).sum() + 1e-12) / len(w))


def eff_coef(p, axis):
    z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, None))
    return float(np.polyfit(axis, z, 1)[0])


def layer_variogram(stack, meta, lags_km=(0.5, 1, 2, 5, 10, 25, 50), n_pairs=250000):
    """varrho(h) of the standardized 19-layer stack, from random pixel pairs."""
    L, H, W = stack.shape
    Z = stack.reshape(L, -1)
    mu = np.nanmean(Z, axis=1, keepdims=True); sd = np.nanstd(Z, axis=1, keepdims=True) + 1e-12
    Z = (Z - mu) / sd
    valid = np.isfinite(Z).all(axis=0)
    idx_valid = np.where(valid)[0]
    km_per_px_y = abs(meta["dy"]) * 110.54
    km_per_px_x = abs(meta["dx"]) * 111.32 * np.cos(np.radians(51.0))
    i = RNG.choice(idx_valid, n_pairs); j = RNG.choice(idx_valid, n_pairs)
    ri, ci = np.divmod(i, W); rj, cj = np.divmod(j, W)
    dist = np.sqrt(((ri - rj) * km_per_px_y) ** 2 + ((ci - cj) * km_per_px_x) ** 2)
    prod = np.nanmean(Z[:, i] * Z[:, j], axis=0)
    rows, edges = [], [0.0] + list(lags_km)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist > lo) & (dist <= hi)
        rows.append(dict(lag_km=hi, n_pairs=int(m.sum()),
                         rho=float(np.nanmean(prod[m])) if m.sum() > 100 else np.nan))
    return pd.DataFrame(rows)


def rho_at(u_km, curve):
    """Interpolate varrho at radius u (km); below the first lag -> 1.0."""
    lags = curve.lag_km.values; rho = curve.rho.values
    ok = np.isfinite(rho)
    return float(np.interp(u_km, np.concatenate([[0.0], lags[ok]]),
                           np.concatenate([[1.0], rho[ok]])))


def disk_mean_env(stack, meta, lon, lat, u_km):
    """Monte-Carlo mean of every layer over the uncertainty disk."""
    out = np.full((len(lon), stack.shape[0]), np.nan, np.float32)
    for k in range(len(lon)):
        u = min(u_km[k], POOL_CAP_KM)
        if u <= 0.5:
            continue
        ang = RNG.uniform(0, 2 * np.pi, K_DISK)
        rad = u * np.sqrt(RNG.uniform(0, 1, K_DISK))
        dlat = rad * np.sin(ang) / 110.54
        dlon = rad * np.cos(ang) / (111.32 * np.cos(np.radians(lat[k])))
        col = np.floor((lon[k] + dlon - meta["x0"]) / meta["dx"]).astype(int)
        row = np.floor((lat[k] + dlat - meta["y0"]) / meta["dy"]).astype(int)
        H, W = stack.shape[1:]
        ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        if ok.sum() < 4:
            continue
        out[k] = np.nanmean(stack[:, row[ok], col[ok]], axis=1)
    return out


def battery(name, unit, g, curve, stack, meta, rows):
    r = (g.uncertainty_clean <= 100.0).astype(int).values
    hi = r == 1
    xy = project_xy(g.decimalLatitude.values, g.decimalLongitude.values)
    X = g[BIOS].values
    sc = StandardScaler().fit(X); Z = sc.transform(X)
    pc1 = PCA(n_components=1, random_state=0).fit_transform(Z)[:, 0]
    Xd = np.hstack([Z, Z ** 2])
    p_obs = oof_propensity(Xd, r, blocks_of(xy))

    # calib: Low records replaced by disk-mean env at their own radius
    low = ~hi
    Xc = X.copy()
    dm = disk_mean_env(stack, meta,
                       g.decimalLongitude.values[low], g.decimalLatitude.values[low],
                       g.uncertainty_clean.values[low] / 1000.0)
    okd = np.isfinite(dm).all(axis=1)
    li = np.where(low)[0]
    Xc[li[okd]] = dm[okd]
    Zc = sc.transform(Xc)
    p_cal = oof_propensity(np.hstack([Zc, Zc ** 2]), r, blocks_of(xy))

    w_o = 1 / np.clip(p_obs[hi], 1e-3, None); w_o = np.clip(w_o, None, np.percentile(w_o, 99))
    w_c = 1 / np.clip(p_cal[hi], 1e-3, None); w_c = np.clip(w_c, None, np.percentile(w_c, 99))
    ub = np.clip(g.uncertainty_clean.values[low] / 1000.0, 0, POOL_CAP_KM)
    rho_bar = float(np.mean([rho_at(u, curve) for u in ub])) if low.any() else 1.0
    smd0 = float(np.abs(Z[hi].mean(0) - Z.mean(0)).mean())
    smd1 = float(np.abs((Z[hi] * (w_o / w_o.sum())[:, None]).sum(0) - Z.mean(0)).mean())

    rows.append(dict(
        name=name, unit=unit, n=len(g), retain=float(hi.mean()),
        med_unc_low_m=float(np.median(g.uncertainty_clean.values[low])) if low.any() else np.nan,
        rho_bar_u=rho_bar, contamination_scale=1.0 - rho_bar,
        auc=float(roc_auc_score(r, p_obs)), brier=float(brier_score_loss(r, p_obs)),
        ess_p99=ess_frac(w_o), dess=ess_frac(w_c) - ess_frac(w_o),
        eff_coef_obs=eff_coef(p_obs, pc1), eff_coef_calib=eff_coef(p_cal, pc1),
        coef_gap=abs(eff_coef(p_cal, pc1) - eff_coef(p_obs, pc1)),
        calib_coverage=float(okd.mean()) if low.any() else np.nan,
        smd_before=smd0, smd_after=smd1))
    r0 = rows[-1]
    print(f"  {name[:30]:30s} [{unit}] n={r0['n']:6d} AUC={r0['auc']:.3f} ESS={r0['ess_p99']:.3f} "
          f"rho_bar={rho_bar:.3f} gap={r0['coef_gap']:.3f} dess={r0['dess']:+.3f}")


def main():
    os.makedirs(REPORTS, exist_ok=True)
    z = np.load(CACHE)
    stack = z["stack"]
    meta = dict(x0=float(z["x0"]), dx=float(z["dx"]), y0=float(z["y0"]), dy=float(z["dy"]))
    curve = layer_variogram(stack, meta)
    curve.to_csv(os.path.join(REPORTS, "survey_rho_curve.csv"), index=False)
    print("layer variogram:")
    for _, rr in curve.iterrows():
        print(f"  h<={rr.lag_km:5.1f} km  rho={rr.rho:.3f}")

    rows = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*_annotated.csv"))):
        slug = os.path.basename(path).replace("_annotated.csv", "")
        d = pd.read_csv(path, low_memory=False)
        d = d[d[BIOS].notna().all(axis=1) & d.uncertainty_clean.notna()].reset_index(drop=True)
        r = (d.uncertainty_clean <= 100.0).astype(int)
        if len(d) < MIN_USABLE or not (MIN_CLASS_FRAC <= r.mean() <= 1 - MIN_CLASS_FRAC):
            print(f"  {slug}: fails selection (n={len(d)}, high={r.mean():.2f}) — recorded, skipped")
            rows.append(dict(name=slug, unit="group", n=len(d), retain=float(r.mean())))
            continue
        battery(slug, "group", d, curve, stack, meta, rows)
        for sp, gsp in d.groupby("species"):
            if len(gsp) >= MIN_SP:
                rr = (gsp.uncertainty_clean <= 100.0).astype(int)
                if MIN_CLASS_FRAC <= rr.mean() <= 1 - MIN_CLASS_FRAC:
                    battery(str(sp), "species", gsp.reset_index(drop=True), curve, stack, meta, rows)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(REPORTS, "survey_battery.csv"), index=False)

    ok = out.dropna(subset=["coef_gap", "contamination_scale"]) if "coef_gap" in out else out.iloc[:0]
    lines = ["=" * 100, "SURVEY — DIAGNOSTIC BATTERY ACROSS TAXON GROUPS", "=" * 100]
    if len(ok) >= 4:
        r_test = np.corrcoef(ok.contamination_scale, ok.coef_gap)[0, 1]
        lines.append(f"\nFORMULA TEST across {len(ok)} units: corr(1 - rho_bar(u), coef_gap) = {r_test:.3f}")
    lines.append("\nSIGNATURE SPACE (real units vs simulation regime means):")
    lines.append("  unit                                AUC     ESS    dess")
    for nm, auc, ess, _, dess in SIM_REFERENCE:
        lines.append(f"  {nm:32s} {auc:6.3f} {ess:7.3f} {dess:+7.3f}")
    for _, rr in ok.iterrows():
        lines.append(f"  {rr['name'][:28]:28s}[{rr.unit[:2]}] {rr.auc:6.3f} {rr.ess_p99:7.3f} {rr.dess:+7.3f}")
    text = "\n".join(lines)
    with open(os.path.join(REPORTS, "survey_verdict.txt"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
