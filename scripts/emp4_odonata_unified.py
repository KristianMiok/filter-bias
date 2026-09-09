"""
emp4_odonata_unified.py
=======================
EMPIRICAL BLOCK, step 4 — the Odonata system, rebuilt around the new framework.

Differences from the crayfish system that make this the right place for each
piece: quality here is a CONTINUOUS per-record uncertainty radius (GBIF
coordinateUncertaintyInMeters, cleaned), the environment is raster-derived
(bio_1..bio_19 annotated per record), and reference density is LOW — so:

  Part A  per-species analysis at the paper's 100 m threshold:
          unified propensity (std bios + quadratic logistic, GroupKFold(5) by
          50 km spatial blocks), Brier, ESS battery, balance; strategies
          ALL / FILTER / IPW / SMITH_env / ALL_calib with SMITH & calib pools
          drawn within each Low record's OWN uncertainty radius (capped 25 km);
          POOL COVERAGE is measured and reported — relocation and calibration
          are only actionable where reference density suffices, and that is a
          finding, not an assumption; kappa-envelope (L2 primary, L4 supp);
          ESS < 0.30 gating rule as in emp2.
  Part B  continuous-uncertainty threshold sweep (R1 #9): thresholds
          {50, 100, 250, 500, 1000} m, per species and pooled — diagnostic
          quantities only (retention, AUC, Brier, ESS, balance), no SDMs.
  Part C  empirical bio-space autocorrelation varrho(h) from the table itself
          (random record pairs, distance bins) — locates the uncertainty scale
          at which contamination can begin to bite relative to layer structure,
          the rho(D) result applied to the practitioner's own data.

Outputs
    reports/emp4_species_table.csv, reports/emp4_threshold_sweep.csv,
    reports/emp4_variogram.csv, reports/emp4_verdict.txt

Run from the repo root:   python scripts/emp4_odonata_unified.py
(EMP4_TREES=30 for a dry run; expects reports/odonata_de_annotated.csv)
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "reports", "odonata_de_annotated.csv")
REPORTS = os.path.join(ROOT, "reports")

BIOS = [f"bio_{i}" for i in range(1, 20)]
THRESH_MAIN = 100.0
THRESH_SWEEP = (50.0, 100.0, 250.0, 500.0, 1000.0)
MIN_N = int(os.environ.get("EMP4_MIN_N", "2000"))
MIN_CLASS_FRAC = 0.15
POOL_CAP_KM = 25.0
KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0)
ESS_GATE = 0.30
BLOCK_KM = 50.0
N_BG, SEED = 20000, 42
N_TREES = int(os.environ.get("EMP4_TREES", "300"))
RNG = np.random.default_rng(SEED)


def make_model(seed):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(n_estimators=N_TREES, max_depth=14,
                                                   min_samples_leaf=10,
                                                   class_weight="balanced_subsample",
                                                   random_state=seed, n_jobs=-1))])


def project_xy(lat, lon):
    lat0 = np.nanmean(lat)
    return np.stack([lon * 111.320 * np.cos(np.radians(lat0)), lat * 110.540], axis=1)  # km


def blocks_of(xy, size_km=BLOCK_KM):
    b = np.floor(xy / size_km).astype(int)
    return b[:, 0] * 100000 + b[:, 1]


def design_quad(X, scaler=None):
    if scaler is None:
        scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    return np.hstack([Z, Z ** 2]), scaler


def oof_propensity(Xd, r, groups):
    p = np.full(len(r), np.nan)
    gk = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gk.split(Xd, r, groups):
        if len(np.unique(r[tr])) < 2:
            p[te] = r[tr].mean(); continue
        m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000)
        m.fit(Xd[tr], r[tr]); p[te] = m.predict_proba(Xd[te])[:, 1]
    return p


def rescale(p, kappa, target):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, None))
    zc = kappa * (z - z.mean())
    lo, hi = -50.0, 50.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (1 / (1 + np.exp(-(mid + zc)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + np.exp(-(0.5 * (lo + hi) + zc)))


def ess_frac(w):
    return float((w.sum() ** 2) / ((w ** 2).sum() + 1e-12) / len(w))


def _norm(x):
    x = np.clip(x, 0, None); return x / (x.sum() + 1e-300)


def schoener_d(a, b):
    return 1.0 - 0.5 * np.abs(_norm(a) - _norm(b)).sum()


def smd_mean(Zstd, r, w_hi=None):
    mu_all = Zstd.mean(0)
    Xh = Zstd[r == 1]
    if w_hi is None:
        mu_hi = Xh.mean(0)
    else:
        w = w_hi / w_hi.sum()
        mu_hi = (Xh * w[:, None]).sum(0)
    return float(np.abs(mu_hi - mu_all).mean())


def variogram(xy, Zstd, lags_km=(0.5, 1, 2, 5, 10, 25, 50), n_samp=4000):
    from scipy.spatial.distance import cdist
    idx = RNG.choice(len(xy), size=min(n_samp, len(xy)), replace=False)
    P, Z = xy[idx], Zstd[idx]
    Dm = cdist(P, P)
    rows = []
    edges = [0.0] + list(lags_km)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (Dm > lo) & (Dm <= hi)
        ii, jj = np.where(np.triu(m, 1))
        if len(ii) < 200:
            rows.append(dict(lag_km=hi, n_pairs=len(ii), bio_corr=np.nan)); continue
        if len(ii) > 200000:
            k = RNG.choice(len(ii), 200000, replace=False); ii, jj = ii[k], jj[k]
        gamma = 0.5 * ((Z[ii] - Z[jj]) ** 2).mean()
        rows.append(dict(lag_km=hi, n_pairs=len(ii), bio_corr=float(1.0 - gamma)))
    return pd.DataFrame(rows)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    t00 = time.time()
    d = pd.read_csv(DATA, low_memory=False)
    d = d[d[BIOS].notna().all(axis=1)].reset_index(drop=True)
    usable = d[d.uncertainty_clean.notna()].reset_index(drop=True)
    n_nounc = len(d) - len(usable)
    xy_all = project_xy(d.decimalLatitude.values, d.decimalLongitude.values)
    tree_all = cKDTree(xy_all)
    Z_all_std = StandardScaler().fit_transform(d[BIOS].values)
    print(f"records: {len(d)} total, {len(usable)} with usable uncertainty ({n_nounc} without, set aside)")

    # ---- Part C: empirical variogram of bio space ----
    vg = variogram(xy_all, Z_all_std)
    vg.to_csv(os.path.join(REPORTS, "emp4_variogram.csv"), index=False)

    # ---- focal species ----
    counts = usable.groupby("species").agg(n=("key", "size"),
                                           hi=("accuracy_class", lambda s: s.eq("High").mean()))
    focal = counts[(counts.n >= MIN_N) & (counts.hi >= MIN_CLASS_FRAC) &
                   (counts.hi <= 1 - MIN_CLASS_FRAC)].index.tolist()
    print(f"focal species ({len(focal)}):", ", ".join(focal))

    lines, sp_rows, sweep_rows = [], [], []

    # ---- Part B: threshold sweep (diagnostics only) ----
    for th in THRESH_SWEEP:
        for name, g in [("POOLED", usable)] + [(sp, usable[usable.species.eq(sp)]) for sp in focal]:
            g = g.reset_index(drop=True)
            r = (g.uncertainty_clean <= th).astype(int).values
            if r.mean() in (0.0, 1.0) or len(g) < 500:
                continue
            xy = project_xy(g.decimalLatitude.values, g.decimalLongitude.values)
            Xd, sc = design_quad(g[BIOS].values)
            p = oof_propensity(Xd, r, blocks_of(xy))
            hi = r == 1
            w = 1 / np.clip(p[hi], 1e-3, None)
            w = np.clip(w, None, np.percentile(w, 99))
            Zs = sc.transform(g[BIOS].values)
            sweep_rows.append(dict(dataset=name, threshold_m=th, n=len(g),
                                   retain=float(hi.mean()),
                                   auc=float(roc_auc_score(r, p)),
                                   brier=float(brier_score_loss(r, p)),
                                   ess_p99=ess_frac(w),
                                   smd_before=smd_mean(Zs, r),
                                   smd_after=smd_mean(Zs, r, w_hi=w)))
    pd.DataFrame(sweep_rows).to_csv(os.path.join(REPORTS, "emp4_threshold_sweep.csv"), index=False)

    # ---- Part A: per-species full analysis at 100 m ----
    for sp in focal:
        t0 = time.time()
        g = usable[usable.species.eq(sp)].reset_index(drop=True)
        r = (g.uncertainty_clean <= THRESH_MAIN).astype(int).values
        hi = r == 1
        xy = project_xy(g.decimalLatitude.values, g.decimalLongitude.values)
        Xd, sc = design_quad(g[BIOS].values)
        p_oof = oof_propensity(Xd, r, blocks_of(xy))
        auc = float(roc_auc_score(r, p_oof)); brier = float(brier_score_loss(r, p_oof))
        w_pt = 1 / np.clip(p_oof[hi], 1e-3, None)
        w_pt = np.clip(w_pt, None, np.percentile(w_pt, 99))
        ess = ess_frac(w_pt)
        gated = ess < ESS_GATE
        Zs = sc.transform(g[BIOS].values)
        bal0, bal1 = smd_mean(Zs, r), smd_mean(Zs, r, w_hi=w_pt)

        # SMITH / calib pools within each Low record's own radius (donors = all records)
        e_obs = g[BIOS].values
        e_smith = e_obs.copy(); e_calib = e_obs.copy()
        low_idx = np.where(~hi)[0]
        radii = np.clip(g.uncertainty_clean.values[low_idx] / 1000.0, 0.0, POOL_CAP_KM)
        cen = e_obs[hi].mean(0)
        sc_all = StandardScaler().fit(d[BIOS].values)
        cen_z = sc_all.transform(cen.reshape(1, -1))[0]
        n_cov = 0
        for k_i, i in enumerate(low_idx):
            pool = tree_all.query_ball_point(project_xy(
                np.array([g.decimalLatitude.values[i]]), np.array([g.decimalLongitude.values[i]]))[0],
                r=max(radii[k_i], 1e-6))
            pool = np.asarray(pool)
            if len(pool) < 2:
                continue
            n_cov += 1
            Ez = sc_all.transform(d[BIOS].values[pool])
            e_smith[i] = d[BIOS].values[pool[np.argmin(((Ez - cen_z) ** 2).sum(1))]]
            e_calib[i] = d[BIOS].values[pool].mean(0)
        pool_cov = n_cov / max(len(low_idx), 1)

        # consequence: target-group background
        bgp = usable[~usable.species.eq(sp)]
        if len(bgp) > N_BG:
            bgp = bgp.sample(n=N_BG, random_state=SEED)
        Xb = bgp[BIOS].values; yb = np.zeros(len(bgp))

        def fit_predict(Xpres, w=None, so=0):
            X = np.vstack([Xpres, Xb]); y = np.concatenate([np.ones(len(Xpres)), yb])
            sw = np.concatenate([np.ones(len(Xpres)) if w is None else w, np.ones(len(bgp))])
            m = make_model(SEED + so)
            m.fit(X, y, rf__sample_weight=sw)
            return m.predict_proba(Xb)[:, 1]

        surf = {"ALL": fit_predict(e_obs),
                "FILTER": fit_predict(e_obs[hi]),
                "IPW": fit_predict(e_obs[hi], w=w_pt),
                "SMITH": fit_predict(e_smith),
                "CALIB": fit_predict(e_calib)}
        kap_surf = {}
        for kap in KAPPAS:
            if kap == 1.0:
                kap_surf[kap] = surf["IPW"]; continue
            p_k = rescale(p_oof, kap, float(hi.mean()))
            w_k = 1 / np.clip(p_k[hi], 1e-3, None)
            w_k = np.clip(w_k, None, np.percentile(w_k, 99))
            kap_surf[kap] = fit_predict(e_obs[hi], w=w_k, so=int(kap * 10))
        envL4 = schoener_d(kap_surf[0.25], kap_surf[4.0])
        envL2 = schoener_d(kap_surf[0.5], kap_surf[2.0])

        row = dict(species=sp, n=len(g), retain=float(hi.mean()), auc=auc, brier=brier,
                   ess_p99=ess, gated=gated, bal_before=bal0, bal_after=bal1,
                   pool_coverage=pool_cov,
                   D_ALL_FILTER=schoener_d(surf["ALL"], surf["FILTER"]),
                   D_FILTER_IPW=schoener_d(surf["FILTER"], surf["IPW"]),
                   D_ALL_SMITH=schoener_d(surf["ALL"], surf["SMITH"]),
                   D_ALL_CALIB=schoener_d(surf["ALL"], surf["CALIB"]),
                   sp_ALL_SMITH=float(spearmanr(surf["ALL"], surf["SMITH"]).statistic),
                   env_width_L4=envL4, env_width_L2=envL2)
        sp_rows.append(row)
        lines.append(f"  {sp[:26]:26s} n={len(g):5d} AUC={auc:.3f} ESS={ess:.3f} gated={'YES' if gated else 'no '} "
                     f"bal {bal0:.3f}->{bal1:.3f}  poolcov={pool_cov:.2f}  "
                     f"D(A,F)={row['D_ALL_FILTER']:.3f} D(F,I)={row['D_FILTER_IPW']:.3f} "
                     f"D(A,S)={row['D_ALL_SMITH']:.3f}  L2={envL2:.3f}  [{time.time()-t0:.0f}s]")
        print(lines[-1])

    pd.DataFrame(sp_rows).to_csv(os.path.join(REPORTS, "emp4_species_table.csv"), index=False)
    out = ["=" * 112,
           f"EMP 4 — ODONATA: unified propensity, per-record-radius SMITH/calib, threshold sweep, variogram",
           "=" * 112,
           f"records without usable uncertainty (set aside): {n_nounc}",
           "bio-space autocorrelation varrho(h):"]
    for _, rr in vg.iterrows():
        out.append(f"    h<={rr.lag_km:5.1f} km   corr={rr.bio_corr:.3f}   (n_pairs={int(rr.n_pairs)})")
    out += lines + ["", "threshold sweep -> emp4_threshold_sweep.csv", ""]
    text = "\n".join(out)
    with open(os.path.join(REPORTS, "emp4_verdict.txt"), "w") as fh:
        fh.write(text)
    print(f"total {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
