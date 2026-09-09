"""
emp3_semisynthetic.py
=====================
EMPIRICAL BLOCK, step 3 — semi-synthetic ground-truth experiment on the real
crayfish database (answers R1 #7: "empirical evidence cannot validate which
correction is correct" — here it can, because we manufacture the damage on real
environmental structure and know the truth).

Design (network-honest: no rasters, every environment is a REAL record's
environment):

  TRUTH      the certified-coordinate (High) records of one focal species are
             taken as true occurrences with true environments.
  QUALITY    synthetic label r ~ Bernoulli(sigmoid(alpha + beta * u)), where u is
             the species' REAL leading coupling axis (the local descriptor with
             the largest observed High-vs-Low SMD in the full database), so the
             synthetic coupling mimics the documented real one. alpha calibrated
             to the species' real retention rate.
  DISPLACE   each synthetic low-quality record swaps its environment for that of
             a real neighbouring record within radius D km (candidate pool from a
             KD-tree on projected coordinates):
                 iso        random candidate
                 toward_HQ  candidate with the HIGHEST u (toward environments
                            where retained records sit; centroid-snap analogue)
                 away_HQ    candidate with the LOWEST u
  STRATEGIES ALL / FILTER / IPW_est (propensity on OBSERVED env, PCA15+quad
             logistic, basin-blocked on the DONOR record's basin) / IPW_oracle /
             SMITH_env (within the same pool, env closest to the env centroid of
             retained records) / ALL_calib (pool-mean env = regression
             calibration) — all network-honest analogues of the simulation's.
  LEARNER    RF as in the paper (300 trees, depth 14, leaf 10, balanced_subsample,
             median imputation), target-group background (20k other-species).
  TRUTH REF  the same learner fitted on the UNDISPLACED truth set; recovery =
             Spearman + Schoener D of each strategy surface vs the reference
             surface over the background.
  SIGNATURE  observable (AUC, Brier, ESS) recorded per cell -> does the
             294-dimensional real-structure signature match the simulation's?

Grid: regimes x D in {2, 10, 25} km x seeds (3), beta = 2.5 fixed; the quality
assignment is shared across regimes within a (D, seed) cell, so regimes are
compared on identical retention patterns (paired design).

Run:   EMP3_TREES=30 python scripts/emp3_semisynthetic.py    (dry run)
       python scripts/emp3_semisynthetic.py                  (full)

Outputs
    reports/emp3_semisynth_sweep.csv, reports/emp3_semisynth_summary.csv,
    reports/emp3_semisynth_verdict.txt
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "raw", "combined_data_true_master.csv")
REPORTS = os.path.join(ROOT, "reports")

FOCAL_SP = "Astacus astacus"
BETA = 2.5
RETAIN_FALLBACK = 0.60
D_KMS = (2.0, 10.0, 25.0)
REGIMES = ("iso", "toward_HQ", "away_HQ")
AXES_ORDER = ("long_range", "short_range")
SEEDS = (0, 1, 2)
N_BACKGROUND, SEED0 = 20000, 42
N_TREES = int(os.environ.get("EMP3_TREES", "300"))


def local_cols(d):
    return [c for c in d.columns if c.startswith(("l_CLI", "l_TOP", "l_LAC", "l_SOL"))]


def make_model(seed):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(n_estimators=N_TREES, max_depth=14,
                                                   min_samples_leaf=10,
                                                   class_weight="balanced_subsample",
                                                   random_state=seed, n_jobs=-1))])


def project_xy(lat, lon):
    lat0 = np.nanmean(lat)
    x = lon * 111.320 * np.cos(np.radians(lat0))
    y = lat * 110.540
    return np.stack([x, y], axis=1)   # km


def calibrate_alpha(u, beta, target):
    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (1 / (1 + np.exp(-(mid + beta * u)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def oof_propensity(X_design, r, groups):
    p = np.full(len(r), np.nan)
    gk = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gk.split(X_design, r, groups):
        if len(np.unique(r[tr])) < 2:
            p[te] = r[tr].mean(); continue
        m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000)
        m.fit(X_design[tr], r[tr]); p[te] = m.predict_proba(X_design[te])[:, 1]
    return p


def pca_design(X):
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=15, random_state=0).fit(scaler.transform(X))
    P = pca.transform(scaler.transform(X))
    return np.hstack([P, P ** 2])


def ess_frac(w):
    return float((w.sum() ** 2) / ((w ** 2).sum() + 1e-12) / len(w))


def _norm(x):
    x = np.clip(x, 0, None); return x / (x.sum() + 1e-300)


def schoener_d(a, b):
    return 1.0 - 0.5 * np.abs(_norm(a) - _norm(b)).sum()


def main():
    os.makedirs(REPORTS, exist_ok=True)
    t00 = time.time()
    d = pd.read_csv(DATA, low_memory=False)
    lc = local_cols(d)
    d = d[d[lc].notna().all(axis=1) & d.lat_or.notna() & d.long_or.notna()].reset_index(drop=True)
    lc = [c for c in lc if d[c].nunique() > 1]

    # coupling axes: overall leading (long-range, climatic) and leading TOPOGRAPHIC
    # (short-range) High-vs-Low SMD axis — the axis correlation length is the moderator
    rq = d.Accuracy.eq("High").astype(int).values
    Z_all = StandardScaler().fit_transform(d[lc].values)
    smd = (Z_all[rq == 1].mean(0) - Z_all[rq == 0].mean(0))
    j_long = int(np.argmax(np.abs(smd)))
    top_idx = [i for i, c in enumerate(lc) if c.startswith("l_TOP")]
    j_short = top_idx[int(np.argmax(np.abs(smd[np.array(top_idx)])))]
    AXES = {"long_range": j_long, "short_range": j_short}
    for a, j in AXES.items():
        print(f"coupling axis [{a}]: {lc[j]}  (High-vs-Low SMD {smd[j]:+.3f})")

    # truth set = High records of the focal species
    tr_mask = d.Crayfish_scientific_name.eq(FOCAL_SP) & d.Accuracy.eq("High")
    truth = d[tr_mask].reset_index(drop=True)
    retain = float(d[d.Crayfish_scientific_name.eq(FOCAL_SP)].Accuracy.eq("High").mean()) or RETAIN_FALLBACK
    print(f"truth records: {len(truth)}   target retention: {retain:.2f}")

    # candidate pool = every record in the database (any species)
    XY_all = project_xy(d.lat_or.values, d.long_or.values)
    tree = cKDTree(XY_all)
    XY_tr = project_xy(truth.lat_or.values, truth.long_or.values)

    U_ALL, U_TR, U_TR_STD = {}, {}, {}
    for a, j in AXES.items():
        sgn = np.sign(smd[j])
        U_ALL[a] = sgn * Z_all[:, j]
        ut = sgn * (truth[lc[j]].values.astype(float) - d[lc[j]].mean()) / (d[lc[j]].std() + 1e-12)
        U_TR[a] = ut
        U_TR_STD[a] = (ut - ut.mean()) / (ut.std() + 1e-12)

    # background: other species
    bg = d[~d.Crayfish_scientific_name.eq(FOCAL_SP)]
    if len(bg) > N_BACKGROUND:
        bg = bg.sample(n=N_BACKGROUND, random_state=SEED0)
    Xb = bg[lc].values
    yb = np.zeros(len(bg))

    def fit_predict(Xpres, w=None, seed=SEED0):
        X = np.vstack([Xpres, Xb]); y = np.concatenate([np.ones(len(Xpres)), yb])
        sw = np.concatenate([np.ones(len(Xpres)) if w is None else w, np.ones(len(bg))])
        m = make_model(seed)
        m.fit(X, y, rf__sample_weight=sw)
        return m.predict_proba(Xb)[:, 1]

    print("fitting truth reference + ceilings ...")
    ref = fit_predict(truth[lc].values)
    ref2 = fit_predict(truth[lc].values, seed=SEED0 + 777)
    ceil_seed = float(spearmanr(ref2, ref).statistic)
    rng0 = np.random.default_rng(7)
    sub = rng0.choice(len(truth), size=int(retain * len(truth)), replace=False)
    mcar = fit_predict(truth[lc].values[sub], seed=SEED0 + 778)
    ceil_mcar = float(spearmanr(mcar, ref).statistic)
    print(f"ceilings: seed-refit {ceil_seed:.3f}   MCAR-{retain:.0%} deletion {ceil_mcar:.3f}")

    Zc_sc = StandardScaler().fit(d[lc].values)
    rows, lines = [], []
    for D_km in D_KMS:
        pools = [np.asarray(pl) for pl in tree.query_ball_point(XY_tr, r=D_km)]
        for axis_name in AXES_ORDER:
          for regime in REGIMES:
            for seed in SEEDS:
                u_all = U_ALL[axis_name]; u_tr_std = U_TR_STD[axis_name]; u_tr_z = U_TR[axis_name]
                t0 = time.time()
                rng = np.random.default_rng(1000 * seed + int(D_km))
                alpha = calibrate_alpha(u_tr_std, BETA, retain)
                p_true = 1 / (1 + np.exp(-(alpha + BETA * u_tr_std)))
                r = (rng.uniform(size=len(truth)) < p_true).astype(int)
                hi = r == 1

                e_obs = truth[lc].values.copy()
                donor_basin = truth.basin_id.values.copy()
                n_disp = 0
                du_rec, du_don = [], []
                for i in np.where(~hi)[0]:
                    pool = pools[i]
                    if len(pool) < 2:
                        continue
                    if regime == "iso":
                        jj = rng.choice(pool)
                    elif regime == "toward_HQ":
                        jj = pool[np.argmax(u_all[pool])]
                    else:
                        jj = pool[np.argmin(u_all[pool])]
                    e_obs[i] = d[lc].values[jj]
                    donor_basin[i] = d.basin_id.values[jj]
                    n_disp += 1
                    du_rec.append(float(u_tr_z[i])); du_don.append(float(u_all[jj]))

                Xd = pca_design(e_obs)
                p_oof = oof_propensity(Xd, r, donor_basin)
                auc = roc_auc_score(r, p_oof)
                brier = brier_score_loss(r, p_oof)
                w_est = 1 / np.clip(p_oof[hi], 1e-3, None)
                w_est = np.clip(w_est, None, np.percentile(w_est, 99))
                w_orc = 1 / np.clip(p_true[hi], 1e-3, None)
                w_orc = np.clip(w_orc, None, np.percentile(w_orc, 99))
                ess = ess_frac(w_est)

                e_smith = e_obs.copy(); e_calib = e_obs.copy()
                cen_z = Zc_sc.transform(e_obs[hi].mean(0).reshape(1, -1))[0]
                for i in np.where(~hi)[0]:
                    pool = pools[i]
                    if len(pool) < 2:
                        continue
                    Ez = Zc_sc.transform(d[lc].values[pool])
                    e_smith[i] = d[lc].values[pool[np.argmin(((Ez - cen_z) ** 2).sum(1))]]
                    e_calib[i] = d[lc].values[pool].mean(0)

                surfaces = {
                    "ALL": fit_predict(e_obs, seed=SEED0 + seed),
                    "FILTER": fit_predict(e_obs[hi], seed=SEED0 + seed),
                    "IPW_est": fit_predict(e_obs[hi], w=w_est, seed=SEED0 + seed),
                    "IPW_oracle": fit_predict(e_obs[hi], w=w_orc, seed=SEED0 + seed),
                    "SMITH_env": fit_predict(e_smith, seed=SEED0 + seed),
                    "ALL_calib": fit_predict(e_calib, seed=SEED0 + seed),
                }
                mu_u = float(np.mean(np.array(du_don) - np.array(du_rec))) if du_rec else 0.0
                rho_u = float(np.corrcoef(du_rec, du_don)[0, 1]) if len(du_rec) > 5 else float("nan")
                row = dict(axis=axis_name, regime=regime, D_km=D_km, seed=seed,
                           n_truth=len(truth), n_high=int(hi.sum()), n_displaced=n_disp,
                           mu_u=mu_u, rho_u=rho_u,
                           ceil_seed=ceil_seed, ceil_mcar=ceil_mcar,
                           auc=auc, brier=brier, ess=ess)
                for k, s in surfaces.items():
                    row[f"sp_{k}"] = float(spearmanr(s, ref).statistic)
                    row[f"D_{k}"] = schoener_d(s, ref)
                rows.append(row)
                best = max(surfaces, key=lambda k: row[f"sp_{k}"])
                lines.append(f"  {axis_name[:5]:5s} {regime:10s} D={D_km:4.0f}km seed={seed}  "
                             f"mu_u={mu_u:+.2f} rho_u={rho_u:.2f}  AUC={auc:.3f} Brier={brier:.3f} "
                             f"ESS={ess:.3f}  " +
                             " ".join(f"{k}={row[f'sp_{k}']:.3f}" for k in surfaces) +
                             f"  best={best}  [{time.time()-t0:.0f}s]")
                print(lines[-1])

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(REPORTS, "emp3_semisynth_sweep.csv"), index=False)
    sm = raw.groupby(["axis", "regime", "D_km"]).mean(numeric_only=True).reset_index()
    sm.to_csv(os.path.join(REPORTS, "emp3_semisynth_summary.csv"), index=False)

    out = ["=" * 116,
           f"EMP 3 — SEMI-SYNTHETIC GROUND TRUTH ON REAL DATA  ({FOCAL_SP}, beta={BETA}, axis by largest real SMD)",
           "=" * 116] + lines + ["", "means -> emp3_semisynth_summary.csv", ""]
    text = "\n".join(out)
    with open(os.path.join(REPORTS, "emp3_semisynth_verdict.txt"), "w") as fh:
        fh.write(text)
    print(f"total {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
