"""
emp1_unified_propensity.py
==========================
EMPIRICAL BLOCK, step 1 — ONE propensity model for everything (fixes the
diagnostic-vs-weights inconsistency: the submitted paper reported a basin-blocked
logistic AUC in Table 2 but computed IPW weights from a randomly-CV'd random
forest at AUC 0.94). Here a single basin-blocked model produces the diagnostic
AND the weights, with the full battery both reviewers asked for:

  - energy distance HIGH vs LOW (disjoint samples; the submitted high-vs-all
    comparison was statistically dependent), permutation p reported as a floor,
    plus a standardized effect size                                (R2 minor)
  - propensity: PCA(15)+quadratic logistic, GroupKFold(5) by basin_id, out-of-fold
    probabilities; Brier score and a 10-bin reliability table       (R2 #7)
  - IPW weights from those SAME out-of-fold probabilities; positivity battery:
    weight quantiles, max/mean, ESS under four trimming rules       (R1 #6)
  - covariate balance: mean|SMD| and #(|SMD|>0.1) across the 302 local
    descriptors, all-vs-high BEFORE weighting and all-vs-IPW-weighted-high AFTER
                                                                    (R2 #7, Boyd 2024)
  - specification sensitivity: n_PC x C grid + basin-blocked RF     (R1 #5)

Outputs
    reports/emp1_species_table.csv    per-species diagnostic + positivity
    reports/emp1_balance.csv          per-species balance before/after
    reports/emp1_sensitivity.csv      global spec-sensitivity grid
    reports/emp1_reliability.csv      10-bin calibration, global + per species
    reports/emp1_verdict.txt          printed summary

Run from the repo root:   python scripts/emp1_unified_propensity.py
(expects data/raw/combined_data_true_master.csv, git-ignored)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "raw", "combined_data_true_master.csv")
REPORTS = os.path.join(ROOT, "reports")

FOCAL = ["Astacus astacus", "Austropotamobius pallipes", "Pacifastacus leniusculus",
         "Procambarus clarkii", "Faxonius limosus", "Pontastacus leptodactylus",
         "Austropotamobius torrentium"]
N_PC, C_PRIMARY, N_FOLDS = 15, 1.0, 5
TRIMS = {"none": None, "cap_p99": 99.0, "cap_p95": 95.0, "cap_10x": "10x"}
RNG = np.random.default_rng(2026)


# ------------------------------------------------------------------ helpers --
def local_cols(d):
    return [c for c in d.columns if c.startswith(("l_CLI", "l_TOP", "l_LAC", "l_SOL"))]


def design_pca(X, n_pc):
    Z = StandardScaler().fit_transform(X)
    P = PCA(n_components=min(n_pc, Z.shape[1]), random_state=0).fit_transform(Z)
    return np.hstack([P, P ** 2])


def oof_propensity(X, r, groups, C=1.0, learner="logit", n_folds=N_FOLDS, seed=0):
    """Out-of-fold P(High | X), grouped by basin."""
    p = np.full(len(r), np.nan)
    gk = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    for tr, te in gk.split(X, r, groups):
        if len(np.unique(r[tr])) < 2:
            p[te] = r[tr].mean(); continue
        if learner == "logit":
            m = LogisticRegression(C=C, solver="lbfgs", max_iter=4000)
        else:
            m = RandomForestClassifier(n_estimators=100, min_samples_leaf=50,
                                       n_jobs=-1, random_state=seed)
        m.fit(X[tr], r[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def energy_distance_hi_lo(Z, r, n_sub=1500, n_perm=199):
    """E-statistic between HIGH and LOW rows of standardized Z (disjoint samples).
    The pooled pairwise-distance matrix is computed ONCE; each permutation is
    pure index arithmetic on it."""
    from scipy.spatial.distance import cdist
    hi_i = np.where(r == 1)[0]; lo_i = np.where(r == 0)[0]
    if len(hi_i) < 30 or len(lo_i) < 30:
        return np.nan, np.nan, np.nan
    hi_i = RNG.choice(hi_i, size=min(n_sub, len(hi_i)), replace=False)
    lo_i = RNG.choice(lo_i, size=min(n_sub, len(lo_i)), replace=False)
    pool = Z[np.concatenate([hi_i, lo_i])]
    nA, nTot = len(hi_i), len(hi_i) + len(lo_i)
    Dm = cdist(pool, pool)

    def edist(ia, ib):
        dab = Dm[np.ix_(ia, ib)].mean()
        daa = Dm[np.ix_(ia, ia)].mean()
        dbb = Dm[np.ix_(ib, ib)].mean()
        return 2 * dab - daa - dbb

    ia0, ib0 = np.arange(nA), np.arange(nA, nTot)
    obs = edist(ia0, ib0)
    cnt = 0
    for _ in range(n_perm):
        idx = RNG.permutation(nTot)
        if edist(idx[:nA], idx[nA:]) >= obs:
            cnt += 1
    p_floor = (cnt + 1) / (n_perm + 1)
    ref = Dm.mean()   # standardized effect size: E / mean pooled pairwise distance
    return obs, obs / ref, p_floor


def smd_table(X, r, w_hi=None):
    """SMD of each column, ALL vs HIGH (weighted if w_hi given)."""
    mu_all, sd_all = X.mean(0), X.std(0) + 1e-12
    Xh = X[r == 1]
    if w_hi is None:
        mu_hi = Xh.mean(0)
    else:
        w = w_hi / w_hi.sum()
        mu_hi = (Xh * w[:, None]).sum(0)
    return (mu_hi - mu_all) / sd_all


def ess_frac(w):
    return float((w.sum() ** 2) / ((w ** 2).sum() + 1e-12) / len(w))


def trim_weights(w, rule):
    if rule is None:
        return w
    if rule == "10x":
        return np.clip(w, None, 10.0 * w.mean())
    return np.clip(w, None, np.percentile(w, rule))


# ------------------------------------------------------------------ analyses --
def analyse(name, d, lc, lines, species_rows, balance_rows, reli_rows):
    r = d.Accuracy.eq("High").astype(int).values
    groups = d.basin_id.values
    X_raw = d[lc].values
    Z = StandardScaler().fit_transform(X_raw)
    Xd = design_pca(X_raw, N_PC)

    e_obs, e_eff, e_p = energy_distance_hi_lo(Z, r)
    p_oof = oof_propensity(Xd, r, groups, C=C_PRIMARY)
    ok = np.isfinite(p_oof)
    auc = roc_auc_score(r[ok], p_oof[ok]) if len(np.unique(r[ok])) > 1 else np.nan
    brier = brier_score_loss(r[ok], p_oof[ok])

    # reliability (10 bins)
    bins = np.clip((p_oof[ok] * 10).astype(int), 0, 9)
    for b in range(10):
        m = bins == b
        if m.sum() > 0:
            reli_rows.append(dict(dataset=name, bin=b, n=int(m.sum()),
                                  p_mean=float(p_oof[ok][m].mean()),
                                  frac_high=float(r[ok][m].mean())))

    # weights + positivity battery
    hi = r == 1
    w_raw = 1.0 / np.clip(p_oof[hi], 1e-3, None)
    row = dict(dataset=name, n=len(d), n_high=int(hi.sum()), pct_high=float(hi.mean()),
               n_basins=int(d.basin_id.nunique()),
               energy_dist=e_obs, energy_eff=e_eff, energy_p_floor=e_p,
               env_auc=auc, brier=brier,
               w_mean=float(np.nanmean(w_raw)), w_p50=float(np.nanpercentile(w_raw, 50)),
               w_p99=float(np.nanpercentile(w_raw, 99)), w_max=float(np.nanmax(w_raw)),
               w_max_over_mean=float(np.nanmax(w_raw) / np.nanmean(w_raw)))
    for tname, rule in TRIMS.items():
        row[f"ess_{tname}"] = ess_frac(trim_weights(w_raw, rule))
    species_rows.append(row)

    # balance
    smd0 = smd_table(Z, r)
    smd1 = smd_table(Z, r, w_hi=trim_weights(w_raw, 99.0))
    balance_rows.append(dict(dataset=name,
                             mean_abs_smd_before=float(np.abs(smd0).mean()),
                             mean_abs_smd_after=float(np.abs(smd1).mean()),
                             n_gt01_before=int((np.abs(smd0) > 0.1).sum()),
                             n_gt01_after=int((np.abs(smd1) > 0.1).sum()),
                             max_abs_smd_before=float(np.abs(smd0).max()),
                             max_abs_smd_after=float(np.abs(smd1).max())))
    lines.append(f"  {name[:32]:32s} n={len(d):6d}  high={hi.mean():.2f}  AUC={auc:.3f}  "
                 f"Brier={brier:.3f}  E-eff={e_eff if np.isfinite(e_eff) else float('nan'):.3f} "
                 f"(p<{e_p:.4f})  ESS(p99)={row['ess_cap_p99']:.3f}  balance {np.abs(smd0).mean():.3f}->{np.abs(smd1).mean():.3f}")


def main():
    os.makedirs(REPORTS, exist_ok=True)
    t0 = time.time()
    d = pd.read_csv(DATA, low_memory=False)
    lc = local_cols(d)
    d = d[d[lc].notna().all(axis=1)].reset_index(drop=True)
    # drop constant local cols (paper: 302 features; constants carry nothing)
    keep = [c for c in lc if d[c].nunique() > 1]
    lc = keep
    print(f"loaded {len(d)} complete-local rows, {len(lc)} usable local descriptors  ({time.time()-t0:.0f}s)")

    lines, species_rows, balance_rows, reli_rows = [], [], [], []

    print("global ..."); analyse("GLOBAL (all species)", d, lc, lines, species_rows, balance_rows, reli_rows)
    for sp in FOCAL:
        g = d[d.Crayfish_scientific_name == sp]
        if len(g) < 200:
            print(f"  skip {sp} (n={len(g)})"); continue
        print(f"{sp} ...")
        analyse(sp, g.reset_index(drop=True), lc, lines, species_rows, balance_rows, reli_rows)

    # spec sensitivity (global; RF on a 40k stratified subsample for runtime)
    print("sensitivity grid ...")
    r_all = d.Accuracy.eq("High").astype(int).values
    sens = []
    for n_pc in (10, 15, 25):
        Xd = design_pca(d[lc].values, n_pc)
        for C in (0.1, 1.0, 10.0):
            p = oof_propensity(Xd, r_all, d.basin_id.values, C=C)
            ok = np.isfinite(p)
            sens.append(dict(spec=f"logit_pc{n_pc}_C{C:g}",
                             auc=roc_auc_score(r_all[ok], p[ok]),
                             brier=brier_score_loss(r_all[ok], p[ok]),
                             ess_p99=ess_frac(trim_weights(1/np.clip(p[r_all==1],1e-3,None), 99.0))))
    idx = np.arange(len(d))
    if len(d) > 40000:
        sub = np.sort(np.concatenate([RNG.choice(idx[r_all==1], 30000, replace=False),
                                      RNG.choice(idx[r_all==0], 10000, replace=False)]))
    else:
        sub = idx
    Zs = StandardScaler().fit_transform(d[lc].values[sub])
    p = oof_propensity(Zs, r_all[sub], d.basin_id.values[sub], learner="rf")
    ok = np.isfinite(p)
    sens.append(dict(spec="rf_basinblocked_40k",
                     auc=roc_auc_score(r_all[sub][ok], p[ok]),
                     brier=brier_score_loss(r_all[sub][ok], p[ok]),
                     ess_p99=ess_frac(trim_weights(1/np.clip(p[r_all[sub]==1],1e-3,None), 99.0))))

    pd.DataFrame(species_rows).to_csv(os.path.join(REPORTS, "emp1_species_table.csv"), index=False)
    pd.DataFrame(balance_rows).to_csv(os.path.join(REPORTS, "emp1_balance.csv"), index=False)
    pd.DataFrame(sens).to_csv(os.path.join(REPORTS, "emp1_sensitivity.csv"), index=False)
    pd.DataFrame(reli_rows).to_csv(os.path.join(REPORTS, "emp1_reliability.csv"), index=False)

    out = []
    out.append("=" * 108)
    out.append("EMP 1 — UNIFIED BASIN-BLOCKED PROPENSITY: diagnostic, positivity, balance  (one model for everything)")
    out.append("=" * 108)
    out.extend(lines)
    out.append("\nSPEC SENSITIVITY (global):")
    for s in sens:
        out.append(f"  {s['spec']:22s} AUC={s['auc']:.3f}  Brier={s['brier']:.3f}  ESS(p99)={s['ess_p99']:.3f}")
    out.append("\nTRIMMING (per dataset, ESS under each rule) -> emp1_species_table.csv")
    out.append("RELIABILITY (10-bin) -> emp1_reliability.csv")
    text = "\n".join(out)
    with open(os.path.join(REPORTS, "emp1_verdict.txt"), "w") as fh:
        fh.write(text)
    print(text)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
