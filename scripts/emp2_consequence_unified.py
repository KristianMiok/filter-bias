"""
emp2_consequence_unified.py
===========================
EMPIRICAL BLOCK, step 2 — consequence analysis rerun with the UNIFIED propensity.

Same presence-background pipeline as the submitted paper (presence = target
species; background = 20,000 records of other species; RF 300 trees, depth 14,
min_leaf 10, balanced_subsample, median imputation; surface = predictions on the
background environment). Three changes, all reviewer-driven:

  1. IPW weights come from the SAME basin-blocked PCA(15)+quadratic logistic
     propensity as the diagnostic (emp1) — not from the randomly-CV'd RF.
  2. kappa-envelope per species: the centred logit of the OOF propensity is
     rescaled by kappa in {1/4, 1/2, 1, 2, 4} (recalibrated to the observed
     retention rate) and the SDM is refit for each kappa. The reported object is
     the ENVELOPE of surfaces, not a single corrected map. (empirical Exp 4)
  3. Gating rule from emp1: species with ESS(p99) < 0.30 or worsened covariate
     balance get their IPW flagged NOT REPORTABLE as a point estimate — envelope
     only. (operationalizes R1 #6 / R1 #12)

No ground truth exists here, so all comparisons are BETWEEN strategies:
Schoener D, Warren I, Spearman, range change at the top-30% threshold of ALL,
and top-10% Jaccard — matching the submitted Supplementary S2 for continuity.

Outputs
    reports/emp2_surfaces_summary.csv   per species x strategy-pair metrics
    reports/emp2_envelope.csv           per species kappa-envelope widths
    reports/emp2_verdict.txt

Run from the repo root:   python scripts/emp2_consequence_unified.py
(expects data/raw/combined_data_true_master.csv, git-ignored; ~45-90 min;
 EMP2_TREES=30 python scripts/emp2_consequence_unified.py  for a fast dry run)
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "raw", "combined_data_true_master.csv")
REPORTS = os.path.join(ROOT, "reports")

FOCAL = ["Astacus astacus", "Austropotamobius pallipes", "Pacifastacus leniusculus",
         "Procambarus clarkii", "Faxonius limosus", "Pontastacus leptodactylus",
         "Austropotamobius torrentium"]
N_BACKGROUND, SEED = 20000, 42
N_TREES = int(os.environ.get("EMP2_TREES", "300"))   # set EMP2_TREES=30 for a fast dry run
KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0)
ESS_GATE = 0.30
RNG = np.random.default_rng(SEED)


def local_cols(d):
    return [c for c in d.columns if c.startswith(("l_CLI", "l_TOP", "l_LAC", "l_SOL"))]


def make_model(seed):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(n_estimators=N_TREES, max_depth=14,
                                                   min_samples_leaf=10,
                                                   class_weight="balanced_subsample",
                                                   random_state=seed, n_jobs=-1))])


def oof_propensity(d, lc):
    """Unified emp1 spec: PCA(15)+quad logistic, GroupKFold(5) by basin."""
    r = d.Accuracy.eq("High").astype(int).values
    Z = StandardScaler().fit_transform(d[lc].values)
    P = PCA(n_components=15, random_state=0).fit_transform(Z)
    X = np.hstack([P, P ** 2])
    p = np.full(len(r), np.nan)
    gk = GroupKFold(n_splits=min(5, d.basin_id.nunique()))
    for tr, te in gk.split(X, r, d.basin_id.values):
        if len(np.unique(r[tr])) < 2:
            p[te] = r[tr].mean(); continue
        m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000)
        m.fit(X[tr], r[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return r, p


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


def warren_i(a, b):
    return 1.0 - 0.5 * ((np.sqrt(_norm(a)) - np.sqrt(_norm(b))) ** 2).sum()


def surf_metrics(a, b):
    from scipy.stats import spearmanr
    t30 = np.quantile(a, 0.70); t10a = np.quantile(a, 0.90); t10b = np.quantile(b, 0.90)
    ja, jb = b >= t10b, a >= t10a
    return dict(schoener_D=schoener_d(a, b), warren_I=warren_i(a, b),
                spearman=float(spearmanr(a, b).statistic),
                range_top30_pct=float(((b >= t30).mean() - 0.30) / 0.30 * 100),
                top10_jaccard=float((ja & jb).sum() / max((ja | jb).sum(), 1)))


def main():
    os.makedirs(REPORTS, exist_ok=True)
    t00 = time.time()
    d = pd.read_csv(DATA, low_memory=False)
    lc = local_cols(d)
    d = d[d[lc].notna().all(axis=1)].reset_index(drop=True)
    lc = [c for c in lc if d[c].nunique() > 1]
    print(f"loaded {len(d)} rows, {len(lc)} descriptors")

    rows, env_rows, lines = [], [], []
    for sp_i, sp in enumerate(FOCAL):
        pres = d[d.Crayfish_scientific_name == sp].reset_index(drop=True)
        if len(pres) < 200:
            print(f"skip {sp}"); continue
        t0 = time.time()
        bg = d[d.Crayfish_scientific_name != sp]
        if len(bg) > N_BACKGROUND:
            bg = bg.sample(n=N_BACKGROUND, random_state=SEED)
        bg = bg.reset_index(drop=True)

        r, p_oof = oof_propensity(pres, lc)
        hi = r == 1
        auc = roc_auc_score(r, p_oof) if len(np.unique(r)) > 1 else np.nan
        w_point = np.clip(1 / np.clip(p_oof[hi], 1e-3, None), None,
                          np.percentile(1 / np.clip(p_oof[hi], 1e-3, None), 99))
        ess = ess_frac(w_point)
        gated = ess < ESS_GATE

        Xp, Xb = pres[lc].values, bg[lc].values
        yb = np.zeros(len(bg))

        def fit_predict(Xpres, w=None, seed_off=0):
            X = np.vstack([Xpres, Xb]); y = np.concatenate([np.ones(len(Xpres)), yb])
            sw = np.concatenate([np.ones(len(Xpres)) if w is None else w, np.ones(len(bg))])
            m = make_model(SEED + sp_i + seed_off)
            m.fit(X, y, rf__sample_weight=sw)
            return m.predict_proba(Xb)[:, 1]

        surfaces = {"ALL": fit_predict(Xp),
                    "FILTER": fit_predict(Xp[hi]),
                    "IPW": fit_predict(Xp[hi], w=w_point)}
        for kap in KAPPAS:
            if kap == 1.0:
                continue
            p_k = rescale(p_oof, kap, float(hi.mean()))
            w_k = 1 / np.clip(p_k[hi], 1e-3, None)
            w_k = np.clip(w_k, None, np.percentile(w_k, 99))
            surfaces[f"IPW_k{kap:g}"] = fit_predict(Xp[hi], w=w_k, seed_off=int(kap * 10))

        base_pairs = [("ALL", "FILTER"), ("ALL", "IPW"), ("FILTER", "IPW")]
        for a, b in base_pairs:
            rows.append(dict(species=sp, pair=f"{a}->{b}", ipw_gated=gated,
                             env_auc=auc, ess_p99=ess, **surf_metrics(surfaces[a], surfaces[b])))
        ks = [f"IPW_k{k:g}" if k != 1.0 else "IPW" for k in KAPPAS]
        env_D = schoener_d(surfaces[ks[0]], surfaces[ks[-1]])
        d_to_filter = {k: schoener_d(surfaces[k], surfaces["FILTER"]) for k in ks}
        env_rows.append(dict(species=sp, env_auc=auc, ess_p99=ess, ipw_gated=gated,
                             env_width_D=env_D,
                             **{f"D_vs_FILTER_{k}": v for k, v in d_to_filter.items()}))
        lines.append(f"  {sp[:30]:30s} AUC={auc:.3f} ESS={ess:.3f} gated={'YES' if gated else 'no '}  "
                     f"D(ALL,FILTER)={surf_metrics(surfaces['ALL'],surfaces['FILTER'])['schoener_D']:.3f}  "
                     f"D(FILTER,IPW)={surf_metrics(surfaces['FILTER'],surfaces['IPW'])['schoener_D']:.3f}  "
                     f"env_width={env_D:.3f}   [{time.time()-t0:.0f}s]")
        print(lines[-1])

    pd.DataFrame(rows).to_csv(os.path.join(REPORTS, "emp2_surfaces_summary.csv"), index=False)
    pd.DataFrame(env_rows).to_csv(os.path.join(REPORTS, "emp2_envelope.csv"), index=False)
    out = ["=" * 108,
           "EMP 2 — CONSEQUENCE RERUN, UNIFIED PROPENSITY + KAPPA-ENVELOPE  (RF learner, target-group background)",
           "=" * 108] + lines + [
           "",
           f"GATING RULE: ESS(p99) < {ESS_GATE} -> IPW point estimate NOT REPORTABLE; envelope only.",
           "pairwise metrics -> emp2_surfaces_summary.csv ; envelope -> emp2_envelope.csv", ""]
    text = "\n".join(out)
    with open(os.path.join(REPORTS, "emp2_verdict.txt"), "w") as fh:
        fh.write(text)
    print(text)
    print(f"total {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
