"""
s5_xgb_tuned.py
===============
SUPPLEMENT S5 — learner-dependence check with a TUNED gradient-boosted learner
(R1 #10: "an untuned comparison is insufficient").

Twin of emp2_consequence_unified.py with one change: the SDM learner is
XGBoost, tuned per species by basin-blocked GroupKFold(3) AUC over a small
grid (max_depth {4, 6} x learning_rate {0.05, 0.1}; 400 trees, subsample 0.8,
colsample_bytree 0.8). Everything else identical: unified basin-blocked
PCA(15)+quadratic logistic propensity (emp1 spec), strategies ALL / FILTER /
IPW, Lambda = 2 envelope (kappa in {0.5, 1, 2}), ESS(p99) < 0.30 gating.

The verdict prints XGB next to the committed RF values (emp2_envelope.csv,
emp2_surfaces_summary.csv) so the S5 table is one paste.

Outputs
    reports/s5_xgb_species_table.csv
    reports/s5_xgb_verdict.txt

Run from the repo root:   python scripts/s5_xgb_tuned.py
(expects data/raw/combined_data_true_master.csv; needs `pip install xgboost`;
 ~25-45 min. Set S5_TREES=50 for a fast dry run.)
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "raw", "combined_data_true_master.csv")
REPORTS = os.path.join(ROOT, "reports")

FOCAL = ["Astacus astacus", "Austropotamobius pallipes", "Pacifastacus leniusculus",
         "Procambarus clarkii", "Faxonius limosus", "Pontastacus leptodactylus",
         "Austropotamobius torrentium"]
N_BACKGROUND, SEED = 20000, 42
N_TREES = int(os.environ.get("S5_TREES", "400"))
GRID = [dict(max_depth=d, learning_rate=lr) for d in (4, 6) for lr in (0.05, 0.1)]
KAPPAS = (0.5, 1.0, 2.0)          # Lambda = 2, the paper's primary envelope
ESS_GATE = 0.30
RNG = np.random.default_rng(SEED)


def local_cols(d):
    return [c for c in d.columns if c.startswith(("l_CLI", "l_TOP", "l_LAC", "l_SOL"))]


def make_xgb(seed, **hp):
    return XGBClassifier(n_estimators=N_TREES, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", eval_metric="logloss", n_jobs=-1,
                         random_state=seed, **hp)


def oof_propensity(d, lc):
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


def tune(Xp, Xb, gp, gb, sp_i):
    """Basin-blocked GroupKFold(3) AUC on the presence-background task."""
    X = np.vstack([Xp, Xb]); y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xb))])
    g = np.concatenate([gp, gb])
    best, best_auc = GRID[0], -1.0
    gk = GroupKFold(n_splits=3)
    for hp in GRID:
        aucs = []
        for tr, te in gk.split(X, y, g):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = make_xgb(SEED + sp_i, **hp)
            m.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        a = float(np.mean(aucs)) if aucs else np.nan
        if np.isfinite(a) and a > best_auc:
            best, best_auc = hp, a
    return best, best_auc


def main():
    os.makedirs(REPORTS, exist_ok=True)
    d = pd.read_csv(DATA, low_memory=False)
    lc = local_cols(d)
    d = d[d[lc].notna().all(axis=1)].reset_index(drop=True)
    lc = [c for c in lc if d[c].nunique() > 1]
    print(f"loaded {len(d)} rows, {len(lc)} descriptors, trees={N_TREES}")

    try:
        rf_env = pd.read_csv(os.path.join(REPORTS, "emp2_envelope.csv")).set_index("species")
        rf_sur = pd.read_csv(os.path.join(REPORTS, "emp2_surfaces_summary.csv"))
    except FileNotFoundError:
        rf_env, rf_sur = None, None

    rows, lines = [], []
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
        w_raw = 1 / np.clip(p_oof[hi], 1e-3, None)
        w_point = np.clip(w_raw, None, np.percentile(w_raw, 99))
        ess = ess_frac(w_point)
        gated = ess < ESS_GATE

        Xp, Xb = pres[lc].values, bg[lc].values
        gp, gb = pres.basin_id.values, bg.basin_id.values
        hp, cv_auc = tune(Xp, Xb, gp, gb, sp_i)

        def fit_predict(Xpres, w=None, seed_off=0):
            X = np.vstack([Xpres, Xb]); y = np.concatenate([np.ones(len(Xpres)), np.zeros(len(bg))])
            sw = np.concatenate([np.ones(len(Xpres)) if w is None else w, np.ones(len(bg))])
            m = make_xgb(SEED + sp_i + seed_off, **hp)
            m.fit(X, y, sample_weight=sw)
            return m.predict_proba(Xb)[:, 1]

        surfaces = {"ALL": fit_predict(Xp), "FILTER": fit_predict(Xp[hi]),
                    "IPW": fit_predict(Xp[hi], w=w_point)}
        for kap in KAPPAS:
            if kap == 1.0:
                continue
            p_k = rescale(p_oof, kap, float(hi.mean()))
            w_k = 1 / np.clip(p_k[hi], 1e-3, None)
            w_k = np.clip(w_k, None, np.percentile(w_k, 99))
            surfaces[f"IPW_k{kap:g}"] = fit_predict(Xp[hi], w=w_k, seed_off=int(kap * 10))

        d_af = schoener_d(surfaces["ALL"], surfaces["FILTER"])
        d_fi = schoener_d(surfaces["FILTER"], surfaces["IPW"])
        envw = schoener_d(surfaces["IPW_k0.5"], surfaces["IPW_k2"])
        rows.append(dict(species=sp, learner="xgb_tuned", max_depth=hp["max_depth"],
                         learning_rate=hp["learning_rate"], cv_auc_sdm=cv_auc,
                         env_auc=auc, ess_p99=ess, ipw_gated=gated,
                         D_ALL_FILTER=d_af, D_FILTER_IPW=d_fi, env_width_L2=envw))
        rf_af = rf_fi = rf_w = np.nan
        if rf_sur is not None:
            m = rf_sur[(rf_sur.species == sp)]
            if len(m):
                rf_af = float(m[m.pair == "ALL->FILTER"].schoener_D.iloc[0])
                rf_fi = float(m[m.pair == "FILTER->IPW"].schoener_D.iloc[0])
        if rf_env is not None and sp in rf_env.index:
            if "env_width_D_L2" in rf_env.columns:
                rf_w = float(rf_env.loc[sp, "env_width_D_L2"])
        lines.append(f"  {sp[:28]:28s} d{hp['max_depth']}/lr{hp['learning_rate']:g} "
                     f"cvAUC={cv_auc:.3f}  gated={'YES' if gated else 'no '}  "
                     f"D(A,F) xgb={d_af:.3f} rf={rf_af:.3f}   "
                     f"D(F,I) xgb={d_fi:.3f} rf={rf_fi:.3f}   "
                     f"envL2 xgb={envw:.3f} rf={rf_w:.3f}   [{time.time()-t0:.0f}s]")
        print(lines[-1])

    pd.DataFrame(rows).to_csv(os.path.join(REPORTS, "s5_xgb_species_table.csv"), index=False)
    txt = ["=" * 110,
           "S5 — TUNED XGBOOST vs RF (same unified propensity, same strategies; Lambda=2 envelope)",
           "=" * 110] + lines + [
           "",
           "Reading: if regime-relevant quantities (which pairs diverge, who is gated, envelope",
           "widths' ORDER across species) match the RF column, the framework's conclusions are",
           "learner-robust; absolute D values are expected to differ by learner smoothness.", ""]
    with open(os.path.join(REPORTS, "s5_xgb_verdict.txt"), "w") as fh:
        fh.write("\n".join(txt))
    print("\n".join(txt[-5:]))


if __name__ == "__main__":
    main()
