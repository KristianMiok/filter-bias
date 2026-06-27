"""
qfbias_diagnose.py
==================
The OBSERVABLE half of the quality-filtering-bias workflow.

Runs on a real occurrence table with NO ground truth, NO rasters, NO background --
just a flat dataframe of {environmental values per record, a quality flag,
optional metadata}. It reports exactly the quantities that locate a real dataset
on the simulation's decision map (qfbias_sim.py):

  A. Per-axis environmental shift induced by filtering   (which niche axes move)
        {all records}  vs  {high-quality records}
  B. Multivariate shift magnitude (model-free)           (is filtering env-neutral?)
        energy distance + permutation p-value
  C. Propensity + positivity + confounder audit          (how far can IPW be trusted?)
        P(high | env), P(high | metadata), P(high | env+metadata)  with spatial/group CV
        IPW weight distribution + effective sample size

What it deliberately does NOT do: estimate rho/D (not identifiable without true
locations) or fit the SDM consequence (that plugs into the user's existing
Protocol B pipeline once a non-trivial regime is confirmed).

Reading the output against the simulation:
  * propensity AUC (env-only) ~ the beta axis: 0.5 = no coupling (filtering harmless),
    higher = stronger coupling (filtering increasingly acts as sampling bias).
  * ESS fraction = positivity stress: low -> IPW reconstruction is fragile, trim/cap.
  * incremental env-over-metadata AUC: how much of the quality structure is genuinely
    environmental vs a metadata (source/era/method) proxy -> mechanism & guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class DiagnosisResult:
    n_total: int
    n_high: int
    n_low: int
    retain_rate: float
    per_axis: pd.DataFrame                 # SMD + KS per environmental variable
    energy_distance: float                 # {all} vs {high}, standardized env space
    energy_p: float
    auc_env: float
    auc_meta: Optional[float]
    auc_env_meta: Optional[float]
    incr_env_over_meta: Optional[float]
    incr_meta_over_env: Optional[float]
    ess_frac: float
    w_max_over_mean: float
    w_p99: float
    n_extreme_weight: int
    cv_scheme: str
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        L = []
        L.append("=" * 68)
        L.append("QUALITY-FILTERING BIAS DIAGNOSTIC  (observable half, no ground truth)")
        L.append("=" * 68)
        L.append(f"records: {self.n_total}  |  high-quality: {self.n_high} "
                 f"({self.retain_rate:.1%})  |  low: {self.n_low}")
        L.append(f"propensity CV scheme: {self.cv_scheme}")
        L.append("")
        L.append("A. Per-axis environmental shift  {all} -> {high}  (filtering effect)")
        L.append("   (positive SMD = high-quality records sit at higher values of that axis)")
        with pd.option_context("display.max_rows", None, "display.width", 120):
            L.append(self.per_axis.round(3).to_string(index=False))
        L.append("")
        L.append("B. Multivariate shift magnitude (model-free)")
        L.append(f"   energy distance = {self.energy_distance:.4f}   "
                 f"permutation p = {self.energy_p:.4f}")
        sig = "non-neutral" if self.energy_p < 0.05 else "not distinguishable from neutral"
        L.append(f"   -> filtering is {sig} in environmental space")
        L.append("")
        L.append("C. Propensity + confounder audit + positivity")
        L.append(f"   AUC  P(high | env)              = {self.auc_env:.3f}")
        if self.auc_meta is not None:
            L.append(f"   AUC  P(high | metadata)         = {self.auc_meta:.3f}")
            L.append(f"   AUC  P(high | env + metadata)   = {self.auc_env_meta:.3f}")
            L.append(f"   incremental env  over metadata  = {self.incr_env_over_meta:+.3f}")
            L.append(f"   incremental meta over env       = {self.incr_meta_over_env:+.3f}")
        L.append(f"   IPW effective sample size       = {self.ess_frac:.1%} of n_high")
        L.append(f"   weight max/mean = {self.w_max_over_mean:.1f}   "
                 f"p99 = {self.w_p99:.2f}   extreme(>10x mean) = {self.n_extreme_weight}")
        L.append("")
        L.append("READING (vs simulation):")
        L.append(self._reading())
        if self.notes:
            L.append("")
            L.append("notes: " + " | ".join(self.notes))
        return "\n".join(L)

    def _reading(self) -> str:
        # coupling regime from env-only propensity AUC
        a = self.auc_env
        if a < 0.58:
            reg = ("   * Weak coupling (AUC~0.5): filtering is close to harmless. "
                   "Hard filtering is defensible; the paper's effect is small here.")
        elif a < 0.72:
            reg = ("   * Moderate coupling: filtering acts as a real but bounded "
                   "sampling-bias mechanism. IPW is usable WITH weight hygiene.")
        else:
            reg = ("   * Strong coupling: filtering is a serious sampling-bias "
                   "mechanism; but positivity is likely stressed (check ESS).")
        ess = ("" if self.ess_frac > 0.5 else
               "\n   * LOW ESS: parts of the niche have few/no high-quality records; "
               "IPW cannot reconstruct them -- trim weights and report sensitivity.")
        conf = ""
        if self.incr_env_over_meta is not None:
            if self.incr_env_over_meta < 0.02:
                conf = ("\n   * Quality structure is largely a METADATA proxy (env adds "
                        "little over source/era/method). Frame as: the consequence holds "
                        "via env-space non-randomness regardless of causal label; consider "
                        "modeling collection method as the alternative to discarding.")
            else:
                conf = ("\n   * Quality structure has a residual ENVIRONMENTAL component "
                        "beyond metadata -- this is the defensible core of the claim.")
        return reg + ess + conf


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _quad(X: np.ndarray) -> np.ndarray:
    """[X, X^2] -- log-intensity of a Gaussian niche lives in this span."""
    return np.hstack([X, X ** 2])


def _cv(y: np.ndarray, groups: Optional[np.ndarray]) -> tuple:
    if groups is not None and len(np.unique(groups)) >= 3:
        k = min(5, len(np.unique(groups)))
        return GroupKFold(n_splits=k), groups, f"GroupKFold(k={k}) on group column"
    k = 5
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=0), None, \
        f"StratifiedKFold(k={k}) [no group column -> AUC may be optimistic under spatial autocorrelation]"


def _oof_auc(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray]) -> tuple:
    """Out-of-fold AUC and OOF positive-class probabilities."""
    cv, g, scheme = _cv(y, groups)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)
    proba = cross_val_predict(clf, X, y, cv=cv, groups=g, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba)), proba, scheme


def _energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    a = cdist(X, Y).mean()
    b = cdist(X, X).mean()
    c = cdist(Y, Y).mean()
    return float(2 * a - b - c)


def _energy_perm_test(Xall: np.ndarray, Xhigh: np.ndarray, n_sub: int, n_perm: int, rng) -> tuple:
    """Subsampled energy distance {all} vs {high} with a permutation p-value."""
    na = min(len(Xall), n_sub)
    nh = min(len(Xhigh), n_sub)
    A = Xall[rng.choice(len(Xall), na, replace=False)]
    H = Xhigh[rng.choice(len(Xhigh), nh, replace=False)]
    obs = _energy_distance(A, H)
    pool = np.vstack([A, H])
    n1 = len(A)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(pool))
        e = _energy_distance(pool[perm[:n1]], pool[perm[n1:]])
        if e >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def diagnose_quality_shift(
    df: pd.DataFrame,
    env_cols: List[str],
    quality_col: str,
    quality_threshold: Optional[float] = None,
    smaller_is_better: bool = True,
    metadata_cols: Optional[List[str]] = None,
    group_col: Optional[str] = None,
    propensity_pca: Optional[int] = None,
    n_sub: int = 2000,
    n_perm: int = 200,
    seed: int = 0,
) -> DiagnosisResult:
    """
    Parameters
    ----------
    env_cols : environmental covariate columns.
    quality_col : binary quality flag (1 = high quality) OR a continuous quality/
        uncertainty column (then supply quality_threshold).
    quality_threshold : if given, quality_col is continuous and is thresholded.
    smaller_is_better : for a continuous uncertainty column (e.g.
        coordinateUncertaintyInMeters), smaller value -> higher quality.
    metadata_cols : optional categorical/numeric metadata for the confounder audit
        (e.g. source, basisOfRecord, year-bin, collection method).
    group_col : optional spatial group for CV (e.g. basin) -> GroupKFold.
    """
    rng = np.random.default_rng(seed)
    notes: List[str] = []
    d = df.copy()

    # --- build binary high-quality label ---
    if quality_threshold is not None:
        q = pd.to_numeric(d[quality_col], errors="coerce")
        hi = (q <= quality_threshold) if smaller_is_better else (q >= quality_threshold)
        miss = q.isna().sum()
        if miss:
            notes.append(f"{miss} records have missing '{quality_col}' (dropped); "
                         "NOTE missingness is itself often non-random.")
        d = d.loc[q.notna()].copy()
        hi = hi.loc[q.notna()]
    else:
        hi = d[quality_col].astype(int) == 1

    # --- drop rows with missing env ---
    before = len(d)
    d = d.loc[d[env_cols].notna().all(axis=1)].copy()
    hi = hi.loc[d.index]
    if len(d) < before:
        notes.append(f"{before - len(d)} records dropped for missing environmental values.")

    y = hi.astype(int).to_numpy()
    E = d[env_cols].to_numpy(dtype=float)
    n_total, n_high = len(d), int(y.sum())
    n_low = n_total - n_high
    if n_high < 30 or n_low < 30:
        notes.append("WARNING: very few records in one quality class; estimates unstable.")

    # standardize env once (for shift metrics and as propensity base)
    scaler = StandardScaler().fit(E)
    Ez = scaler.transform(E)

    # ---------------- A. per-axis shift {all} vs {high} ----------------
    rows = []
    for j, name in enumerate(env_cols):
        x_all = E[:, j]
        x_hi = E[y == 1, j]
        pooled_sd = x_all.std() + 1e-12
        smd = (x_hi.mean() - x_all.mean()) / pooled_sd
        ks = ks_2samp(x_hi, E[y == 0, j]).statistic
        rows.append({"env_var": name, "mean_all": x_all.mean(),
                     "mean_high": x_hi.mean(), "SMD_all_to_high": smd,
                     "KS_high_vs_low": ks})
    per_axis = pd.DataFrame(rows).sort_values("SMD_all_to_high",
                                              key=lambda s: s.abs(), ascending=False)

    # ---------------- B. multivariate shift magnitude ----------------
    energy, energy_p = _energy_perm_test(Ez, Ez[y == 1], n_sub, n_perm, rng)

    # ---------------- C. propensity + confounder audit ----------------
    groups = d[group_col].to_numpy() if group_col else None
    if propensity_pca is not None and propensity_pca < Ez.shape[1]:
        pca = PCA(n_components=propensity_pca, random_state=seed).fit(Ez)
        Xenv = _quad(pca.transform(Ez))
        notes.append(f"propensity env reduced to {propensity_pca} PCs "
                     f"({pca.explained_variance_ratio_.sum():.0%} variance) for a "
                     "well-conditioned, convergent model; per-axis shift and energy "
                     "distance still use all raw features.")
    else:
        Xenv = _quad(Ez)
    auc_env, proba_env, cv_scheme = _oof_auc(Xenv, y, groups)

    auc_meta = auc_env_meta = incr_env_over_meta = incr_meta_over_env = None
    if metadata_cols:
        M = pd.get_dummies(d[metadata_cols].astype("category"), drop_first=False)
        if M.shape[1] > 200:
            notes.append(f"metadata expands to {M.shape[1]} dummies; consider binning "
                         "high-cardinality columns.")
        Xmeta = M.to_numpy(dtype=float)
        auc_meta, _, _ = _oof_auc(Xmeta, y, groups)
        auc_env_meta, _, _ = _oof_auc(np.hstack([Xenv, Xmeta]), y, groups)
        incr_env_over_meta = auc_env_meta - auc_meta
        incr_meta_over_env = auc_env_meta - auc_env

    # ---------------- positivity diagnostics (IPW weights on high-q) -----
    p_hi = proba_env[y == 1]
    w = 1.0 / np.clip(p_hi, 1e-3, None)
    ess = (w.sum() ** 2) / (np.sum(w ** 2) + 1e-12)
    w_mean = w.mean()
    n_extreme = int((w > 10 * w_mean).sum())

    return DiagnosisResult(
        n_total=n_total, n_high=n_high, n_low=n_low, retain_rate=n_high / n_total,
        per_axis=per_axis, energy_distance=energy, energy_p=energy_p,
        auc_env=auc_env, auc_meta=auc_meta, auc_env_meta=auc_env_meta,
        incr_env_over_meta=incr_env_over_meta, incr_meta_over_env=incr_meta_over_env,
        ess_frac=ess / len(w), w_max_over_mean=w.max() / w_mean,
        w_p99=float(np.percentile(w, 99)), n_extreme_weight=n_extreme,
        cv_scheme=cv_scheme, notes=notes,
    )


# --------------------------------------------------------------------------- #
# Demo: generate a low- and a high-coupling world, confirm the diagnostic reads them
# --------------------------------------------------------------------------- #
def _synthetic_table(beta: float, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # two correlated environmental axes
    e1 = rng.standard_normal(n)
    e2 = 0.5 * e1 + np.sqrt(1 - 0.25) * rng.standard_normal(n)
    # gaussian niche thinning -> keep occurrences
    S = np.exp(-0.5 * ((e1 / 0.8) ** 2 + (e2 / 0.8) ** 2))
    keep = rng.uniform(size=n) < S / S.max()
    e1, e2 = e1[keep], e2[keep]
    m = e1.size
    # quality coupled to e1 (calibrated to ~60% high), plus a metadata proxy
    u = (e1 - e1.mean()) / (e1.std() + 1e-12)
    lo, hi = -50, 50
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (1 / (1 + np.exp(-(mid + beta * u)))).mean() < 0.6:
            lo = mid
        else:
            hi = mid
    p = 1 / (1 + np.exp(-(0.5 * (lo + hi) + beta * u)))
    q = (rng.uniform(size=m) < p).astype(int)
    source = np.where(u + 0.5 * rng.standard_normal(m) > 0, "gps", "gazetteer")  # correlated metadata
    basin = pd.qcut(e2 + 0.3 * rng.standard_normal(m), 6, labels=False)          # spatial-ish group
    return pd.DataFrame({"e1": e1, "e2": e2, "high_quality": q,
                         "source": source, "basin": basin})


if __name__ == "__main__":
    for beta, tag in [(0.4, "WEAK coupling"), (2.5, "STRONG coupling")]:
        print(f"\n########## synthetic world: {tag} (beta={beta}) ##########")
        tbl = _synthetic_table(beta, n=6000, seed=1)
        res = diagnose_quality_shift(
            tbl, env_cols=["e1", "e2"], quality_col="high_quality",
            metadata_cols=["source"], group_col="basin", n_perm=150,
        )
        print(res.summary())
