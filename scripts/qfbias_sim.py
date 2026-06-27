"""
qfbias_sim.py
=============
Simulation skeleton for the MEE paper:
    "When quality filtering becomes environmental sampling bias in SDMs."

GENERATIVE MODEL (signed off)
-----------------------------
  * Spatially autocorrelated environmental field E(g) = (E1, E2) over a 2D grid.
    E1 doubles as an "accessibility / remoteness" axis.
  * True Gaussian niche S(e); true occurrences are an inhomogeneous Poisson
    point process with intensity proportional to S(E(g)).
  * Record quality r in {0,1} with
        P(r = 1 | e) = sigmoid(alpha + beta * u(e)),   u(e) = standardized E1.
    alpha is *calibrated* to a fixed marginal retention rate, so that beta is a
    pure coupling knob (Knob 1) and is not confounded with the filtering fraction.
  * High-quality records keep their TRUE coordinates (but are environmentally
    selected -> p_high is a thinned version of p_true).
  * Low-quality records get displaced coordinates:
        g_obs = g_true + delta,
        delta = rho * D * v_hat + (1 - rho) * D * eps_iso
    rho = directedness of the error (Knob 2), D = error magnitude in grid units,
    v_hat = fixed geographic pull direction (a "road/coast corridor"); because E
    is autocorrelated this fixed geographic pull yields a consistent environmental
    shift. (A gradient-following variant is available via cfg.error_follow_gradient.)
  * Presence-background SDM with a quality-INDEPENDENT background (available env).

CORE IDENTITY
-------------
    p_high(e) ∝ p_true(e) * P(r = 1 | e)
        =>  p_true(e) ∝ p_high(e) / P(r = 1 | e)
so inverse-propensity weights  w = 1 / P(r = 1 | e)  applied to the high-quality
sample reconstruct the true niche in the clean limit. The three treatments

    ALL     : every record at its OBSERVED coordinates (contaminated by error)
    FILTER  : high-quality only, true coords (environmentally selected -> biased)
    IPW     : high-quality, weighted by 1 / P(r=1|e)  (oracle or estimated)

trace out the decision space over (beta, rho, D). The honest punchline is that
none of the three dominates globally; which is least-bad is a function of the knobs.

The learner is a logistic GLM with linear + quadratic env terms. Because log S is
quadratic in e, this GLM is *correctly specified* for the Gaussian niche, so under
beta=0, rho=0 the algorithmic bias is ~0 and any residual error is sampling bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    # grid / environment
    grid: int = 200                 # H = W = grid
    corr_len: float = 14.0          # Gaussian smoothing sigma (cells) -> autocorrelation
    # true niche (in standardized env units)
    niche_center: Tuple[float, float] = (0.0, 0.0)
    niche_sd: Tuple[float, float] = (0.65, 0.65)
    n_occ: int = 3000               # expected number of true occurrences
    # quality model
    beta: float = 0.0               # Knob 1: env<->quality coupling strength
    retain_rate: float = 0.60       # marginal P(r=1); alpha calibrated to hit this
    # coordinate error for low-quality records
    rho: float = 0.0                # Knob 2: directedness (0 = isotropic, 1 = systematic)
    D: float = 12.0                 # error magnitude (grid cells)
    error_dir: Tuple[float, float] = (0.0, 1.0)   # fixed geographic pull (row, col)
    error_follow_gradient: bool = True            # snap UP the E1 (accessibility) gradient
    # SDM
    n_bg: int = 10000               # background points (quality-independent)
    weight_trim_pct: float = 99.0   # cap IPW weights at this percentile (stability)
    # rng
    env_seed: int = 0               # field is fixed across occurrence seeds by default


# --------------------------------------------------------------------------- #
# Environment & true niche
# --------------------------------------------------------------------------- #
def make_env_field(cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two spatially autocorrelated, standardized environmental layers + grad(E1)."""
    rng = np.random.default_rng(cfg.env_seed)
    layers = []
    for _ in range(2):
        white = rng.standard_normal((cfg.grid, cfg.grid))
        smooth = gaussian_filter(white, sigma=cfg.corr_len, mode="reflect")
        smooth = (smooth - smooth.mean()) / smooth.std()
        layers.append(smooth)
    E1, E2 = layers
    gE1_r, gE1_c = np.gradient(E1)            # gradient of the accessibility axis
    return E1, E2, gE1_r, gE1_c


def true_suitability(E1: np.ndarray, E2: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Gaussian response surface, normalized to max 1."""
    cx, cy = cfg.niche_center
    sx, sy = cfg.niche_sd
    S = np.exp(-0.5 * (((E1 - cx) / sx) ** 2 + ((E2 - cy) / sy) ** 2))
    return S / S.max()


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #
def _env_at(coords: np.ndarray, fieldlist: List[np.ndarray], grid: int) -> np.ndarray:
    """Nearest-cell lookup with clipping. coords: (N,2) float (row,col)."""
    idx = np.clip(np.round(coords).astype(int), 0, grid - 1)
    return np.stack([f[idx[:, 0], idx[:, 1]] for f in fieldlist], axis=1)


def sample_true_occurrences(S: np.ndarray, cfg: SimConfig, rng) -> np.ndarray:
    """Thin the grid by S to get true occurrence coordinates (continuous, jittered)."""
    p = (S / S.sum()).ravel()
    flat = rng.choice(p.size, size=cfg.n_occ, replace=True, p=p)
    rows, cols = np.divmod(flat, cfg.grid)
    coords = np.stack([rows, cols], axis=1).astype(float)
    coords += rng.uniform(-0.5, 0.5, size=coords.shape)     # sub-cell jitter
    return coords


def _calibrate_alpha(u: np.ndarray, beta: float, target: float) -> float:
    """Solve E[sigmoid(alpha + beta*u)] = target for alpha (bisection)."""
    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        m = 1.0 / (1.0 + np.exp(-(mid + beta * u)))
        if m.mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def assign_quality(e_occ: np.ndarray, cfg: SimConfig, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Return (r, p_true_quality) where p = P(r=1|e). u(e) = standardized E1."""
    u = e_occ[:, 0]
    u = (u - u.mean()) / (u.std() + 1e-12)
    alpha = _calibrate_alpha(u, cfg.beta, cfg.retain_rate)
    p = 1.0 / (1.0 + np.exp(-(alpha + cfg.beta * u)))
    r = (rng.uniform(size=p.size) < p).astype(int)
    return r, p


def apply_coord_error(coords: np.ndarray, cfg: SimConfig, grad: Tuple[np.ndarray, np.ndarray], rng) -> np.ndarray:
    """Displace low-quality coordinates: rho*systematic + (1-rho)*isotropic, scaled by D."""
    n = coords.shape[0]
    if cfg.error_follow_gradient:
        gE1_r, gE1_c = grad
        gv = _env_at(coords, [gE1_r, gE1_c], cfg.grid)      # +grad(E1): snap toward accessible
        norm = np.linalg.norm(gv, axis=1, keepdims=True) + 1e-12
        v = gv / norm
    else:
        v = np.broadcast_to(np.array(cfg.error_dir, float), (n, 2))
        v = v / (np.linalg.norm(cfg.error_dir) + 1e-12)
    iso = rng.standard_normal((n, 2))
    iso /= (np.linalg.norm(iso, axis=1, keepdims=True) + 1e-12)
    delta = cfg.rho * cfg.D * v + (1.0 - cfg.rho) * cfg.D * iso
    return np.clip(coords + delta, 0, cfg.grid - 1)


def sample_background(cfg: SimConfig, rng) -> np.ndarray:
    """Quality-independent background = uniform over available environment."""
    return rng.uniform(0, cfg.grid - 1, size=(cfg.n_bg, 2))


# --------------------------------------------------------------------------- #
# SDM (correctly-specified GLM) and surface prediction
# --------------------------------------------------------------------------- #
def _design(e: np.ndarray) -> np.ndarray:
    """[e1, e2, e1^2, e2^2] — log-intensity of a Gaussian niche is in this span."""
    e1, e2 = e[:, 0], e[:, 1]
    return np.stack([e1, e2, e1 ** 2, e2 ** 2], axis=1)


def fit_sdm(e_pres: np.ndarray, e_bg: np.ndarray, w_pres: np.ndarray | None = None) -> LogisticRegression:
    X = np.vstack([_design(e_pres), _design(e_bg)])
    y = np.concatenate([np.ones(len(e_pres)), np.zeros(len(e_bg))])
    w = np.concatenate([
        np.ones(len(e_pres)) if w_pres is None else w_pres,
        np.ones(len(e_bg)),
    ])
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    clf.fit(X, y, sample_weight=w)
    return clf


def predict_surface(clf: LogisticRegression, E1: np.ndarray, E2: np.ndarray) -> np.ndarray:
    """Relative suitability = exp(linear predictor), as a 2D grid."""
    e = np.stack([E1.ravel(), E2.ravel()], axis=1)
    lp = clf.decision_function(_design(e))
    lp -= lp.max()
    return np.exp(lp).reshape(E1.shape)


# --------------------------------------------------------------------------- #
# Distance-to-truth metrics (SDM-familiar)
# --------------------------------------------------------------------------- #
def _norm(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, None)
    return x / x.sum()


def schoener_d(pred: np.ndarray, true: np.ndarray) -> float:
    p, q = _norm(pred).ravel(), _norm(true).ravel()
    return 1.0 - 0.5 * np.abs(p - q).sum()


def warren_i(pred: np.ndarray, true: np.ndarray) -> float:
    p, q = _norm(pred).ravel(), _norm(true).ravel()
    return 1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)


def spearman_surface(pred: np.ndarray, true: np.ndarray) -> float:
    return float(spearmanr(pred.ravel(), true.ravel()).statistic)


# --------------------------------------------------------------------------- #
# Overlap / positivity diagnostics for IPW
# --------------------------------------------------------------------------- #
def overlap_diagnostics(p_hat_hi: np.ndarray) -> Dict[str, float]:
    """Positivity diagnostics from the IPW weights on the high-quality sample."""
    w = 1.0 / np.clip(p_hat_hi, 1e-3, None)
    ess = (w.sum() ** 2) / (np.sum(w ** 2) + 1e-12)
    return {
        "ess_frac": ess / len(w),
        "w_max_over_mean": w.max() / w.mean(),
        "w_p99": np.percentile(w, 99),
    }


# --------------------------------------------------------------------------- #
# One run: build a world, fit the three strategies, score against truth
# --------------------------------------------------------------------------- #
def run_one(cfg: SimConfig, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    E1, E2, gE1_r, gE1_c = make_env_field(cfg)
    S = true_suitability(E1, E2, cfg)

    # true occurrences
    coords = sample_true_occurrences(S, cfg, rng)
    e_true = _env_at(coords, [E1, E2], cfg.grid)

    # quality + observed coordinates
    r, p_q = assign_quality(e_true, cfg, rng)
    coords_obs = coords.copy()
    low = r == 0
    coords_obs[low] = apply_coord_error(coords[low], cfg, (gE1_r, gE1_c), rng)
    e_obs = _env_at(coords_obs, [E1, E2], cfg.grid)        # env at OBSERVED coords

    # background (fixed across strategies)
    bg = sample_background(cfg, rng)
    e_bg = _env_at(bg, [E1, E2], cfg.grid)

    hi = r == 1
    e_hi_true = e_true[hi]                                  # high-quality, correct coords

    # estimated propensity from the occurrence sample (practitioner view)
    prop = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    prop.fit(_design(e_true), r)
    p_hat_all = prop.predict_proba(_design(e_true))[:, 1]
    p_hat = p_hat_all[hi]
    prop_auc = float(roc_auc_score(r, p_hat_all)) if r.min() != r.max() else float("nan")

    def trim(w):
        cap = np.percentile(w, cfg.weight_trim_pct)
        return np.clip(w, None, cap)

    w_oracle = trim(1.0 / np.clip(p_q[hi], 1e-3, None))
    w_est = trim(1.0 / np.clip(p_hat, 1e-3, None))

    surfaces = {
        "ALL":        predict_surface(fit_sdm(e_obs, e_bg), E1, E2),
        "FILTER":     predict_surface(fit_sdm(e_hi_true, e_bg), E1, E2),
        "IPW_oracle": predict_surface(fit_sdm(e_hi_true, e_bg, w_oracle), E1, E2),
        "IPW_est":    predict_surface(fit_sdm(e_hi_true, e_bg, w_est), E1, E2),
    }

    out: Dict[str, float] = {
        "beta": cfg.beta, "rho": cfg.rho, "D": cfg.D, "seed": seed,
        "retain_obs": float(hi.mean()),
    }
    for name, surf in surfaces.items():
        out[f"D_{name}"] = schoener_d(surf, S)
        out[f"I_{name}"] = warren_i(surf, S)
        out[f"rho_sp_{name}"] = spearman_surface(surf, S)
    out.update({f"diag_{k}": v for k, v in overlap_diagnostics(p_hat).items()})
    out["diag_propensity_auc"] = prop_auc
    return out


# --------------------------------------------------------------------------- #
# Sweep over the two knobs
# --------------------------------------------------------------------------- #
def run_sweep(betas, rhos, Ds, seeds, base: SimConfig):
    import pandas as pd
    rows = []
    for beta in betas:
        for rho in rhos:
            for Dv in Ds:
                cfg = SimConfig(**{**base.__dict__, "beta": beta, "rho": rho, "D": Dv})
                for s in seeds:
                    rows.append(run_one(cfg, s))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)

    base = SimConfig(grid=160, n_occ=3000, n_bg=8000, corr_len=12.0, D=10.0)
    betas = [0.0, 3.0]
    rhos = [0.0, 1.0]
    seeds = list(range(8))

    print("Running validation sweep (beta x rho), 8 seeds each ...\n")
    df = run_sweep(betas, rhos, [base.D], seeds, base)

    metric_cols = ["rho_sp_ALL", "rho_sp_FILTER", "rho_sp_IPW_oracle", "rho_sp_IPW_est"]
    g = df.groupby(["beta", "rho"])[metric_cols + ["diag_propensity_auc", "diag_ess_frac"]].mean()
    print("Mean Spearman(predicted vs true suitability) and IPW diagnostics:\n")
    print(g.round(3).to_string())
