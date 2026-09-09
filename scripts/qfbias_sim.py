"""
qfbias_sim.py
=============
Simulation for the MEE paper:
    "When cleaning becomes biasing: quality-induced bias in SDMs."

CHANGELOG (R1 resubmission, Sept 2026)
--------------------------------------
Two implementation issues raised by Reviewer 1 (point 13) are fixed here.
Both are exposed as config flags so the ORIGINAL behaviour remains
reproducible for the old-vs-new comparison in the response to reviewers.

  FIX 1 - coordinate-error magnitude was not held constant across rho.
    Old:  delta = rho*D*v + (1-rho)*D*eps
          -> E|delta|^2 = D^2 * (rho^2 + (1-rho)^2); at rho=0.5, |delta| ~ 0.71 D.
          beta and rho were therefore NOT orthogonal as described.
    New:  cfg.fixed_magnitude=True normalises the blended direction so that
          |delta| == D for every record, for every rho. rho now changes ONLY
          the direction. (cfg.error_mode="mixture" is an alternative in which
          each low-quality record is displaced either fully along v with
          probability rho, or isotropically with probability 1-rho; magnitude
          is D in both branches.)

  FIX 2 - the retention propensity P(r=1|e) was fitted on e_true, including
    for records whose observed coordinates had been displaced. A practitioner
    only ever has e_obs.
    Old:  prop.fit(design(e_true), r)
    New:  cfg.propensity_on="obs" fits prop on e_obs (high-quality records
          have e_obs == e_true by construction; low-quality records enter at
          their DISPLACED environment). "true" reproduces the old behaviour
          and is retained only as an oracle-covariate reference.

  EXP 1 additions (Smith et al. 2023 comparator + uncertainty-aware propensity)
    Both use the SAME extra information a practitioner has: the uncertainty
    radius D around each low-quality record. They use it in opposite ways.
      SMITH_env  : impute each low-quality record to the cell within radius D
                   whose environment is CLOSEST to the environmental centroid of
                   the high-quality records (Smith et al. 2023 GEB, method 2)
      SMITH_geo  : same, but closest to the GEOGRAPHIC centroid (method 1)
      ALL_calib  : all records, low-quality at the MEAN environment over the disk
                   of radius D (regression calibration; Carroll et al.)
      IPW_calib  : IPW with the propensity fitted on disk-mean environment for
                   low-quality records (cfg.propensity_on="calib" makes IPW_est
                   use this too)

  Extra diagnostics recorded per run (for R1 #5/#6 and R2 #7):
    diag_prop_coef_E1   : fitted propensity coefficient on E1 (sign flip /
                          attenuation under directed error is the mechanism
                          by which fix 2 bites)
    diag_prop_brier     : Brier score of the propensity model (calibration,
                          not just discrimination)
    diag_lowq_dE1_shift : mean (E1_obs - E1_true) over low-quality records,
                          i.e. what the error model actually does in env space
    diag_ess_frac, diag_w_max_over_mean, diag_w_p99 : as before

GENERATIVE MODEL (unchanged)
----------------------------
  * Spatially autocorrelated environmental field E(g) = (E1, E2) over a 2D grid.
    E1 doubles as an "accessibility / remoteness" axis.
  * True Gaussian niche S(e); true occurrences are an inhomogeneous Poisson
    point process with intensity proportional to S(E(g)).
  * Record quality r in {0,1} with
        P(r = 1 | e) = sigmoid(alpha + beta * u(e)),   u(e) = standardized E1.
    alpha is calibrated to a fixed marginal retention rate, so beta is a pure
    coupling knob and is not confounded with the filtering fraction.
  * High-quality records keep their TRUE coordinates.
  * Low-quality records are displaced: g_obs = g_true + delta (see FIX 1).
  * Presence-background SDM with a quality-INDEPENDENT background.

CORE IDENTITY
-------------
    p_high(e) ∝ p_true(e) * P(r = 1 | e)
        =>  p_true(e) ∝ p_high(e) / P(r = 1 | e)

STRATEGIES
----------
    ALL        : every record at its OBSERVED coordinates
    FILTER     : high-quality only
    IPW_oracle : high-quality, weighted by 1 / p_q  (TRUE retention prob;
                 upper bound on what any propensity estimate can achieve)
    IPW_est    : high-quality, weighted by 1 / p_hat, p_hat fitted per
                 cfg.propensity_on  (this is the practitioner's IPW)

The learner is a logistic GLM with linear + quadratic env terms, correctly
specified for the Gaussian niche.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


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
    error_follow_gradient: bool = True            # snap along the E1 (accessibility) gradient
    error_gradient_sign: float = +1.0             # +1: toward high E1 (where high-quality sit)
                                                  # -1: toward low E1 (away from them)
    # ---- FIX 1 ----
    fixed_magnitude: bool = True    # True: |delta| == D for all rho. False: original (buggy)
    error_mode: str = "blend"       # "blend" (normalised rho*v + (1-rho)*eps) | "mixture"
    # ---- FIX 2 ----
    propensity_on: str = "obs"      # "obs": practitioner (e_obs). "true": original (e_true)
                                    # "calib": disk-mean env for low-quality (regression calibration)
    # ---- EXP 3: misspecification knobs (R1 #4) ----
    niche_shape: str = "gaussian"   # "gaussian" | "bimodal" | "skewed"
    quality_link: str = "linear"    # f(u) in logit: "linear" | "quadratic" | "threshold"
    env_corr: float = 0.0           # correlation induced between E1 and E2
    propensity_learner: str = "logit_quad"   # "logit_quad" | "logit_linear" | "rf"
    # ---- EXP 2: analytical grain ----
    grain: int = 1                  # block-average the env layers the modeller reads, by this
                                    # factor. Truth stays at the fine scale. grain > 1 raises the
                                    # effective field autocorrelation at lag D.
    # ---- EXP 4: kappa-envelope (sensitivity to multiplicative propensity corruption) ----
    kappa_grid: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)   # Lambda = 4
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
    if abs(cfg.env_corr) > 1e-9:
        rho_e = float(np.clip(cfg.env_corr, -0.99, 0.99))
        E2 = rho_e * E1 + np.sqrt(1 - rho_e ** 2) * E2
        E2 = (E2 - E2.mean()) / E2.std()
    gE1_r, gE1_c = np.gradient(E1)
    return E1, E2, gE1_r, gE1_c


def true_suitability(E1: np.ndarray, E2: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Gaussian response surface, normalized to max 1."""
    cx, cy = cfg.niche_center
    sx, sy = cfg.niche_sd
    if cfg.niche_shape == "gaussian":
        S = np.exp(-0.5 * (((E1 - cx) / sx) ** 2 + ((E2 - cy) / sy) ** 2))
    elif cfg.niche_shape == "bimodal":
        # two optima on E1: the quadratic learner cannot represent this
        S = (np.exp(-0.5 * (((E1 + 0.9) / (0.5 * sx)) ** 2 + ((E2 - cy) / sy) ** 2))
             + np.exp(-0.5 * (((E1 - 0.9) / (0.5 * sx)) ** 2 + ((E2 - cy) / sy) ** 2)))
    elif cfg.niche_shape == "skewed":
        # skew-normal-like on E1: asymmetric tolerance
        z = (E1 - cx) / sx
        S = np.exp(-0.5 * (z ** 2 + ((E2 - cy) / sy) ** 2)) * (1.0 / (1.0 + np.exp(-3.0 * z)))
    else:
        raise ValueError(f"unknown niche_shape {cfg.niche_shape!r}")
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
    coords += rng.uniform(-0.5, 0.5, size=coords.shape)
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
    if cfg.quality_link == "linear":
        f = u
    elif cfg.quality_link == "quadratic":
        f = u + 0.6 * (u ** 2 - 1.0)
    elif cfg.quality_link == "threshold":
        f = np.tanh(2.5 * u)
    else:
        raise ValueError(f"unknown quality_link {cfg.quality_link!r}")
    f = (f - f.mean()) / (f.std() + 1e-12)
    alpha = _calibrate_alpha(f, cfg.beta, cfg.retain_rate)
    p = 1.0 / (1.0 + np.exp(-(alpha + cfg.beta * f)))
    r = (rng.uniform(size=p.size) < p).astype(int)
    return r, p


def apply_coord_error(coords: np.ndarray, cfg: SimConfig,
                      grad: Tuple[np.ndarray, np.ndarray], rng) -> np.ndarray:
    """
    Displace low-quality coordinates.

    FIX 1: when cfg.fixed_magnitude is True, every displacement has |delta| == D
    regardless of rho, so rho controls direction only.
    """
    n = coords.shape[0]
    if cfg.error_follow_gradient:
        gE1_r, gE1_c = grad
        gv = _env_at(coords, [gE1_r, gE1_c], cfg.grid)
        norm = np.linalg.norm(gv, axis=1, keepdims=True) + 1e-12
        v = cfg.error_gradient_sign * gv / norm
    else:
        v = np.broadcast_to(np.array(cfg.error_dir, float), (n, 2))
        v = v / (np.linalg.norm(cfg.error_dir) + 1e-12)
    iso = rng.standard_normal((n, 2))
    iso /= (np.linalg.norm(iso, axis=1, keepdims=True) + 1e-12)

    if not cfg.fixed_magnitude:
        # ORIGINAL (buggy): magnitude shrinks at intermediate rho
        delta = cfg.rho * cfg.D * v + (1.0 - cfg.rho) * cfg.D * iso
    elif cfg.error_mode == "blend":
        # blended direction, renormalised to unit length, then scaled by D
        blend = cfg.rho * v + (1.0 - cfg.rho) * iso
        bnorm = np.linalg.norm(blend, axis=1, keepdims=True) + 1e-12
        delta = cfg.D * blend / bnorm
    elif cfg.error_mode == "mixture":
        # each record is EITHER fully directed (prob rho) OR isotropic (prob 1-rho)
        directed = rng.uniform(size=n) < cfg.rho
        unit = np.where(directed[:, None], v, iso)
        delta = cfg.D * unit
    else:
        raise ValueError(f"unknown error_mode {cfg.error_mode!r}")

    return np.clip(coords + delta, 0, cfg.grid - 1)


def _disk_offsets(radius: float) -> np.ndarray:
    """Integer (dr, dc) offsets of all cells within `radius` of the origin."""
    R = int(np.ceil(radius))
    dr, dc = np.mgrid[-R:R + 1, -R:R + 1]
    m = (dr ** 2 + dc ** 2) <= radius ** 2
    return np.stack([dr[m], dc[m]], axis=1)


def _disk_cells(coords: np.ndarray, radius: float, grid: int) -> np.ndarray:
    """(N, K, 2) integer cell indices of the disk around each coordinate, clipped."""
    off = _disk_offsets(radius)
    base = np.round(coords).astype(int)[:, None, :]
    cells = base + off[None, :, :]
    return np.clip(cells, 0, grid - 1)


def disk_mean_env(coords: np.ndarray, radius: float, fieldlist: List[np.ndarray], grid: int) -> np.ndarray:
    """Regression-calibration proxy: mean environment over the uncertainty disk."""
    cells = _disk_cells(coords, radius, grid)
    return np.stack([f[cells[:, :, 0], cells[:, :, 1]].mean(axis=1) for f in fieldlist], axis=1)


def smith_impute_env(coords_lq: np.ndarray, radius: float, fieldlist: List[np.ndarray],
                     grid: int, env_centroid: np.ndarray) -> np.ndarray:
    """Smith et al. 2023, method 2: within the disk, take the cell whose environment
    is closest (Euclidean) to the environmental centroid of the precise records."""
    cells = _disk_cells(coords_lq, radius, grid)
    env = np.stack([f[cells[:, :, 0], cells[:, :, 1]] for f in fieldlist], axis=2)   # (N, K, 2)
    d2 = ((env - env_centroid[None, None, :]) ** 2).sum(axis=2)
    j = np.argmin(d2, axis=1)
    return env[np.arange(len(coords_lq)), j, :]


def smith_impute_geo(coords_lq: np.ndarray, radius: float, fieldlist: List[np.ndarray],
                     grid: int, geo_centroid: np.ndarray) -> np.ndarray:
    """Smith et al. 2023, method 1: within the disk, take the cell closest to the
    geographic centroid of the precise records."""
    cells = _disk_cells(coords_lq, radius, grid)
    d2 = ((cells - geo_centroid[None, None, :]) ** 2).sum(axis=2)
    j = np.argmin(d2, axis=1)
    pick = cells[np.arange(len(coords_lq)), j, :]
    return np.stack([f[pick[:, 0], pick[:, 1]] for f in fieldlist], axis=1)


def coarsen(field: np.ndarray, grain: int) -> np.ndarray:
    """Block-average by `grain`, then upsample back: the modeller reads a field that is
    piecewise-constant on grain x grain blocks. Truth is unaffected."""
    if grain <= 1:
        return field
    g, n = grain, field.shape[0]
    m = (n // g) * g
    blocks = field[:m, :m].reshape(m // g, g, m // g, g).mean(axis=(1, 3))
    up = np.repeat(np.repeat(blocks, g, axis=0), g, axis=1)
    out = field.copy()
    out[:m, :m] = up
    return out


def field_autocorr(field: np.ndarray, lag: float, rng, n_pairs: int = 40000) -> float:
    """Empirical correlation between field values separated by `lag` in a random direction.
    This is exactly the quantity the displacement operator sees."""
    n = field.shape[0]
    p = rng.uniform(0, n - 1, size=(n_pairs, 2))
    ang = rng.uniform(0, 2 * np.pi, size=n_pairs)
    q = p + lag * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    keep = np.all((q >= 0) & (q <= n - 1), axis=1)
    p, q = p[keep], q[keep]
    a = field[np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)]
    b = field[np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)]
    return float(np.corrcoef(a, b)[0, 1])


def rescale_propensity(p_hat_all: np.ndarray, kappa: float, target_rate: float) -> np.ndarray:
    """Multiply the centred logit of an estimated propensity by kappa and recalibrate the
    intercept so the marginal retention rate is preserved. kappa < 1 = attenuated
    propensity, kappa > 1 = exaggerated. This is a one-parameter marginal sensitivity
    model whose parameter is the coefficient-corruption factor."""
    z = np.log(np.clip(p_hat_all, 1e-6, 1 - 1e-6) / np.clip(1 - p_hat_all, 1e-6, None))
    zc = kappa * (z - z.mean())
    lo, hi = -50.0, 50.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        m = 1.0 / (1.0 + np.exp(-(mid + zc)))
        if m.mean() < target_rate:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    return 1.0 / (1.0 + np.exp(-(a + zc)))


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


def fit_sdm(e_pres: np.ndarray, e_bg: np.ndarray,
            w_pres: np.ndarray | None = None) -> LogisticRegression:
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
# Distance-to-truth metrics
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
# One run
# --------------------------------------------------------------------------- #
def run_one(cfg: SimConfig, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    E1f, E2f, gE1_r, gE1_c = make_env_field(cfg)
    S = true_suitability(E1f, E2f, cfg)          # truth always at the fine scale

    # EXP 2: the modeller reads the environment at the analysis grain
    E1 = coarsen(E1f, cfg.grain)
    E2 = coarsen(E2f, cfg.grain)

    # true occurrences (drawn from fine-scale truth)
    coords = sample_true_occurrences(S, cfg, rng)
    e_true = _env_at(coords, [E1, E2], cfg.grid)

    # quality + observed coordinates
    r, p_q = assign_quality(e_true, cfg, rng)
    coords_obs = coords.copy()
    low = r == 0
    coords_obs[low] = apply_coord_error(coords[low], cfg, (gE1_r, gE1_c), rng)
    e_obs = _env_at(coords_obs, [E1, E2], cfg.grid)

    # background (fixed across strategies)
    bg = sample_background(cfg, rng)
    e_bg = _env_at(bg, [E1, E2], cfg.grid)

    hi = r == 1
    e_hi = e_obs[hi]          # == e_true[hi] by construction (high-quality keep true coords)

    # ---- EXP 1: uncertainty-aware environment for low-quality records ----
    # practitioner knows the uncertainty radius D (e.g. coordinateUncertaintyInMeters)
    e_calib = e_obs.copy()
    if low.any():
        e_calib[low] = disk_mean_env(coords_obs[low], cfg.D, [E1, E2], cfg.grid)

    # ---- FIX 2: propensity fitted on what the practitioner can observe ----
    if cfg.propensity_on == "obs":
        e_prop = e_obs
    elif cfg.propensity_on == "true":
        e_prop = e_true       # ORIGINAL (oracle covariates) — reference only
    elif cfg.propensity_on == "calib":
        e_prop = e_calib
    else:
        raise ValueError(f"unknown propensity_on {cfg.propensity_on!r}")

    def fit_prop(e_):
        if cfg.propensity_learner == "logit_quad":
            m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
            X = _design(e_)
        elif cfg.propensity_learner == "logit_linear":
            m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
            X = e_
        elif cfg.propensity_learner == "rf":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                       n_jobs=-1, random_state=seed)
            X = e_
        else:
            raise ValueError(f"unknown propensity_learner {cfg.propensity_learner!r}")
        m.fit(X, r)
        return m, m.predict_proba(X)[:, 1]

    def effective_coef(p_all, e_):
        """Learner-agnostic attenuation measure: OLS slope of logit(p_hat) on E1.
        For a linear-logit propensity this recovers the coefficient itself; for RF or
        any other learner it is the effective retention gradient the weights encode."""
        z = np.log(np.clip(p_all, 1e-4, 1 - 1e-4) / np.clip(1 - p_all, 1e-4, None))
        x = e_[:, 0]
        return float(np.polyfit(x, z, 1)[0])

    prop, p_hat_all = fit_prop(e_prop)
    p_hat = p_hat_all[hi]
    prop_auc = float(roc_auc_score(r, p_hat_all)) if r.min() != r.max() else float("nan")
    prop_brier = float(brier_score_loss(r, p_hat_all))
    prop_coef_E1 = float(prop.coef_[0, 0]) if hasattr(prop, "coef_") else float("nan")

    # always also fit the calibrated propensity, for side-by-side comparison
    prop_c, p_hat_all_c = fit_prop(e_calib)
    p_hat_c = p_hat_all_c[hi]
    prop_coef_E1_calib = float(prop_c.coef_[0, 0]) if hasattr(prop_c, "coef_") else float("nan")

    # oracle-covariate propensity: reference for the attenuation formula
    prop_t, p_hat_all_t = fit_prop(e_true)
    prop_coef_E1_true = float(prop_t.coef_[0, 0]) if hasattr(prop_t, "coef_") else float("nan")
    eff_coef_obs = effective_coef(p_hat_all, e_prop)
    eff_coef_true = effective_coef(p_hat_all_t, e_true)

    def trim(w):
        cap = np.percentile(w, cfg.weight_trim_pct)
        return np.clip(w, None, cap)

    w_oracle = trim(1.0 / np.clip(p_q[hi], 1e-3, None))
    w_est = trim(1.0 / np.clip(p_hat, 1e-3, None))
    w_calib = trim(1.0 / np.clip(p_hat_c, 1e-3, None))

    # ---- EXP 1: Smith et al. 2023 imputation of low-quality records ----
    e_smith_env = e_obs.copy()
    e_smith_geo = e_obs.copy()
    if low.any() and hi.any():
        env_centroid = e_hi.mean(axis=0)
        geo_centroid = coords_obs[hi].mean(axis=0)
        e_smith_env[low] = smith_impute_env(coords_obs[low], cfg.D, [E1, E2], cfg.grid, env_centroid)
        e_smith_geo[low] = smith_impute_geo(coords_obs[low], cfg.D, [E1, E2], cfg.grid, geo_centroid)

    # ---- EXP 4: kappa-envelope around the practitioner's propensity ----
    kappa_surfaces = {}
    for kap in cfg.kappa_grid:
        p_k = rescale_propensity(p_hat_all, kap, float(hi.mean()))
        w_k = trim(1.0 / np.clip(p_k[hi], 1e-3, None))
        kappa_surfaces[kap] = predict_surface(fit_sdm(e_hi, e_bg, w_k), E1, E2)

    surfaces = {
        "ALL":        predict_surface(fit_sdm(e_obs, e_bg), E1, E2),
        "FILTER":     predict_surface(fit_sdm(e_hi, e_bg), E1, E2),
        "IPW_oracle": predict_surface(fit_sdm(e_hi, e_bg, w_oracle), E1, E2),
        "IPW_est":    predict_surface(fit_sdm(e_hi, e_bg, w_est), E1, E2),
        "IPW_calib":  predict_surface(fit_sdm(e_hi, e_bg, w_calib), E1, E2),
        "ALL_calib":  predict_surface(fit_sdm(e_calib, e_bg), E1, E2),
        "SMITH_env":  predict_surface(fit_sdm(e_smith_env, e_bg), E1, E2),
        "SMITH_geo":  predict_surface(fit_sdm(e_smith_geo, e_bg), E1, E2),
    }

    out: Dict[str, float] = {
        "beta": cfg.beta, "rho": cfg.rho, "D": cfg.D, "seed": seed,
        "fixed_magnitude": int(cfg.fixed_magnitude),
        "error_mode": cfg.error_mode,
        "propensity_on": cfg.propensity_on,
        "retain_obs": float(hi.mean()),
    }
    for name, surf in surfaces.items():
        out[f"D_{name}"] = schoener_d(surf, S)
        out[f"I_{name}"] = warren_i(surf, S)
        out[f"rho_sp_{name}"] = spearman_surface(surf, S)
    # pairwise: how far the practitioner's IPW is from its oracle, and from FILTER
    out["D_IPWest_vs_IPWoracle"] = schoener_d(surfaces["IPW_est"], surfaces["IPW_oracle"])
    out["D_IPWest_vs_FILTER"] = schoener_d(surfaces["IPW_est"], surfaces["FILTER"])

    out.update({f"diag_{k}": v for k, v in overlap_diagnostics(p_hat).items()})
    out["diag_propensity_auc"] = prop_auc
    out["diag_prop_brier"] = prop_brier
    out["diag_prop_coef_E1"] = prop_coef_E1
    out["diag_prop_coef_E1_calib"] = prop_coef_E1_calib
    out["diag_prop_coef_E1_true"] = prop_coef_E1_true
    out["diag_eff_coef_obs"] = eff_coef_obs
    out["diag_eff_coef_true"] = eff_coef_true
    out["cfg_niche_shape"] = cfg.niche_shape
    out["cfg_quality_link"] = cfg.quality_link
    out["cfg_env_corr"] = cfg.env_corr
    out["cfg_propensity_learner"] = cfg.propensity_learner
    out["cfg_retain_rate"] = cfg.retain_rate
    out["cfg_n_occ"] = cfg.n_occ
    # attenuation-formula ingredients (true class separation on E1, mean displacement on E1)
    out["diag_true_sep_E1"] = float(e_true[hi, 0].mean() - e_true[low, 0].mean()) if low.any() else float("nan")
    out["diag_obs_sep_E1"] = float(e_obs[hi, 0].mean() - e_obs[low, 0].mean()) if low.any() else float("nan")
    # pooled within-class variance on E1, true vs observed (reliability ratio for the formula)
    pi1 = float(hi.mean())
    pv_true = pi1 * e_true[hi, 0].var() + (1 - pi1) * e_true[low, 0].var() if low.any() else float("nan")
    pv_obs = pi1 * e_obs[hi, 0].var() + (1 - pi1) * e_obs[low, 0].var() if low.any() else float("nan")
    out["diag_pooled_var_E1_true"] = float(pv_true)
    out["diag_pooled_var_E1_obs"] = float(pv_obs)
    # ---- EXP 2: field autocorrelation at the displacement lag, and its predictions ----
    rho_D = field_autocorr(E1, cfg.D, np.random.default_rng(10_000 + seed))
    out["diag_rho_D"] = rho_D
    out["diag_grain"] = cfg.grain
    out["diag_corr_len"] = cfg.corr_len
    if low.any():
        m0 = float(e_true[low, 0].mean())
        v1 = float(e_true[hi, 0].var()); v0 = float(e_true[low, 0].var())
        out["pred_mu_delta"] = (rho_D - 1.0) * m0
        out["pred_pooled_var_obs"] = pi1 * v1 + (1 - pi1) * (rho_D ** 2 * v0 + (1 - rho_D ** 2))
        out["pred_reliability"] = float(pv_true) / out["pred_pooled_var_obs"]
    else:
        out["pred_mu_delta"] = float("nan")
        out["pred_pooled_var_obs"] = float("nan")
        out["pred_reliability"] = float("nan")
    # kappa envelope
    for kap, surf in kappa_surfaces.items():
        out[f"rho_sp_IPW_k{kap:g}"] = spearman_surface(surf, S)
    ks = list(kappa_surfaces.keys())
    out["env_width_D"] = schoener_d(kappa_surfaces[min(ks)], kappa_surfaces[max(ks)])
    out["env_best_rec"] = max(spearman_surface(kappa_surfaces[k], S) for k in ks)
    out["env_best_kappa"] = float(max(ks, key=lambda k: spearman_surface(kappa_surfaces[k], S)))
    out.update({f"diag_{k}_calib": v for k, v in overlap_diagnostics(p_hat_c).items()})
    out["diag_lowq_dE1_shift"] = float(np.mean(e_obs[low, 0] - e_true[low, 0])) if low.any() else 0.0
    out["diag_realised_disp"] = float(np.mean(np.linalg.norm(coords_obs[low] - coords[low], axis=1))) if low.any() else 0.0
    return out


# --------------------------------------------------------------------------- #
# Sweep
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

    print("Running validation sweep (beta x rho), 8 seeds each, FIXED build ...\n")
    df = run_sweep(betas, rhos, [base.D], seeds, base)
    metric_cols = ["rho_sp_ALL", "rho_sp_FILTER", "rho_sp_IPW_oracle", "rho_sp_IPW_est"]
    g = df.groupby(["beta", "rho"])[metric_cols + ["diag_propensity_auc", "diag_ess_frac",
                                                    "diag_prop_coef_E1", "diag_realised_disp"]].mean()
    print(g.round(3).to_string())
