# MEE paper — findings, framing, and plan (supersedes MEE_STRATEGY.md)

*Quality filtering as environmental sampling bias in species distribution models.*
Status as of this run on the World of Crayfish occurrence table (n = 113,907).

---

## 1. Central claim (corrected)

Spatial-accuracy filtering of occurrence records is normally treated as neutral data
hygiene. We show that when record quality is **environmentally structured**, filtering
acts as an environmental sampling-bias mechanism: it does not merely shrink the sample,
it **displaces the environmental distribution the SDM learns from**. The contribution is
a workflow that (i) diagnoses whether this structure exists, (ii) separates magnitude
from confounding (metadata) and from positivity, and (iii) maps when hard filtering,
inverse-propensity weighting, or stratified weighting is the least-bad choice.

This is a **distributional** thesis (filtering = niche displacement), not a classifier
thesis. Whether record quality is *predictable* from environment is secondary; whether
low-quality records sit non-randomly in environmental space such that removing them
distorts the niche is the load-bearing question, and it can hold even when the quality
structure is largely a metadata proxy.

---

## 2. Simulation (validation backbone — done)

Two-knob generative model: spatially autocorrelated environment, Gaussian niche, true
occurrences thinned by suitability; quality `P(high | e) = sigmoid(alpha + beta*u(e))`
with `alpha` calibrated to a fixed retention rate so **beta** is a pure coupling knob;
low-quality records displaced by `delta = rho*D*v + (1-rho)*D*eps` with **rho** the
directedness of the coordinate error.

Core identity: `p_high ∝ p_true * P(high|e)` ⇒ inverse-propensity weights `w = 1/P(high|e)`
on the high-quality sample reconstruct the true niche in the clean limit. Three treatments
(ALL / FILTER / IPW) trace the decision space. Validated result (Spearman vs true surface):

| regime | ALL | FILTER | IPW |
|---|---|---|---|
| beta=0 (no coupling) | ~1.00 | ~1.00 | ~1.00 |
| beta=0, rho=1 (directed error only) | 0.86 | ~1.00 | ~1.00 |
| beta high, rho=0 (coupling, isotropic error) | 1.00 | 0.86 | 0.97 (positivity-capped) |
| beta high, rho=1 (coupling + directed error) | 0.83 | 0.86 | 0.97 |

**No strategy dominates; the best one flips with the regime.** `beta` governs when
*filtering* hurts; `rho` governs when *keeping everything* hurts. Propensity AUC and ESS
are the only quantities observable in real data, and they locate a dataset on this map.
Code: `scripts/qfbias_sim.py`.

---

## 3. Crayfish empirical anchor (this run)

GroupKFold(k=5) on `basin_id`; propensity env reduced to 15 PCs (68% variance) for a
well-conditioned model; per-axis shift and energy distance use all 302 raw `l_*` features.
Local (`l_*`) features only — `u_*` are NaN for the ~43% headwater segments and would
silently drop them.

**Robust (model-free / CV-free — these carry the empirical claim):**
- Energy distance {all} vs {high} = 0.0835, **permutation p = 0.005**. Filtering is
  environmentally non-neutral.
- Per-axis shift is coherent along climate axes: high-quality records over-represent
  certain `l_CLI` conditions (SMD ≈ +0.16 across many correlated climate features:
  l_CLI25/27/13/15/26/14 ...) and under-represent some `l_SOL` (l_SOL41/42/43 ≈ −0.13 to −0.15).
  Filtering demonstrably distorts the niche along specific, interpretable axes. Magnitude
  is moderate (~0.16 SMD), not extreme.

**Coupling magnitude (spatial CV):**
- AUC P(high | env) = 0.722; P(high | metadata = Status + year_bin) = 0.688;
  P(high | env + metadata) = 0.763.
- Incremental env over metadata = **+0.075**; incremental metadata over env = +0.040.
- Reading: a **moderate, spatially robust, unique environmental component** above status
  and record year. Environment carries *more* unique signal than this metadata set — the
  reverse of the earlier picture (see §3a).

### 3a. Why earlier numbers are not trusted

Pre-correction this run reported env AUC 0.857 and incremental-env +0.160, and a separate
earlier analysis had env 0.816 < metadata 0.835. Both were artifacts:
- **Random CV** inflated env AUC under spatial autocorrelation (0.857 → 0.722 under
  basin CV — the gap *is* the inflation; matches the prior basin-CV value of 0.816).
- **Underfed metadata** (Status only) understated metadata AUC and overstated incremental env.
- **604 collinear features** (302 + quadratics) gave a non-converged, overfit propensity
  model; this also produced the artifactual ESS of 1.3% and exploding weights.

The corrected, spatially-honest decomposition is the one in §3.

### 3b. Honest caveats to state in the manuscript

- **+0.075 is a floor.** PCA-15 captures only 68% of env variance; the full-feature
  basin-CV env AUC was 0.816, so the true unique environmental contribution is likely
  larger. Pending: incremental env at pca ∈ {15, 30, 50} to see where it stabilizes.
- **Source / contributor / batch not controlled** — those columns are absent from this
  table. The residual environmental component is "above status and record year," *not*
  above database/batch effects. This is the single most important thing Lucian must
  clarify (see §6).
- Absolute energy-distance magnitude is dimension-dependent and not comparable to the
  simulation scale; only its significance is interpretable.

---

## 4. Positivity finding → shape of the contribution

Even with a well-conditioned model and spatial CV, **ESS = 5.6% of n_high**; weights are
dominated by a handful of records (max/mean = 664, but p99 = 3.78, so ~24 records carry
extreme weight). On data this coupled, **naive global IPW is fragile** — a few records in
under-sampled environmental regions dictate the reconstruction, and regions with no
high-quality records cannot be reconstructed at all.

This is a finding, not a failure, and it sets the contribution: the paper is not "use
weights," it is a **decision framework** — when filtering biases the niche, when IPW
works versus fails (positivity), and **stratified / Mondrian weighting (separate quantiles
per subgroup) plus transparent sensitivity reporting** as the principled response under
positivity stress. This connects directly to the Mondrian-conformal approach already in
use elsewhere.

---

## 5. Next actions (prioritized)

1. **SDM consequence (the punchline) — blocked on pipeline details.** Does the +0.075
   environmental quality-structure move predictions enough to matter? Plug ALL / FILTER /
   IPW(trimmed) / Mondrian-stratified weights into the existing Protocol B SDM and measure
   Schoener's D / Warren's I / Spearman **between strategies** on the prediction surfaces.
   This is the number that quantifies the real cost of filtering. Standalone diagnostics
   cannot produce it; it needs the actual rasters/background and multi-algorithm setup.
2. **Lucian** — revised, sharpened questions (§6). This is unblocked now.
3. **PCA robustness** — incremental env at pca ∈ {15,30,50}; expect it to rise and settle.
4. **Second taxon** — the MEE single-system risk. GBIF `coordinateUncertaintyInMeters`
   (continuous) is the natural second system and *also* the cleaner demonstration of the
   threshold-vs-weighting argument. Caveat: GBIF uncertainty is sparse and often defaulted,
   and its missingness is itself non-random — handle as a third confounder.
5. **Article type** — Research Article (second taxon effectively mandatory) vs Workflows
   (new 2026 type, but still excludes single-taxon/narrow problems and additionally
   requires broad applicability + error-propagation analysis; it does **not** relieve the
   second-taxon requirement). Decide after the consequence result.

---

## 6. Questions for Lucian (revised by the findings)

Reframed from the original eight — what we now actually need:

1. **Source / batch columns.** Are there `source`, `contributor`, `database`, or batch /
   import-event columns anywhere (not in the modelling table) that we can join? This is
   the one confounder we cannot currently control, and the strength of the
   residual-environmental claim depends on it.
2. **Labelling homogeneity over time.** Was the High/Low Accuracy protocol consistent
   across the database's history, or did the labelling rule itself change? If it changed,
   Accuracy is partly a temporal artifact and the year effect needs that caveat.
3. **Ecological reading of the climate-axis shift.** High-quality records over-represent
   specific `l_CLI` conditions and under-represent some `l_SOL`. Is this ecologically
   interpretable, or an artifact of where GPS-era / accessible sampling concentrated?
   (We can supply the exact climate variables behind the top SMDs.)
4. **Alien vs Native, post-correction.** Year is now in the model. Is the residual
   Alien↔Low association after controlling for year a real ecological signal, or remaining
   batch effect from how alien records entered the database?
5. **Co-authorship and worked-example framing.** The crayfish dataset is the empirical
   worked example, not the whole paper. Confirm framing and his role.
