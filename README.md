# filter-bias

**When cleaning becomes biasing: diagnosing and bounding quality-induced bias in species distribution models**

Code and analysis for a methods paper (targeting *Methods in Ecology and Evolution*) on how
**filtering occurrence records by positional quality** — a near-universal, seemingly harmless
cleaning step before species distribution modelling (SDM) — can itself introduce sampling bias
when record quality is structured with respect to the environment.

The repository contains: a two-parameter **simulation** that establishes when each filtering
strategy is appropriate; a ground-truth-free **diagnostic** that detects environmentally
structured quality in a flat occurrence table; and a **consequence** analysis measuring how much
the filtering decision actually moves SDM predictions. It is demonstrated on two systems —
curated freshwater crayfish (binary accuracy label) and German Odonata from GBIF (continuous
coordinate uncertainty).

---

## The idea in one paragraph

Filtering keeps high-quality records and discards low-quality ones. This is neutral **only if**
record quality is unrelated to the environment. When it is not — when high-accuracy records
over-represent some environments — discarding the rest removes records non-randomly across
environmental space, and the distribution the model learns from shifts. Filtering then becomes a
covariate-shift operation, biasing the niche the same way uneven survey effort does. This work
formalises that, provides a diagnostic to detect it before a threshold is chosen, and maps when
hard filtering, trust weighting, or inverse-propensity weighting (IPW) is the least-bad choice.

The central identity: if `p_true(e)` is the environmental density of correctly located
occurrences and `P(r=1|e)` the retention propensity, then the filtered (high-quality) density is

```
p_high(e)  ∝  p_true(e) · P(r=1|e)
```

so IPW weights `w = 1 / P(r=1|e)` reconstruct `p_true` in the limit of a correct propensity
model and adequate overlap (positivity).

---

## Key findings

- **No strategy dominates.** In simulation, quality–environment coupling governs when hard
  filtering biases the niche; the directedness of coordinate error governs when keeping all
  records does; IPW recovers the truth only up to a positivity ceiling. Hard filtering is never
  uniquely optimal.
- **Filtering shifts the crayfish niche toward headwaters** (energy-distance permutation
  *p* = 0.005; coherent climatic shift), with a moderate, spatially robust environmental
  component beyond record metadata.
- **The downstream consequence is real but status-blind.** It does not track native/alien status,
  niche breadth, or presence–background separability (all three tested); the largest shift falls
  on a native species.
- **Hard filtering and IPW move predictions in opposite directions**, and the spread between them
  is irreducible because the coordinate-error structure is not identifiable from the data — but
  on both systems that spread is narrow, placing the data in a *filter-with-confidence* regime.
- **A second system (Odonata) reproduces the structure** from a different label type and adds a
  dose-response: stricter filtering induces *more* shift, and there is no bias-free threshold.

---

## Repository layout

```
filter-bias/
├── src/filter_bias/        # small importable package (simulation helpers, ID tools)
├── scripts/                # analysis pipeline (see below)
├── reports/                # derived result tables (CSV) backing the figures
├── figures/                # final figures (PDF + PNG, 300 dpi)
└── pyproject.toml          # dependencies (managed with uv)
```

### Pipeline scripts (the paper, in order)

**Simulation (validation half)**
- `qfbias_sim.py` — two-knob generative model (β = quality–environment coupling, ρ =
  directedness of coordinate error); fits ALL / FILTER / IPW strategies against known truth.
- `fig_decision_heatmap.py` — runs the fine β×ρ sweep and draws **Figure 1** (best-strategy and
  recovery maps).

**Diagnostic (observable half, no ground truth)**
- `qfbias_diagnose.py` — the core diagnostic: per-axis shift (SMD, KS), multivariate energy
  distance with a permutation test, propensity AUC under spatial CV, metadata confounding, and
  IPW effective sample size.
- `run_qfbias_diagnostic.py` — driver for the crayfish system.
- `build_variable_dictionary.py` — names the top shifted axes from the GeoFRESH dictionary
  (→ **Figure 2** labels).

**Consequence (does the shift move SDM predictions?)**
- `sdm_consequence_strategies.py` — five strategies (ALL / FILTER / TRUST / IPW / stratified) per
  species; between-strategy divergence.
- `sdm_consequence_reconcile.py` — Schoener's *D*, Warren's *I*, predicted-range-area change.
- `sdm_consequence_learners.py` — learner sensitivity (random forest vs XGBoost).
- `build_species_table.py` — per-species descriptives, incl. per-species propensity env-AUC and
  IPW ESS (**Table 2**).
- `compute_niche_breadth.py`, `compute_pb_overlap.py` — tests that the correction sensitivity
  tracks neither status nor niche breadth.

**Second system (Odonata)**
- `gbif_odonata_gate.py` — checks data adequacy before committing.
- `gbif_odonata_pull.py` — stratified GBIF pull (captures museum + recent records).
- `annotate_env_chelsa.py` — CHELSA bioclim annotation via cloud-optimised GeoTIFFs.
- `run_odonata_diagnostic.py` — the diagnostic on Odonata.
- `sweep_odonata_threshold.py` — threshold dose-response (→ **Figure 4**).
- `fig_odonata_dose_response.py` — draws Figure 4.

**Figures**
- `make_figures.py` — Figures 2 and 3 (diagnostic; consequence and divergence).

Other scripts (`audit_*`, `interpret_*`, `test_*`, `*_pilot`) are exploratory analyses from
earlier stages, including a geometric/intrinsic-dimension hypothesis that was tested and did not
pan out; they are retained for provenance but are not part of the final pipeline.

---

## Reproducing the analysis

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/).

```bash
# install dependencies into a local environment
uv sync

# simulation + Figure 1
uv run python scripts/qfbias_sim.py            # validation table
uv run python scripts/fig_decision_heatmap.py  # Figure 1

# crayfish diagnostic + consequence  (requires the World of Crayfish extract)
uv run python scripts/run_qfbias_diagnostic.py
uv run python scripts/sdm_consequence_strategies.py
uv run python scripts/sdm_consequence_reconcile.py
uv run python scripts/sdm_consequence_learners.py
uv run python scripts/build_species_table.py
uv run python scripts/make_figures.py          # Figures 2 and 3

# Odonata second system  (pulls from GBIF; no account needed)
uv run python scripts/gbif_odonata_gate.py
uv run python scripts/gbif_odonata_pull.py
uv run python scripts/annotate_env_chelsa.py
uv run python scripts/run_odonata_diagnostic.py
uv run python scripts/sweep_odonata_threshold.py
uv run python scripts/fig_odonata_dose_response.py
```

The simulation and Odonata pipelines run from public data. The crayfish occurrence extract is
from the curated **World of Crayfish®** database (see Data below) and is not redistributed here.

---

## Data

- **Freshwater crayfish** — World of Crayfish® database (Ion et al., 2024),
  <https://world.crayfish.ro>. Environmental predictors: Hydrography90m (Amatulli et al., 2022)
  and GeoFRESH (Domisch et al., 2024).
- **Odonata** — GBIF, order Odonata, Germany (pulled via the GBIF API). Environment:
  CHELSA bioclim v2.1 (Karger et al., 2017).

The `reports/` tables are derived summaries that back the figures; they can be regenerated by the
scripts above.

---

## Citation

If you use this code, please cite the paper (details to follow on acceptance). A frozen, citable
archive with a persistent identifier will be minted at that point.

## License

Released under the MIT License. See `LICENSE`.
