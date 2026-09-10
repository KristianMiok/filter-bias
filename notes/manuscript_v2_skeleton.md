# m20 resubmission — manuscript skeleton (v2, Sept 2026)

Deadline: 29 Nov 2026 (new-manuscript resubmission, MEE-26-07-629 lineage).
Target: Research Article, 7,000–8,000 words, 5 main figures, 3–4 tables.

## Title candidates (decide with Lucian)

1. **When cleaning becomes biasing — and when correcting fails: a diagnostic
   and sensitivity framework for coordinate-quality filtering in species
   distribution models**  (keeps the brand the editor knows; subtitle carries
   the reframing)
2. The error you correct is the error that corrupts: diagnosing and bounding
   quality-filtering bias in SDMs
3. Quality filtering as environmental sampling: diagnosis, sensitivity
   envelopes, and the limits of correction in SDMs

## The one-paragraph story

Filtering occurrence records by coordinate quality is nonrandom environmental
sampling (known). We show the standard fix inherits the disease: the retention
propensity is estimated on observed environments, which are displaced by the
very error being corrected. A one-line formula — alignment x reliability —
predicts when the correction collapses; everything about error magnitude,
correlation length and analysis grain enters through a single quantity,
varrho(D). The dangerous regime is partially identifiable from an observable
signature; where it cannot be identified, a mechanistically interpretable
kappa-envelope replaces the point estimate, gated by positivity (ESS). Two
real systems with 302 and 19 descriptors, a semi-synthetic ground-truth
experiment, and a seven-group GBIF survey (34 units) show: filtering costs
more than random deletion of the same fraction; even oracle reweighting cannot
buy back coverage; both flagship systems are structurally protected — for two
different, formula-predicted reasons — while the dangerous signature exists in
the wild exactly where the theory says (herbarium data with km uncertainties).

## Sections, budgets, claims -> evidence -> display

### 1 Introduction (~1,100 w)
- Problem + stakes; quality filtering ubiquitous (cite CoordinateCleaner use).
- Prior work, honestly: Meng 2018; Boyd 2021/2024/2025; van Eupen 2021;
  Moudry 2024; Gabor; Smith et al. 2023 (closest); Zizka. Gap: corrections
  treated as available; no one models the corruption of the correction itself.
- Contributions (6): mechanism+formula; regimes; varrho(D) scale result;
  observable signature (partial identification); envelope+gating protocol;
  two-system + survey evidence with ground truth.

### 2 Theory (~1,400 w) — Fig 1a, Fig 2a
- Setup, core identity p_true ∝ p_high / P(r=1|e).
- Corruption mechanism: propensity on e_obs. **Proposition (attenuation):**
  coef_obs/coef_true ≈ (1 − μ_Δ/Δm)(σ²_true/σ²_obs); validated r = 0.998,
  slope 0.966, MAE 0.049 (exp4). Collapse μ_Δ=Δm; sign flip beyond; false
  coupling at β=0 (coef −2.12 with directed error).
- **Scale result:** D, correlation length, grain enter ONLY via varrho(D);
  closed form r = 0.997 (exp2); coarsening moves along the same curve.
- Three alignment regimes; alignment, not directedness, decides (exp0b).
- Observable signature (AUC, ESS, dess): regime ~90% identifiable at β ≥ 1;
  safe-side rule precision 91%, recall 46% -> matched-simulation protocol,
  not universal thresholds (exp0c).
- kappa-envelope (Tan-type MSM, multiplicative on the coefficient; κ<1 =
  toward-regime attenuation, κ>1 = away-regime exaggeration); Λ=2 primary.
- Gating rule: ESS(p99) < 0.30 -> point IPW not reportable; envelope only.
  (Balance deteriorates under IPW below ~0.30: emp1.)

### 3 Simulation experiments (~1,200 w) — Fig 1b–d
- State openly: R1 identified two implementation flaws; the honest build
  reverses the submitted headline (IPW vs FILTER at β>0: 17/2/16). This is
  the paper's pivot, said plainly (also in cover letter).
- Four operations — keep / drop / reweight / relocate — none dominates;
  winner set by regime (exp1; Smith wins both directed regimes incl. over
  oracle IPW, harms under isotropic).
- Misspecification: regime ordering 10/11 variants (bimodal niche the
  exception); formula right in sign everywhere, calibrated under LDA (exp3).

### 4 Crayfish system: 302 descriptors, 113,907 records (~1,100 w) — Fig 3 (squares), Fig 4
- One basin-blocked propensity for diagnosis AND weights (emp1): global AUC
  0.724 (submitted: 0.722). Submitted ESS values were artifacts of a
  randomly-CV'd RF (P. leniusculus 6.3%→70.9%, A. pallipes 6.6%→2.8%).
- **Positivity paradox:** strongest coupling = worst positivity
  (A. pallipes AUC 0.96, ESS 2.8%).
- Consequence rerun (emp2): 5/7 species gated; D(FILTER,IPW) widens exactly
  for low-ESS species (A. astacus 0.93→0.826) -> the submitted "narrow
  bracket" was partly the artifact. Λ=2 envelope widths 0.50–0.89.
- **Semi-synthetic ground truth** (emp3; A. astacus High records as truth,
  donor-swap displacement, both axes):
  (1) filtering costs 0.12 Spearman beyond the random-deletion ceiling
      (0.72 vs 0.841) on the long axis; 0.04 on the short;
  (2) IPW_est ≈ IPW_oracle ≈ FILTER ≪ MCAR ceiling — positivity binds even
      the oracle; keep-all strategies (ALL/relocate/calibrate 0.82–0.89)
      dominate;
  (3) km-scale error cannot create the dangerous regime here: ρ_u(25 km) ≈
      0.5 even on the leading topographic axis; μ_u/Δm ≤ ~0.3; AUC signature
      separates in the predicted direction on the long axis (0.76/0.82/0.86).
  Caveats: donor-swap drifts toward database mean (iso μ_u ≠ 0, reported);
  distance_m median 53 m, p95 226 m supports e_obs ≈ e_true for High records.

### 5 Odonata + GBIF survey (~1,000 w) — Fig 3 (circles), Fig 5, Fig 2b
- Odonata = the opposite endpoint: AUC 0.63–0.68, ESS 0.81–0.94, 0/6 gated,
  balance improves everywhere, Λ=2 envelopes 0.90–0.95 -> point IPW
  defensible. Same framework, opposite recommendation.
- Relocation/calibration feasibility depends on the FORM of reference data:
  donor pools cover 16–23% of imprecise records at GBIF densities (emp4)
  vs ~100% with raster disk-means (survey). Bounds Smith 2023 in practice;
  their setting (polygon-only records) stated fairly.
- Threshold sweep (50–1000 m): coupling concentrated in the extreme-
  uncertainty tail (AUC 0.66→0.83 as the threshold loosens); retention
  0.59→0.93; IPW balances across the whole range (smd_after ≤ 0.012).
- Survey: 7 groups, ~352k records, 34 units, pre-registered criteria (all
  pass). **Formula test across datasets: corr(1 − ρ̄(u), coef_gap) = 0.87.**
  Dangerous signature ~3/34 units, all with km-scale uncertainties
  (Orchis militaris ESS 0.20, AUC 0.45 = spatial confounding;
  Pelophylax lessonae 0.21). Orchidaceae: dess = −0.25 — de-attenuation
  caught in the wild, direction as predicted.
- Two protection mechanisms, one formula: crayfish (long-range coupling
  axis) vs Odonata (uncertainty ≪ layer correlation length); danger =
  short-range predictors + km uncertainty (that is where the survey finds it).

### 6 Discussion (~1,300 w)
- Practitioner protocol (box): diagnose (AUC/Brier/ESS/balance) -> gate ->
  if gated: envelope, keep-all repairs; if healthy: point IPW defensible;
  choose operation by georeferencing mechanism, argued not assumed.
- Relation to prior work TABLE (rows: Meng; Boyd x3; van Eupen; Moudry;
  Smith; this paper — columns: models the correction's own corruption? /
  scale result? / observable signature? / bounds?).
- Limitations: one country (data-rich -> prevalence conservative); one layer
  family (CHELSA bio); search-API stratified sample (pre-registered strata,
  derived-dataset DOI); survey diagnostics-only; donor-swap drift; formula
  calibrated under LDA (sign robust); bimodal-niche exception; learner set.
- Future: sharp bounds / general-learner theory (separate paper).

### 7 Data & code
- Repo (public), commit-hash pinned; GBIF derived-dataset DOI from
  survey_datasetkeys_*; crayfish master per original paper's terms;
  reusable audit pipeline sentence (survey1–3 generalize to any region).

## Fig 6 / provenance block (Section 5 + caption material)
- Geography of quality (Fig 6): provenance-structured, not habitat-structured.
  naturgucker 79k records 77% imprecise vs Observation.org 71k 11% -> recording
  convention (site-level vs GPS) drives the map.
- Orchidaceae Low is bimodal: <5 km mass (62%; site-level recording conventions
  across portals — Obs.org, naturgucker, ArtenFinder all fall here) and a
  15-40 km mode (35%) that is EXCLUSIVELY iNaturalist geoprivacy obscuring
  (6,042/6,046 records; ~0.2 deg cell). Pl@ntNet (no obscuring) 4% imprecise.
  Trough at 5-15 km (573 records).
- Discussion insight: conservation privacy policy manufactures the dangerous
  regime for exactly the taxa SDMs matter most for; corrections then operate at
  scales where varrho(u) genuinely falls (0.87 at 25 km). Ties O. militaris
  signature + dess = -0.25 + Fig 6b into one mechanism chain.
- Methods sentence: obscured coordinates are randomised within cells, i.e.
  genuine displacement, not mere imprecision — consistent with the framework's
  e_obs formalism.

## Tables
- T1 notation + regimes + what is (un)identifiable.
- T2 crayfish species battery (emp1+emp2 merged; gated column).
- T3 survey battery (34 units; ρ̄(u), gap, dess).
- T4 relation to prior work.

## Supplement
- S1 Λ=4 envelopes; S2 exp3 misspecification detail; S3 Odonata species
  table + variogram detail; S4 exp0/exp0b full tables (old-vs-new builds);
- S5 tuned-XGB twin (DONE): diagnosis is learner-independent (same unified
  propensity, same 5/7 gated); filtering divergence is LARGER under the tuned
  learner for 7/7 species (A. torrentium D(ALL,FILTER) 0.245 vs RF 0.675),
  while D(FILTER,IPW) is 0.82-0.90 under both -> contrast, not global
  roughness; envelope magnitude is pipeline-dependent -> protocol line:
  compute the envelope with the learner you deploy.

## Response matrix (separate doc, next step)
- R1: 13 points — #13 (both fixes) answered by exp0 + open cover-letter
  admission; #6/#12 by gating+balance; #7 by emp3; #9 by threshold sweep; ...
- R2: 10 points — novelty by contributions 1–5 + T4; "bracket" by envelope;
  Smith comparison by exp1 + emp4/survey feasibility contrast; ...

## Writing split (proposal for the verbal briefing)
- Kristian: Theory, Simulation, Empirical sections (verdicts -> prose),
  figures, response matrix numbers.
- Lucian: Introduction, Discussion ecology framing, prior-work table wording,
  cover letter voice; joint: abstract, title decision.
