# Response matrix — MEE-26-07-629 resubmission (working doc, Sept 2026)

Format per point: reviewer's ask (key quote) -> what we did -> where -> number.
Status: ALL 26 points have a concrete answer already computed and committed.

## Reviewer 1

**R1.1 — IPW identification assumptions.** "formally state the identification
assumptions and discuss situations in which IPW does not target p_true"
-> Theory now states A1 selection-on-observables in E_TRUE, A2 retained records
carry e_obs = e_true, A3 positivity, A4 propensity form; and the paper's core
result is precisely the failure mode: the practitioner's propensity is fitted
on E_obs, and Prop. 1 quantifies the gap. -> Sec 2, Fig 1a. -> coef ratio
= (1 - mu/dm)(s2t/s2o), validated r = 0.998, slope 0.966 (exp4).

**R1.2 — "bracket the truth" too strong.** -> Terminology withdrawn everywhere;
replaced by a kappa sensitivity ENVELOPE (Manski identification-vs-sensitivity
distinction, per R2.4) with a mechanistically interpretable parameter
(kappa<1 = toward-regime attenuation; kappa>1 = away-regime exaggeration).
In simulation the envelope always contains the best practitioner strategy but
cannot name kappa*. -> Sec 2, Fig 5a. -> Lambda=2 primary; exp4 tables.

**R1.3 — decision map vs non-identifiable directedness.** "how can the
simulation map prescribe a treatment when one of its dimensions is
unobservable?" -> The beta x rho decision surface is DROPPED. Reframed as
partial identification: the dangerous regime leaves an observable signature
(propensity AUC, ESS, dess); matched-simulation protocol instead of universal
placement; safe-side rule stated with its operating characteristics.
-> Sec 2, Fig 3. -> regime ~90% recoverable at beta>=1; safe-side precision
0.91, recall 0.46 (exp0c).

**R1.4 — simulation too restricted.** -> exp3 misspecification battery: 11
variants (bimodal/skewed niches, nonlinear and threshold coupling,
heterogeneous D, prevalence and n, correlated predictors, misspecified
learners incl. LDA). -> Sec 3, Supp S2. -> regime ordering holds 10/11
(bimodal-niche exception documented); formula sign-correct in all variants.

**R1.5 — propensity-model sensitivity with 302 predictors.** -> One unified
basin-blocked PCA(15)+quadratic-logistic propensity for diagnosis AND weights;
sensitivity across estimators and dimensions reported; the submitted RF-CV ESS
values are shown to be artifacts (P. leniusculus 6.3%->70.9%, A. pallipes
6.6%->2.8%). -> Sec 4, Table T2, Supp. -> emp1 tables (global AUC 0.724 vs
paper's 0.722).

**R1.6 — positivity severity; quantitative criteria for abandoning IPW.**
-> Weight distributions, p99/max, trimming sensitivity reported; explicit GATE:
ESS(p99) < 0.30 => point IPW not reportable, envelope only — justified
empirically (covariate balance under IPW deteriorates below ~0.30). 5/7
crayfish species gated. -> Sec 2 protocol + Sec 4. -> emp1/emp2; positivity
paradox named (A. pallipes AUC 0.96, ESS 0.028).

**R1.7 — "correct with confidence" unsupported; agreement != correctness.**
-> Phrase removed. And we now MEASURE correctness where it is measurable:
semi-synthetic ground truth shows FILTER ~ IPW ~ IPW_oracle all well below the
random-deletion ceiling — convergence with the truth absent, demonstrated.
-> Sec 4, Fig 4. -> emp3: FILTER 0.72 vs MCAR ceiling 0.841 (long axis);
IPW_est ~ oracle ~ FILTER.

**R1.8 — crayfish provenance confounder; Odonata does not resolve it.**
-> Odonata no longer offered as resolving anything; provenance stated as a
limitation for the crayfish system; and provenance structure is now shown
DIRECTLY in the GBIF survey (portal conventions and geoprivacy obscuring make
the quality geography). -> Sec 4 caveat + Sec 5, Fig 6. -> naturgucker 77% vs
Observation.org 11% imprecise; iNat obscuring = 6,042/6,046 of the 15-40 km
mode (Orchidaceae).

**R1.9 — thresholding discards continuous uncertainty.** -> Continuous
formulation is now the framework's spine: everything enters through rho(u),
the layer autocorrelation at the uncertainty scale; per-unit continuous
rho-bar(u) in the survey; threshold sweep retained as the practitioner view.
-> Sec 2 (Fig 2), Sec 5 (Fig 5b). -> cross-dataset formula test
corr(1 - rho-bar(u), coef gap) = 0.87 over 34 units; Odonata sweep 50-1000 m:
AUC 0.66->0.83, retention 0.59->0.93, IPW balances throughout (SMD<=0.012).

**R1.10 — untuned learner comparison insufficient.** -> S5: tuned XGBoost twin
of the full pipeline (basin-blocked GroupKFold tuning), same propensity and
gating. Finding strengthens the paper: filtering divergence is LARGER under
the tuned flexible learner for 7/7 species, while FILTER-vs-IPW distances
match — the phenomenon is not an artifact of an under-tuned learner.
Envelope magnitudes are pipeline-dependent -> protocol line: compute the
envelope with the learner you deploy. -> Supp S5. -> A. torrentium D(ALL,
FILTER) 0.245 (XGB) vs 0.675 (RF); D(FILTER,IPW) 0.82-0.90 both learners.

**R1.11 — A. torrentium "variance rather than bias" too strong.** -> Replaced
by "no resolvable environmental structure in retention at the chosen spatial
scale"; 4/5 of records in two basins stated; S5 adds that the full-vs-filtered
difference for this species is large and learner-dependent, reinforcing
caution. -> Sec 4. (Same as R2.8.)

**R1.12 — operational decision rule or why universal thresholds cannot
exist.** -> Both: an explicit protocol box (diagnose -> gate at ESS 0.30 ->
envelope vs point IPW -> choose operation by georeferencing mechanism) AND the
formula-level explanation of why universal AUC/SMD thresholds cannot exist:
the damage depends on dm and rho(u), which are dataset-specific — hence the
matched-simulation protocol. -> Sec 6 box + Sec 2. -> gate justified by
balance breakdown below ESS 0.30 (emp1).

**R1.13 — two code flaws; essential rerun.** -> Both fixed exactly as
specified (|delta| = D for all rho; propensity strictly on e_obs), old
behaviour retained behind flags for comparison; full four-build rerun. The
honest build reverses the submitted headline and the paper now says so
plainly (also in the cover letter): IPW vs FILTER at beta>0 goes 17 wins /
2 ties / 16 losses, and the reversal is the paper's central mechanism.
-> Sec 3, Supp S4, cover letter. -> exp0 gate tables; mechanism: propensity
coefficient at beta=2 goes +2.23 -> -0.01 as alignment turns toward-precise;
false coupling coef -2.12 at beta=0.

## Reviewer 2

**R2.1 — novelty vs Meng 2018 / Boyd 2021, 2024ab / BSP 2024 / Van Eupen 2021
/ Moudry 2024.** "the particular combination ... within an SDM workflow" ->
Landing point adopted verbatim: the paper is framed as an SDM-specific
diagnostic and sensitivity workflow for an established selection-bias problem;
"gone largely unexamined" deleted; prior-work Table T4 maps each work to what
it does and does not cover; contributions restated as (i) the corruption-of-
the-correction mechanism + closed-form attenuation, (ii) the rho(D) scale
result, (iii) partial identification via an observable signature, (iv)
envelope + positivity gating, (v) ground-truth quantification. -> Intro, T4.

**R2.2 — covariate shift != bias to the estimand; define "true niche".**
-> Three-level distinction adopted (shift / potential to affect the fitted
SDM / bias to a stated target); estimand defined as the relative intensity
surface over the stated prediction domain; terminology table separates
original sample / retained sample / occurrence population / prediction
population. -> Sec 2, Table T1.

**R2.3 — P(R=1|E_obs) vs P(R=1|E_true); IPW cannot fix selection and
measurement error at once.** -> This is now the paper's central result rather
than an unacknowledged gap: Prop. 1 gives the corruption in closed form;
identification assumptions stated with MAR/selection-on-observables citations
(Dumelle 2025; Pescott 2023); and the semi-synthetic experiment demonstrates
the joint failure empirically (even oracle weights cannot recover what
positivity has destroyed). -> Sec 2 + Fig 4. -> emp3: IPW_oracle ~ IPW_est ~
FILTER << MCAR ceiling.

**R2.4 — identification bounds vs sensitivity envelope (Manski).** -> Manski's
distinction adopted explicitly; FILTER/IPW are presented as sensitivity
scenarios; the kappa-envelope is the formal sensitivity object; "bracket",
"irreducible uncertainty", "correct with confidence" removed. -> Sec 2.
(Same fix as R1.2.)

**R2.5 — error magnitude and analytical grain must be central; coarsening as
an alternative response.** -> Now a headline result: D, correlation length and
grain enter ONLY through rho(D); closed form validated; coarsening moves a
dataset along the same curve, so it is quantified as one of the four
operations (keep / drop / reweight-or-relocate / coarsen); retention fraction
varied in exp3. Boyd et al. 2024's resolution-accuracy trade-off connects
through the same curve. -> Sec 2, Fig 2. -> exp2: corr(rho(D), reliability)
= 0.997 across parts A+B pooled.

**R2.6 — mapping diagnostics onto the beta x rho surface unjustified.**
-> Surface dropped; sensitivity-over-the-unidentified-dimension formulation
adopted; what IS recoverable stated with operating characteristics.
-> Sec 2, Fig 3. (Same numbers as R1.3.)

**R2.7 — calibration + proper scoring + balance, not AUC; trimming changes the
estimator.** -> Brier score and calibration curves added to every propensity;
covariate balance before/after weighting reported for every unit (Makela-type
diagnostic); trimming acknowledged as a bias-variance trade with sensitivity
across rules; survey battery carries smd_before/smd_after throughout.
-> Sec 4/5, tables. -> e.g. Odonata: IPW balances the whole threshold range
(SMD<=0.012); crayfish balance deteriorates exactly in the gated set.

**R2.8 — A. torrentium.** -> As R1.11; reviewer's exact wording ("no
resolvable environmental structure in retention at the chosen spatial scale")
adopted.

**R2.9 — metadata/environment AUC is not a decomposition.** -> Rephrased as
"environmental variables carry additional predictive information about record
quality conditional on the included metadata"; decomposition/causal wording
removed. -> Sec 4.

**R2.10 — native-vs-alien claims with n=7.** -> Presented descriptively;
"status-blind" and null-hypothesis language removed. -> Sec 4.

**R2.m1 — energy distance: retained vs discarded, not subset vs superset.**
-> Recomputed retained-vs-discarded with permutation across the union;
dependence issue resolved. -> Supp.

**R2.m2 — effect sizes over p-values.** -> SMD and distance effect sizes
reported throughout; significance de-emphasised (sample sizes make trivial
departures significant). -> all tables.

**R2.m3 — terminology: sample / population / target.** -> Table T1. (With
R2.2.)

## Cover-letter spine (drafting note)
1. Both implementation issues fixed; honest build reverses the headline; we
   say so and the reversal became the paper's mechanism (R1.13).
2. Novelty reframed exactly on R2's suggested landing point (R2.1).
3. Bracket -> envelope; decision map -> partial identification + matched
   simulation (R1.2/3, R2.4/6).
4. New evidence beyond asks: ground-truth ceilings (emp3), rho(D) scale law
   (exp2), 34-unit survey with cross-dataset formula test, geography-of-
   quality map (Fig 6).
