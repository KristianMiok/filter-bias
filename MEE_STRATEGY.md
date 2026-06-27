# MEE strategy for EARS paper

## Target

Primary target: Methods in Ecology and Evolution.

Fallback targets:
1. Ecography
2. Ecological Informatics
3. Diversity and Distributions, if the invasion/conservation angle becomes central

## Working method name

EARS: Environmental Accuracy-Risk Scoring

## MEE framing

This should be framed as a general workflow for diagnosing and propagating non-random spatial-accuracy uncertainty in species occurrence data.

The empirical crayfish dataset is a worked example, not the entire contribution.

## Core methodological contribution

EARS has five steps:

1. Learn an environmental accuracy-risk model from occurrence records with quality/accuracy labels.
2. Validate whether low-quality records are non-random using out-of-fold, group, and spatial/basin holdout designs.
3. Stratify records into environmental accuracy-risk groups.
4. Check whether the risk gradient persists after major ecological or metadata confounders are controlled.
5. Propagate risk into SDM workflows through filtering or weighting and quantify prediction shifts.

## What MEE likely needs

1. Simulation benchmark showing when EARS works and when it should not.
2. Broadly applicable workflow, not only a crayfish audit.
3. Clear ecological motivation and interpretation.
4. Reproducible code.
5. Consequence analysis showing that risk-aware treatment changes SDM predictions.

## Current empirical support

- Local environmental features predict Low accuracy with OOF ROC-AUC = 0.938.
- Low-rate increases from 0.6% in the lowest-risk tertile to 71.1% in the highest-risk tertile.
- Top risk decile has 96.1% Low accuracy.
- The gradient persists within Alien, Native, and Introduced records.
- In seven species, removing high-risk presences changes top predicted suitability regions:
  - mean top-10% Jaccard = 0.507 for all vs remove-high-risk
  - mean top-10% Jaccard = 0.737 for all vs risk-weighted

## Role of DADApy / Laio methods

DADApy should be used only if it improves the MEE contribution.

Potentially useful:
1. density-based environmental risk regimes;
2. information imbalance / metric-space comparison for feature groups;
3. intrinsic dimension only as a secondary diagnostic.

Not useful if it only adds complexity without improving interpretation or generality.

## What to ask Lucian

1. What exactly does the Accuracy label mean and how was it assigned?
2. Are Low accuracy labels comparable across time/status/source?
3. Why are Alien records much more often Low accuracy than Native records?
4. Is the >2020 increase in Low accuracy real, or due to batch/source effects?
5. Are the selected species good case studies for SDM consequence analysis?
6. Are there source, country, contributor, or database columns that should be included as confounders?
7. What ecological interpretation should we avoid?
8. What ecological story would make this valuable to SDM/invasion ecology readers?

## Next methodological work

1. Write concept note for Lucian.
2. Build simulation benchmark for EARS.
3. Refactor scripts into reusable EARS workflow functions.
4. Add grouped permutation importance.
5. Decide whether density-based risk regimes add value.
