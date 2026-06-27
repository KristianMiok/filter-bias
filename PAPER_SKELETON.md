# Environmentally structured spatial-accuracy risk in species occurrence records

## Working title options

1. Environmentally structured spatial-accuracy risk in species occurrence records
2. Low spatial accuracy is not random: environmental structure of data-quality risk in SDM occurrence data
3. Mapping environmental risk of low spatial accuracy in species distribution modelling data
4. Data-quality uncertainty is environmentally structured in crayfish occurrence records

## Core research question

Are low-spatial-accuracy occurrence records randomly distributed, or do they occupy structured regions of environmental feature space?

## Secondary research question

Does environmental accuracy-risk affect SDM-like predictions when high-risk occurrence records are removed or down-weighted?

## Main claim

Low spatial accuracy in crayfish occurrence records is not random noise. It is strongly structured in local environmental feature space and can be predicted from environmental covariates alone. This structure persists within Alien, Native, and Introduced records and has measurable consequences for SDM-like suitability predictions.

## Data

- Dataset: global crayfish occurrence records.
- Records: 115,191 occurrence records with `Accuracy` labels.
- Accuracy labels:
  - High: 83,545
  - Low: 31,646
- Environmental predictors:
  - local climate
  - local topography
  - local soil
  - local lake variables
  - upstream/network variables used in secondary analyses

## Analysis 1: Are Low accuracy records predictable?

Models predicting `Accuracy=Low` from environmental features showed strong performance.

Key results:

- Random split, all environmental features: ROC-AUC = 0.937
- Random split, local environmental features only: ROC-AUC = 0.938
- Species holdout, local environmental features only: ROC-AUC = 0.944
- Repeated basin holdout, local environmental features only: mean ROC-AUC = 0.813

Interpretation:

The signal is strong and primarily local environmental rather than upstream/network-based. Generalization is stronger across species than across basins.

## Analysis 2: Is the signal explained by metadata?

Metadata-only models were also predictive.

Key results:

- Year only: ROC-AUC = 0.767
- Status only: ROC-AUC = 0.719
- Distance only: ROC-AUC = 0.646
- Strahler only: ROC-AUC = 0.639
- All metadata: ROC-AUC = 0.858

Low-rate by status:

- Alien: 46.2%
- Introduced: 23.0%
- Native: 11.6%

Interpretation:

Low spatial accuracy is associated with record period and invasion status, but environmental features remain highly predictive and provide complementary information.

## Analysis 3: Does environmental signal persist within status groups?

Yes. Environmental features predicted Low accuracy even within each status group.

Random split:

- Alien only: ROC-AUC = 0.884
- Native only: ROC-AUC = 0.974
- Introduced only: ROC-AUC = 0.949

Repeated basin holdout:

- Alien only: mean ROC-AUC = 0.794
- Native only: mean ROC-AUC = 0.835
- Introduced only: mean ROC-AUC = 0.810

Interpretation:

The environmental signal is not merely an Alien-vs-Native confounder.

## Analysis 4: Environmental accuracy-risk score

A 5-fold out-of-fold model trained only on local environmental covariates produced an environmental accuracy-risk score.

Performance:

- OOF ROC-AUC = 0.938
- OOF average precision = 0.865

Observed Low-rate by risk tertile:

- low risk: 0.6%
- medium risk: 10.7%
- high risk: 71.1%

Observed Low-rate by risk decile:

- decile 1: 0.1%
- decile 8: 54.3%
- decile 9: 75.2%
- decile 10: 96.1%

Within-status gradient:

Alien:

- low risk: 4.4%
- high risk: 71.1%

Native:

- low risk: 0.2%
- high risk: 73.1%

Introduced:

- low risk: 1.4%
- high risk: 57.5%

Interpretation:

Local environmental space sharply stratifies spatial-accuracy risk, even within biogeographic status groups.

## Analysis 5: SDM-like consequence pilot

Two candidate species were used:

- Pacifastacus leniusculus
- Astacus astacus

Three training strategies were compared:

1. all presences
2. remove high-risk presences
3. weight presences by `1 - p_low_accuracy`

Pacifastacus leniusculus:

- all: AUC = 0.970, AP = 0.908
- remove high-risk: AUC = 0.944, AP = 0.872
- weighted: AUC = 0.961, AP = 0.893
- top-10% suitability Jaccard:
  - all vs remove high-risk = 0.608
  - all vs weighted = 0.724

Astacus astacus:

- all: AUC = 0.981, AP = 0.926
- remove high-risk: AUC = 0.928, AP = 0.827
- weighted: AUC = 0.974, AP = 0.905
- top-10% suitability Jaccard:
  - all vs remove high-risk = 0.432
  - all vs weighted = 0.723

Interpretation:

Risk-aware filtering and weighting change SDM-like prediction surfaces. Standard test AUC decreases when high-risk records are removed, but this is expected because the test distribution includes the original high-risk occurrence pattern. The more important result is that suitability rankings and top predicted suitable areas shift substantially.

## Candidate figures

Figure 1. Accuracy prediction performance across validation designs.

Figure 2. Low-accuracy rate by environmental accuracy-risk decile.

Figure 3. Low-accuracy rate across risk tertiles within Alien, Native, and Introduced records.

Figure 4. SDM consequence pilot: AUC/AP by training strategy.

Figure 5. SDM consequence pilot: top-10% suitability overlap and prediction shifts.

Figure 6. Scatter plots of all-record vs risk-aware predictions.

## Main limitations

1. `Accuracy` labels are dataset-specific and may reflect source-specific rules.
2. The risk score is learned from observed labels, not independently validated georeferencing error.
3. SDM consequence analysis is currently a simplified presence-background pilot.
4. Background is sampled from other occurrence records, not from a full environmental raster grid.
5. Spatial autocorrelation remains important; basin holdout helps but does not solve all spatial dependence.
6. Random test AUC is not a sufficient measure of trustworthy SDM performance.

## Next validation steps

1. Extend SDM consequence analysis to 5–6 species.
2. Use repeated background samples.
3. Add basin/spatial holdout to the SDM consequence pilot.
4. Compare all-record, high-only, remove-high-risk, and risk-weighted models.
5. If possible, project predictions to real environmental raster/grid space.
6. Write a compact methods/results draft.
