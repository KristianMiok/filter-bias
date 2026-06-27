# Working results: environmental structure of spatial accuracy

## Core finding

Spatial accuracy in the crayfish occurrence dataset is strongly non-random. Low-accuracy records are structured by metadata, invasion status, stream context, and local environmental feature space.

## Accuracy labels are predictable

Using all environmental features, random row-split prediction of `Accuracy=Low` reached ROC-AUC ≈ 0.94. Local environmental features alone performed similarly, indicating that the signal is primarily local rather than upstream/network-based.

Under repeated basin holdout validation, prediction remained informative but weaker:

- environmental + metadata: mean ROC-AUC = 0.853
- metadata only: mean ROC-AUC = 0.835
- environmental all: mean ROC-AUC = 0.816
- local environmental only: mean ROC-AUC = 0.813

This suggests that metadata captures the strongest basin-generalizable signal, but environmental covariates add complementary information.

## Metadata structure

Low-accuracy records are strongly associated with status and record period.

Low-rate by status:

- Alien: 46.2%
- Introduced: 23.0%
- Native: 11.6%

Low-rate by record period showed a strong increase in recent records, especially after 2020:

- <=1950: 9.6%
- 1951–1970: 6.0%
- 1971–1990: 7.9%
- 1991–2000: 33.8%
- 2001–2010: 25.7%
- 2011–2020: 26.9%
- >2020: 56.1%

This suggests that low spatial accuracy is not simply an old-record problem.

## Environmental signal within status groups

The environmental signal persists within biogeographic status groups, indicating that it is not only an Alien-vs-Native confounder.

Within-status random split:

- Alien only: ROC-AUC = 0.884
- Native only: ROC-AUC = 0.974
- Introduced only: ROC-AUC = 0.949

Within-status repeated basin holdout:

- Alien only: mean ROC-AUC = 0.794
- Native only: mean ROC-AUC = 0.835
- Introduced only: mean ROC-AUC = 0.810

Local environmental features performed almost identically to all environmental features, again suggesting that the signal is mainly local.

## Environmental accuracy-risk score

A 5-fold out-of-fold model trained only on local environmental covariates produced a strong accuracy-risk score:

- OOF ROC-AUC = 0.938
- OOF average precision = 0.865

Observed low-rate by environmental risk tertile:

- low risk: 0.6%
- medium risk: 10.7%
- high risk: 71.1%

Observed low-rate by environmental risk decile:

- decile 1: 0.1%
- decile 5: 6.8%
- decile 8: 54.3%
- decile 9: 75.2%
- decile 10: 96.1%

The gradient persists within status groups:

Alien:

- low risk: 4.4%
- medium risk: 14.8%
- high risk: 71.1%

Native:

- low risk: 0.2%
- medium risk: 6.9%
- high risk: 73.1%

Introduced:

- low risk: 1.4%
- medium risk: 7.9%
- high risk: 57.5%

## Working interpretation

Low spatial accuracy should not be treated as random noise. It is structured in local environmental space and linked to invasion status and record history. This implies that SDM workflows may inherit systematic data-quality bias if they ignore spatial-accuracy uncertainty.

## Candidate paper claim

Data-quality uncertainty in species occurrence records is environmentally structured. In the crayfish dataset, local environmental covariates alone identify regions of occurrence space where low spatial accuracy is nearly absent and regions where it dominates. This structure persists within Alien, Native, and Introduced records and generalizes moderately across river basins.

## Next step

Connect the environmental accuracy-risk score to SDM consequences:

1. Train SDMs with all records.
2. Train SDMs excluding high-risk records.
3. Train SDMs weighting records by `1 - p_low_accuracy`.
4. Compare prediction maps, calibration, and uncertainty.
