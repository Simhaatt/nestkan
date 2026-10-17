# Pre-registered protocol: tuned benchmark v5

Written before any v5 result exists. Code: scripts/bench3.py, scripts/run_tuned3.py. Output tag: _v5.

## Changes from v4 (decided from v4 / simulation diagnostics, applied to all methods where relevant)

1. NestKAN's search space gains `use_residual ∈ {True, False}`: tuning may switch off the
   time-varying surface. Budget unchanged (K = 8 random configurations per fold for every method).
2. Every neural method (NestKAN, NestKAN-PH, Logistic-Hazard, PMF, MTLR, DeepHit, PC-Hazard,
   DeepSurv, CoxKAN) is selected on a single seed, then the selected configuration is refitted
   with 2 further seeds and the 3 survival curves are averaged for the test metrics. Single-seed
   test metrics are stored too (`*_single`). Cox and random survival forest are not ensembled.
3. Two comparators added: PC-Hazard (piecewise-constant hazard, Kvamme & Borgan 2021) and CoxKAN
   (Cox partial likelihood with a KAN risk function, Breslow baseline; no symbolic step).
4. Repeated cross-validation: 3 repeats × 5 folds (StratifiedKFold seeds 0, 1, 2). Repeat 0
   reuses the v4 splits.
5. Datasets: the 6 SurvSet datasets of v4 plus three discrete-time cohorts of the grouped-PH
   project: support2_slos (n = 9,105), drsa_clinic (n = 4,828), sparcs_drg302 (n = 34,233).
   sparcs_drg302 uses 1 repeat only (n > 20,000; compute).

## Unchanged

Selection on validation rNLL (30-interval KM-quantile reference grid); metrics rNLL, Antolini
C_td, IPCW IBS, IBS with a Cox censoring model; leakage drops as in v4; grid ends at last
training event; evaluation grid = uniform(200) + reference cuts + 300 event-time quantiles.

## Analysis (fixed now)

- Primary: rNLL. Secondary: C_td, IBS, IBS_cx.
- Paired differences NestKAN − competitor per dataset over all repeat × fold units; SE inflated
  by the Nadeau–Bengio correction for repeated k-fold: var × (1/(r·k) + n_test/n_train),
  n_test/n_train = 0.2/0.64.
- "Resolved" = |difference| > 1.96 corrected SE. Report counts better / worse / unresolved and
  mean ranks across datasets, for all datasets including those where NestKAN loses.
- v4 remains reported; v5 does not replace it silently.
