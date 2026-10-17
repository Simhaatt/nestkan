# Pre-registered protocol: v7 — SurvKAN training-budget sensitivity check

Registered 2026-10-15, before any v7 run was launched. Nothing in v5/v6 results is modified or
deleted; v7 only adds new files under the tag `_v6survkan1000`.

## Motivation

In v6 (PROTOCOL_v6.md) the SurvKAN re-implementation (`src/dkan/ctkan.py`, driven by
`scripts/bench4.py`, family `SurvKAN`, `max_epochs = 300`, early-stopping patience 15) often ran its
tuning trials to the 300-epoch cap (trials files `results/trials_v6survkan_<ds>.csv`, column
`epochs`: rott2 38/40, nwtco 40/40, support2 26/40 trials at 300). A reviewer could therefore argue
that SurvKAN was undertrained. v7 tests whether a larger epoch budget changes the conclusions.

## B1. New family `SurvKAN-1000`

Identical to the v6 family `SurvKAN` in every respect except `max_epochs = 1000`:

- same model (`CTKAN`, one hidden layer of width `hidden`, spline order k = 2), same loss
  (`SurvKANLoss(50)`), batch 256, Adam, early stopping on validation loss with patience 15
  (unchanged), best-validation state restored;
- same search space `SURVKAN_SPACE` (hidden ∈ {0,1,2,3}, grid ∈ {3,5,8}, base_fn ∈ {identity, silu},
  lam ∈ {0, 1e-4, 1e-3, 1e-2}, lr ∈ {1e-3, 3e-3, 1e-2}), min–max scaling;
- K = 8 random configurations per fold (same config RNG as v6, via `bench3.run`), selection on
  validation rNLL, 3-seed ensemble of the selected configuration;
- same splits: repeated-CV repetition 0, folds 0–4 (the v5/v6 split seeds).

Implementation: `scripts/bench4.py` gains the family entry
`"SurvKAN-1000": ("ctkan", "SurvKAN1000", "minmax", SURVKAN_SPACE, {})`, and `fit_one` treats
mkey `"SurvKAN1000"` exactly as `"SurvKAN"` but with `max_ep = 1000`. No other family is affected.
Run via `scripts/run_v6.py SurvKAN-1000 4 rott2,nwtco,support2` (output tag `_v6survkan1000`:
`results/tuned_v6survkan1000_<ds>.csv`, `results/trials_v6survkan1000_<ds>.csv`).

## B2. Datasets

rott2, nwtco, support2; rep 0; folds 0–4 (15 fold-level fits). No other datasets.

## B3. Analysis (fixed in advance)

Comparisons, each paired over the 5 folds of rep 0, per dataset:

- (a) SurvKAN-1000 (`results/tuned_v6survkan1000_<ds>.csv`) vs SurvKAN-300
  (`results/tuned_v6survkan_<ds>.csv`);
- (b) NestKAN (`results/tuned_v5_<ds>.csv`, rows with `model == "NestKAN"` and `rep == 0`) vs
  SurvKAN-1000.

Metrics: rNLL (lower better), C_td (higher better), IBS (lower better) — the ensemble columns
`rNLL`, `C_td`, `IBS` of the tuned files.

For each comparison, dataset and metric: d_f = metric(first) − metric(second) per fold f;
mean diff d̄; Nadeau–Bengio corrected SE = sqrt(var(d, ddof=1) · (1/5 + 0.2/0.64));
z = d̄ / SE; "resolved" iff |z| > 1.96. The verdict is "first better" / "first worse" accounting
for metric direction, else "unresolved". n_folds, both means, diff, se, z and verdict are written
to `results/v7_survkan_budget.csv`.

Training-budget diagnostics: from `results/trials_v6survkan1000_<ds>.csv`, column `epochs`
(number of epochs actually run, including the patience window), report per dataset the mean,
median and max epochs and the number of trials that hit the cap (`epochs == 1000`), alongside the
same quantities for v6 (cap 300).

## B4. Decision rule

If SurvKAN-1000 is resolved-better than SurvKAN-300 on rNLL on any dataset, the paper reports
SurvKAN-1000 alongside SurvKAN-300 for those datasets. Otherwise the paper reports SurvKAN-300 as
in v6 and cites v7 as a sensitivity check showing the budget does not change the conclusions.
Comparison (b) is reported in either case. All outcomes are reported, including unfavourable ones.

## B5. Execution and integrity

One worker process only (no two processes on the same tag/dataset), 4 torch threads, under a
watchdog that keeps Windows awake and relaunches the resumable worker if it dies. If the run cannot
be completed, only completed folds are reported, with the missing folds listed; no extrapolation.

## Amendment rev1 (2026-10-15, before launch; no v7 result existed)

`bench3.run` seeds the configuration sampler with `default_rng([rep, fold, sum(map(ord, family_name))])`,
so the new family name `SurvKAN-1000` alone would have drawn a *different* set of K = 8
configurations from v6 SurvKAN, confounding the epoch budget with configuration-sampling noise.
To make B1 hold as written ("same config RNG as v6"), `scripts/bench3.py` gains a module-level
`RNG_ALIAS = {}` (empty by default, so every existing family is seeded exactly as before) used as
`sum(map(ord, RNG_ALIAS.get(fam, fam)))`, and `scripts/bench4.py` sets
`bench3.RNG_ALIAS["SurvKAN-1000"] = "SurvKAN"`. Checked before launch: on rott2, nwtco and support2
(rep 0, folds 0–4) the 8 configurations drawn for SurvKAN-1000 are identical, in order, to the
`cfg` column of `results/trials_v6survkan_<ds>.csv` (15/15 folds). Training seeds (`1000*rep + fold`)
are also identical, so SurvKAN-1000 differs from SurvKAN-300 only in the epoch cap (plus any
floating-point differences from thread count). Backups: `scripts/bench3.py.bak_pre_v7`,
`scripts/bench4.py.bak_pre_v7`.
