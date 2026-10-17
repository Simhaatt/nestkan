# PROTOCOL v8: robustness analyses for the NestKAN manuscript (pre-registered)

Written 2026-10-15, before any v8 run. It covers five analyses (items 1–5). Item 6, the covariate-level PH-diagnostic scatter, is a descriptive post hoc analysis of existing refits, and the paper labels it as exploratory.

**Global rules**
- Existing result files are never modified. Every v8 run writes to new tags.
- Methods, splits, seeds, tuning budgets and metrics are as in v5, v6 and v7 unless stated otherwise.
- Resolved means |z| > 1.96.
- Partial completion is reported as such, with the units completed. Nothing is extrapolated.

**Compute.** Everything runs on the laptop, two worker processes at a time, under a keep-awake, auto-restart watchdog.
- Track *Simulation*: item 1, then item 4.
- Track *Benchmarks*: item 5, then item 2, then item 3.

---

## Item 1: more simulation replicates, plus surface-selection rates

**Design**
- Use the existing `scripts/sim.py` design unchanged:
  - 4 scenarios: ph_linear, ph_additive, nonph, nonph_inter;
  - the same 8 families, K = 6, selection on validation rNLL;
  - the same replicate RNG, `default_rng([2026, rep, n, sum(ord(scen))])`.
- Replicate targets:
  - n = 1,000: replicates 0–19 (the new ones are 5–19);
  - n = 4,000: replicates 0–9 (new: 2–9 for ph_linear, 3–9 for the others).
- **New family `NestKAN-sel`.** NestKAN with the v5 search space, so `use_residual ∈ {True, False}` is tuned. It is run on every replicate, including the existing ones. Its `best_cfg` records whether the surface was kept. The existing `NestKAN` family (surface always on) is unchanged.
- Output files: `results/sim_<scen>_n<n>.csv`. New rows are appended for new replicates; replicates already present are skipped. `NestKAN-sel` rows go to `results/sim8_sel_<scen>_n<n>.csv`.
- Run order: all n = 1,000 cells first, then n = 4,000.
- Minimum to report: 10 replicates at n = 1,000 and 5 at n = 4,000 per scenario.

**Analysis (fixed)**
- Mean ISE × 100 per (scenario, n, method), with SE over replicates.
- Paired differences NestKAN − method per (scenario, n), using a plain paired SE (replicates are independent datasets).
- The same for `NestKAN-sel`.
- Surface-kept rate of `NestKAN-sel` per (scenario, n), with a Wilson 95% CI.
- Pre-stated expectation, from Proposition 4: the kept rate is lower under the PH scenarios than under the non-PH scenarios at both n, and the gap is larger at n = 4,000. The result is reported whether or not this holds.

## Item 4: dose-response calibration of the PH diagnostic

**Design**
- Generate data with $\log h = \log h_0(t) + g_{\text{additive}}(x) + \delta\,(1-2F_t(t))\,x_4$. Here g_additive is the ph_additive scenario, so δ = 0 is exactly ph_additive and δ = 1 is the nonph scenario.
- Grid and size: δ ∈ {0, 0.25, 0.5, 0.75, 1.0}, n = 1,000, 20 replicates per δ, censoring as in sim.py.
- Replicate RNG: `default_rng([2027, rep, round(100δ)])`.
- Per replicate:
  - tune `NestKAN-sel` (v5 space, K = 6, validation rNLL) and record whether the selected configuration kept the surface;
  - refit the selected configuration (seed = rep) and compute the functional-ANOVA time-varying share (`scripts/interp.py: anova`);
  - compute the x4 partial-dependence time-varying SD, i.e. the SD over intervals of the per-unit log-HR slope;
  - fit a Grambsch–Therneau test (lifelines `proportional_hazard_test`, rank transform, Cox ridge 0.01) on the training data, and record the x4 p-value and the minimum covariate p-value.
- Output: `results/sim8_dose.csv`.

**Analysis (fixed)**
- Per δ: median and IQR of the time-varying share; surface-kept rate with a Wilson CI; Grambsch–Therneau rejection rate for x4 at α = 0.05.
- Spearman correlation of the share with δ, over all replicates.
- The false-positive rate is the surface-kept rate at δ = 0. Also report the share distribution at δ = 0, which is the null distribution.

## Item 5: likelihood ladder on all 9 datasets

**Design**
- New family `NestKAN-lin`: the NestKAN code with `use_additive=False, use_residual=False`, i.e. the grouped Cox model with a penalised interval baseline.
- Search space: lam {1e-4, 1e-3, 1e-2}, lr {1e-3, 3e-3}, wd {1e-5, 1e-4}, lam_alpha {0.1, 1, 10}, T {10, 20, 40}.
- Run on the v5 design: 3 × 5 folds (SPARCS 1 × 5), K = 8, rNLL selection, 3-seed ensemble. Output tag `_v8ladder`.

**Analysis (fixed)**
- Per dataset, paired over all v5 units with the Nadeau–Bengio repeated-CV SE, var × (1/J + 0.2/0.64):
  - step 1: NestKAN-PH − NestKAN-lin (the additive block);
  - step 2: NestKAN − NestKAN-PH (the surface).
- Metrics: rNLL (primary), C_td.
- Report the v5 surface-kept fraction per dataset alongside.

## Item 2: SurvKAN-1000 on the remaining 6 datasets

**Design**
- Exactly as v7, with the `SurvKAN-1000` family (1,000-epoch cap, patience 15). The configuration sampler is seeded as for SurvKAN (alias), so the configurations match v6.
- Datasets: smarto, drsa_clinic, rhc, flchain, support2_slos, sparcs_drg302, in that order; rep 0, folds 0–4.
- Minimum to report: all datasets except sparcs_drg302.

**Analysis (fixed)**
- As in v7: (a) SurvKAN-1000 vs SurvKAN-300, and (b) NestKAN vs SurvKAN-1000, on rNLL, C_td and IBS. Also report epochs used and cap hits.

**Decision rule (fixed now).** The main-text SurvKAN comparison is replaced by SurvKAN-1000 on every dataset where it completed. SurvKAN-300 moves to the supplement.

## Item 3: sensitivity to the tuning metric (Brier-score selection)

**Design**
- All 11 v5 families with `DKAN_SELECT=IBS`: the configuration with the lowest validation IPCW integrated Brier score is selected.
- Everything else as in v5: K = 8, the same splits and seeds, 3-seed ensembles.
- Scope: rep 0, folds 0–4, all 9 datasets. Output tag `_v8ibs`.
- Order: datasets by size, smallest first.
- Minimum to report: the datasets on which all 11 methods completed.

**Analysis (fixed)**
- Mean ranks per metric (rNLL, C_td, IBS, IBS_cx) over the completed datasets, using the rep-0 units of the IBS-selected run.
- Compared side by side with the rNLL-selected v5 ranks restricted to the same rep-0 units and datasets.
- Paired NestKAN vs each method, with the Nadeau–Bengio single-run SE var × (1/5 + 0.2/0.64): counts of better / unresolved / worse per metric.


---

## Amendment / scheduling note, 2026-10-15 (Benchmarks track; changes no design or analysis)

On the lead's instruction, item 3 no longer waits for item 2 to finish. It starts as a second worker in parallel with the ongoing item-2 SurvKAN-1000 SPARCS run. Trigger: `results/v8_dose_summary.csv` exists (the Simulation track has stopped its worker), or free RAM > 1.5 GB, checked every 10 min.
- Each worker keeps DKAN_THREADS=4.
- Item-3 jobs, order (smallest dataset first), tag `_v8ibs`, DKAN_SELECT=IBS, splits, seeds, K = 8 and ensembles are unchanged.
- Implementation: `scripts/watchdog_v8_ibs.py` runs the item-3 jobs. `scripts/bench8.py` now takes an exclusive per-(tag, dataset) lock file, so the original Benchmarks watchdog (`scripts/watchdog_v8_bench.py`, which still lists the item-3 jobs after item 2) can never run the same dataset at the same time. A second launch waits for the lock, then resumes, skipping completed units.
- Item 5 had already completed under the unlocked bench8 (125/125 units) and is unaffected.
- Timing: item-3 secs per unit may be inflated by the parallel load. Run times are not an analysed metric.
- 2026-10-15 23:2x: item 2 finished at 23:15 (all 6 datasets). The original Benchmarks watchdog was then stopped, so all item-3 jobs run in the single lane of watchdog_v8_ibs.py: one item-3 worker, DKAN_THREADS=4. Nothing else changes.
- 2026-10-16 00:10: lead asked for item 3 to run as two workers on disjoint dataset lists. Lane A, `watchdog_v8_ibs.py`, keeps the smallest-first list. Lane B, `watchdog_v8_ibsB.py`, takes sparcs_drg302 then support2, each worker with DKAN_THREADS=4.
  - Lane B starts only when free RAM > 1.5 GB, checked every 10 min; at the time of this note free RAM was 0.2–0.4 GB, so one worker continues.
  - Lane B skips any dataset that is already complete or whose bench8 lock is held by a live process. Lane A waits on the lock for any dataset lane B holds, then resumes, skipping completed units.
  - Jobs, settings and analysis are unchanged.
- 2026-10-16 04:41: lane B never started, because free RAM stayed between 0.07 and 0.43 GB. Lane A ran all item-3 datasets in the pre-registered smallest-first order. Lane B was stopped once only sparcs_drg302 remained, already running in lane A.
