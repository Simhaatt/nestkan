# Reproducing the analyses

All commands run from the repository root. Outputs are written to `results/`, which is created on first use.

## Conventions
- **Environment variables** read by the scripts:
  - `DKAN_TAG`: output tag; for example `_v5` gives `results/tuned_v5_<ds>.csv`.
  - `DKAN_SELECT`: selection metric, `rNLL` (default) or `IBS`.
  - `DKAN_THREADS`: torch threads (default 4).
- **Resumable.** Every driver appends one row per (rep, fold, method) to `results/tuned<tag>_<ds>.csv`, plus every tuning trial to `results/trials<tag>_<ds>.csv`. It skips units already present. Never run two processes on the same (tag, dataset).
- **Splits and seeds.**
  - Folds: `StratifiedKFold(5, shuffle=True, random_state=rep)` on the event indicator.
  - Validation set: 20% of the training part, `random_state = 1000*rep + fold`.
  - Configuration sampler: seeded by `(rep, fold, family name)`. `RNG_ALIAS` in `bench3.py` maps SurvKAN-1000 to SurvKAN's seed.
- **Inference.** Paired NestKAN − comparator differences with the Nadeau–Bengio corrected variance var × (1/J + 0.2/0.64). A difference is "resolved" when |z| > 1.96.

## Analysis map

| Paper analysis | Protocol | Command(s) | Output |
|---|---|---|---|
| Main benchmark: 9 datasets, 11 methods, 3 × 5 CV (SPARCS 1 × 5), K = 8, 3-seed ensembles | v5 | `DKAN_TAG=_v5 python -u scripts/run_tuned3.py <ds>:<rep>,... 8` | `results/tuned_v5_<ds>.csv`; `python scripts/summarize_v5.py` → `results/v5_{summary,paired,ranks,counts}.csv` |
| KAN comparators: SurvKAN, KAPLAN-HR, CoxKAN-sym (rep 0) | v6 | `python -u scripts/run_v6.py <SurvKAN\|KAPLAN-HR\|CoxKAN-sym> 4 [ds,...]` | `results/tuned_v6survkan_*`, `_v6kaplanhr2_*`, `_v6coxkansym_*`; `python scripts/summarize_v6.py` |
| SurvKAN at a 1000-epoch budget | v7, v8 item 2 | `python -u scripts/run_v6.py SurvKAN-1000 4 [ds,...]` | `results/tuned_v6survkan1000_*`; `python scripts/summarize_v7.py`, `python scripts/summarize_v8_survkan1000.py` |
| Likelihood ladder: grouped Cox → additive → surface | v8 item 5 | `DKAN_TAG=_v8ladder python -u scripts/bench8.py <ds> 8 <rep> NestKAN-lin` | `results/tuned_v8ladder_*`; `python scripts/summarize_v8_ladder.py` |
| Brier-score-selected sensitivity | v8 item 3 | `DKAN_TAG=_v8ibs DKAN_SELECT=IBS python -u scripts/bench8.py <ds> 8 0 LogisticHazard,PMF,MTLR,DeepHit,DeepSurv,NestKAN,PCHazard,CoxKAN,NestKAN-PH,CoxPH,RSF` | `results/tuned_v8ibs_*`; `python scripts/summarize_v8_ibs.py` |
| Known-truth simulation: 4 scenarios, n = 1,000 (reps 0–19) and 4,000 (reps 0–9) | v8 item 1 | `python -u scripts/sim.py <scenario> <n> <a>-<b> 6`, then `python -u scripts/sim8.py` (new replicates + NestKAN-sel) | `results/sim_<scen>_n<n>.csv`, `results/sim8_sel_*`; `python scripts/sim8_summary.py` |
| Dose-response calibration of the PH diagnostic | v8 item 4 | `python -u scripts/sim8_dose.py` | `results/sim8_dose.csv`; `python scripts/sim8_dose_summary.py` |
| Interpretability: functional ANOVA, shapes, partial-dependence curves, Grambsch–Therneau tests | post hoc | `python -u scripts/interp.py <ds> [n_seeds]`; `python -u scripts/interp.py sim` | `results/interp/` |
| Ablations: ECDF vs min-max, grid size, blocks | v4 folds | `python -u scripts/ablate.py rott2,rhc,...` | `results/ablate_<ds>.csv` |
| Within-interval censoring bias (exploratory) | — | `python scripts/sim_censor_bias.py results/aux_censor_bias.csv [gamma]` | CSV of bias by convention |

Scenario names: `ph_linear`, `ph_additive`, `nonph`, `nonph_inter`.

Datasets: `rott2`, `smarto`, `nwtco`, `drsa_clinic`, `rhc`, `flchain`, `support2`, `support2_slos`, `sparcs_drg302`.

## Run times (laptop, 12-core CPU, 4 torch threads per process)
- **v5:** about 16 h for repeat 0 of all datasets.
- **SurvKAN:** 0.5–6 h per dataset; SPARCS is the slowest.
- **Simulation:** about 3 min per replicate at n = 1,000 and 12 min at n = 4,000.
- **Dose-response study:** about 1.5 h.
- **Memory:** 0.5–1.5 GB per process.

## Notes
- `scripts/bench3.py`, `interp.py` and `coxkan_official.py` add `../pylib` and `../pylib_ck` to `sys.path` if those folders exist. This supports pip `--target` installs; it is harmless otherwise.
- If Python is started with `PYTHONSAFEPATH` set (or `-P`), scripts launched as subprocesses cannot import sibling modules. Unset it, or run from `scripts/` with that folder on `PYTHONPATH`.
- Results depend slightly on thread count and library versions, typically in the last digits.
- The discretiser counts a subject censored inside interval k as event-free through k − 1. Held-out rNLL uses interval-censored contributions and is unaffected. See the paper's Web Appendix on within-interval censoring.
