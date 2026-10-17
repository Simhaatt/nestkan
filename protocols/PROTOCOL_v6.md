# Pre-registered protocol: v6 — KAN survival comparators

This protocol was written before any v6 production result existed. The only v6 numbers seen beforehand are one smoke-test row per method on rott2 (rep 0, fold 0), used solely to measure cost. Those smoke rows are tagged `_v6smoke` and are excluded from the analysis.

- Code: `scripts/bench4.py` (reuses `bench3.run`, so splits, grids and metrics are identical to v5), `src/dkan/ctkan.py`, `src/dkan/coxkan_official.py`, `scripts/run_v6.py`.
- Output files: `results/tuned_v6<family>_<ds>.csv`.

## Methods added

1. **SurvKAN** (arXiv 2602.02179). No code has been released, so this is a re-implementation from the paper.
   - Model: $\log h(t\mid x)=\mathrm{KAN}([x,t])$ with width $[d{+}1,m,1]$.
   - Edge functions: $w_b\,b(\cdot)+\text{quadratic B-spline}$, with $b\in\{\text{identity},\text{SiLU}\}$.
   - Training: full-likelihood NLL, with $\Lambda(t_i\mid x_i)$ computed by the trapezoid rule on 50 uniform points in $[0,t_i]$; log-hazard clamped to $[-20,20]$; L1 activation-type regulariser.
   - Search space: $m\in\{0,1,2,3\}$, grid $\in\{3,5,8\}$, base $\in\{\text{identity},\text{SiLU}\}$, $\lambda\in\{0,10^{-4},10^{-3},10^{-2}\}$, lr $\in\{10^{-3},3\cdot10^{-3},10^{-2}\}$.
   - Optimisation: batch 256, at most 300 epochs.
2. **KAPLAN-HR** (arXiv 2605.23082). No code has been released, so this is a re-implementation from the paper.
   - Model: $\log\lambda(t\mid x)=\mathrm{KAN}(x,t)$ with $D$ layers; $D=1$ is the additive/PH model with a time function.
   - Training: $\Lambda$ by a right-endpoint rectangle rule on a shared grid of $L=128$ points over $[0,t_{\max}]$; shrinkage (L1) penalty.
   - Search space: $D\in\{1,2,3\}$, hidden width $Q\in\{2,4,8,16,32\}$ (paper's range), spline order $\in\{2,3,4\}$, knots $\in\{3,5,8,10\}$, $\lambda_{\rm shrink}\in\{0,10^{-5},10^{-4},10^{-3}\}$, lr $\in\{10^{-3},3\cdot10^{-3}\}$.
   - Optimisation: batch 128, at most 200 epochs.
3. **CoxKAN-sym**: the official `coxkan` 0.0.2 package (with its pinned `pykan` 0.0.2), including the symbolic step.
   - Pipeline: train with the package's own early stopping → prune edges (threshold 0.02) → `auto_symbolic` → fine-tune.
   - Symbolic library: bounded on bounded inputs — x, x², x³, sqrt, abs, sin, tanh, sigmoid, arctan, gaussian, 0. It excludes 1/x, log, exp, tan and arctanh, which overflowed in testing.
   - Fine-tune: affine parameters at lr/10 for 25 steps. It is kept only if the validation partial likelihood improves; without this rule it destroyed the fit (rott2 C 0.71 → 0.57).
   - Search space: hidden $\in\{0,2\}$ (width $[d,1]$ or $[d,2,1]$), grid $\in\{3,5\}$, lr $\in\{10^{-3},10^{-2}\}$, $\lambda\in\{0,10^{-3},10^{-2}\}$; 100 steps.
   - Survival curves: Breslow baseline (same code as DeepSurv / CoxKAN).

Implementation choices common to SurvKAN and KAPLAN-HR, stated in the paper as re-implementation details:
- inputs are min-max scaled to $[-1,1]$;
- time is mapped to $[-1,1]$ via $t/t_{\max}(\text{train})$;
- hidden activations pass through tanh onto a fixed uniform knot grid, instead of pykan grid updates;
- B-splines use an exact closed form (verified equal to the Cox–de Boor recursion to $7\times10^{-7}$);
- Adam, with early stopping on validation NLL (patience 15), as for all v5 neural methods;
- prediction by trapezoid integration of the hazard on 400 points over $[0,t_{\max}]$, with the hazard held constant beyond $t_{\max}$.

## Unchanged from v5

- K = 8 random configurations per fold;
- selection on validation rNLL;
- the same StratifiedKFold splits and train/validation split seeds;
- the same evaluation grid, metrics (rNLL, C_td, IBS, IBS_cx), leakage drops and censoring models.

## Deviations from the full v5 design (compute)

All deviations are fixed now and are reported in the paper.

| # | Deviation | Reason |
|---|---|---|
| D1 | **Repeat 0 only** (the 5 folds shared by v4 and v5) for all three new methods. | Measured cost per rott2 row: SurvKAN 360 s, KAPLAN-HR 1,623 s, CoxKAN-sym 1,730 s. Three repeats would take more than 100 CPU-hours. |
| D2 | *Withdrawn (revision 1).* A cap of $Q\le16$ was first written here and launched. It was withdrawn before any KAPLAN-HR production row existed, because the smoke test (rott2, fold 0) selected a $Q=32$ configuration: validation rNLL 1.7155, against 1.7523 for the best $Q\le16$ trial. The cap would therefore have handicapped the comparator. KAPLAN-HR uses the paper's full range $Q\in\{2,\dots,32\}$. Output tag `_v6kaplanhr2`; any `_v6kaplanhr` rows come from the withdrawn capped launch and are excluded. | — |
| D3 | **KAPLAN-HR and CoxKAN-sym are not run on sparcs_drg302** (n = 34,233). | One deep KAPLAN configuration would take several hours there; for CoxKAN-sym, the edge count × symbolic sweep cost is large. SurvKAN is run on all 9 datasets. |
| D4 | **CoxKAN-sym is not ensembled.** | The symbolic step's purpose is a single closed-form formula, and averaging three formulas' survival curves is not how the method is used. Its symbolic step is applied once, to the configuration selected on the spline stage (tune, then symbolify). Symbolic fitting uses cached activations of at most 2,000 random training rows. SurvKAN and KAPLAN-HR are ensembled over 3 seeds exactly as in v5. |

## Analysis (fixed now)

**Primary comparison.**
- NestKAN (v5, rep 0 rows) − comparator, paired over the 5 rep-0 folds per dataset.
- Ensembled vs. ensembled for SurvKAN and KAPLAN-HR.
- For CoxKAN-sym (single formula), the primary comparison is NestKAN single-seed (`rNLL_single`, `C_single`, `IBS_single`) vs. CoxKAN-sym. NestKAN ensembled vs. CoxKAN-sym is also reported.

**Standard errors and resolution.**
- SE uses the Nadeau–Bengio correction for a single 5-fold run: $\text{var}\times(1/5+0.2/0.64)$.
- "Resolved" means $|z|>1.96$.
- Report counts better / worse / unresolved per metric, and mean ranks over the rep-0 units of all 14 methods on the datasets each method covers.

**Interpretability side-result.** Report the fraction of CoxKAN-sym fits whose symbolic step succeeded, and how often the fine-tune was kept (`results/coxkan_sym_log.csv`).

**Integrity.** All results are reported, including datasets where NestKAN loses. v5 results are not altered.

## Execution note (2026-10-10, written before any further v6 rows were seen)

The remaining v6 rows are run on Kaggle CPU sessions (4 vCPU, 12 h limit) instead of the laptop. This changes nothing in the methods, search spaces, K, seeds, splits, metrics or analysis above. `scripts/kaggle_v6.py` launches the same command as `run_v6.py` (`bench4.py <ds> 8 0 <family> [folds]`), with the same output tags. Work is sharded across sessions by (family, dataset, fold), using 1–2 torch threads per process, and results are merged on (dataset, rep, fold, model). Package versions on Kaggle may differ slightly from the laptop. coxkan 0.0.2 and pykan 0.0.2 stay pinned. Thread count and library versions can change floating-point results at the last digits only.


## Revision 2 — analysis note (2026-10-14, after CoxKAN-sym completed, before KAPLAN-HR completed)

On support2_slos, fold 0, the CoxKAN-sym symbolic formula fitted the validation data, which had a finite validation partial likelihood. On some test subjects, however, it produced non-finite risk scores, so rNLL and IBS were undefined (C_td was reported as 0).

Rule adopted for every method: a fit with non-finite test survival curves is a **failed fit**.
- Its metrics are set to missing.
- The unit is dropped from the paired comparison, and the number of paired folds is reported.
- The failure is listed in `results/v6_failed_fits.csv`.

Dropping a failed fit can only favour the comparator, never NestKAN.

Separately, as already implemented before launch, 3 of 40 CoxKAN-sym fits fell back to the spline model because the symbolic step gave a non-finite validation loss.
