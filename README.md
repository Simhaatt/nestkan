# NestKAN: A Nested Kolmogorov–Arnold Model for Discrete-Time Survival Data: reproducibility package

Code for the paper *NestKAN: A Nested Kolmogorov–Arnold Model for Discrete-Time Survival Data* by Simhaa T. T. and Sudheesh Kumar Kattumannil.

NestKAN is a discrete-time hazard model whose complementary log-log predictor nests the grouped-data Cox model, an additive spline model on empirical-distribution-scaled covariates, and a gated Kolmogorov–Arnold network surface in covariates and time.

This repository contains:
- the `dkan` library (models, data handling, metrics, re-implemented comparators);
- the scripts for every benchmark, simulation, interpretability and robustness analysis in the paper;
- the pre-registration protocols (v5–v8).

**It contains code only.** No result files, figures or third-party data are included; see [data/README.md](data/README.md) for how to obtain the data.

## Contents

| Path | What it is |
|---|---|
| `src/dkan/` | library: `data.py` (preprocessing, ECDF transform, discretisation), `models.py` (NestKAN and the discrete-time / neural comparators), `metrics.py` (rNLL, C_td, IBS), `ctkan.py` (SurvKAN, KAPLAN-HR re-implementations), `coxkan_official.py` (wrapper for the official `coxkan` package) |
| `scripts/` | analysis drivers and summaries (map in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)) |
| `protocols/` | pre-registration protocols v5–v8, with dated revisions and amendments |
| `data/README.md` | data sources, cohort definitions, checksums of the prepared cohorts |
| `tests/smoke_test.py` | quick installation check |
| `MANIFEST.sha256` | SHA-256 of every file in this release |

## Installation

Python 3.11 was used; see `requirements.txt` for the versions used in the paper.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate     Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install --no-deps coxkan==0.0.2 pykan==0.0.2      # only for the CoxKAN-sym comparator
pip install -e .
python tests/smoke_test.py
```

**Do not install `pykan>=0.2`.** `coxkan` 0.0.2 requires `pykan` 0.0.2.

## Reproducing the analyses

Every analysis is resumable: rerunning a command skips units already present in its output CSV. Run at most one process per (output tag, dataset). The full command list, expected outputs, seeds and approximate run times are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). In brief:

```bash
# v5 benchmark (11 methods, 3 x 5-fold CV), e.g. one dataset and repeat
DKAN_TAG=_v5 python -u scripts/run_tuned3.py rott2:0,rott2:1,rott2:2 8
python scripts/summarize_v5.py
# v6 KAN comparators; v7/v8 SurvKAN at 1000 epochs
python -u scripts/run_v6.py SurvKAN 4      # also KAPLAN-HR, CoxKAN-sym, SurvKAN-1000
python scripts/summarize_v6.py; python scripts/summarize_v7.py; python scripts/summarize_v8_survkan1000.py
# known-truth simulation and v8 robustness analyses
python -u scripts/sim.py nonph 4000 0-9 6; python -u scripts/sim8.py; python -u scripts/sim8_dose.py
```

On Windows, set environment variables with `set DKAN_TAG=_v5` (cmd) or `$env:DKAN_TAG="_v5"` (PowerShell).

## Pre-registration

Each benchmark stage was specified in `protocols/PROTOCOL_vX.md` before its results were seen. Later changes are recorded in the protocol files as dated revisions or amendments.

## Citation

Please cite the archived release (DOI on the Zenodo record; see `CITATION.cff`) and the paper.

## Licence

The authors' original material is released under CC BY 4.0 (see `LICENSE` and `NOTICE.md`). Third-party packages and datasets keep their own terms. No raw third-party data are distributed.
