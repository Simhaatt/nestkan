# Data

No data are distributed in this repository.

## SurvSet datasets (no download needed)
`rott2`, `smarto`, `nwtco`, `rhc`, `flchain` and `support2` are read from the [SurvSet](https://github.com/ErikinBC/SurvSet) Python package (`dkan.data.load_survset`).

Leakage-prone columns are dropped in `scripts/bench3.py` (`DROP`):
- support2: `num_surv2m`, `num_surv6m`, `fac_sfdm2`, `num_hday`;
- rhc: `num_surv2md1`;
- nwtco: `fac_in_subcohort`;
- flchain: `fac_chapter`, the cause-of-death chapter, which is defined only for deaths.

Cite the original studies as in the paper.

## Prepared discrete-time cohorts
These three cohorts are built by `scripts/prep_cohorts.py` from raw files that you obtain from the providers under their own terms. Check each provider's terms before sharing any derived file.

| Cohort | Source | Definition |
|---|---|---|
| `support2_slos` | SUPPORT study (`support2.csv`, Vanderbilt Biostatistics datasets, https://hbiostat.org/data/) | length of stay in days; event = discharged alive; in-hospital death censors; administrative horizon 60 days; 20 baseline covariates (columns with > 25% missing dropped) |
| `sparcs_drg302` | New York State SPARCS 2012 Hospital Inpatient Discharges (De-Identified), Health Data NY dataset `u4ud-w55t` | APR-DRG 302; length of stay in days ("120 +" → 120); event = discharge; horizon 30 days; covariates age band, severity, gender, risk of mortality, admission type, ED indicator |
| `drsa_clinic` | DRSA benchmark data, CLINIC train split | observed = min(z, b), event = 1{z ≤ b}; one categorical column per one-hot field with ≤ 100 levels |

Build them with:
```bash
export SUPPORT2_ZIP=/path/to/support2csv.zip
export SPARCS_CSV="/path/to/Hospital_Inpatient_Discharges_(SPARCS_De-Identified)__2012.csv"
export DRSA_CLINIC_DIR=/path/to/drsa-data/CLINIC
python scripts/prep_cohorts.py              # or name a subset, e.g. support2_slos
```

### Checksums of the files used in the paper
If your files match, the analyses use identical data.

| File `data/<cohort>.csv` | rows | events | SHA-256 |
|---|---|---|---|
| `support2_slos` | 9,105 | 6,464 | `aece5ddc8cd2b1519b986f9548745295ffc8b55a0197b7aa522a4045fdd824dc` |
| `drsa_clinic` | 4,828 | 4,238 | `36c3c893094a29d402bbade204f8105080441412e53d5cccc1d57cc3d76c982c` |
| `sparcs_drg302` | 34,233 | 34,203 | `b9cd31efa9fafbaa819510e0ff2c95b08d453c4d3ef4698fc84a77ef5747fed8` |

Checksums can differ if a provider re-issues its file. In that case compare row and event counts.
