# Release checklist

- [ ] Authors, ORCID, affiliations and licence (CC BY 4.0, as for the grouped-PH package) confirmed by both authors.
- [ ] Decide whether code should instead be MIT/Apache-2.0, a common choice for software. CC BY 4.0 is currently applied to everything.
- [ ] `python tests/smoke_test.py` passes in a clean environment.
- [ ] `git status` shows no data, results, credentials or archives before the first commit.
- [ ] Repository public; Zenodo GitHub toggle switched on **before** the release.
- [ ] Release `v1.0.0` published; Zenodo record checked and downloaded once.
- [ ] DOI added to CITATION.cff, README and the manuscript's Code Availability statement.
- [ ] Optional: add frozen result CSVs, the figure/table scripts and the manuscript in a later version (v1.1.0).
