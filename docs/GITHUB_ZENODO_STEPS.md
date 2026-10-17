# Publishing on GitHub and Zenodo

Do these steps in order. Use only this folder, never the parent research directory.

## 1. Check the package locally
```powershell
python -m venv .venv; .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt; pip install -e .
python tests/smoke_test.py
```
Review `README.md`, `CITATION.cff` and `.zenodo.json`: author names, ORCID, affiliations and licence. Each file needs your approval before publication.

## 2. Create ONE GitHub repository
1. On GitHub: **+ → New repository**.
2. Owner `Simhaatt`, name **nestkan**.
3. Description: "Reproducibility package for NestKAN: A Nested Kolmogorov–Arnold Model for Discrete-Time Survival Data".
4. Leave "Add README", ".gitignore" and "licence" **unchecked**, because the files already exist.
5. You may start the repository private for co-author review, but make it public before step 4.

## 3. Push this folder
Open PowerShell **inside this folder**, then run:
```powershell
git init -b main
git add .
git status --short          # check: no data/, results/, .venv/, zip or secrets
git commit -m "NestKAN reproducibility package v1.0.0"
git remote add origin https://github.com/Simhaatt/nestkan.git
git push -u origin main
```

## 4. Connect Zenodo BEFORE the first release
1. Log in to https://zenodo.org with your GitHub account.
2. Go to **Account → GitHub** and switch the toggle **on** for `Simhaatt/nestkan`. The repository must be public.
3. Zenodo reads `.zenodo.json` for the title, creators, licence and keywords.

## 5. Create the GitHub release
1. On GitHub: **Releases → Draft a new release**.
2. Tag `v1.0.0` on `main`, titled "v1.0.0".
3. Publish.
4. Within a few minutes Zenodo archives the release and mints two DOIs: a **version DOI** for v1.0.0 and a **concept DOI** for all versions.

## 6. Afterwards
- Put the DOI in `CITATION.cff` (uncomment `doi:` and `date-released:`), and add a DOI badge to `README.md`.
- In the manuscript: the Code Availability statement, the reference list and `\todo{repository URL}`.
- Commit these metadata edits to `main`. You do not need a new release just to add the DOI. If you want the archived copy to contain it, release v1.0.1.
- Never edit or re-upload a published version, and never reuse an earlier version's DOI for a new one.
