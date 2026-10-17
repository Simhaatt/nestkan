"""Convert the user's discrete-time cohorts to the SurvSet layout (num_*, fac_*, time, event)
in data/<name>.csv.  Definitions follow kanrel/data.py of the grouped-PH project.

  support2_slos  : SUPPORT length of stay (days); event = discharged alive; in-hospital death
                   censors; administrative horizon 60 days.  20 baseline covariates (kanrel list).
  sparcs_drg302  : SPARCS 2012, APR-DRG 302 (knee replacement); LOS (days, "120 +" -> 120);
                   event = discharge; horizon 30 days.  Covariates: age band, severity, gender,
                   risk of mortality, admission type, ED indicator.
  drsa_clinic    : DRSA CLINIC train split; observed = min(z, b), event = 1{z <= b};
                   one categorical column per one-hot FIELD with <= 100 levels.
"""
import os, sys, zipfile
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)
SUPPORT_COVARIATES = ["age", "sex", "dzgroup", "dzclass", "num.co", "scoma", "race", "sps", "aps",
                      "meanbp", "hrt", "resp", "temp", "wblc", "crea", "sod", "diabetes", "dementia",
                      "ca", "adlsc"]
SUPPORT_FACTORS = {"sex", "dzgroup", "dzclass", "race", "diabetes", "dementia", "ca"}


def support2_slos(path=None, horizon=60):
    path = path or os.environ["SUPPORT2_ZIP"]
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(z.open("support2.csv"))
    out = pd.DataFrame()
    for c in SUPPORT_COVARIATES:
        if df[c].isna().mean() > 0.25:      # same rule as kanrel._prepare
            continue
        name = c.replace(".", "_")
        out[("fac_" if c in SUPPORT_FACTORS else "num_") + name] = df[c]
    t = pd.to_numeric(df["slos"], errors="coerce").fillna(1).to_numpy(float)
    ev = 1 - df["hospdead"].to_numpy(int)
    beyond = t > horizon
    out["time"] = np.minimum(t, horizon); out["event"] = np.where(beyond, 0, ev)
    return out


def sparcs(drg="302", horizon=30, path=None):
    path = path or os.environ["SPARCS_CSV"]
    cols = ["APR DRG Code", "Length of Stay", "Age Group", "Gender", "APR Severity of Illness Code",
            "APR Risk of Mortality", "Type of Admission", "Emergency Department Indicator"]
    parts = []
    for ch in pd.read_csv(path, usecols=cols, dtype=str, chunksize=200_000, low_memory=False):
        parts.append(ch[ch["APR DRG Code"].astype(str).str.strip().str.zfill(3) == drg.zfill(3)])
    df = pd.concat(parts, ignore_index=True)
    age = {"0 to 17": 0, "18 to 29": 1, "30 to 49": 2, "50 to 69": 3, "70 or Older": 4}
    t = pd.to_numeric(df["Length of Stay"].str.replace("+", "", regex=False).str.strip(), errors="coerce").fillna(120).to_numpy(float)
    out = pd.DataFrame({"num_age_band": df["Age Group"].map(age).astype(float),
                        "num_severity": pd.to_numeric(df["APR Severity of Illness Code"], errors="coerce"),
                        "fac_gender": df["Gender"], "fac_risk_mortality": df["APR Risk of Mortality"],
                        "fac_admission": df["Type of Admission"], "fac_ed": df["Emergency Department Indicator"]})
    beyond = t > horizon
    out["time"] = np.minimum(t, horizon); out["event"] = np.where(beyond, 0, 1)
    return out


def drsa_clinic(root=None, max_card=100):
    root = root or os.environ["DRSA_CLINIC_DIR"]
    fmap = {}
    for line in open(os.path.join(root, "featindex.txt"), encoding="utf-8", errors="replace"):
        key, _, idx = line.rstrip("\n").partition("\t")
        if idx:
            try:
                fmap[int(idx)] = int(key.split(":", 1)[0])
            except ValueError:
                pass
    nf = max(fmap.values()) + 1
    rows, zs, bs = [], [], []
    for line in open(os.path.join(root, "train.yzbx.txt"), encoding="utf-8", errors="replace"):
        p = line.split()
        zs.append(int(p[1])); bs.append(int(p[2]))
        v = np.full(nf, -1, dtype=np.int64)
        for tok in p[3:]:
            i = int(tok.split(":", 1)[0]); f = fmap.get(i)
            if f is not None:
                v[f] = i
        rows.append(v)
    X = np.vstack(rows)
    card = np.array([len(np.unique(X[:, j])) for j in range(nf)])
    keep = [j for j in range(nf) if 1 < card[j] <= max_card]
    out = pd.DataFrame({f"fac_f{j}": X[:, j].astype(str) for j in keep})
    z, b = np.asarray(zs, float), np.asarray(bs, float)
    out["time"] = np.maximum(np.minimum(z, b), 1e-3); out["event"] = (z <= b).astype(int)
    return out


if __name__ == "__main__":
    # Raw inputs are located via environment variables (see data/README.md):
    #   SUPPORT2_ZIP     path to support2csv.zip (containing support2.csv)
    #   SPARCS_CSV       path to the SPARCS 2012 de-identified inpatient discharge CSV
    #   DRSA_CLINIC_DIR  path to the DRSA CLINIC folder (featindex.txt, train.yzbx.txt)
    # Optional arguments restrict the cohorts built, e.g.  python scripts/prep_cohorts.py support2_slos
    wanted = set(sys.argv[1:])
    for name, fn in [("support2_slos", support2_slos), ("drsa_clinic", drsa_clinic), ("sparcs_drg302", sparcs)]:
        if wanted and name not in wanted:
            continue
        d = fn()
        d = d[d["time"] > 0].reset_index(drop=True)
        d.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(name, d.shape, "event rate", round(d.event.mean(), 3), "unique times", d.time.nunique(),
              "cols", [c for c in d.columns if c not in ("time", "event")][:12], flush=True)
