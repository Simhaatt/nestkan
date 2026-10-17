"""Quick installation check (about 1-3 minutes on a laptop CPU).

Runs one cross-validation fold of NestKAN and Cox PH on the Rotterdam data (from the SurvSet
package; no download), with K = 1 configuration, and checks that finite metrics are produced.
Output goes to results/tuned_smoke_rott2.csv and is deleted afterwards.
    python tests/smoke_test.py
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ["DKAN_TAG"] = "_smoke"; os.environ.setdefault("DKAN_THREADS", "2")
import pandas as pd
import bench3

out = os.path.join(ROOT, "results", "tuned_smoke_rott2.csv")
tri = os.path.join(ROOT, "results", "trials_smoke_rott2.csv")
for p in (out, tri):
    if os.path.exists(p):
        os.remove(p)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
bench3.run("rott2", K=1, families=["NestKAN", "CoxPH"], folds=[0], rep=0)
d = pd.read_csv(out)
assert set(d.model) == {"NestKAN", "CoxPH"}, d.model.tolist()
for _, r in d.iterrows():
    assert all(math.isfinite(r[m]) for m in ("rNLL", "C_td", "IBS")), r
    print(f"{r.model:8s} rNLL={r.rNLL:.4f}  C_td={r.C_td:.4f}  IBS={r.IBS:.4f}")
for p in (out, tri):
    os.remove(p)
print("smoke test passed")
