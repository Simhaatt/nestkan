"""v6 supervisor: one family per worker, datasets smallest first, resumable, auto-restart.
usage: python run_v6.py <family> <threads>"""
import os, sys, subprocess, time
import pandas as pd
R = os.environ.get("DKAN_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
fam, thr = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(R, "tmp"), exist_ok=True); os.makedirs(os.path.join(R, "results"), exist_ok=True)
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None   # optional dataset subset (parallel workers)
ORDER = ["rott2", "smarto", "nwtco", "drsa_clinic", "rhc", "flchain", "support2", "support2_slos", "sparcs_drg302"]
if fam != "SurvKAN":
    ORDER.remove("sparcs_drg302")          # PROTOCOL_v6 D3
tag = "_v6" + fam.replace("-", "").lower() + ("2" if fam == "KAPLAN-HR" else "")   # full-Q rerun; "_v6kaplanhr" = withdrawn capped launch
env = dict(os.environ, DKAN_TAG=tag, DKAN_THREADS=thr)
for ds in (ONLY or ORDER):
    for attempt in range(4):
        f = os.path.join(R, "results", f"tuned{tag}_{ds}.csv")
        if os.path.exists(f) and len(pd.read_csv(f).query("rep == 0")) >= 5:
            break
        print(time.strftime("%H:%M:%S"), fam, ds, "attempt", attempt, flush=True)
        rc = subprocess.run([sys.executable, "-u", os.path.join(R, "scripts", "bench4.py"), ds, "8", "0", fam],
                            cwd=os.path.join(R, "tmp"), env=env).returncode
        print(time.strftime("%H:%M:%S"), fam, ds, "rc", rc, flush=True)
print("DONE", fam, flush=True)
