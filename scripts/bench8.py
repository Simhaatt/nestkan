"""v8 (pre-registered in PROTOCOL_v8.md) wrapper around the v5 driver bench3.

Item 5 (likelihood ladder): adds the family 'NestKAN-lin' = NestKAN code with
use_additive=False, use_residual=False, i.e. the grouped Cox model
eta_k = alpha_k + beta'z with the second-difference-penalised interval baseline.
hidden / grid are not searched: with both KAN blocks off the model has only alpha
and beta, so the NestKAN defaults (hidden=8, grid=5) create no parameters.

Item 3 (IBS-selected sensitivity) also runs through this file with the 11 v5
families and DKAN_SELECT=IBS; existing families are untouched (bench3 is imported,
not modified; bench4's monkey-patches are NOT loaded here).

usage: python -u scripts/bench8.py <dataset> <K> <rep> <families comma-separated> [folds]
env:   DKAN_TAG (required, e.g. _v8ladder / _v8ibs), DKAN_SELECT, DKAN_THREADS
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
if "DKAN_TAG" not in os.environ:
    raise SystemExit("bench8: set DKAN_TAG explicitly (never write into v5 files)")
import bench3

LIN_SPACE = dict(lam=[1e-4, 1e-3, 1e-2], lr=[1e-3, 3e-3], wd=[1e-5, 1e-4],
                 lam_alpha=[0.1, 1.0, 10.0], T=[10, 20, 40])
bench3.FAMILIES["NestKAN-lin"] = ("disc", "NestKAN", "dual", LIN_SPACE,
                                  dict(use_additive=False, use_residual=False))
V5_FAMILIES = ["LogisticHazard", "PMF", "MTLR", "DeepHit", "DeepSurv", "NestKAN", "PCHazard",
               "CoxKAN", "NestKAN-PH", "CoxPH", "RSF"]

if __name__ == "__main__":
    ds = sys.argv[1]
    K = int(sys.argv[2]); rep = int(sys.argv[3])
    fams = V5_FAMILIES if sys.argv[4] == "v5" else sys.argv[4].split(",")
    folds = [int(f) for f in sys.argv[5].split(",")] if len(sys.argv) > 5 else range(5)
    assert os.environ["DKAN_TAG"] not in ("_v5", ""), "refusing to write v5 tag"
    # 2026-10-15 scheduling note (PROTOCOL_v8): two watchdogs may launch item-3 jobs; an exclusive
    # per-(tag, dataset) lock guarantees that no two processes ever write the same output file.
    # A second launch waits for the lock, then resumes (completed units are skipped by bench3.run).
    import time, ctypes, atexit
    LOCK = os.path.join(_HERE, "..", "tmp", f"lock{os.environ['DKAN_TAG']}_{ds}.lock")

    def _alive(pid):
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        c = ctypes.c_ulong(); ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(c))
        ctypes.windll.kernel32.CloseHandle(h)
        return bool(ok) and c.value == 259
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
            break
        except FileExistsError:
            try:
                owner = open(LOCK).read().strip()
            except OSError:
                owner = ""
            if owner.isdigit() and _alive(owner):
                print("bench8: lock held by", owner, "- waiting", flush=True); time.sleep(60)
            else:
                try:
                    os.remove(LOCK)
                except OSError:
                    time.sleep(5)
    atexit.register(lambda: os.path.exists(LOCK) and open(LOCK).read().strip() == str(os.getpid()) and os.remove(LOCK))
    print("bench8", ds, "K", K, "rep", rep, "tag", os.environ["DKAN_TAG"],
          "select", os.environ.get("DKAN_SELECT", "rNLL"), "families", fams, flush=True)
    bench3.run(ds, K, fams, folds, rep)
