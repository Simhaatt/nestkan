"""Run v5 jobs sequentially (resumable).  usage: python -u scripts/run_tuned3.py ds:rep,ds:rep,... [K=8]"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import bench3

if __name__ == "__main__":
    jobs = [j.split(":") for j in sys.argv[1].split(",")]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for ds, rep in jobs:
        t0 = time.time()
        print(f"### {ds} rep{rep} start", flush=True)
        bench3.run(ds, K, None, range(5), int(rep))
        print(f"### {ds} rep{rep} done in {time.time() - t0:.0f}s", flush=True)
