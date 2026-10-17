"""PROTOCOL_v8 item 4 fixed analysis -> results/v8_dose_summary.csv.
Per delta: median / IQR of the time-varying share (share_interaction_of_cov); surface-kept rate (Wilson 95% CI);
Grambsch-Therneau x4 rejection rate at alpha = 0.05 (Wilson CI).  Row delta = 'all': Spearman rho of the share
with delta over all replicates.  False-positive rate = kept rate at delta = 0; the delta = 0 share distribution
(null) is given as quantiles.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sim8_summary import wilson
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main():
    d = pd.read_csv(os.path.join(R, "sim8_dose.csv")).drop_duplicates(["delta", "rep"], keep="first")
    d["kept"] = d.use_residual.astype(str).str.lower().eq("true")
    rows = []
    for dl, g in d.groupby("delta"):
        s = g.share_interaction_of_cov
        k, m = int(g.kept.sum()), len(g)
        rj = int((g.gt_p_x4 < 0.05).sum())
        klo, khi = wilson(k, m); glo, ghi = wilson(rj, m)
        rows.append(dict(delta=dl, reps=m, share_median=s.median(), share_q25=s.quantile(0.25), share_q75=s.quantile(0.75),
                         share_q90=s.quantile(0.9), share_q95=s.quantile(0.95), share_max=s.max(),
                         kept=k, kept_rate=k / m, kept_lo=klo, kept_hi=khi,
                         gt_x4_reject=rj, gt_x4_reject_rate=rj / m, gt_lo=glo, gt_hi=ghi,
                         gt_minp_reject_rate=float((g.gt_min_p < 0.05).mean()),
                         x4_slope_sd_median=g.x4_slope_sd.median(), x4_true_slope_sd_median=g.x4_true_slope_sd.median(),
                         share_median_kept_only=g.loc[g.kept, "share_interaction_of_cov"].median() if k else np.nan))
    rho, p = spearmanr(d.share_interaction_of_cov, d.delta)
    rows.append(dict(delta="all", reps=len(d), spearman_rho_share_delta=rho, spearman_p=p))
    out = pd.DataFrame(rows)
    z = out[out.delta.astype(str) == "0.0"]
    out["false_positive_rate"] = np.nan
    if len(z):
        out.loc[out.delta.astype(str) == "0.0", "false_positive_rate"] = z.kept_rate.values
    out.to_csv(os.path.join(R, "v8_dose_summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
