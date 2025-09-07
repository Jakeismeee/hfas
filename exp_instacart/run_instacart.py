
from pathlib import Path
import pandas as pd, numpy as np
from .instacart_prep import build_instacart_panel
from .baselines.lightgbm_baseline import run_lightgbm
from .metrics import rmse, smape, r2
from scipy.stats import t as student_t

def dm_test(e1, e2, h=1, power=2):
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    d = (np.abs(e1)-np.abs(e2)) if power==1 else (e1**2 - e2**2)
    T = d.size; dbar = d.mean()
    if h>1:
        s = d.var(ddof=1)
        for L in range(1, h):
            w = 1 - L/h
            cov = np.cov(d[L:], d[:-L], ddof=1)[0,1]
            s += 2*w*cov
        var = s
    else:
        var = d.var(ddof=1)
    dm = dbar / np.sqrt(var/T + 1e-12)
    k = ((T + 1 - 2*h)/T)**0.5 if T > 2*h else 1.0
    dm *= k
    p = 2*(1 - student_t.cdf(abs(dm), df=T-1))
    return float(dm), float(p)

def main(data_dir, out_base="exp_instacart", horizon=28, thesis_preds_path=None):
    base = Path(out_base)
    processed = base / "processed"
    outputs   = base / "outputs"
    base.mkdir(parents=True, exist_ok=True)

    # 1) Preprocess
    build_instacart_panel(data_dir, out_base)

    # 2) Train baseline on SAME cutoff/horizon
    run_lightgbm(processed, outputs, horizon=horizon)

    # 3) If thesis predictions are provided, compute DM comparison
    if thesis_preds_path:
        hfas = (pd.read_parquet(outputs / "predictions_baseline.parquet")
                  .rename(columns={"yhat":"yhat_baseline"}))
        thesis = pd.read_parquet(thesis_preds_path)
        if "yhat" in thesis.columns and "series_id" in thesis.columns and "date" in thesis.columns:
            thesis = thesis.rename(columns={"yhat":"yhat_thesis"})
        joined = hfas.merge(thesis, on=["series_id","date"], how="inner")
        joined["e_b"] = joined["y"] - joined["yhat_baseline"]
        joined["e_t"] = joined["y"] - joined["yhat_thesis"]

        # Per-series DM
        rows = []
        for sid, g in joined.groupby("series_id"):
            if len(g) >= max(8, 2*horizon):
                dm, p = dm_test(g["e_b"], g["e_t"], h=horizon, power=2)
                rows.append({"series_id": sid, "DM": dm, "p_value": p})
        pd.DataFrame(rows).to_csv(outputs / "dm_per_series.csv", index=False)

        # Pooled DM
        dm_pool, p_pool = dm_test(joined["e_b"], joined["e_t"], h=horizon, power=2)
        with open(outputs / "dm_pooled.txt", "w") as f:
            f.write(f"DM={dm_pool:.3f}, p={p_pool:.4f}\n")
        print("DM comparison done. See outputs/ for files.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with Instacart CSVs")
    ap.add_argument("--out_base", default="exp_instacart")
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--thesis_preds_path", default=None, help="Parquet with thesis predictions (series_id,date,yhat)")
    args = ap.parse_args()
    main(args.data_dir, args.out_base, args.horizon, args.thesis_preds_path)
