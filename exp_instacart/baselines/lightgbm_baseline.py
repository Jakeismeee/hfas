# exp_instacart/baselines/lightgbm_baseline.py

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump

# --- metrics import (works whether run as module or script)
try:
    from ..metrics import rmse, smape, r2
except ImportError:
    from exp_instacart.metrics import rmse, smape, r2

# --- model (LightGBM). If you prefer XGBoost, swap imports + model below.
from lightgbm import LGBMRegressor

# ------------------ Tunables ------------------
LAGS = [1, 7, 14]       # reduce to [1,7,14] if your panel is short
ROLLS = [7, 14]         # reduce to [7,14] if many rows drop
CONTEXT_DAYS = 180         # history rows added before test to compute lags
# ---------------------------------------------

def _fe(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["dow"]   = d["date"].dt.weekday
    d["dom"]   = d["date"].dt.day
    d["month"] = d["date"].dt.month
    for L in LAGS:
        d[f"lag{L}"] = d.groupby("series_id")["y"].shift(L)
    for w in ROLLS:
        d[f"roll{w}"] = d.groupby("series_id")["y"].shift(1).rolling(w).mean()
    return d

def run_lightgbm(processed_dir, out_dir, horizon=28, seed=42):
    processed_dir = Path(processed_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_dir / "series_bottom.parquet").sort_values(["series_id","date"])

    last_date = df["date"].max()
    cutoff = last_date - pd.Timedelta(days=horizon)

    train = df[df["date"] <= cutoff].copy()
    test  = df[df["date"] >  cutoff].copy()

    # Feature engineering + clean indices
    train_fe = _fe(train).dropna().reset_index(drop=True)

    # Provide extra context rows so test lags/rolls are computable
    ctx = train.tail(CONTEXT_DAYS)
    test_fe  = _fe(pd.concat([ctx, test], ignore_index=True)).dropna()
    test_fe  = test_fe[test_fe["date"] > cutoff].copy().reset_index(drop=True)

    # Guard: no test rows after lagging/rolling
    if test_fe.empty:
        print("No test rows after lag/rolling features. "
              "Try reducing LAGS/ROLLS, lowering horizon, or increasing CONTEXT_DAYS.")
        return

    preds = []

    for h in range(1, horizon+1):
        # Build y at t+h aligned with train_fe rows
        y_future = (
            train_fe
            .assign(date_plus=train_fe["date"] + pd.Timedelta(days=h))
            .merge(
                train[["series_id","date","y"]]
                .rename(columns={"date":"date_plus","y":"y_future"}),
                on=["series_id","date_plus"], how="left"
            )["y_future"]
            .reset_index(drop=True)
        )

        X = train_fe.drop(columns=["y"]).reset_index(drop=True)

        # Positional boolean mask
        mask = y_future.notna().to_numpy()
        if mask.sum() == 0:
            # nothing to fit for this horizon; skip
            continue

        X_num = X.select_dtypes(include=[np.number]).to_numpy(dtype="float32")
        y_vec = y_future.to_numpy(dtype="float32")

        # Define model and fit
        model = LGBMRegressor(
            random_state=seed,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        )
        model.fit(X_num[mask], y_vec[mask])
        dump(model, out / f"lgbm_h{h}.joblib")

        # Build test matrix; skip if empty for this horizon
        Xtest_num = (
            test_fe.drop(columns=["y"])
                   .select_dtypes(include=[np.number])
                   .to_numpy(dtype="float32")
        )
        if Xtest_num.shape[0] == 0:
            continue

        yhat = model.predict(Xtest_num)

        step = pd.DataFrame({
            "series_id": test_fe["series_id"].values,
            "date":      test_fe["date"].values + np.timedelta64(h, "D"),
            "yhat":      yhat
        })
        preds.append(step)

    # Guard: no horizons produced predictions
    if not preds:
        print("No predictions generated (all test-feature rows empty after lagging). "
              "Reduce LAGS/ROLLS or horizon, or increase CONTEXT_DAYS.")
        return

    pred = (
        pd.concat(preds, ignore_index=True)
          .merge(test[["series_id","date","y"]], on=["series_id","date"], how="inner")
          .dropna(subset=["y"])
    )

    pred.to_parquet(out / "predictions_baseline.parquet", index=False)

    overall = {"RMSE": rmse(pred["y"], pred["yhat"]),
               "sMAPE": smape(pred["y"], pred["yhat"]),
               "R2": r2(pred["y"], pred["yhat"])}
    pd.DataFrame([overall]).to_csv(out / "metrics_baseline_overall.csv", index=False)

    per = (
        pred.groupby("series_id")
            .apply(lambda g: pd.Series({
                "RMSE": rmse(g["y"], g["yhat"]),
                "sMAPE": smape(g["y"], g["yhat"]),
                "R2": r2(g["y"], g["yhat"])
            }))
            .reset_index()
    )
    per.to_csv(out / "metrics_baseline_per_series.csv", index=False)

    print("Baseline done. Outputs in", out)
