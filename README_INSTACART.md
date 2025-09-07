
# Instacart Retail Experiment (No thesis code included)

This adds an **Instacart** experiment to the HFAS repo to produce **comparable results** to your thesis pipeline **without** including any of your thesis code.

## What this does
- Replicates common retail preprocessing used in recent studies:
  - daily SKU counts from Instacart `order_products__prior`
  - zero-filled missing days per SKU
  - hierarchy: `SKU -> Aisle -> Department`
- Trains a **global LightGBM baseline** (HFAS-style interface) on the same cutoff & horizon.
- Computes **sMAPE, RMSE, R²** with the exact same implementation for all models.
- (Optional) Compares against your thesis predictions via **Diebold–Mariano** test — you pass in a Parquet file of your predictions; your code is **not** used or included here.

## Files
- `exp_instacart/instacart_prep.py` — build panel & hierarchy
- `exp_instacart/baselines/lightgbm_baseline.py` — baseline trainer
- `exp_instacart/metrics.py` — sMAPE, RMSE, R²
- `exp_instacart/run_instacart.py` — one-shot runner (prep → baseline → optional DM)

## How to run
1) Place Instacart CSVs in a folder (must include: `orders.csv`, `order_products__prior.csv`, `products.csv`, `aisles.csv`, `departments.csv`).

2) Run the experiment (choose horizon e.g. 28):
```bash
python -m exp_instacart.run_instacart --data_dir /path/to/instacart --out_base exp_instacart --horizon 28
```

Outputs:
- `exp_instacart/outputs/predictions_baseline.parquet`
- `exp_instacart/outputs/metrics_baseline_overall.csv`
- `exp_instacart/outputs/metrics_baseline_per_series.csv`

3) (Optional) Compare against your thesis predictions
Export your thesis model forecasts to **Parquet** with columns: `series_id, date, yhat` (dates must align with the baseline test window), then run:
```bash
python -m exp_instacart.run_instacart --data_dir /path/to/instacart --out_base exp_instacart --horizon 28 --thesis_preds_path /path/to/your/predictions_hybrid.parquet
```
This writes:
- `exp_instacart/outputs/dm_per_series.csv`
- `exp_instacart/outputs/dm_pooled.txt`

## Notes / Protocol
- Frequency: **daily**
- Horizon: configurable (`--horizon H`), default **28**
- Split: fixed-origin (train up to last_date - H; forecast H days). Swap to rolling-origin later if needed.
- Scale: all metrics computed on **original scale** (no log transforms) to align with your thesis.
- Missing days: zero-filled per series to keep leakage out of lag windows.
- Intermittency: if desired, also compute MASE/MAE in a follow-up.

No thesis code is present in this folder. Only a baseline pipeline and the ability to **read** an external file of thesis predictions for comparison.
