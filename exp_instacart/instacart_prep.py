
"""
Instacart preprocessing (replicates common retail-study protocols):
- Build daily SKU-level demand (counts of line items per product per day)
- Zero-fill missing days per SKU to avoid ragged windows
- Create simple hierarchy: SKU -> Aisle -> Department
Outputs:
- processed/series_bottom.parquet  [date, series_id, y, product_id, aisle_id, department_id]
- processed/hierarchy_map.parquet  [series_id, aisle_series, dept_series]
- processed/meta.parquet           [min_date, max_date, n_series]
"""

from pathlib import Path
import pandas as pd

def build_instacart_panel(data_dir, out_dir):
    data_dir = Path(data_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    orders = pd.read_csv(data_dir / "orders.csv")
    priors = pd.read_csv(data_dir / "order_products__prior.csv")
    prods  = pd.read_csv(data_dir / "products.csv")
    aisles = pd.read_csv(data_dir / "aisles.csv")
    depts  = pd.read_csv(data_dir / "departments.csv")

    # Use only 'prior' orders to avoid leakage from train/test labels
    orders = orders[orders["eval_set"].astype(str).str.lower() == "prior"].copy()

    # Build a pseudo-calendar from users' days_since_prior_order (cumulative)
    orders["days_since_prior_order"] = orders["days_since_prior_order"].fillna(0)
    orders = orders.sort_values(["user_id","order_number"])
    orders["cum_days"] = orders.groupby("user_id")["days_since_prior_order"].cumsum()
    origin = pd.Timestamp("2015-01-01")
    orders["date"] = origin + pd.to_timedelta(orders["cum_days"], unit="D")

    # Join order lines to dates
    lines = priors.merge(orders[["order_id","date"]], on="order_id", how="left")

    # Attach taxonomy
    prod_tax = (prods.merge(aisles, on="aisle_id", how="left")
                     .merge(depts, on="department_id", how="left"))
    lines = lines.merge(prod_tax[["product_id","aisle_id","department_id"]], on="product_id", how="left")

    # Aggregate to daily SKU demand (single-store assumption)
    daily = (lines.groupby(["product_id","aisle_id","department_id", pd.Grouper(key="date", freq="D")])
                  .size().rename("y").reset_index())

    # Zero-fill per SKU
    def fill_days(g):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx).fillna(0.0).rename_axis("date").reset_index()
        g["product_id"]    = g["product_id"].iloc[0]
        g["aisle_id"]      = g["aisle_id"].iloc[0]
        g["department_id"] = g["department_id"].iloc[0]
        return g

    daily = (daily.groupby(["product_id","aisle_id","department_id"], group_keys=False)
                  .apply(fill_days).reset_index(drop=True))

    daily["series_id"] = daily["product_id"].astype(str)  # bottom=SKU
    panel = daily[["date","series_id","y","product_id","aisle_id","department_id"]].copy()
    panel = panel.sort_values(["series_id","date"])

    # Save panel
    proc = out / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(proc / "series_bottom.parquet", index=False)

    # Save hierarchy mapping
    hier = (panel[["series_id","aisle_id","department_id"]].drop_duplicates()
            .assign(aisle_series=lambda d: "A" + d["aisle_id"].astype(str),
                    dept_series =lambda d: "D" + d["department_id"].astype(str)))
    hier.to_parquet(proc / "hierarchy_map.parquet", index=False)

    meta = pd.DataFrame({
        "min_date":[panel["date"].min()],
        "max_date":[panel["date"].max()],
        "n_series":[panel["series_id"].nunique()]
    })
    meta.to_parquet(proc / "meta.parquet", index=False)
    print("Prepared processed files under", proc)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with Instacart CSVs")
    ap.add_argument("--out_dir", default="exp_instacart", help="Output base folder")
    args = ap.parse_args()
    build_instacart_panel(args.data_dir, args.out_dir)
