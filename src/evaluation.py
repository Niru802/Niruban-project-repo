import pandas as pd
import os
from typing import Dict, List

def ensure_results_dir(path: str = "results"):
    os.makedirs(path, exist_ok=True)

def save_ml_metrics(linear_lasso: Dict, rf: Dict, path: str = "results/ml_metrics.csv"):
    ensure_results_dir(os.path.dirname(path) or ".")

    rows: List[Dict] = []

    for entry in linear_lasso.get("linear", []):
        rows.append(
            {
                "model": "LinearRegression",
                "year": entry["year"],
                "mse": entry["mse"],
                "r2": entry["r2"],
            }
        )
    for entry in linear_lasso.get("lasso", []):
        rows.append(
            {
                "model": f"Lasso(alpha={entry['alpha']:.3f})",
                "year": entry["year"],
                "mse": entry["mse"],
                "r2": entry["r2"],
            }
        )
    for entry in rf.get("rf_results", []):
        rows.append(
            {
                "model": "RandomForest",
                "year": entry["year"],
                "mse": entry["mse"],
                "r2": entry["r2"],
            }
        )
    
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print("\nSaved ML metrics to:", path)
    print(df)


def save_feature_importances(
    feature_importances: pd.DataFrame,
    path: str = "results/feature_importances.csv",
):
    ensure_results_dir(os.path.dirname(path) or ".")
    feature_importances.to_csv(path, index=False)
    print("\nSaved feature importances to:", path)
    print(feature_importances)
