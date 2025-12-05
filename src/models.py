import numpy as np
import pandas as pd
from typing import Dict, List

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
Features: List[str] = ["ESG Global Score","Size", "Momentum_1y_w", "Beta_Value", "ROE_Value_w",
    "PE_Ratio_w", "Debt-to-Equity_w", "ESG_x_Sector", "High_Impact_ESG",]

Target = "Next_Year_Return_w"

def _get_windows(data: pd.DataFrame, test_years=(2022, 2023)):
    """Yield (train_df, test_df, year) for an expanding-window scheme."""
    min_year = data["Year"].min()
    for test_year in test_years:
        train_mask = (data["Year"] >= min_year) & (data["Year"] < test_year)
        test_mask = data["Year"] == test_year
        train = data.loc[train_mask].copy()
        test = data.loc[test_mask].copy()

        # Drop rows with missing target or features
        train = train.dropna(subset=Features + [Target])
        test = test.dropna(subset=Features + [Target])

        if train.empty or test.empty:
            continue

        yield train, test, test_year

def run_linear_models(data: pd.DataFrame) -> Dict:
    """Run Linear Regression and Lasso with expanding windows."""
    results = {
        "linear": [],
        "lasso": [],
    }

    for train, test, year in _get_windows(data):
        X_train = train[Features].values
        y_train = train[Target].values
        X_test = test[Features].values
        y_test = test[Target].values

        # Linear Regression (with scaling)
        lin_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
        lin_pipe.fit(X_train, y_train)
        preds_lin = lin_pipe.predict(X_test)
        results["linear"].append(
            {
                "year": year,
                "mse": float(mean_squared_error(y_test, preds_lin)),
                "r2": float(r2_score(y_test, preds_lin)),
            }
        )
        # Lasso (with CV for alpha)
        lasso_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LassoCV(cv=5, random_state=RANDOM_STATE)),
            ]
        )
        lasso_pipe.fit(X_train, y_train)
        preds_lasso = lasso_pipe.predict(X_test)
        best_alpha = float(lasso_pipe.named_steps["model"].alpha_)
        results["lasso"].append(
            {
                "year": year,
                "alpha": best_alpha,
                "mse": float(mean_squared_error(y_test, preds_lasso)),
                "r2": float(r2_score(y_test, preds_lasso)),
            }
        )

    return results

def run_random_forest(data: pd.DataFrame) -> Dict:
    """Run Random Forest with expanding windows."""
    rf_results = []
    all_importances = []

    for train, test, year in _get_windows(data):
        X_train = train[Features].values
        y_train = train[Target].values
        X_test = test[Features].values
        y_test = test[Target].values

        rf = RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)

        rf_results.append(
            {
                "year": year,
                "mse": float(mean_squared_error(y_test, preds_rf)),
                "r2": float(r2_score(y_test, preds_rf)),
            }
        )

        all_importances.append(rf.feature_importances_)

    if all_importances:
        mean_importances = np.mean(np.vstack(all_importances), axis=0)
        feature_importances = pd.DataFrame(
            {"feature": Features, "importance": mean_importances}
        ).sort_values("importance", ascending=False)
    else:
        feature_importances = pd.DataFrame(columns=["feature", "importance"])

    return {"rf_results": rf_results, "feature_importances": feature_importances}

        
