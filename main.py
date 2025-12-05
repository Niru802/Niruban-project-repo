from src.data_loader import load_data
from src.models import run_linear_models, run_random_forest
from src.evaluation import save_ml_metrics, save_feature_importances

def main():
    print("\n=== ESG and Stock Returns – Niruban Final Project ===\n")

    # 1. Load data
    data = load_data()
    print(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns.")

    # 2. Run Linear + Lasso models
    print("\nRunning Linear Regression and Lasso...")
    lin_lasso_results = run_linear_models(data)

    # 3. Run Random Forest model
    print("\nRunning Random Forest...")
    rf_results = run_random_forest(data)

    # 4. Save evaluation metrics
    save_ml_metrics(lin_lasso_results, rf_results)
    save_feature_importances(rf_results["feature_importances"])
    print("\nDone. Results are stored in the 'results/' folder.\n")

if __name__ == "__main__":
    main()