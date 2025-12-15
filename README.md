# ESG Scores and Stock Returns – Niruban Final Project

This project studies the explanatory and predictive power of ESG (Environmental, Social, Governance) scores for stock returns using S&P 500 firms between 2015-2023.

## Requirements

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate Niruban-project

```
## Project Structure

```
Niruban-project/
├── src/
│   ├── __init__.py              # Makes src a Python package
│   ├── data_loader.py           # Data loading utilities
│   ├── models.py                # ML models (Linear, Lasso, Random Forest)
│   └── evaluation.py            # Results evaluation and saving
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── 01. esg_data_exploration.ipynb    # ESG data exploration and visualization
│   ├── 02. stat_analysis.ipynb           # Statistical analysis
│   └── 03. machine_learning.ipynb    # ML analysis and experimentation
├── data/
│   └── main_dataset.csv         # Main dataset (ESG + financial data)
├── results/                     # Generated results
├── main.py                      # Main script to run the analysis
├── environment.yml              # Conda environment file
├── README.md                    # This file
└── Niruban_project_report.pdf   # Detailed project report and analysis

```

## Usage

Run the complete analysis:

```bash
python main.py
```
This will:
1. Load the dataset
2. Run Linear Regression and Lasso models with expanding windows
3. Run Random Forest model
4. Save results to `results/` folder

## Data

The dataset combines:
- ESG scores from [Refinitiv]
- Financial data from S&P 500 companies (2015-2023)
- Features: ESG Global Score, Size, Momentum, Beta, ROE, PE Ratio, Debt-to-Equity
- Target: Next Year Return

## Models

- **Linear Regression**: Baseline model with standardized features
- **Lasso Regression**: Feature selection with cross-validated alpha
- **Random Forest**: Non-linear model with feature importance analysis

## Results

Results are saved in the `results/` folder:
- `ml_metrics.csv`: Model performance metrics (MSE, R²)
- `feature_importances.csv`: Random Forest feature importance rankings

## Author

Niruban Jeyarajah - [01.12.2025]
