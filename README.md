# Predictive Modeling of Microeconomic Indicators via Ensemble Learning

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Optuna](https://img.shields.io/badge/Optuna-AutoML-orange)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## 📌 Project Overview
This repository contains a publication-ready machine learning pipeline for analyzing and predicting continuous target variables (`Y_target`) based on multidimensional feature sets (temporal, spatial, and continuous metrics). The core architecture leverages a **Stacking Ensemble Regressor** combining hyperparameter-optimized XGBoost, LightGBM, and Random Forest models to achieve superior predictive accuracy.

All visualizations are generated adhering strictly to leading academic journal standards (e.g., *Nature*, *IEEE*), featuring restrained color palettes, high-resolution SVG/PNG outputs, and rigorous statistical diagnostics.

## 🛠 Environmental Requirements
- Python 3.8+
- Scikit-learn
- XGBoost & LightGBM
- Optuna (for Bayesian hyperparameter tuning)
- Matplotlib & Seaborn (for academic plotting)
- Pandas & Numpy

## 🚀 Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/academic-ensemble-pipeline.git](https://github.com/your-username/academic-ensemble-pipeline.git)
   cd academic-ensemble-pipeline
Install the required dependencies:

Bash
pip install -r requirements.txt
Prepare your local font (TimesSimSunRegular.ttf) or adjust the FONT_PATH in the script to point to a valid system font.

📊 Usage Guide
Place your dataset (data.csv) in the root directory. Ensure it matches the expected schema defined in the data ingestion module.

Execute the main pipeline:

Bash
python src/ensemble_pipeline.py
The script will automatically:

Perform cyclic temporal encoding and feature engineering.

Run Optuna to find the global minima for XGBoost hyperparameters.

Train the RidgeCV-backed stacking meta-learner.

Output predictive metrics (R², RMSE, MAE).

Generate 6 academic-grade figures in the root directory.

🔬 Methodology & Architecture
This project treats the prediction task through a rigorous statistical learning framework:

Deep Feature Engineering: Cyclical time transformations (sine/cosine encodings) are applied to capture non-linear seasonality without imposing strict ordinality.

AutoML Optimization: Optuna minimizes the objective function via Tree-structured Parzen Estimator (TPE), tuning depth, learning rate, and subsampling to prevent overfitting.

Stacking Generalization: Base estimators handle distinct data distributions (e.g., LightGBM for speed/efficiency, Random Forest for variance reduction), while the Meta-Learner (RidgeCV) uses cross-validated penalization to assign optimal weights to the base models.

Furthermore, the pipeline outputs interpretable diagnostic checks, including Partial Dependence Plots (PDP) and heteroskedasticity residual evaluations, bridging the gap between "black-box" predictive power and econometric interpretability.


```text
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
optuna>=2.10.0
matplotlib>=3.5.0
seaborn>=0.11.2
Plaintext
# Environments
.env
.venv
env/
venv/
ENV/

# Python Specific
__pycache__/
*.py[cod]
*$py.class

# Data Files (Crucial for Anonymity)
*.csv
*.xlsx
*.xls
*.dta
data/
raw_data/
processed_data/

# Generated Outputs
*.png
*.svg
*.pdf
*.log

# IDE & OS
.vscode/
.idea/
*.DS_Store
