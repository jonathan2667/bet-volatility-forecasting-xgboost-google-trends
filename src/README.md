# Case Study: BET Volatility Prediction with XGBoost

This folder contains the initial case study implementation demonstrating the methodology for predicting BET index volatility spikes using XGBoost.

## Project Structure

```
src/
├── data_loader.py         # Load and preprocess BET data
├── feature_engineering.py # Create technical indicators
├── baseline_xgboost.py    # Train and evaluate XGBoost model
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Case Study

```bash
python baseline_xgboost.py
```

This will:
1. Load BET data from `../data/BET-2010-2025.csv`
2. Compute returns and volatility
3. Create binary target (high/low volatility)
4. Engineer technical features
5. Split data (train/val/test)
6. Train XGBoost classifier
7. Evaluate performance (ROC AUC, precision, recall)
8. Plot feature importance

### Individual Components

**Load data:**
```python
from data_loader import BETDataLoader

loader = BETDataLoader()
df = loader.load_data()
df = loader.compute_returns()
df = loader.compute_volatility()
df = loader.create_target(threshold_percentile=75)
```

**Engineer features:**
```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer(df)
df_features = engineer.add_all_features()
```

**Train model:**
```python
from baseline_xgboost import BaselineXGBoost

model = BaselineXGBoost(n_estimators=100, max_depth=5)
X_train, y_train = model.prepare_data(train_df)
model.train(X_train, y_train)
results = model.evaluate(X_test, y_test)
```

## Methodology

### Problem Formulation
- **Task:** Binary classification (high vs. low volatility)
- **Target:** Volatility spike = volatility > 75th percentile
- **Features:** Lagged volatility, moving averages, RSI, momentum

### Model
- **Algorithm:** XGBoost (Gradient Boosting)
- **Hyperparameters:** 100 trees, max depth 5, learning rate 0.1
- **Validation:** Time series split (2010-2019 train, 2020-2022 val, 2023-2024 test)

### Evaluation Metrics
- ROC AUC (primary metric)
- Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importance

## Expected Results

Initial baseline should achieve:
- **Training AUC:** ~0.85-0.90 (may overfit)
- **Validation AUC:** ~0.70-0.75
- **Test AUC:** ~0.65-0.75

Top features expected: lagged volatility, RSI, momentum

## Next Steps

1. Add Google Trends preprocessing
2. Implement LSTM baseline for comparison
3. Hyperparameter tuning with grid search
4. SHAP analysis for explainability
5. Economic significance testing (VaR, Sharpe ratio)

