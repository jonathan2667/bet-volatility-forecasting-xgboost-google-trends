# BET Volatility Forecasting with XGBoost and Google Trends

## Research Project Overview

This repository contains the implementation and research materials for predicting Romanian BET (Bucharest Exchange Trading) index volatility using XGBoost machine learning and Google Trends behavioral data.

## Project Structure

```
bet-volatility-forecasting-xgboost-google-trends/
├── data/               # Raw and processed datasets
│   └── BET-2010-2025.csv
├── docs/               # Research documentation and reports
│   ├── Research-lab3.tex
│   └── Research.pdf
├── src/                # Source code for experiments
├── notebooks/          # Jupyter notebooks for analysis
├── results/            # Experimental results and figures
└── README.md
```

## Research Objectives

1. **Main Hypothesis**: XGBoost ensemble learning with preprocessed Google Trends features achieves superior volatility prediction accuracy compared to traditional time series models (GARCH) and deep learning approaches (LSTM) for emerging market data.

2. **Key Contributions**:
   - First study combining Google Trends + Machine Learning for Romanian BET index
   - Advanced preprocessing pipeline for sparse behavioral data
   - Comparative analysis: XGBoost vs. LSTM vs. GARCH
   - SHAP-based feature importance interpretation

## Methodology

- **Data**: BET daily index (2010-2025), Romanian Google Trends queries, EUR/RON exchange rates
- **Models**: XGBoost, LSTM, GARCH baseline
- **Validation**: Time series cross-validation with out-of-sample testing
- **Metrics**: ROC AUC, Precision, Recall, F1-Score, VaR breach analysis

## Timeline

- **Weeks 1-2**: Data collection and exploration
- **Week 3**: Google Trends preprocessing (clustering, denoising, detrending)
- **Weeks 4-5**: Model development and training
- **Week 6**: Out-of-sample testing (2023-2024)
- **Weeks 7-8**: Robustness checks and final report

## Requirements

TBD (will be added as code is implemented)

## References

See `docs/Research.pdf` for complete bibliography and literature review.

## License

Academic research project - Semester 5, 2025

