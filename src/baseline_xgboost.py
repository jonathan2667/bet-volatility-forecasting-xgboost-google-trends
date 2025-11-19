"""
Baseline XGBoost Model for Volatility Prediction
Simple initial implementation to demonstrate methodology
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt

from data_loader import BETDataLoader
from feature_engineering import FeatureEngineer


class BaselineXGBoost:
    """Simple XGBoost classifier for volatility spikes"""
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            eval_metric='logloss'
        )
        self.feature_names = None
        
    def prepare_data(self, df, target_col='high_volatility'):
        """Prepare X and y from dataframe"""
        # Exclude target and intermediate columns
        exclude_cols = [target_col, 'returns', 'volatility', 'Close', 'Price', 
                       'Open', 'High', 'Low', 'Volume', 'Adj Close', 'Vol.', 'Change %']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        self.feature_names = feature_cols
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Positive class: {y.sum()} ({100*y.mean():.1f}%)")
        
        return X, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model"""
        print("\n=== Training XGBoost ===")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        print("Training complete!")
        return self.model
    
    def evaluate(self, X, y, dataset_name="Test"):
        """Evaluate model performance"""
        print(f"\n=== {dataset_name} Set Evaluation ===")
        
        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y, y_pred_proba)
        print(f"ROC AUC: {auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Low Vol', 'High Vol']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        return {
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def plot_feature_importance(self, top_n=15):
        """Plot top N most important features"""
        from pathlib import Path
        
        importance = self.model.feature_importances_
        
        # Create dataframe for plotting
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Create results directory if it doesn't exist
        results_dir = Path('../results')
        results_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - XGBoost')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(results_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        print(f"\nFeature importance plot saved to {results_dir}/feature_importance.png")
        plt.close()
        
        return importance_df


def run_case_study():
    """Run complete case study pipeline"""
    
    print("=" * 60)
    print("BET VOLATILITY PREDICTION - CASE STUDY")
    print("XGBoost Baseline Model")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading BET data...")
    loader = BETDataLoader()
    df = loader.load_data()
    df = loader.compute_returns()
    df = loader.compute_volatility()
    df = loader.create_target(threshold_percentile=75)
    
    # Step 2: Engineer features
    print("\n[Step 2] Engineering features...")
    engineer = FeatureEngineer(df)
    df_features = engineer.add_all_features()
    
    # Step 3: Split data
    print("\n[Step 3] Splitting data...")
    train_df, val_df, test_df = loader.split_data()
    
    # Re-apply feature engineering to each split
    train_eng = FeatureEngineer(train_df).add_all_features()
    val_eng = FeatureEngineer(val_df).add_all_features()
    test_eng = FeatureEngineer(test_df).add_all_features()
    
    # Step 4: Prepare data for ML
    print("\n[Step 4] Preparing data for ML...")
    model = BaselineXGBoost(n_estimators=100, max_depth=5, learning_rate=0.1)
    
    X_train, y_train = model.prepare_data(train_eng)
    X_val, y_val = model.prepare_data(val_eng)
    X_test, y_test = model.prepare_data(test_eng)
    
    # Step 5: Train model
    print("\n[Step 5] Training model...")
    model.train(X_train, y_train, X_val, y_val)
    
    # Step 6: Evaluate
    print("\n[Step 6] Evaluating model...")
    train_results = model.evaluate(X_train, y_train, "Training")
    val_results = model.evaluate(X_val, y_val, "Validation")
    test_results = model.evaluate(X_test, y_test, "Test")
    
    # Step 7: Feature importance
    print("\n[Step 7] Analyzing feature importance...")
    importance_df = model.plot_feature_importance(top_n=15)
    print("\nTop 10 Features:")
    print(importance_df.head(10))
    
    # Summary
    print("\n" + "=" * 60)
    print("CASE STUDY RESULTS SUMMARY")
    print("=" * 60)
    print(f"Training AUC:   {train_results['auc']:.4f}")
    print(f"Validation AUC: {val_results['auc']:.4f}")
    print(f"Test AUC:       {test_results['auc']:.4f}")
    print("=" * 60)
    
    return {
        'model': model,
        'train_results': train_results,
        'val_results': val_results,
        'test_results': test_results,
        'importance': importance_df
    }


if __name__ == "__main__":
    results = run_case_study()
    print("\n✓ Case study complete!")

