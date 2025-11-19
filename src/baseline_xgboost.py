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
from google_trends_loader import load_all_trends, expand_monthly_to_daily, create_trends_features, compute_trends_correlation


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
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        print(f"\n=== {dataset_name} Set Evaluation ===")
        
        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y, y_pred_proba)
        precision = precision_score(y, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y, y_pred, pos_label=1, zero_division=0)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"ROC AUC: {auc_score:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Low Vol', 'High Vol']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        # Extract confusion matrix values
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'auc': auc_score,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'y_true': y,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
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
    
    # Step 2.5: Load and integrate Google Trends data
    print("\n[Step 2.5] Loading Google Trends behavioral data...")
    trends_monthly = load_all_trends('../data')
    
    if trends_monthly is not None:
        print("\n📊 Computing Google Trends Correlations...")
        trends_corr = compute_trends_correlation(trends_monthly)
        
        # Expand monthly to daily and align with BET data
        print("\n🔄 Expanding monthly trends to daily frequency...")
        trends_daily = expand_monthly_to_daily(trends_monthly, df_features.index)
        
        # Create behavioral features from trends
        print("\n🔧 Creating Google Trends features...")
        trends_features = create_trends_features(trends_daily)
        
        # Merge with BET data
        print("\n🔗 Merging Google Trends with BET data...")
        df_features = df_features.join(trends_features, how='left')
        df_features.fillna(method='ffill', inplace=True)
        df_features.fillna(0, inplace=True)
        print(f"✓ Total features after merge: {len(df_features.columns)}")
        print(f"✓ Feature columns: Technical (12) + Google Trends ({len(trends_features.columns)})")
    else:
        print("⚠️  No Google Trends data found. Continuing with technical indicators only.")
    
    # Update loader's dataframe with merged features
    loader.df = df_features
    
    # Step 3: Split data
    print("\n[Step 3] Splitting data...")
    train_df, val_df, test_df = loader.split_data()
    
    print(f"✓ Train: {len(train_df)} samples, {len(train_df.columns)} features")
    print(f"✓ Val:   {len(val_df)} samples, {len(val_df.columns)} features")
    print(f"✓ Test:  {len(test_df)} samples, {len(test_df.columns)} features")
    
    # Step 4: Prepare data for ML
    print("\n[Step 4] Preparing data for ML...")
    model = BaselineXGBoost(n_estimators=100, max_depth=5, learning_rate=0.1)
    
    X_train, y_train = model.prepare_data(train_df)
    X_val, y_val = model.prepare_data(val_df)
    X_test, y_test = model.prepare_data(test_df)
    
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
    
    # Step 8: Generate comprehensive visualizations
    print("\n[Step 8] Creating comprehensive visualizations...")
    from visualizations import CaseStudyVisualizer
    
    visualizer = CaseStudyVisualizer()
    results_table = visualizer.generate_all_visualizations(
        train_results, val_results, test_results, df=df_features
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("CASE STUDY RESULTS SUMMARY")
    print("=" * 60)
    print(f"Training AUC:   {train_results['auc']:.4f}")
    print(f"Validation AUC: {val_results['auc']:.4f}")
    print(f"Test AUC:       {test_results['auc']:.4f}")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("GENERATED VISUALIZATIONS")
    print("=" * 60)
    print("✓ roc_curves.png - ROC curves for all datasets")
    print("✓ confusion_matrices.png - Confusion matrix heatmaps")
    print("✓ prediction_distributions.png - Probability distributions")
    print("✓ metrics_comparison.png - Performance metrics bar chart")
    print("✓ feature_correlation.png - Feature correlation heatmap")
    print("✓ feature_importance.png - Top features importance")
    print("✓ results_table.csv - Detailed metrics table")
    print("✓ results_table.png - Results table visualization")
    print("=" * 60)
    
    return {
        'model': model,
        'train_results': train_results,
        'val_results': val_results,
        'test_results': test_results,
        'importance': importance_df,
        'results_table': results_table
    }


if __name__ == "__main__":
    results = run_case_study()
    print("\n✓ Case study complete!")

