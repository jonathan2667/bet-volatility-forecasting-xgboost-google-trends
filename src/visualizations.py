"""
Advanced Visualizations for Case Study
Creates comprehensive plots and tables to demonstrate model effectiveness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


class CaseStudyVisualizer:
    """Create comprehensive visualizations for case study"""
    
    def __init__(self, results_dir='../results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def plot_roc_curves(self, train_results, val_results, test_results):
        """Plot ROC curves for all datasets"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        datasets = [
            ('Training', train_results, 'blue'),
            ('Validation', val_results, 'green'),
            ('Test', test_results, 'red')
        ]
        
        for idx, (name, results, color) in enumerate(datasets):
            y_true = results['y_true']
            y_pred_proba = results['probabilities']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            axes[idx].plot(fpr, tpr, color=color, lw=2, 
                          label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[idx].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
            axes[idx].set_xlim([0.0, 1.0])
            axes[idx].set_ylim([0.0, 1.05])
            axes[idx].set_xlabel('False Positive Rate', fontsize=11)
            axes[idx].set_ylabel('True Positive Rate', fontsize=11)
            axes[idx].set_title(f'{name} Set ROC Curve', fontsize=12, fontweight='bold')
            axes[idx].legend(loc="lower right")
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
        print(f"✓ ROC curves saved to {self.results_dir}/roc_curves.png")
        plt.close()
        
    def plot_confusion_matrices(self, train_results, val_results, test_results):
        """Plot confusion matrices as heatmaps"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        datasets = [
            ('Training', train_results),
            ('Validation', val_results),
            ('Test', test_results)
        ]
        
        for idx, (name, results) in enumerate(datasets):
            y_true = results['y_true']
            y_pred = results['predictions']
            
            cm = confusion_matrix(y_true, y_pred)
            
            # Normalize by row (actual values)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Low Vol', 'High Vol'],
                       yticklabels=['Low Vol', 'High Vol'],
                       ax=axes[idx], cbar=True)
            
            axes[idx].set_xlabel('Predicted', fontsize=11)
            axes[idx].set_ylabel('Actual', fontsize=11)
            axes[idx].set_title(f'{name} Set Confusion Matrix', 
                               fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrices saved to {self.results_dir}/confusion_matrices.png")
        plt.close()
        
    def plot_prediction_distribution(self, train_results, val_results, test_results):
        """Plot distribution of predicted probabilities"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        datasets = [
            ('Training', train_results),
            ('Validation', val_results),
            ('Test', test_results)
        ]
        
        for idx, (name, results) in enumerate(datasets):
            y_true = results['y_true']
            y_pred_proba = results['probabilities']
            
            # Separate by actual class
            low_vol_probs = y_pred_proba[y_true == 0]
            high_vol_probs = y_pred_proba[y_true == 1]
            
            axes[idx].hist(low_vol_probs, bins=30, alpha=0.6, 
                          label='Actual Low Vol', color='blue', density=True)
            axes[idx].hist(high_vol_probs, bins=30, alpha=0.6, 
                          label='Actual High Vol', color='red', density=True)
            axes[idx].axvline(x=0.5, color='black', linestyle='--', 
                             linewidth=1, label='Decision threshold')
            
            axes[idx].set_xlabel('Predicted Probability', fontsize=11)
            axes[idx].set_ylabel('Density', fontsize=11)
            axes[idx].set_title(f'{name} Set: Prediction Distribution', 
                               fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'prediction_distributions.png', dpi=150, bbox_inches='tight')
        print(f"✓ Prediction distributions saved to {self.results_dir}/prediction_distributions.png")
        plt.close()
        
    def plot_performance_comparison(self, train_results, val_results, test_results):
        """Plot bar chart comparing metrics across datasets"""
        metrics_data = {
            'Dataset': ['Training', 'Validation', 'Test'] * 4,
            'Metric': ['AUC']*3 + ['Precision']*3 + ['Recall']*3 + ['F1-Score']*3,
            'Value': [
                train_results['auc'], val_results['auc'], test_results['auc'],
                train_results['precision'], val_results['precision'], test_results['precision'],
                train_results['recall'], val_results['recall'], test_results['recall'],
                train_results['f1'], val_results['f1'], test_results['f1']
            ]
        }
        
        df = pd.DataFrame(metrics_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create grouped bar chart
        metrics = df['Metric'].unique()
        x = np.arange(len(metrics))
        width = 0.25
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        datasets = df['Dataset'].unique()
        
        for i, dataset in enumerate(datasets):
            values = df[df['Dataset'] == dataset]['Value'].values
            ax.bar(x + i*width, values, width, label=dataset, color=colors[i])
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Performance Metrics Comparison Across Datasets', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✓ Metrics comparison saved to {self.results_dir}/metrics_comparison.png")
        plt.close()
        
    def plot_feature_correlation(self, df, top_n=15):
        """Plot correlation heatmap of top features"""
        # Exclude non-feature columns
        exclude_cols = ['high_volatility', 'returns', 'volatility', 
                       'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Select top N features by variance (most informative)
        variances = df[feature_cols].var().sort_values(ascending=False)
        top_features = variances.head(top_n).index.tolist()
        
        # Compute correlation matrix
        corr_matrix = df[top_features].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Feature Correlation Matrix (Top 15 Features)', 
                    fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_correlation.png', dpi=150, bbox_inches='tight')
        print(f"✓ Feature correlation saved to {self.results_dir}/feature_correlation.png")
        plt.close()
        
    def create_results_table(self, train_results, val_results, test_results):
        """Create detailed results table and save as CSV and image"""
        
        # Create comprehensive results dataframe
        results_data = {
            'Metric': ['ROC AUC', 'Precision (High Vol)', 'Recall (High Vol)', 
                      'F1-Score (High Vol)', 'Accuracy', 'True Negatives', 
                      'False Positives', 'False Negatives', 'True Positives'],
            'Training': [
                f"{train_results['auc']:.4f}",
                f"{train_results['precision']:.4f}",
                f"{train_results['recall']:.4f}",
                f"{train_results['f1']:.4f}",
                f"{train_results['accuracy']:.4f}",
                train_results['tn'],
                train_results['fp'],
                train_results['fn'],
                train_results['tp']
            ],
            'Validation': [
                f"{val_results['auc']:.4f}",
                f"{val_results['precision']:.4f}",
                f"{val_results['recall']:.4f}",
                f"{val_results['f1']:.4f}",
                f"{val_results['accuracy']:.4f}",
                val_results['tn'],
                val_results['fp'],
                val_results['fn'],
                val_results['tp']
            ],
            'Test': [
                f"{test_results['auc']:.4f}",
                f"{test_results['precision']:.4f}",
                f"{test_results['recall']:.4f}",
                f"{test_results['f1']:.4f}",
                f"{test_results['accuracy']:.4f}",
                test_results['tn'],
                test_results['fp'],
                test_results['fn'],
                test_results['tp']
            ]
        }
        
        df = pd.DataFrame(results_data)
        
        # Save as CSV
        df.to_csv(self.results_dir / 'results_table.csv', index=False)
        print(f"✓ Results table saved to {self.results_dir}/results_table.csv")
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colWidths=[0.35, 0.22, 0.22, 0.22])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Comprehensive Performance Results', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'results_table.png', dpi=150, bbox_inches='tight')
        print(f"✓ Results table image saved to {self.results_dir}/results_table.png")
        plt.close()
        
        return df
    
    def generate_all_visualizations(self, train_results, val_results, test_results, df=None):
        """Generate all visualizations at once"""
        print("\n" + "="*60)
        print("GENERATING CASE STUDY VISUALIZATIONS")
        print("="*60)
        
        self.plot_roc_curves(train_results, val_results, test_results)
        self.plot_confusion_matrices(train_results, val_results, test_results)
        self.plot_prediction_distribution(train_results, val_results, test_results)
        self.plot_performance_comparison(train_results, val_results, test_results)
        results_df = self.create_results_table(train_results, val_results, test_results)
        
        if df is not None:
            self.plot_feature_correlation(df)
        
        print("="*60)
        print("✓ All visualizations generated successfully!")
        print(f"✓ Location: {self.results_dir.absolute()}")
        print("="*60 + "\n")
        
        return results_df


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("Use CaseStudyVisualizer class to generate plots.")

