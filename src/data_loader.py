"""
Data Loader for BET Index
Loads and prepares BET historical data for volatility prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path

class BETDataLoader:
    """Load and preprocess BET index data"""
    
    def __init__(self, data_path='../data/BET-2010-2025.csv'):
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self):
        """Load BET CSV data"""
        print(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # Convert date column to datetime
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
        
        print(f"Loaded {len(self.df)} observations")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        return self.df
    
    def compute_returns(self):
        """Compute log returns"""
        if 'Close' in self.df.columns:
            self.df['returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        elif 'Price' in self.df.columns:
            self.df['returns'] = np.log(self.df['Price'] / self.df['Price'].shift(1))
        else:
            raise ValueError("No price column found (expected 'Close' or 'Price')")
        
        print(f"Computed returns. Mean: {self.df['returns'].mean():.6f}, Std: {self.df['returns'].std():.6f}")
        return self.df
    
    def compute_volatility(self):
        """Compute realized volatility (squared returns)"""
        if 'returns' not in self.df.columns:
            self.compute_returns()
        
        self.df['volatility'] = self.df['returns'] ** 2
        print(f"Computed volatility. Mean: {self.df['volatility'].mean():.6f}")
        return self.df
    
    def create_target(self, threshold_percentile=75):
        """
        Create binary target: 1 if high volatility, 0 otherwise
        
        Args:
            threshold_percentile: percentile to define high volatility (default 75)
        """
        if 'volatility' not in self.df.columns:
            self.compute_volatility()
        
        threshold = np.percentile(self.df['volatility'].dropna(), threshold_percentile)
        self.df['high_volatility'] = (self.df['volatility'] > threshold).astype(int)
        
        n_spikes = self.df['high_volatility'].sum()
        pct_spikes = 100 * n_spikes / len(self.df)
        print(f"Created target variable: {n_spikes} high volatility days ({pct_spikes:.1f}%)")
        print(f"Threshold: {threshold:.8f}")
        
        return self.df
    
    def get_basic_stats(self):
        """Print basic statistics"""
        print("\n=== BET Index Statistics ===")
        print(self.df.describe())
        
        if 'returns' in self.df.columns:
            print(f"\nReturns - Skewness: {self.df['returns'].skew():.4f}")
            print(f"Returns - Kurtosis: {self.df['returns'].kurtosis():.4f}")
        
        return self.df.describe()
    
    def split_data(self, train_end='2019-12-31', val_end='2022-12-31'):
        """
        Split data into train, validation, and test sets
        
        Args:
            train_end: last date of training set
            val_end: last date of validation set
        """
        train = self.df[self.df.index <= train_end]
        val = self.df[(self.df.index > train_end) & (self.df.index <= val_end)]
        test = self.df[self.df.index > val_end]
        
        print(f"\n=== Data Split ===")
        print(f"Training: {len(train)} days ({train.index.min()} to {train.index.max()})")
        print(f"Validation: {len(val)} days ({val.index.min()} to {val.index.max()})")
        print(f"Test: {len(test)} days ({test.index.min()} to {test.index.max()})")
        
        return train, val, test


if __name__ == "__main__":
    # Example usage
    loader = BETDataLoader()
    df = loader.load_data()
    df = loader.compute_returns()
    df = loader.compute_volatility()
    df = loader.create_target(threshold_percentile=75)
    loader.get_basic_stats()
    
    train, val, test = loader.split_data()
    print("\nData loading complete!")

