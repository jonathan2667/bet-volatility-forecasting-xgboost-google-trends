"""
Feature Engineering for Volatility Prediction
Creates technical indicators and lagged features
"""

import pandas as pd
import numpy as np

class FeatureEngineer:
    """Create features for ML models"""
    
    def __init__(self, df):
        self.df = df.copy()
        
    def add_lagged_volatility(self, lags=[1, 2, 5, 10, 20]):
        """Add lagged volatility features"""
        for lag in lags:
            self.df[f'vol_lag_{lag}'] = self.df['volatility'].shift(lag)
        
        print(f"Added {len(lags)} lagged volatility features")
        return self.df
    
    def add_moving_averages(self, windows=[5, 10, 20, 50]):
        """Add moving average features"""
        # Use Price column (BET CSV format)
        price_col = 'Price' if 'Price' in self.df.columns else 'Close'
        
        for window in windows:
            self.df[f'ma_{window}'] = self.df[price_col].rolling(window=window).mean()
            
        print(f"Added {len(windows)} moving average features")
        return self.df
    
    def add_momentum(self, window=10):
        """Add momentum indicator"""
        # Use Price column (BET CSV format)
        price_col = 'Price' if 'Price' in self.df.columns else 'Close'
        self.df['momentum'] = self.df[price_col] - self.df[price_col].shift(window)
        
        print(f"Added momentum (window={window})")
        return self.df
    
    def add_rsi(self, window=14):
        """
        Add Relative Strength Index
        RSI = 100 - 100/(1 + RS), where RS = avg_gain / avg_loss
        """
        if 'returns' not in self.df.columns:
            raise ValueError("Returns column required for RSI calculation")
        
        # Calculate gains and losses
        gains = self.df['returns'].clip(lower=0)
        losses = -self.df['returns'].clip(upper=0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=window, min_periods=1).mean()
        avg_losses = losses.rolling(window=window, min_periods=1).mean()
        
        # Calculate RSI
        rs = avg_gains / (avg_losses + 1e-10)  # avoid division by zero
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        print(f"Added RSI (window={window})")
        return self.df
    
    def add_volatility_of_volatility(self, window=20):
        """Add rolling standard deviation of volatility"""
        self.df['vov'] = self.df['volatility'].rolling(window=window).std()
        
        print(f"Added volatility of volatility (window={window})")
        return self.df
    
    def add_all_features(self):
        """Add all features at once"""
        print("\n=== Creating Features ===")
        self.add_lagged_volatility()
        self.add_moving_averages()
        self.add_momentum()
        self.add_rsi()
        self.add_volatility_of_volatility()
        
        # Drop NaN rows created by feature engineering
        initial_len = len(self.df)
        self.df = self.df.dropna()
        dropped = initial_len - len(self.df)
        
        print(f"\nDropped {dropped} rows with NaN values")
        print(f"Final dataset: {len(self.df)} observations")
        print(f"Total features: {len(self.df.columns)}")
        
        return self.df
    
    def get_feature_names(self, exclude=['high_volatility', 'returns', 'volatility']):
        """Get list of feature column names"""
        feature_cols = [col for col in self.df.columns if col not in exclude]
        return feature_cols


if __name__ == "__main__":
    from data_loader import BETDataLoader
    
    # Load data
    loader = BETDataLoader()
    df = loader.load_data()
    df = loader.compute_returns()
    df = loader.compute_volatility()
    df = loader.create_target()
    
    # Create features
    engineer = FeatureEngineer(df)
    df_features = engineer.add_all_features()
    
    print("\n=== Feature Summary ===")
    print(df_features.head())
    print(f"\nFeature columns: {engineer.get_feature_names()}")

