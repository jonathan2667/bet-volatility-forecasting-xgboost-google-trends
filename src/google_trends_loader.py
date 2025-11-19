"""
Google Trends Data Loader and Feature Engineering
Loads multiTimeline CSV files from Google Trends and creates behavioral features
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_google_trends_csv(filepath, skip_rows=1):
    """
    Load a single Google Trends multiTimeline CSV file
    
    Args:
        filepath: Path to the CSV file
        skip_rows: Number of rows to skip (default 1 for "Category: All categories")
    
    Returns:
        DataFrame with Date index and search volume column
    """
    df = pd.read_csv(filepath, skiprows=skip_rows)
    
    # Rename columns: first is Month, second is the search term
    cols = df.columns.tolist()
    df.rename(columns={cols[0]: 'Month', cols[1]: 'SearchVolume'}, inplace=True)
    
    # Convert Month to datetime
    df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
    
    # Set Month as index
    df.set_index('Month', inplace=True)
    
    # Convert SearchVolume to numeric (handle '<1' values as 0)
    df['SearchVolume'] = pd.to_numeric(df['SearchVolume'], errors='coerce').fillna(0)
    
    return df


def load_all_trends(data_dir='../data'):
    """
    Load all Google Trends files and merge into single DataFrame
    
    Args:
        data_dir: Directory containing the multiTimeline CSV files
    
    Returns:
        DataFrame with all Google Trends series as columns
    """
    data_path = Path(data_dir)
    
    trends_files = {
        'inflatie': data_path / 'multiTimeline-inflatie.csv',
        'recesiune': data_path / 'multiTimeline-recesiune.csv',
        'eurocurs': data_path / 'multiTimeline-eurocurs.csv'
    }
    
    trends_data = {}
    
    for name, filepath in trends_files.items():
        if filepath.exists():
            df = load_google_trends_csv(filepath)
            trends_data[name] = df['SearchVolume']
            print(f"✓ Loaded {name}: {len(df)} months, range {df.index.min()} to {df.index.max()}")
        else:
            print(f"✗ File not found: {filepath}")
    
    # Combine all trends into single DataFrame
    if trends_data:
        trends_df = pd.DataFrame(trends_data)
        print(f"\n✓ Combined trends data: {len(trends_df)} rows, {len(trends_df.columns)} series")
        return trends_df
    else:
        return None


def expand_monthly_to_daily(monthly_df, daily_index):
    """
    Expand monthly Google Trends data to daily frequency using forward fill
    
    Args:
        monthly_df: DataFrame with monthly data (datetime index)
        daily_index: Daily datetime index to expand to
    
    Returns:
        DataFrame with daily data
    """
    # Reindex to daily frequency with forward fill
    daily_df = monthly_df.reindex(
        pd.date_range(monthly_df.index.min(), daily_index.max(), freq='D'),
        method='ffill'
    )
    
    # Filter to match the daily_index
    daily_df = daily_df.reindex(daily_index, method='ffill')
    
    return daily_df


def create_trends_features(trends_df, lags=[1, 5, 10, 20]):
    """
    Create behavioral features from Google Trends data
    
    Args:
        trends_df: DataFrame with Google Trends series
        lags: List of lag periods for rate of change calculations
    
    Returns:
        DataFrame with engineered features
    """
    features = trends_df.copy()
    
    for col in trends_df.columns:
        # Rate of change (momentum)
        for lag in lags:
            roc_col = f'{col}_roc_{lag}'
            features[roc_col] = trends_df[col].pct_change(periods=lag) * 100
            
        # Moving averages
        features[f'{col}_ma_7'] = trends_df[col].rolling(window=7, min_periods=1).mean()
        features[f'{col}_ma_30'] = trends_df[col].rolling(window=30, min_periods=1).mean()
        
        # Volatility (rolling std)
        features[f'{col}_vol_7'] = trends_df[col].rolling(window=7, min_periods=1).std()
        
    # Replace inf values with NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Forward fill NaN values
    features.fillna(method='ffill', inplace=True)
    features.fillna(0, inplace=True)
    
    print(f"\n✓ Created {len(features.columns)} Google Trends features")
    
    return features


def compute_trends_correlation(trends_df):
    """
    Compute correlation matrix for Google Trends series
    
    Args:
        trends_df: DataFrame with Google Trends data
    
    Returns:
        Correlation matrix
    """
    corr_matrix = trends_df.corr()
    
    print("\n📊 Google Trends Correlation Matrix:")
    print(corr_matrix.round(3))
    
    return corr_matrix


if __name__ == '__main__':
    # Test the loader
    print("Testing Google Trends Loader\n" + "="*50)
    
    trends_df = load_all_trends('../data')
    
    if trends_df is not None:
        print("\n" + "="*50)
        print("Sample data:")
        print(trends_df.head(10))
        
        print("\n" + "="*50)
        print("Data summary:")
        print(trends_df.describe())
        
        print("\n" + "="*50)
        corr = compute_trends_correlation(trends_df)
        
        print("\n" + "="*50)
        print("Creating features...")
        features = create_trends_features(trends_df)
        print(f"Total features: {len(features.columns)}")
        print("\nFeature names:")
        for col in features.columns:
            print(f"  - {col}")

