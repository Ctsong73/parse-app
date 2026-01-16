import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path, date_col='Date', sort_data=True):
    """
    Load Excel or CSV file with financial data
    Expected columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Convert date column
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    
    # Ensure OHLCV columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns and col.lower() in [c.lower() for c in df.columns]:
            # Handle case-insensitive column names
            match = [c for c in df.columns if c.lower() == col.lower()]
            if match:
                df[col] = df[match[0]]
    
    # Sort by date
    if sort_data:
        df.sort_index(inplace=True)
    
    return df

def validate_data(df):
    """Validate data has required columns"""
    required = ['Open', 'High', 'Low', 'Close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True