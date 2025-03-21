"""
Data loading and saving utilities for email data.
"""
import pandas as pd
import os

def load(path: str) -> pd.DataFrame:
    """Load raw email data from CSV file.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame containing raw email data
        
    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} emails from {path}")
    return df

def save(df: pd.DataFrame, path: str) -> None:
    """Save processed email data in Parquet format.
    
    Args:
        df: DataFrame containing processed emails
        path: Path to save the Parquet file
    """
    df.to_parquet(path, index=False)
    print(f"Saved cleaned data at {path}")