import pandas as pd
import os

def load(path):
    """
    Loads raw data from CSV

    :param path
    :return: Pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} emails from {path}")
    return df

def save(df, path):
    """
    Save cleaned demail data in a Parquet format.

    :param df: DataFrame containing processed emails.
    :param path
    """
    df.to_parquet(path, index=False)
    print(f"Saved cleaned data at {path}")