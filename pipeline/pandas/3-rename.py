#!/usr/bin/env python3
"""
Module that provides a function to rename the 'Timestamp' column to 'Datetime'
and convert its values to datetime objects, returning only 'Datetime' and 'Close'.
"""

import pandas as pd


def rename(df):
    """
    Renames the 'Timestamp' column to 'Datetime', converts values to datetime,
    and returns only 'Datetime' and 'Close' columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing a column named 'Timestamp'.
    
    Returns:
        pd.DataFrame: The modified DataFrame with 'Datetime' and 'Close' columns.
    """
    # Rename the 'Timestamp' column to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert the timestamp values (assuming they're in seconds) to datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Keep only 'Datetime' and 'Close' columns
    df = df[['Datetime', 'Close']]

    return df
