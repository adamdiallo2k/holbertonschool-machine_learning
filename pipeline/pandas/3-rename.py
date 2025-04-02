#!/usr/bin/env python3
"""
Module that renames the 'Timestamp' column to 'Datetime', converts its
values to datetime objects, and returns only 'Datetime' and 'Close'.
"""

import pandas as pd


def rename(df):
    """
    Renames the 'Timestamp' column to 'Datetime', converts values to datetime,
    and returns only 'Datetime' and 'Close' columns.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: Modified DataFrame with 'Datetime' and 'Close' columns.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})

    df['Datetime'] = pd.to_datetime(
        df['Datetime'],
        unit='s'
    )

    df = df[['Datetime', 'Close']]

    return df
