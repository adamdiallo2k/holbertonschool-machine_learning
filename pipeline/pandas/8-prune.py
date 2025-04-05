#!/usr/bin/env python3
"""
Module that removes any entries in a DataFrame where the 'Close' value is NaN.
"""


def prune(df):
    """
    Removes rows where the 'Close' column contains NaN values.

    Args:
        df: A DataFrame-like object that includes a 'Close' column.

    Returns:
        A modified DataFrame without rows where 'Close' is NaN.
    """
    df = df.dropna(subset=['Close'])
    return df
