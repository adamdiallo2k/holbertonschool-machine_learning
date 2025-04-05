#!/usr/bin/env python3
"""
Module that sets the 'Timestamp' column as the index of a DataFrame.
"""


def index(df):
    """
    Sets 'Timestamp' as the index of df, returning the modified DataFrame.

    Args:
        df: A DataFrame-like object with a 'Timestamp' column.

    Returns:
        The modified DataFrame with 'Timestamp' as its index.
    """
    return df.set_index('Timestamp')
