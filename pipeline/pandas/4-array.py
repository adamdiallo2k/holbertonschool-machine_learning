#!/usr/bin/env python3
"""
Module that selects the last 10 rows from 'High' and 'Close' columns
of a DataFrame and returns them as a NumPy array.
"""


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns from df
    and converts them to a NumPy array.

    Args:
        df: A DataFrame-like object with 'High' and 'Close' columns.

    Returns:
        The selected values as a NumPy array.
    """
    # Select the last 10 rows of 'High' and 'Close'
    last_ten = df[['High', 'Close']].tail(10)
    # Convert to a NumPy array
    return last_ten.to_numpy()
