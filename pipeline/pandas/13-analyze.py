#!/usr/bin/env python3
"""
Module that computes descriptive statistics for all columns
except 'Timestamp'.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except 'Timestamp'.

    Args:
        df: A DataFrame-like object, including a 'Timestamp' column.

    Returns:
        A new DataFrame with the descriptive statistics
        of the non-'Timestamp' columns.
    """
    # Drop the 'Timestamp' column and compute descriptive stats
    stats = df.drop(columns=['Timestamp']).describe()
    return stats
