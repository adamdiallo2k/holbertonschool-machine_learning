#!/usr/bin/env python3
"""
Module that sorts a DataFrame in descending order by the 'High' column.
"""


def high(df):
    """
    Sorts df in descending order by the 'High' column.

    Args:
        df: A DataFrame-like object with a 'High' column.

    Returns:
        The sorted DataFrame.
    """
    sorted_df = df.sort_values(by='High', ascending=False)
    return sorted_df
