#!/usr/bin/env python3
"""
Module that sorts a DataFrame in reverse chronological order
by its row index and then transposes the result.
"""


def flip_switch(df):
    """
    Sorts df in descending order by row index, then transposes df.

    Args:
        df: A DataFrame-like object with a numeric or chronological index.

    Returns:
        The transposed DataFrame after reversing the row index order.
    """
    # Sort rows by descending index
    df = df.iloc[::-1]
    # Transpose the DataFrame
    df = df.T
    return df
