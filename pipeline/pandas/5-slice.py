#!/usr/bin/env python3
"""
Module that extracts specific columns from a DataFrame and
returns every 60th row of those columns.
"""


def slice(df):
    """
    Extracts columns 'High', 'Low', 'Close', and 'Volume_(BTC)',
    then returns every 60th row.

    Args:
        df: A DataFrame-like object with 'High', 'Low',
            'Close', and 'Volume_(BTC)' columns.

    Returns:
        A DataFrame-like object containing every 60th row
        of the specified columns.
    """
    # Select the desired columns
    df_sliced = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    # Take every 60th row
    df_sliced = df_sliced.iloc[::60]
    return df_sliced
