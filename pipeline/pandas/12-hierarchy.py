#!/usr/bin/env python3
"""
Module that rearranges two DataFrames in a MultiIndex so that 'Timestamp'
is the first level (keys second), selecting only timestamps from 1417411980
through 1417417980, inclusive.
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    1) Sets 'Timestamp' as the index for df1 and df2.
    2) Restricts both to timestamps in [1417411980, 1417417980], inclusive.
    3) Concatenates them with keys=['bitstamp', 'coinbase'].
    4) Rearranges the MultiIndex so that 'Timestamp' is the first level.
    5) Returns the resulting DataFrame in ascending chronological order.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame
        df2 (pd.DataFrame): Bitstamp DataFrame

    Returns:
        pd.DataFrame: The multi-index DataFrame with 'Timestamp' as level 0.
    """
    # Step 1: Index both DataFrames by 'Timestamp'
    df1 = index(df1)
    df2 = index(df2)

    # Step 2: Filter rows to the specified timestamp range
    df1_filtered = df1.loc[1417411980:1417417980]
    df2_filtered = df2.loc[1417411980:1417417980]

    # Step 3: Concatenate with labeled keys
    df_concat = pd.concat([df2_filtered, df1_filtered],
                          keys=['bitstamp', 'coinbase'])

    # Step 4: Swap the levels so that 'Timestamp' is first
    df_concat = df_concat.swaplevel(0, 1)

    # Step 5: Sort by 'Timestamp' in ascending order
    df_concat = df_concat.sort_index(level='Timestamp')

    return df_concat
