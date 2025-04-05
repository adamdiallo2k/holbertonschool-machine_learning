#!/usr/bin/env python3
"""
Module that:
- Indexes two DataFrames on their 'Timestamp' columns
- Selects rows from df2 (bitstamp) up to and including timestamp 1417411920
- Concatenates these rows above df1 (coinbase)
- Adds keys to identify each DataFrame in the concatenated result
"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two DataFrames (coinbase above bitstamp),
    after:
      - Indexing both on 'Timestamp'
      - Selecting rows from bitstamp up to timestamp <= 1417411920
      - Assigning multi-level keys: 'bitstamp' and 'coinbase'

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame
        df2 (pd.DataFrame): Bitstamp DataFrame

    Returns:
        pd.DataFrame: Concatenated DataFrame with multi-level indexing
    """
    # 1) Index both DataFrames on 'Timestamp'
    df1 = index(df1)
    df2 = index(df2)

    # 2) Filter df2 to include timestamps up to and including 1417411920
    df2_filtered = df2.loc[:1417411920]

    # 3) Concatenate the rows from df2 above those from df1, adding keys
    df_concat = pd.concat([df2_filtered, df1],
                          keys=['bitstamp', 'coinbase'])

    return df_concat
