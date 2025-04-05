#!/usr/bin/env python3
"""
Module that modifies a DataFrame by:
- Removing the 'Weighted_Price' column
- Filling missing 'Close' values with the previous row's value
- Filling missing 'High', 'Low', and 'Open' values with the 'Close'
  value of the same row
- Setting missing 'Volume_(BTC)' and 'Volume_(Currency)' values to 0
"""


def fill(df):
    """
    Removes 'Weighted_Price' from df, fills missing 'Close' with the previous
    row's value, fills missing 'High', 'Low', and 'Open' with 'Close' of the
    same row, and sets missing 'Volume_(BTC)' and 'Volume_(Currency)' to 0.

    Args:
        df: A DataFrame-like object with the relevant columns.

    Returns:
        The modified DataFrame.
    """
    # 1) Remove 'Weighted_Price' column
    df = df.drop(columns=['Weighted_Price'])

    # 2) Fill missing Close with previous row's value
    df['Close'] = df['Close'].fillna(method='ffill')

    # 3) Fill missing High, Low, Open with the same row's Close
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # 4) Set missing Volume_(BTC) and Volume_(Currency) to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
