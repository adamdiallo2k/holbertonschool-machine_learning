#!/usr/bin/env python3
"""
Module that selects the last 10 rows from 'High' and 'Close' columns
of a DataFrame and returns them as a NumPy array.
"""

import pandas as pd


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns from df
    and converts them to a NumPy array.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns named 'High' and 'Close'.
    
    Returns:
        numpy.ndarray: The selected values as a NumPy array.
    """
    # Select the last 10 rows of 'High' and 'Close'
    selected_df = df[['High', 'Close']].tail(10)
    
    # Convert the selected data to a NumPy array
    arr = selected_df.to_numpy()
    
    return arr
