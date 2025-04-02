#!/usr/bin/env python3
"""
This module creates a DataFrame from a dictionary with:
- A 'First' column containing [0.0, 0.5, 1.0, 1.5]
- A 'Second' column containing ['one', 'two', 'three', 'four']
- Row labels A, B, C, and D
It stores the resulting DataFrame in the variable 'df'.
"""

import pandas as pd

# Define a dictionary with two columns
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Define row labels
index_labels = ['A', 'B', 'C', 'D']

# Create the DataFrame
df = pd.DataFrame(data, index=index_labels)
