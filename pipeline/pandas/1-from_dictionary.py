#!/usr/bin/env python3
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
