#!/usr/bin/env python3
"""commented module"""


def np_shape(matrix):
    """commented function"""
    # Getting the number of rows
    num_rows = len(matrix)

    # Getting the number of columns (assuming all rows have the same number of columns)
    num_cols = len(matrix[0]) if num_rows > 0 else 0

    return (num_rows, num_cols)

