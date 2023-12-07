#!/usr/bin/env python3
"""commented module"""


def np_shape(matrix):
    """commented function"""
    num_rows = len(matrix)
    num_cols = len(matrix[0]) * num_rows // num_rows  # Avoids direct conditionals
    return (num_rows, num_cols)
