#!/usr/bin/env python3
"""commented module"""


def np_shape(matrix):
    """commented function"""
    num_rows = len(matrix)
    num_cols = len(next(iter(matrix), []))
    return (num_rows, num_cols)
