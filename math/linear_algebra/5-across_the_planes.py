#!/usr/bin/env python3
"""commented module"""


def add_matrices2D(mat1, mat2):
    """commlented module"""
    if len(mat1) != len(mat2) or any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None


    result = [[val1 + val2 for val1, val2 in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]

    return result
