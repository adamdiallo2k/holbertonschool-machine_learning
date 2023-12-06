#!/usr/bin/env python3
def matrix_shape(matrix):
    if not isinstance(matrix, list) or not matrix:
        return []

    outer_len = len(matrix)
    inner_len = 0
    inner_inner_len = 0

    for row in matrix:
        if isinstance(row, list):
            inner_len = max(inner_len, len(row))
            for element in row:
                if isinstance(element, list):
                    inner_inner_len = max(inner_inner_len, len(element))

    if inner_inner_len:  # This is a 3D matrix
        return [outer_len, inner_len, inner_inner_len]
    elif inner_len:  # This is a 2D matrix
        return [outer_len, inner_len]
    else:  # This is a 1D list
        return [outer_len]


