#!/usr/bin/env python3
"""commented module"""


def np_transpose(matrix):
    """
    Transpose a given matrix represented as a list of lists.
    
    :param matrix: A matrix to be transposed
    :return: Transposed matrix
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
