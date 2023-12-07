#!/usr/bin/env python3
"""commented module"""


def matrix_transpose(matrix):
    """commented function"""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
