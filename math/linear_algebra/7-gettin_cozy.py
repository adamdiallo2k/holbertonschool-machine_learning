#!/usr/bin/env python3
"""commented module"""


def cat_matrices2D(mat1, mat2, axis=0):
    """commented function"""
    # Vérification de la possibilité de concaténation
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    elif axis == 1 and len(mat1) != len(mat2):
        return None

    # Concaténation des matrices
    if axis == 0:
        # Concaténation verticale
        return mat1 + mat2
    else:
        # Concaténation horizontale
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
