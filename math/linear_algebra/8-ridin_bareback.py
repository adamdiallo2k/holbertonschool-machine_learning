#!/usr/bin/env python3
"""commented module"""


def mat_mul(mat1, mat2):
    """commented function"""
    # Vérifier si les matrices peuvent être multipliées
    if len(mat1[0]) != len(mat2):
        return None

    # Initialiser la matrice résultante
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Effectuer la multiplication matricielle
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
