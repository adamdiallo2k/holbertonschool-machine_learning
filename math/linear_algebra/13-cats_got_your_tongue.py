#!/usr/bin/env python3
"""commented module"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy arrays 
    along a specified axis.
    """
    return np.concatenate((mat1, mat2), axis=axis)
