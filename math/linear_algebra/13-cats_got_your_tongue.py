#!/usr/bin/env python3
"""commented module"""


import numpy as np

def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy arrays along a specified axis.
    
    :param mat1: First numpy array
    :param mat2: Second numpy array
    :param axis: Axis along which the arrays will be concatenated (default is 0)
    :return: Concatenated numpy array
    """
    return np.concatenate((mat1, mat2), axis=axis)
