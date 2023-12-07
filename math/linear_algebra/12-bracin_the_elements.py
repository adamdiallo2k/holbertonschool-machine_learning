#!/usr/bin/env python3
"""commented module"""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise operations on two numpy arrays: addition, subtraction,
    multiplication, and division.

    :param mat1: First numpy array
    :param mat2: Second numpy array
    :return: A tuple containing the element-wise sum, difference
    , product, and quotient
    """
    sum_array = mat1 + mat2  # Element-wise addition
    diff_array = mat1 - mat2  # Element-wise subtraction
    prod_array = mat1 * mat2  # Element-wise multiplication
    quot_array = mat1 / mat2  # Element-wise division

    return sum_array, diff_array, prod_array, quot_array
