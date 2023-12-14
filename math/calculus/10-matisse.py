#!/usr/bin/env python3
"""commented module """


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial represented by a list of coefficients.

    :param poly: List of coefficients, where the index represents the power of x.
    :return: List of coefficients representing the derivative of the polynomial.
    """
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None

    if len(poly) <= 1:  # The polynomial is constant or empty
        return [0]

    # Calculate the derivative coefficients
    # Skip the first element (constant term) and use enumerate to get index and coefficient
    derivative = [coef * index for index, coef in enumerate(poly) if index > 0]

    return derivative
