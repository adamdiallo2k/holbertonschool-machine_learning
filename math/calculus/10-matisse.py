#!/usr/bin/env python3
"""commented module """


def poly_derivative(poly):
    """commented function"""
    # Check if poly is a list and contains only numbers
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None

    # Check if poly represents a constant polynomial (including empty polynomial)
    if len(poly) == 0 or (len(poly) == 1 and poly[0] == 0):
        return [0]

    # Check for a single non-zero constant term
    if len(poly) == 1:
        return [0]

    # Calculate the derivative coefficients
    derivative = [coef * index for index, coef in enumerate(poly) if index > 0]

    return derivative
