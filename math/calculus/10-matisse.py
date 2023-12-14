#!/usr/bin/env python3
"""commented module """


def poly_derivative(poly):
    """commented function"""
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None
    if len(poly) <= 1:  # This covers the case where the polynomial is a constant or empty
        return [0]
    
    # Calculate the derivative coefficients
    derivative = [coef * power for power, coef in enumerate(poly) if power > 0]
    
    # The last term of the polynomial (highest power) will always have a derivative of 0 since it's constant after differentiation
    return derivative[::-1]  # Reverse the list to represent the polynomial correctly
