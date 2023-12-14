#!/usr/bin/env python3
"""commented module """


def summation_i_squared(n):
    """
    Calculate the sum of the squares of all integers from 1 to n.
    The formula for the sum of the squares of the first n positive integers is n(n + 1)(2n + 1)/6.
    
    :param n: integer, the stopping condition
    :return: integer, the sum of the squares from 1 to n, or None if input is not valid
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
