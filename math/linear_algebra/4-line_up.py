#!/usr/bin/env python3
"""commented module"""


def add_arrays(arr1, arr2):
    """commented function"""
    if len(arr1) != len(arr2):
        return None
    result = [a + b for a, b in zip(arr1, arr2)]
    return result
