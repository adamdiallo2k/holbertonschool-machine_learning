#!/usr/bin/env python3
"""
Module containing policy gradient functions
"""
import numpy as np


def policy(state, weight):
    """
    Computes the policy with a weight of a matrix

    Args:
        state: numpy.ndarray containing the state input
        weight: numpy.ndarray containing the weight matrix

    Returns:
        The policy for the given state and weight
    """
    # Compute the softmax of the dot product of state and weight
    z = np.dot(state, weight)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
