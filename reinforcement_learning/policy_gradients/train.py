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


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and a weight matrix
    Args:
        state: matrix representing the current observation of the environment
        weight: matrix of random weight
    Returns:
        action and the gradient (in this order)
    """
    # Reshape state if it's not a 2D array
    if len(state.shape) == 1:
        state = state.reshape(1, -1)
    # Get policy probabilities
    policy_probs = policy(state, weight)
    
    # Sample action from policy distribution
    action = np.random.choice(policy_probs.shape[1], p=policy_probs[0])
    
    # Initialize gradient matrix with zeros
    dsoftmax = np.zeros_like(policy_probs)
    dsoftmax[0, action] = 1.0
    
    # Compute gradient of log policy
    dlog = dsoftmax - policy_probs
    
    # Compute gradient with respect to weights
    gradient = np.outer(state, dlog)[0]
    
    return action, gradient
