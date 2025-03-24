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
    if len(state.shape) == 1:
        state = state.reshape(1, -1)
    
    policy_probs = policy(state, weight)
    action = np.random.choice(policy_probs.shape[1], p=policy_probs[0])
    
    dsoftmax = np.zeros_like(policy_probs)
    dsoftmax[0, action] = 1.0
    
    dlog = dsoftmax - policy_probs
    
    # Typically you want a 2D gradient = np.outer(state, dlog),
    # but some instructions want a 1D version:
    gradient = np.outer(state, dlog)[0]
    
    return action, gradient
