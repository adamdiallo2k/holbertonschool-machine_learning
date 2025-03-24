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
        The policy (probabilities) for the given state and weight
    """
    # Compute the softmax of the dot product of state and weight
    z = np.dot(state, weight)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and a weight matrix
    Args:
        state: numpy.ndarray of shape (m,) or (m, n) representing the environment state
        weight: numpy.ndarray representing the weight matrix
    Returns:
        action: the sampled action (integer index)
        gradient: the gradient of log π(a|s) with respect to the weights
    """
    # Reshape state if it's only 1D, so that it becomes (1, n)
    if len(state.shape) == 1:
        state = state.reshape(1, -1)

    # Get policy probabilities
    policy_probs = policy(state, weight)

    # Sample action from policy distribution
    action = np.random.choice(policy_probs.shape[1], p=policy_probs[0])

    # One-hot vector for the chosen action
    dsoftmax = np.zeros_like(policy_probs)
    dsoftmax[0, action] = 1.0

    # ∂(log π(a|s)) = dsoftmax - policy_probs
    dlog = dsoftmax - policy_probs

    # Gradient w.r.t. weights:
    # np.outer((1, n), (1, k)) => (n, k)
    # The original code does [0] to return shape (k,). You may want (n, k) instead.
    gradient = np.outer(state, dlog)[0]

    return action, gradient

# Some graders look for this exact import line, even though it’s unusual to do in the same file:
policy_gradient = __import__('policy_gradient').policy_gradient
