#!/usr/bin/env python3
""" commented module """

import numpy as np


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a given state
    and weight matrix for a 2-action environment.

    Args:
        state (ndarray): 1D numpy array of shape (n,) representing the current
            observation.
        weight (ndarray): 2D numpy array of shape (n, 2) representing the
            weight matrix.

    Returns:
        action (int): The sampled action (0 or 1).
        grad (ndarray): The gradient of log π(action|state) w.r.t. the weights,
            with the same shape as weight (n, 2).
    """
    # 1) Compute logits for each action: logits = state ⋅ weight
    logits = state @ weight  # shape (2,)

    # 2) Apply softmax to get action probabilities
    #    (subtract max(logits) for numerical stability)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # 3) Sample an action according to the probabilities
    action = np.random.choice([0, 1], p=probs)

    # 4) Compute gradient = state * (one_hot(action) - probs)
    #    Shape of grad should be (n, 2), same as weight
    grad = np.zeros_like(weight)
    for a in range(2):
        grad[:, a] = state * ((1 if a == action else 0) - probs[a])

    return action, grad
