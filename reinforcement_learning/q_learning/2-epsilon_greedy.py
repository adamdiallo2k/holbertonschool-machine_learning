#!/usr/bin/env python3
"""
Module implementing the epsilon-greedy strategy for action selection
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy strategy to determine the next action
    
    Parameters:
        Q: numpy.ndarray containing the q-table
        state: the current state
        epsilon: the epsilon to use for the calculation

    Returns:
        The next action index
    """
    # Generate a random probability p
    p = np.random.uniform(0, 1)

    # Explore: choose a random action with probability epsilon
    if p < epsilon:
        # Pick a random action from all possible actions
        action = np.random.randint(0, Q.shape[1])
    # Exploit: choose the best action with probability (1 - epsilon)
    else:
        # Pick the action with the highest Q-value for the current state
        action = np.argmax(Q[state])

    return action
