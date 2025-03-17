#!/usr/bin/env python3
"""
Module to initialize the Q-table for FrozenLake environment
"""
import numpy as np


def q_init(env):
    """
    Initialize the Q-table for a given FrozenLake environment

    Parameters:
        env: the FrozenLakeEnv instance

    Returns:
        Q-table as a numpy.ndarray of zeros
    """
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    # Initialize Q-table with zeros
    Q = np.zeros((state_space_size, action_space_size))

    return Q
