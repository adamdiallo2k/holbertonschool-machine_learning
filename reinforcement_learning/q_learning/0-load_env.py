#!/usr/bin/env python3
"""
Module to load the FrozenLake environment from gymnasium
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Load the pre-made FrozenLakeEnv environment from gymnasium

    Parameters:
        desc: None or list of lists containing a custom description of the map
        map_name: None or string containing the pre-made map to load
        is_slippery: boolean to determine if the ice is slippery

    Returns:
        The environment
    """
    env = gym.make('FrozenLake-v1', 
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)

    return env
