#!/usr/bin/env python3
"""
Monte Carlo algorithm implementation for
reinforcement learning value estimation
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for reinforcement learning

    Parameters:
        env: environment instance (must have .reset() and .step() methods)
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate (numpy.ndarray of shape (s,))
    """
    for _ in range(episodes):
        # Reset the environment at the start of each episode
        state, _ = env.reset()

        # Lists to store the states visited and rewards received in the episode
        states = []
        rewards = []

        # Generate one full episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            rewards.append(reward)

            state = next_state
            if terminated or truncated:
                break

        # Backward pass to update state values
        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            # Update V for each visited state
            V[states[t]] = V[states[t]] + alpha * (G - V[states[t]])

    return V
