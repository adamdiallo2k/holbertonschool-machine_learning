#!/usr/bin/env python3
"""
Temporal Difference Lambda (TD-λ) Learning Algorithm
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha=0.9, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Applies the TD(λ) algorithm to estimate the value function.

    Parameters:
        env (object): The reinforcement learning environment.
        V (numpy.ndarray): Estimated value function array.
        policy (function): Function mapping a state to an action.
        lambtha (float): Decay factor for eligibility traces (default: 0.9).
        episodes (int): Number of training episodes (default: 5000).
        max_steps (int): Maximum steps per episode (default: 100).
        alpha (float): Learning rate (default: 0.1).
        gamma (float): Discount factor for future rewards (default: 0.99).

    Returns:
        numpy.ndarray: The updated value function after training.
    """
    for _ in range(episodes):
        # Reset environment and get the initial state
        current_state, _ = env.reset()

        # Initialize eligibility traces for all states
        trace_memory = np.zeros_like(V)

        for _ in range(max_steps):
            # Determine action using the given policy
            chosen_action = policy(current_state)

            # Execute action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(chosen_action)

            # Compute the temporal difference (TD) error
            delta = reward + gamma * V[next_state] - V[current_state]

            # Update the eligibility trace for the current state
            trace_memory[current_state] += 1

            # Apply updates to all state values using eligibility traces
            V += alpha * delta * trace_memory

            # Decay the eligibility traces across all states
            trace_memory *= gamma * lambtha

            # Move to the next state
            current_state = next_state

            # Terminate episode if the state is terminal
            if done:
                break

    return V
