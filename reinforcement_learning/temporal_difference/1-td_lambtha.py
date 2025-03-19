#!/usr/bin/env python3
"""
Temporal Difference Lambda (TD-λ) Learning Algorithm
"""
import numpy as np

def td_lambtha(env, V, policy, lambtha=0.9, episodes=5000, max_steps=100, 
               alpha=0.1, gamma=0.99):
    """
    Applies the TD(λ) algorithm to estimate the value function for an environment.

    Parameters:
        env: The reinforcement learning environment.
        V (numpy.ndarray): Array storing the estimated value function for each state.
        policy (function): Function mapping a state to an action.
        lambtha (float): The decay factor for eligibility traces (default: 0.9).
        episodes (int): Number of episodes for training (default: 5000).
        max_steps (int): Maximum steps allowed per episode (default: 100).
        alpha (float): Learning rate controlling the update step (default: 0.1).
        gamma (float): Discount factor for future rewards (default: 0.99).

    Returns:
        numpy.ndarray: The updated value function after training.
    """
    for _ in range(episodes):
        # Reset environment and obtain the starting state
        current_state, _ = env.reset()

        # Initialize the eligibility trace for all states
        trace_memory = np.zeros_like(V)

        for _ in range(max_steps):
            # Determine action using the given policy
            chosen_action = policy(current_state)

            # Execute action and retrieve the next state and reward
            next_state, reward, done, _, _ = env.step(chosen_action)

            # Compute the temporal difference error
            delta = reward + gamma * V[next_state] - V[current_state]

            # Increase eligibility trace for the current state
            trace_memory[current_state] += 1  

            # Apply updates to all state values using eligibility traces
            V += alpha * delta * trace_memory  

            # Reduce eligibility traces across all states
            trace_memory *= gamma * lambtha  

            # Move to the next state
            current_state = next_state

            # Terminate episode if the state is terminal
            if done:
                break  

    return V
