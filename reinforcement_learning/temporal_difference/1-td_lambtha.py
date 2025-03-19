#!/usr/bin/env python3
"""
TD(λ) algorithm for value estimation
"""
import numpy as np

def td_lambtha(env, V, policy, lambtha=0.9, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Implements the TD(λ) algorithm to estimate the value function.

    Parameters:
        env (object): The environment instance.
        V (numpy.ndarray): Array of shape (s,) representing state-value estimates.
        policy (function): Function that takes a state and returns an action.
        lambtha (float): The eligibility trace decay factor (default: 0.9).
        episodes (int): Number of training episodes (default: 5000).
        max_steps (int): Maximum steps per episode (default: 100).
        alpha (float): Learning rate (default: 0.1).
        gamma (float): Discount factor for future rewards (default: 0.99).

    Returns:
        numpy.ndarray: The updated state-value function V.
    """
    for _ in range(episodes):
        # Start a new episode and get the initial state
        state, _ = env.reset()
        
        # Initialize eligibility traces for all states to zero
        traces = np.zeros_like(V)

        for _ in range(max_steps):
            # Choose an action using the policy
            action = policy(state)
            
            # Execute the chosen action and observe the result
            next_state, reward, done, _, _ = env.step(action)

            # Compute the Temporal Difference (TD) error
            td_error = reward + (gamma * V[next_state] - V[state])

            # Increase the eligibility trace for the current state
            traces[state] += 1  

            # Update the value function V using the eligibility traces
            V += alpha * td_error * traces  

            # Decay the eligibility traces for all states
            traces *= gamma * lambtha  

            # Move to the next state
            state = next_state

            # End episode if terminal state is reached
            if done:
                break  

    return V
