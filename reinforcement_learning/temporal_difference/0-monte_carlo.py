#!/usr/bin/env python3
import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the state-value function.

    Parameters:
        env: The environment instance.
        V (numpy.ndarray): The state-value function estimate (shape: (s,)).
        policy (function): A function that takes a state and returns the next action.
        episodes (int): The total number of episodes for training.
        max_steps (int): The maximum number of steps per episode.
        alpha (float): The learning rate.
        gamma (float): The discount factor.

    Returns:
        numpy.ndarray: The updated state-value function V.
    """
    for _ in range(episodes):
        # Generate an episode using the policy
        state = env.reset()[0]  # Reset environment and get initial state
        episode = []
        for _ in range(max_steps):
            action = policy(state)  # Select action based on policy
            next_state, reward, done, _, _ = env.step(action)  # Take action
            episode.append((state, reward))  # Store state and reward
            state = next_state
            if done:
                break

        # Compute the returns and update V using incremental mean update
        G = 0  # Initialize return
        visited_states = set()
        for state, reward in reversed(episode):
            G = gamma * G + reward  # Compute return
            if state not in visited_states:  # First-visit MC update
                visited_states.add(state)
                V[state] = V[state] + alpha * (G - V[state])  # Update state-value estimate

    return V
