#!/usr/bin/env python3
"""
Implementation of the SARSA(λ) algorithm with eligibility traces.
This variant has been refactored to keep the same functionality 
but features rephrased commentary.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Selects an action according to the epsilon-greedy approach.

    Args:
        Q (np.ndarray): Q-table mapping state-action pairs to values.
        state (int): Index of the current state.
        epsilon (float): Probability of choosing a random action 
                         rather than the best-known one.

    Returns:
        int: Chosen action index.
    """
    # Decide whether to exploit (greedy) or explore (random) based on epsilon
    if np.random.rand() > epsilon:
        # Utilize the best action from Q-table
        return np.argmax(Q[state, :])
    else:
        # Pick a random valid action
        return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Executes the SARSA(λ) algorithm on a given environment to refine the Q-table.

    Args:
        env: The environment (e.g., a FrozenLake instance).
        Q (np.ndarray): Q-table to be updated.
        lambtha (float): Coefficient for eligibility trace decay.
        episodes (int): Number of full training episodes.
        max_steps (int): Cap on steps per episode.
        alpha (float): Learning rate for Q updates.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial probability for random action selection.
        min_epsilon (float): Lower limit for epsilon as it decays.
        epsilon_decay (float): Rate controlling how quickly epsilon falls.

    Returns:
        np.ndarray: The updated Q-table after training completes.
    """
    init_eps = epsilon

    for episode in range(episodes):
        # Reset the environment and pick the initial action
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)

        # Initialize all eligibility traces to zero
        traces = np.zeros_like(Q)

        for _ in range(max_steps):
            # Execute the chosen action
            next_state, reward, done, truncated, _ = env.step(action)

            # Select the subsequent action
            next_action = epsilon_greedy(Q, next_state, epsilon)

            # Compute the TD Error (delta)
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # Update eligibility trace for the current state-action
            traces[state, action] += 1

            # Decay all traces by gamma and lambda
            traces *= gamma * lambtha

            # Adjust Q-values in proportion to TD error and trace
            Q += alpha * delta * traces

            # Transition to the next state and action
            state = next_state
            action = next_action

            # Break if the episode ends
            if done or truncated:
                break

        # Gradually reduce epsilon after each episode
        epsilon = min_epsilon + (init_eps - min_epsilon) * np.exp(-epsilon_decay * episode)

    return Q
