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
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
    
    Returns:
        V: the updated value estimate
    """
    # Loop through each episode
    for _ in range(episodes):
        # Reset the environment
        state, _ = env.reset()
        
        # Initialize lists to store the state and reward information
        states = []
        rewards = []
        
        # Run the episode
        for _ in range(max_steps):
            # Choose action based on the policy
            action = policy(state)
            
            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Store state and reward
            states.append(state)
            rewards.append(reward)
            
            # Update state
            state = next_state
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # Calculate returns and update value function
        G = 0
        for t in range(len(states) - 1, -1, -1):
            # Calculate return
            G = gamma * G + rewards[t]
            
            # Update value estimate
            V[states[t]] = V[states[t]] + alpha * (G - V[states[t]])
    
    return V
