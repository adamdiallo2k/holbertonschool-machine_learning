#!/usr/bin/env python3
"""
Monte Carlo algorithm implementation
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
    # Iterate through episodes
    for _ in range(episodes):
        # Initialize episode
        state, _ = env.reset()
        done = False
        episode = []
        
        # Generate episode
        for _ in range(max_steps):
            # Break if episode is done
            if done:
                break
                
            # Get action from policy
            action = policy(state)
            
            # Take action and observe next state and reward
            next_state, reward, done, _, _ = env.step(action)
            
            # Store step information
            episode.append((state, action, reward))
            
            # Update state
            state = next_state
        
        # Calculate returns for each step in the episode
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, _, reward = episode[t]
            
            # Calculate return
            G = gamma * G + reward
            
            # Update value estimate
            V[state] = V[state] + alpha * (G - V[state])
    
    return V
