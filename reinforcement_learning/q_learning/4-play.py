#!/usr/bin/env python3
"""
Module for playing an episode using a trained Q-learning agent
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode using the Q-table for action selection
    
    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode
        
    Returns:
        total_rewards: float, total rewards for the episode
        rendered_outputs: list, rendered outputs representing the board state at each step
    """
    # Reset the environment
    state, _ = env.reset()
    
    # Initialize rewards and rendered outputs
    total_rewards = 0
    rendered_outputs = []
    
    # Get the initial render
    rendered_outputs.append(env.render())
    
    # Play the episode
    for _ in range(max_steps):
        # Choose the action with the highest Q-value for the current state (exploitation)
        action = np.argmax(Q[state, :])
        
        # Take the action and observe the next state and reward
        next_state, reward, done, _, _ = env.step(action)
        
        # Add the action taken to the rendered output
        if not done:
            # Map action to direction
            direction = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
            rendered_outputs[-1] += f"\n  ({direction[action]})"
        
        # Update the state
        state = next_state
        
        # Add the reward to the total rewards
        total_rewards += reward
        
        # Get the new render after taking the action
        rendered_outputs.append(env.render())
        
        # Check if the episode is done
        if done:
            break
    
    return total_rewards, rendered_outputs
