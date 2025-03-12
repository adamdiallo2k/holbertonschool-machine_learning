#!/usr/bin/env python3
"""
Module for playing an episode with a trained agent
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode
    
    Parameters:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode
    
    Returns:
        total_rewards: the total rewards for the episode
        rendered_outputs: list of rendered outputs for the board at each step
    """
    # Reset the environment
    state, _ = env.reset()
    
    # Initialize variables
    done = False
    total_rewards = 0
    rendered_outputs = []
    
    # Get initial render
    render = env.render()
    
    # Play episode
    for step in range(max_steps):
        # Always exploit the Q-table (no exploration)
        action = np.argmax(Q[state])
        
        # Convert action number to direction for display
        action_direction = {
            0: "(Left)",
            1: "(Down)",
            2: "(Right)",
            3: "(Up)"
        }
        
        # Display current state with action
        if render is not None:
            current_render = render + f"\n  {action_direction[action]}"
            rendered_outputs.append(current_render)
        
        # Take action
        new_state, reward, done, _, _ = env.step(action)
        
        # Update state and reward
        state = new_state
        total_rewards += reward
        
        # Get new render
        render = env.render()
        
        # Break if done
        if done:
            break
    
    # Add final state render
    if render is not None:
        rendered_outputs.append(render)
    
    return total_rewards, rendered_outputs
