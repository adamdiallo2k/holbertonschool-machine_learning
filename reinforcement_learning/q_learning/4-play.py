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
        
        # Add current position marker to the render
        state_row = state // env.unwrapped.ncol
        state_col = state % env.unwrapped.ncol
        
        # Create a modified render with marker for current position
        lines = render.strip().split('\n')
        for i, line in enumerate(lines):
            if i == state_row:
                # Replace character at current position with quoted version
                new_line = ''
                for j, char in enumerate(line):
                    if j == state_col:
                        new_line += f'"{char}"'
                    else:
                        new_line += char
                lines[i] = new_line
        
        marked_render = '\n'.join(lines)
        
        # Display current state with action
        current_render = marked_render
        if step < max_steps - 1 and not done:  # Don't add direction to final state
            current_render += f"\n  {action_direction[action]}"
        
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
            # Add final state with position marker
            if render is not None:
                state_row = state // env.unwrapped.ncol
                state_col = state % env.unwrapped.ncol
                
                lines = render.strip().split('\n')
                for i, line in enumerate(lines):
                    if i == state_row:
                        new_line = ''
                        for j, char in enumerate(line):
                            if j == state_col:
                                new_line += f'"{char}"'
                            else:
                                new_line += char
                        lines[i] = new_line
                
                final_render = '\n'.join(lines)
                rendered_outputs.append(final_render)
            break
    
    return total_rewards, rendered_outputs
