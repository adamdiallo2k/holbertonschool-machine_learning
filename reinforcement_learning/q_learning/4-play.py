#!/usr/bin/env python3
"""
Module 4-play

Provides a function `play` that has the trained agent play an episode on the
FrozenLake environment.
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays one episode using a trained Q-table on the FrozenLake environment.

    Args:
        env: FrozenLakeEnv instance with render_mode="ansi".
        Q (numpy.ndarray): Q-table containing the agent's learned values.
        max_steps (int): Maximum number of steps in the episode.

    Returns:
        tuple: (total_rewards, rendered_outputs)
            - total_rewards (float): Total rewards for the episode.
            - rendered_outputs (list): List of rendered outputs representing
              the board state at each step.
    """
    total_rewards = 0
    rendered_outputs = []

    # Retrieve the current state.
    # It is assumed that env.reset() has been called prior to play.
    try:
        state = env.env.s
    except AttributeError:
        state = env.reset()

    # Mapping of action indices to their names.
    actions_dict = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

    done = False
    for _ in range(max_steps):
        # Always exploit the Q-table by choosing the best action.
        action = int(np.argmax(Q[state]))

        # Take the action in the environment.
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward

        # Render the current board state.
        board = env.render()
        if not done:
            # Append the rendered board with the action taken.
            rendered_outputs.append(board + "\n  (" + actions_dict[action] + ")")
        else:
            # Append the final rendered board without an action.
            rendered_outputs.append(board)
            break
        state = next_state

    # Ensure the final state is rendered if max_steps is reached without done.
    if not done:
        rendered_outputs.append(env.render())

    return total_rewards, rendered_outputs
