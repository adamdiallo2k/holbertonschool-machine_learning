#!/usr/bin/env python3
"""
Module implementing Q-learning algorithm for FrozenLake environment
"""
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning on the FrozenLake environment
    
    Parameters:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes
    
    Returns:
        Q: the updated Q-table
        total_rewards: list containing the rewards per episode
    """
    total_rewards = []
    
    for episode in range(episodes):
        # Reset the environment
        state, _ = env.reset()
        
        # Initialize variables for tracking rewards
        done = False
        total_episode_reward = 0
        
        # Explore the environment
        for step in range(max_steps):
            # Choose action using epsilon-greedy policy
            action = epsilon_greedy(Q, state, epsilon)
            
            # Take action and observe new state and reward
            new_state, reward, done, _, _ = env.step(action)
            
            # Update reward to -1 if agent falls in a hole
            if done and reward == 0:
                reward = -1
            
            # Update Q-table using Q-learning formula
            # Q(s,a) = (1-α) * Q(s,a) + α * (r + γ * max(Q(s',a')))
            current_q = Q[state, action]
            max_future_q = np.max(Q[new_state])
            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
            Q[state, action] = new_q
            
            # Update state and reward
            state = new_state
            total_episode_reward += reward
            
            # Break if done
            if done:
                break
                
        # Append episode reward to list
        total_rewards.append(total_episode_reward)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
    
    return Q, total_rewards
