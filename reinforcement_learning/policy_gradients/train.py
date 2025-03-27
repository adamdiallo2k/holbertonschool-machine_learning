#!/usr/bin/env python3
"""
Performs a basic REINFORCE (policy gradient) training procedure.
"""
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Trains an agent using REINFORCE.

    Args:
        env: the environment to train on
        nb_episodes: number of episodes for training
        alpha: the learning rate
        gamma: the discount factor

    Returns:
        scores: a list of total rewards per episode
    """
    # Weights for the policy – you should shape or initialize them according
    # to the dimensions your policy_gradient function expects. For example,
    # if the environment observation space is of dimension D (env.observation_space.shape[0])
    # and there are K discrete actions (env.action_space.n), you might need
    # a weight matrix (D x K). Here is a minimal example:
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    # Simple initialization of weights
    weights = np.random.rand(n_obs, n_actions) * 0.01

    scores = []

    for episode in range(nb_episodes):
        # Reset environment at the start of each episode
        state, _ = env.reset()
        done = False

        # To store (gradients, rewards) for each step in this episode
        grads = []
        rewards = []

        while not done:
            # Use the current policy to get an action and the gradient
            action, grad = policy_gradient(state, weights)
            # Perform the action in the environment
            next_state, reward, done, _, _ = env.step(action)

            # Track gradient and reward
            grads.append(grad)
            rewards.append(reward)

            # Move to the next state
            state = next_state

        # Compute total reward (score) for this episode
        episode_reward = sum(rewards)
        scores.append(episode_reward)

        # At the end of the episode, calculate the return for each time step
        # and update weights (REINFORCE update)
        for t in range(len(grads)):
            # Gt is the discounted return from step t onward
            Gt = 0
            discount = 1
            for k in range(t, len(rewards)):
                Gt += rewards[k] * discount
                discount *= gamma

            # We do a gradient ascent step with alpha * Gt * grad[t]
            weights += alpha * Gt * grads[t]

        # Print tracking info
        print("Episode: {} Score: {}".format(episode, episode_reward))

    return scores
