#!/usr/bin/env python3
"""
TD(λ) algorithm
"""
import numpy as np

def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, 
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm.

    Parameters:
        env: Environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimates.
        policy: Function that takes a state and returns the next action to take.
        lambtha: Eligibility trace factor (controls balance between MC & TD).
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        Updated value estimates V.
    """
    for _ in range(episodes):
        # Réinitialisation de l'environnement
        state, _ = env.reset()

        # Initialisation des traces d'éligibilité
        eligibility_traces = np.zeros_like(V)  

        for _ in range(max_steps):
            action = policy(state)  # Choix de l'action selon la politique
            next_state, reward, done, _, _ = env.step(action)  # Exécuter l'action

            # Calcul de l'erreur TD
            td_error = reward + gamma * V[next_state] * (not done) - V[state]

            # Mise à jour des traces d’éligibilité
            eligibility_traces[state] += 1  # Augmente l'importance de cet état

            # Mise à jour de la valeur des états
            V += alpha * td_error * eligibility_traces  # Correction pondérée par les traces
            eligibility_traces *= gamma * lambtha  # Atténuation exponentielle des traces

            if done:
                break  # Arrêter l'épisode si état terminal

            state = next_state  # Passer à l'état suivant

    return V
