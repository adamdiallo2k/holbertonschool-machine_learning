#!/usr/bin/env python3
"""
TD(λ) algorithm - Optimized version
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
            td_target = reward + (gamma * V[next_state] if not done else 0)
            td_error = td_target - V[state]

            # Mise à jour des traces d’éligibilité (Remplacement au lieu d'addition)
            eligibility_traces[state] = 1  # Remise à 1 au lieu d'ajouter 1

            # Mise à jour de la valeur des états
            V += alpha * td_error * eligibility_traces  # Correction pondérée par les traces
            
            # Atténuation exponentielle des traces
            eligibility_traces *= gamma * lambtha  

            if done:
                break  # Arrêter l'épisode si état terminal

            state = next_state  # Passer à l'état suivant

    return V
