#!/usr/bin/env python3
"""
Monte Carlo algorithm - Alternative Version
"""
import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm for estimating the value function.

    Parameters:
        env: Environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimates.
        policy: Function that takes a state and returns the next action
            to take.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        Updated value estimates V.
    """
    for _ in range(episodes):
        # Réinitialiser l'environnement et obtenir l'état initial
        state, _ = env.reset()
        episode_data = []  # Stocker la séquence (état, récompense)

        # Génération de l'épisode en suivant la politique
        for _ in range(max_steps):
            action = policy(state)  # Sélectionner une action
            next_state, reward, done, _, _ = env.step(action)  # Exécuter l'action
            episode_data.append((state, reward))  # Enregistrer l'état et la récompense

            if done:
                break  # Fin de l'épisode

            state = next_state  # Passer à l'état suivant

        # Mise à jour de la fonction de valeur V
        G = 0  # Retour cumulé
        visited_states = set()  # Suivi des états déjà mis à jour

        for state, reward in reversed(episode_data):
            G = gamma * G + reward  # Calcul du retour actualisé

            if state not in visited_states:  # Première visite uniquement
                visited_states.add(state)
                V[state] += alpha * (G - V[state])  # Mise à jour incrémentale de V

    return V
