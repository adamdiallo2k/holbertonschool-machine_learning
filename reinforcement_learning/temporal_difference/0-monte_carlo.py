#!/usr/bin/env python3
"""
    Monte Carlo algorithm
"""
import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
        Implémente l'algorithme Monte Carlo

    :param env: Instance de l'environnement OpenAI Gym
    :param V: ndarray, valeur estimée des états (shape: (s,))
    :param policy: Fonction qui prend un état et retourne l'action à effectuer
    :param episodes: Nombre d'épisodes d'entraînement
    :param max_steps: Nombre maximal d'étapes par épisode
    :param alpha: Taux d'apprentissage
    :param gamma: Facteur de réduction (discount)

    :return: V mis à jour
    """
    for _ in range(episodes):
        # Réinitialisation de l'environnement
        state, _ = env.reset()  # ✅ Correction ici

        episode_data = []  # Stockage des (state, reward)

        for _ in range(max_steps):
            action = policy(state)  # Appliquer la politique
            next_state, reward, done, _, _ = env.step(action)  # ✅ Correction ici
            episode_data.append((state, reward))  # Enregistrer (état, récompense)

            if done:  # ✅ Correction ici
                break

            state = next_state  # Mise à jour de l'état

        # Mettre à jour la valeur des états
        G = 0  # Retour cumulé
        visited_states = set()  # ✅ Correction ici (suivi des états déjà vus)

        for state, reward in reversed(episode_data):
            G = gamma * G + reward  # Calcul du retour actualisé

            # Première visite uniquement
            if state not in visited_states:
                visited_states.add(state)  # ✅ Ajout de l'état visité
                V[state] = V[state] + alpha * (G - V[state])  # Mise à jour de V

    return V
