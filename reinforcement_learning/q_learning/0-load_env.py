#!/usr/bin/env python3
"""
    Module pour charger l'environnement FrozenLake depuis OpenAI Gymnasium.
"""
import gymnasium as gym  # Mise à jour de l'importation pour Gymnasium


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
        Charge l'environnement FrozenLakeEnv depuis OpenAI Gymnasium.

        :param desc: None ou liste représentant une carte personnalisée.
        :param map_name: None ou str indiquant le nom d'une carte préexistante.
        :param is_slippery: bool, définit si la glace est glissante (True) ou non (False).

        :return: l'environnement Gymnasium correspondant.
    """
    env = gym.make('FrozenLake-v1',  # Gymnasium utilise 'FrozenLake-v1' au lieu de 'FrozenLake-v0'
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)

    return env
