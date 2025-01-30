#!/usr/bin/env python3
"""
    Module pour charger l'environnement FrozenLake depuis OpenAI Gym.
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
        Charge l'environnement FrozenLakeEnv depuis OpenAI Gym.

        :param desc: None ou liste représentant une carte personnalisée.
        :param map_name: None ou str indiquant le nom d'une carte préexistante.
        :param is_slippery: bool, définit si la glace est glissante (True) ou non (False).

        :return: l'environnement Gym correspondant.
    """
    env = gym.make('FrozenLake-v0',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)

    return env
