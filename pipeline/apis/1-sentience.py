#!/usr/bin/env python3
"""
Fetches and returns the list of home planets for all sentient species
from the SWAPI (https://swapi.dev/api/species). A species is considered
sentient if 'sentient' is in either its classification or designation.
"""


import requests


def sentientPlanets():
    """
    Returns:
        A list of planet names (strings) for all sentient species.
        If a homeworld is not listed, 'unknown' is used.
    """
    url = "https://swapi.dev/api/species/"
    planets = []

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data.get('results', []):
            classification = species.get('classification', '').lower()
            designation = species.get('designation', '').lower()

            if 'sentient' in classification or 'sentient' in designation:
                homeworld_url = species.get('homeworld')
                if homeworld_url:
                    try:
                        world_resp = requests.get(homeworld_url).json()
                        planets.append(world_resp.get('name', 'unknown'))
                    except Exception:
                        planets.append('unknown')
                else:
                    planets.append('unknown')

        url = data.get('next')

    return planets
