#!/usr/bin/env python3
"""
Returns a list of ships from the SWAPI (https://swapi.dev/api/starships)
that can hold at least a given number of passengers.
"""


import requests


def availableShips(passengerCount):
    """
    Fetch starships from the SWAPI and return those that can hold
    >= passengerCount passengers.

    Args:
        passengerCount (int): The minimum number of passengers required.

    Returns:
        list of str: List of starship names meeting the criteria.
    """
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for starship in data.get('results', []):
            raw_passengers = starship.get('passengers', '0')
            # Replace commas and strip extra spaces
            raw_passengers = raw_passengers.replace(',', '').strip()

            try:
                passenger_num = int(raw_passengers)
            except ValueError:
                passenger_num = 0

            if passenger_num >= passengerCount:
                ships.append(starship['name'])

        url = data.get('next')

    return ships
