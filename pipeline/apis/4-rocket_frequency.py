#!/usr/bin/env python3
"""
Displays the number of launches per rocket (SpaceX),
sorted by descending count, then by rocket name A-Z.
"""


import requests


if __name__ == '__main__':
    # 1. Fetch all launches
    launches_url = "https://api.spacexdata.com/v4/launches"
    launches_resp = requests.get(launches_url)
    launches_data = launches_resp.json()

    # 2. Count rocket occurrences
    rocket_count = {}
    for launch in launches_data:
        rocket_id = launch.get('rocket')
        if rocket_id:
            rocket_count[rocket_id] = rocket_count.get(rocket_id, 0) + 1

    # 3. Retrieve rocket names and store them in a dict
    #    so we can map rocket_id -> rocket_name
    #    We'll fetch all rockets once and use that data (instead of per ID).
    rockets_url = "https://api.spacexdata.com/v4/rockets"
    rockets_resp = requests.get(rockets_url)
    rockets_data = rockets_resp.json()

    rocket_id_to_name = {}
    for rocket in rockets_data:
        rocket_id_to_name[rocket['id']] = rocket.get('name', 'Unknown')

    # 4. Build a list of (rocket_name, count) tuples
    rocket_stats = []
    for rocket_id, count in rocket_count.items():
        rocket_name = rocket_id_to_name.get(rocket_id, 'Unknown')
        rocket_stats.append((rocket_name, count))

    # 5. Sort by -count (descending) and rocket_name (alphabetical for ties)
    rocket_stats.sort(key=lambda x: (-x[1], x[0]))

    # 6. Print the result
    for rocket_name, count in rocket_stats:
        print(f"{rocket_name}: {count}")
