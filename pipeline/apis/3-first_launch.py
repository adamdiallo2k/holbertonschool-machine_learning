#!/usr/bin/env python3
"""
Displays the first upcoming SpaceX launch in local time, along with
its rocket name and launchpad (name + locality).
"""

import requests


if __name__ == '__main__':
    # 1. Fetch upcoming launches
    url_upcoming = "https://api.spacexdata.com/v4/launches/upcoming"
    upcoming_resp = requests.get(url_upcoming)
    upcoming_launches = upcoming_resp.json()

    # 2. Sort launches by date_unix to get the earliest upcoming launch
    # Python's sort is stable, so if two launches have the same date_unix,
    # the original order from the API is preserved.
    upcoming_sorted = sorted(upcoming_launches, key=lambda x: x['date_unix'])
    first_launch = upcoming_sorted[0]

    # 3. Get rocket info
    rocket_id = first_launch['rocket']
    url_rocket = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_resp = requests.get(url_rocket).json()
    rocket_name = rocket_resp['name']

    # 4. Get launchpad info
    launchpad_id = first_launch['launchpad']
    url_launchpad = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_resp = requests.get(url_launchpad).json()
    launchpad_name = launchpad_resp['name']
    launchpad_locality = launchpad_resp['locality']

    # 5. Get the launch name and local time (SpaceX provides "date_local")
    launch_name = first_launch['name']
    date_local = first_launch['date_local']  # e.g. '2022-10-08T19:05:00-04:00'

    # 6. Print result in the specified format
    # <launch name> (<date_local>) <rocket name> - <launchpad name> (<launchpad locality>)
    print(f"{launch_name} ({date_local}) {rocket_name} - {launchpad_name} ({launchpad_locality})")
