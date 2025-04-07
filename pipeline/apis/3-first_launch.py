#!/usr/bin/env python3
"""
Displays the first upcoming SpaceX launch in local time, along with
its rocket name and launchpad (name + locality).
"""


import requests


if __name__ == '__main__':
    url_upcoming = "https://api.spacexdata.com/v4/launches/upcoming"
    resp_upcoming = requests.get(url_upcoming)
    upcoming_launches = resp_upcoming.json()

    # Sort by 'date_unix' to get the earliest upcoming launch
    upcoming_sorted = sorted(upcoming_launches, key=lambda x: x['date_unix'])
    first_launch = upcoming_sorted[0]

    # Fetch rocket info
    rocket_id = first_launch.get('rocket')
    url_rocket = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_resp = requests.get(url_rocket).json()
    rocket_name = rocket_resp.get('name')

    # Fetch launchpad info
    launchpad_id = first_launch.get('launchpad')
    url_launchpad = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_resp = requests.get(url_launchpad).json()
    launchpad_name = launchpad_resp.get('name')
    launchpad_locality = launchpad_resp.get('locality')

    # Gather launch details
    launch_name = first_launch.get('name')
    date_local = first_launch.get('date_local')

    # Print result with pycodestyle-compliant line lengths
    print(
        f"{launch_name} ({date_local}) {rocket_name} - "
        f"{launchpad_name} ({launchpad_locality})"
    )
