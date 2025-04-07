#!/usr/bin/env python3
"""Prints the location of a GitHub user given the full API URL."""

import sys
import requests
from datetime import datetime


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: ./2-user_location.py <GitHub_user_API_URL>")

    url = sys.argv[1]
    response = requests.get(url)

    # Check if we've hit the rate limit
    if response.status_code == 403:
        reset_timestamp = response.headers.get("X-RateLimit-Reset")
        if reset_timestamp is not None:
            # Convert from string to int, then to a datetime
            reset_timestamp = int(reset_timestamp)
            now_timestamp = int(datetime.utcnow().timestamp())
            diff_in_sec = reset_timestamp - now_timestamp
            # If negative (reset time has already passed), default to 0
            diff_in_min = diff_in_sec // 60 if diff_in_sec > 0 else 0
            print(f"Reset in {diff_in_min} min")
        else:
            # If no X-RateLimit-Reset header is given (unlikely), default to 0
            print("Reset in 0 min")

    elif response.status_code == 404:
        # User does not exist
        print("Not found")

    elif response.status_code == 200:
        # Successful request - parse the JSON
        user_info = response.json()
        # Print the location (could be None or empty string)
        location = user_info.get("location", "")
        print(location)

    else:
        # For any other status code, treat it as "Not found"
        print("Not found")
