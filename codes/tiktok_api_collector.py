"""
These code snippets are simplified examples showing how the TikTok Research API
can be used for academic purposes. They illustrate the basic authentication flow
and how to query public TikTok account information.

They do NOT represent the full data-collection pipeline used in the thesis.
To protect user privacy and API security, raw datasets, full scraping scripts,
usernames, video IDs, and API keys are not included in this repository.

API credentials must be provided via environment variables:
    TIKTOK_CLIENT_KEY
    TIKTOK_CLIENT_SECRET
"""


import os
import requests


# -------------------------------------------------
# 1. Get API credentials from environment variables
# -------------------------------------------------
CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY")
CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET")

if not CLIENT_KEY or not CLIENT_SECRET:
    raise ValueError("Please set TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET.")


# -------------------------------------------------
# 2. Generate an access token
# -------------------------------------------------
def get_access_token() -> str:
    url = "https://open.tiktokapis.com/v2/oauth/token/"
    payload = {
        "client_key": CLIENT_KEY,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials"
    }

    response = requests.post(url, data=payload)
    response.raise_for_status()
    return response.json()["access_token"]


# -------------------------------------------------
# 3. Query TikTok user information
# -------------------------------------------------
def get_user_profile(username: str, access_token: str):
    """
    Fetch public metadata for a TikTok user using the Research API.
    """

    url = (
        "https://open.tiktokapis.com/v2/research/user/info/"
        "?fields=display_name,follower_count,following_count,"
        "is_verified,avatar_url,region_code,username"
    )

    headers = {"Authorization": f"Bearer {access_token}"}

    payload = {
        "query": {
            "and": [
                {"field_name": "username", "operation": "EQ", "field_values": [username]}
            ]
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()
