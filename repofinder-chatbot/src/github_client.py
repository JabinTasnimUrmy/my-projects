# src/github_client.py
import requests
from config import GITHUB_TOKEN

API_URL = "https://api.github.com/search/repositories"

MAX_QUERY_LENGTH = 250

def search_repositories(query: str, count: int = 50):
    """
    Searches GitHub repositories for a given query, ensuring the query is not too long.
    """
    if not GITHUB_TOKEN:
        print("\n---FATAL ERROR---")
        print("GitHub API token is not configured. Please check your .env file.")
        return None

    # SAFETY NET ADDED HERE 
    # If the query is too long (usually from a failed AI fallback), truncate it.
    if len(query) > MAX_QUERY_LENGTH:
        print(f"\nWarning: The generated search query was too long and has been truncated.")
        query = query[:MAX_QUERY_LENGTH]

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": count
    }

    try:
        response = requests.get(API_URL, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get("items", [])
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred while communicating with the GitHub API: {e}")
        return None