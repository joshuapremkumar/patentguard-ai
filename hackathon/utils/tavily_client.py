"""
utils/tavily_client.py
=======================
Shared Tavily search wrapper used by patent_search and prior_art_search.
Centralises API key loading, endpoint constant, and error handling.
"""

from __future__ import annotations

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
TAVILY_ENDPOINT: str = "https://api.tavily.com/search"


def tavily_search(
    query: str,
    domains: list[str],
    max_results: int = 10,
) -> list[dict]:
    """
    Execute a Tavily search and return raw result dicts.

    Parameters
    ----------
    query : str
        Search query string (already formatted by the caller).
    domains : list[str]
        Domain whitelist to restrict Tavily results.
    max_results : int
        Maximum number of results to return (default 10).

    Returns
    -------
    list[dict]
        Raw Tavily result items, or an empty list on any error.

    Raises
    ------
    ValueError
        If ``query`` is empty or ``TAVILY_API_KEY`` is not set.
    """
    if not query.strip():
        raise ValueError("query must not be empty.")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is not set. Add it to your .env file.")

    payload: dict = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_domains": domains,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        resp = requests.post(TAVILY_ENDPOINT, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except requests.exceptions.HTTPError as exc:
        logger.error("Tavily HTTP error: %s", exc)
        return []
    except Exception as exc:
        logger.error("Tavily search failed: %s", exc)
        return []
