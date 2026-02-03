"""HTTP fetching and caching for CSO API.

This module handles all HTTP communication with the CSO API, including
request caching, retry logic, and error handling.

Public Functions:
    fetch_json: Fetch JSON from a URL with caching and retries.
    load_metadata: Load dataset metadata from the CSO API.
    load_dataset: Load dataset data from the CSO API.
    flush_cache: Clear all cached HTTP responses.
    get_cache_info: Get information about the current cache state.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests
from cachetools import TTLCache

from pycsodata.constants import (
    CACHE_TTL_SECONDS,
    CSO_BASE_URL,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    RETRY_DELAY_MULTIPLIER,
)
from pycsodata.exceptions import APIError
from pycsodata.parsers import repair_json

# =============================================================================
# Cache Management
# =============================================================================

# Thread-safe TTL cache for HTTP responses
# maxsize=256 limits memory usage, ttl ensures data freshness
_http_cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=256, ttl=CACHE_TTL_SECONDS)
_cache_lock = threading.Lock()

# Cache statistics stored in a mutable container to avoid global statements
_cache_stats: dict[str, int] = {"hits": 0, "misses": 0}


def flush_cache() -> None:
    """Clear all cached HTTP responses.

    This forces subsequent requests to fetch fresh data from the API.
    Useful when you know data has been updated or during testing.

    Note:
        This also resets the cache hit/miss statistics.
    """
    with _cache_lock:
        _http_cache.clear()
        _cache_stats["hits"] = 0
        _cache_stats["misses"] = 0


def get_cache_info() -> dict[str, Any]:
    """Get information about the current cache state.

    Returns:
        A dictionary with cache statistics including size, maxsize, TTL,
        and hit rate.
    """
    with _cache_lock:
        total_requests = _cache_stats["hits"] + _cache_stats["misses"]
        hit_rate = _cache_stats["hits"] / total_requests if total_requests > 0 else None
        return {
            "size": len(_http_cache),
            "maxsize": _http_cache.maxsize,
            "ttl_seconds": _http_cache.ttl,
            "hit_rate": hit_rate,
        }


# =============================================================================
# Low-Level HTTP Functions
# =============================================================================


def _make_cache_key(url: str, params: dict[str, Any] | None) -> str:
    """Create a normalised cache key from URL and params.

    Args:
        url: The base URL.
        params: Optional query parameters.

    Returns:
        A string cache key that uniquely identifies the request.
    """
    if not params:
        return url
    # Sort params for consistent ordering and convert to tuple
    sorted_params = tuple(sorted((k, str(v)) for k, v in params.items()))
    return f"{url}?{sorted_params}"


def fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    cache: bool = True,
) -> dict[str, Any]:
    """Fetch JSON from a URL with automatic caching and retries.

    This is the main entry point for HTTP requests. It automatically
    caches responses (with TTL-based expiration) and retries failed requests.

    Args:
        url: The URL to fetch.
        params: Optional query parameters.
        cache: Whether to use caching. Defaults to True.

    Returns:
        The parsed JSON response as a dictionary.

    Raises:
        APIError: If the request fails after all retries.
    """
    cache_key = _make_cache_key(url, params)

    # Check cache first (thread-safe)
    if cache:
        with _cache_lock:
            if cache_key in _http_cache:
                _cache_stats["hits"] += 1
                return _http_cache[cache_key]

    # Fetch from network
    result = _fetch_json_impl(url, params=params)

    if cache:
        with _cache_lock:
            _cache_stats["misses"] += 1
            _http_cache[cache_key] = result

    return result


def _fetch_json_impl(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    """Fetch JSON with retry logic (not cached).

    Implements exponential backoff between retry attempts for transient
    failures. Client errors (4xx) are not retried.

    Args:
        url: The URL to fetch.
        params: Optional query parameters.
        timeout: Request timeout in seconds.
        retries: Number of retry attempts.

    Returns:
        The parsed JSON response as a dictionary.

    Raises:
        APIError: If all retry attempts fail.
    """
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except requests.exceptions.Timeout as e:
            last_error = e
        except requests.exceptions.ConnectionError as e:
            last_error = e
        except requests.exceptions.HTTPError as e:
            # Don't retry client errors (4xx)
            if e.response is not None and 400 <= e.response.status_code < 500:
                raise APIError(
                    f"Request to {url} failed: {e}",
                    url=url,
                    status_code=e.response.status_code,
                ) from e
            last_error = e
        except requests.exceptions.RequestException as e:
            last_error = e

        # Exponential backoff between retries
        if attempt < retries - 1:
            time.sleep(RETRY_DELAY_MULTIPLIER * (attempt + 1))

    raise APIError(
        f"Request to {url} failed after {retries} attempts: {last_error}",
        url=url,
    )


# =============================================================================
# CSO API-Specific Functions
# =============================================================================


def load_metadata(table_code: str, *, cache: bool = True) -> dict[str, Any]:
    """Load dataset metadata from the CSO RESTful API.

    Args:
        table_code: The CSO table code (e.g., 'FY003A').
        cache: Whether to use caching. Defaults to True.

    Returns:
        The metadata dictionary with encoding issues repaired.

    Raises:
        APIError: If the metadata cannot be loaded, including if the
            table_code does not correspond to a valid dataset.
    """
    url = f"{CSO_BASE_URL}.ReadMetadata/{table_code}/JSON-stat/2.0/en"
    try:
        data = fetch_json(url, cache=cache)
        return repair_json(data)
    except APIError as e:
        # Provide a more helpful error message for 404 errors (invalid table code)
        if e.status_code == 404:
            raise APIError(
                f"Dataset '{table_code}' not found. The table code does not correspond "
                f"to a valid CSO dataset. Please check the table code and try again. "
                f"You can use CSOCatalogue().search() to find available datasets.",
                url=url,
                status_code=404,
            ) from e
        raise
    except Exception as e:
        raise APIError(f"Failed to load metadata for '{table_code}': {e}") from e


def load_dataset(table_code: str, *, cache: bool = True) -> dict[str, Any]:
    """Load dataset data from the CSO RESTful API.

    Args:
        table_code: The CSO table code.
        cache: Whether to use caching. Defaults to True.

    Returns:
        The dataset JSON with encoding issues repaired.

    Raises:
        APIError: If the dataset cannot be loaded, including if the
            table_code does not correspond to a valid dataset.
    """
    url = f"{CSO_BASE_URL}.ReadDataset/{table_code}/JSON-stat/2.0/en"
    try:
        data = fetch_json(url, cache=cache)
        return repair_json(data)
    except APIError as e:
        # Provide a more helpful error message for 404 errors (invalid table code)
        if e.status_code == 404:
            raise APIError(
                f"Dataset '{table_code}' not found. The table code does not correspond "
                f"to a valid CSO dataset. Please check the table code and try again. "
                f"You can use CSOCatalogue().search() to find available datasets.",
                url=url,
                status_code=404,
            ) from e
        raise
    except Exception as e:
        raise APIError(f"Failed to load dataset '{table_code}': {e}") from e
