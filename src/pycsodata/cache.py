"""Cache management for pycsodata.

This module provides the CSOCache class for managing the HTTP response cache
used by the package. It provides a clean, object-oriented interface for
inspecting and controlling cache behaviour.

Examples:
    >>> from pycsodata import CSOCache
    >>> cache = CSOCache()
    >>> cache.info()
    {'size': 5, 'maxsize': 256, 'ttl_seconds': 86400, 'hit_rate': 0.75}
    >>> cache.flush()
"""

from __future__ import annotations

from dataclasses import dataclass

from pycsodata import fetchers as _fetchers


@dataclass(frozen=True)
class CacheInfo:
    """Information about the current cache state.

    Attributes:
        size: Current number of cached responses.
        maxsize: Maximum cache capacity.
        ttl_seconds: Time-to-live for cached entries in seconds.
        hit_rate: Ratio of cache hits to total requests (0.0-1.0), or None if
            no requests have been made.
    """

    size: int
    maxsize: int
    ttl_seconds: float
    hit_rate: float | None

    def __repr__(self) -> str:
        """Return a string representation of the cache info."""
        hit_rate_str = f"{self.hit_rate:.1%}" if self.hit_rate is not None else "N/A"
        return (
            f"CacheInfo(size={self.size}, maxsize={self.maxsize}, "
            f"ttl_seconds={self.ttl_seconds}, hit_rate={hit_rate_str})"
        )


class CSOCache:
    """Manage the HTTP response cache for CSO API requests.

    This class provides a convenient interface for inspecting and controlling
    the cache used by pycsodata. The cache stores HTTP responses to reduce
    API calls and improve performance.

    The cache is shared across all instances of CSOCache, CSODataset, and
    CSOCatalogue. Operations on one instance affect all users of the cache.

    Methods:
        info: Get information about the current cache state.
        flush: Clear all cached responses.

    Examples:
        >>> from pycsodata import CSOCache
        >>> cache = CSOCache()
        >>>
        >>> # Check cache statistics
        >>> info = cache.info()
        >>> print(f"Cache contains {info.size} entries")
        >>>
        >>> # Clear the cache to force fresh API requests
        >>> cache.flush()
        >>> print("Cache cleared")
    """

    def info(self) -> CacheInfo:
        """Get information about the current cache state.

        Returns:
            A CacheInfo object containing:
                - size: Current number of cached responses.
                - maxsize: Maximum cache capacity.
                - ttl_seconds: Time-to-live for cached entries in seconds.
                - hit_rate: Ratio of cache hits to total requests, or None.

        Examples:
            >>> cache = CSOCache()
            >>> info = cache.info()
            >>> print(f"Cache is {info.size / info.maxsize:.0%} full")
            >>> if info.hit_rate:
            ...     print(f"Hit rate: {info.hit_rate:.1%}")
        """
        raw_info = _fetchers.get_cache_info()
        return CacheInfo(
            size=raw_info["size"],
            maxsize=raw_info["maxsize"],
            ttl_seconds=raw_info["ttl_seconds"],
            hit_rate=raw_info.get("hit_rate"),
        )

    def flush(self) -> None:
        """Clear all cached HTTP responses.

        This forces subsequent API calls to fetch fresh data from the CSO.
        Useful when you know data has been updated or during testing.

        Note:
            This affects all CSODataset and CSOCatalogue instances, as they
            share the same cache.

        Examples:
            >>> cache = CSOCache()
            >>> cache.flush()  # Clear all cached data
            >>> # Now all API calls will fetch fresh data
        """
        _fetchers.flush_cache()

    def __repr__(self) -> str:
        """Return a string representation of the cache."""
        info = self.info()
        return f"CSOCache(size={info.size}, maxsize={info.maxsize})"
