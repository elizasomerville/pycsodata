"""Tests for the CSOCache class and cache management."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from pycsodata import CacheInfo, CSOCache
from pycsodata.fetchers import fetch_json, flush_cache


class TestCacheInfo:
    """Tests for the CacheInfo dataclass."""

    def test_cacheinfo_is_frozen(self):
        """Test that CacheInfo is immutable."""
        info = CacheInfo(size=5, maxsize=256, ttl_seconds=86400.0, hit_rate=0.5)
        with pytest.raises(AttributeError):
            info.size = 10  # type: ignore[misc]

    def test_cacheinfo_repr_with_hit_rate(self):
        """Test CacheInfo repr with a hit rate."""
        info = CacheInfo(size=5, maxsize=256, ttl_seconds=86400.0, hit_rate=0.75)
        repr_str = repr(info)
        assert "size=5" in repr_str
        assert "maxsize=256" in repr_str
        assert "75.0%" in repr_str

    def test_cacheinfo_repr_without_hit_rate(self):
        """Test CacheInfo repr when hit rate is None."""
        info = CacheInfo(size=0, maxsize=256, ttl_seconds=86400.0, hit_rate=None)
        repr_str = repr(info)
        assert "N/A" in repr_str

    def test_cacheinfo_equality(self):
        """Test CacheInfo equality comparison."""
        info1 = CacheInfo(size=5, maxsize=256, ttl_seconds=86400.0, hit_rate=0.5)
        info2 = CacheInfo(size=5, maxsize=256, ttl_seconds=86400.0, hit_rate=0.5)
        assert info1 == info2

    def test_cacheinfo_attributes(self):
        """Test CacheInfo attribute access."""
        info = CacheInfo(size=10, maxsize=512, ttl_seconds=3600.0, hit_rate=0.25)
        assert info.size == 10
        assert info.maxsize == 512
        assert info.ttl_seconds == 3600.0
        assert info.hit_rate == 0.25


class TestCSOCache:
    """Tests for the CSOCache class."""

    def test_csocache_instantiation(self):
        """Test that CSOCache can be instantiated."""
        cache = CSOCache()
        assert cache is not None

    def test_info_returns_cacheinfo(self):
        """Test that info() returns a CacheInfo object."""
        cache = CSOCache()
        flush_cache()  # Start fresh
        info = cache.info()
        assert isinstance(info, CacheInfo)

    def test_info_contains_expected_fields(self):
        """Test that info() contains all expected fields."""
        cache = CSOCache()
        flush_cache()
        info = cache.info()
        assert hasattr(info, "size")
        assert hasattr(info, "maxsize")
        assert hasattr(info, "ttl_seconds")
        assert hasattr(info, "hit_rate")

    def test_info_valid_values(self):
        """Test that info() returns valid values."""
        cache = CSOCache()
        flush_cache()
        info = cache.info()
        assert info.size >= 0
        assert info.maxsize > 0
        assert info.ttl_seconds > 0
        # After flush, hit_rate should be None (no requests)
        assert info.hit_rate is None

    def test_flush_clears_cache(self):
        """Test that flush() clears the cache."""
        cache = CSOCache()
        cache.flush()
        info = cache.info()
        assert info.size == 0

    def test_flush_resets_hit_rate(self):
        """Test that flush() resets hit rate statistics."""
        cache = CSOCache()
        cache.flush()
        info = cache.info()
        assert info.hit_rate is None

    def test_repr(self):
        """Test CSOCache repr."""
        cache = CSOCache()
        cache.flush()
        repr_str = repr(cache)
        assert "CSOCache" in repr_str
        assert "size=" in repr_str
        assert "maxsize=" in repr_str

    def test_multiple_instances_share_cache(self):
        """Test that multiple CSOCache instances share the same underlying cache."""
        cache1 = CSOCache()
        cache2 = CSOCache()

        cache1.flush()
        info1 = cache1.info()
        info2 = cache2.info()

        # Both should show empty cache after flush
        assert info1.size == info2.size == 0

    @patch("pycsodata.fetchers._fetch_json_impl")
    def test_cache_hit_rate_tracking(self, mock_impl):
        """Test that hit rate is tracked correctly."""
        mock_impl.return_value = {"data": "test"}
        cache = CSOCache()
        cache.flush()

        # First fetch is a miss
        fetch_json("http://example.com/test1")
        info = cache.info()
        assert info.hit_rate == 0.0  # 0 hits, 1 miss

        # Second fetch of same URL is a hit
        fetch_json("http://example.com/test1")
        info = cache.info()
        assert info.hit_rate == 0.5  # 1 hit, 1 miss

        # Third fetch of same URL is another hit
        fetch_json("http://example.com/test1")
        info = cache.info()
        assert info.hit_rate == pytest.approx(0.666, rel=0.01)  # 2 hits, 1 miss


class TestCSOCacheThreadSafety:
    """Tests for CSOCache thread safety."""

    def test_flush_thread_safe(self):
        """Test that flush() is thread safe."""
        cache = CSOCache()
        errors = []

        def flush_repeatedly():
            try:
                for _ in range(50):
                    cache.flush()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=flush_repeatedly) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_info_thread_safe(self):
        """Test that info() is thread safe."""
        cache = CSOCache()
        results = []
        errors = []

        def get_info_repeatedly():
            try:
                for _ in range(50):
                    results.append(cache.info())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_info_repeatedly) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 250  # 5 threads * 50 iterations

    def test_concurrent_flush_and_info(self):
        """Test concurrent flush and info calls."""
        cache = CSOCache()
        errors = []

        def flush_loop():
            try:
                for _ in range(30):
                    cache.flush()
            except Exception as e:
                errors.append(e)

        def info_loop():
            try:
                for _ in range(30):
                    cache.info()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=flush_loop),
            threading.Thread(target=info_loop),
            threading.Thread(target=flush_loop),
            threading.Thread(target=info_loop),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCSOCacheIntegration:
    """Integration tests for CSOCache with other classes."""

    @patch("pycsodata.fetchers._fetch_json_impl")
    def test_cache_populated_by_fetch(self, mock_impl):
        """Test that cache is populated by fetch_json calls."""
        mock_impl.return_value = {"data": "test"}
        cache = CSOCache()
        cache.flush()

        assert cache.info().size == 0

        # Make a cached request
        fetch_json("http://example.com/data")
        assert cache.info().size == 1

        # Make another cached request
        fetch_json("http://example.com/data2")
        assert cache.info().size == 2

    @patch("pycsodata.fetchers._fetch_json_impl")
    def test_flush_forces_refetch(self, mock_impl):
        """Test that flush forces data to be refetched."""
        mock_impl.return_value = {"data": "test"}
        cache = CSOCache()
        cache.flush()

        # First fetch
        fetch_json("http://example.com/data")
        assert mock_impl.call_count == 1

        # Second fetch from cache
        fetch_json("http://example.com/data")
        assert mock_impl.call_count == 1  # Still 1

        # Flush and fetch again
        cache.flush()
        fetch_json("http://example.com/data")
        assert mock_impl.call_count == 2  # Now 2


class TestCacheInfoImport:
    """Tests for CacheInfo import accessibility."""

    def test_cacheinfo_importable_from_package(self):
        """Test that CacheInfo can be imported from pycsodata."""
        from pycsodata import CacheInfo

        assert CacheInfo is not None

    def test_cacheinfo_importable_from_cache_module(self):
        """Test that CacheInfo can be imported from cache module."""
        from pycsodata.cache import CacheInfo

        assert CacheInfo is not None

    def test_both_imports_are_same_class(self):
        """Test that both import paths give the same class."""
        from pycsodata import CacheInfo as CacheInfoPkg
        from pycsodata.cache import CacheInfo as CacheInfoMod

        assert CacheInfoPkg is CacheInfoMod
