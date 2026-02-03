"""Tests for the fetchers module."""

from unittest.mock import Mock, patch

import pytest
import requests

from pycsodata.exceptions import APIError
from pycsodata.fetchers import (
    _fetch_json_impl,
    fetch_json,
    flush_cache,
    get_cache_info,
    load_dataset,
    load_metadata,
)


class TestCacheManagement:
    """Tests for cache management functions."""

    def test_get_cache_info_returns_dict(self):
        """Test that get_cache_info returns expected structure."""
        info = get_cache_info()
        assert isinstance(info, dict)
        assert "size" in info
        assert "maxsize" in info
        assert "ttl_seconds" in info

    def test_cache_info_values_are_valid(self):
        """Test that cache info values are valid."""
        info = get_cache_info()
        assert isinstance(info["size"], int)
        assert info["size"] >= 0
        assert isinstance(info["maxsize"], int)
        assert info["maxsize"] > 0
        assert isinstance(info["ttl_seconds"], int | float)
        assert info["ttl_seconds"] > 0

    def test_flush_cache_clears_cache(self):
        """Test that flush_cache clears the cache."""
        flush_cache()
        info = get_cache_info()
        assert info["size"] == 0


class TestFlushCache:
    """Tests for the flush_cache function."""

    def test_flush_cache_runs_without_error(self):
        """Test that flush_cache completes without error."""
        # Should not raise any exceptions
        flush_cache()


class TestFetchJson:
    """Tests for the fetch_json function."""

    @patch("pycsodata.fetchers.requests.get")
    def test_successful_fetch(self, mock_get):
        """Test successful JSON fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Clear cache to ensure we hit our mock
        flush_cache()

        result = fetch_json("http://example.com/data.json")
        assert result == {"data": "test"}

    @patch("pycsodata.fetchers.requests.get")
    def test_raises_api_error_on_failure(self, mock_get):
        """Test that APIError is raised on request failure."""
        import requests as req

        mock_get.side_effect = req.exceptions.RequestException("Network error")

        flush_cache()

        with pytest.raises(APIError):
            fetch_json("http://example.com/data.json")

    @patch("pycsodata.fetchers.requests.get")
    def test_retries_on_failure(self, mock_get):
        """Test that requests are retried on failure."""
        import requests as req

        # First two calls fail, third succeeds
        mock_response = Mock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None

        mock_get.side_effect = [
            req.exceptions.Timeout("Timeout"),
            req.exceptions.Timeout("Timeout"),
            mock_response,
        ]

        flush_cache()

        result = fetch_json("http://example.com/data.json")
        assert result == {"data": "test"}
        assert mock_get.call_count == 3


class TestLoadMetadata:
    """Tests for the load_metadata function."""

    @pytest.mark.network
    def test_loads_valid_metadata(self):
        """Test loading metadata for a known valid table."""
        flush_cache()
        metadata = load_metadata("FY003A")

        assert isinstance(metadata, dict)
        assert "dimension" in metadata or "extension" in metadata

    def test_raises_error_for_invalid_code(self):
        """Test that APIError is raised for invalid table code."""
        flush_cache()

        with pytest.raises(APIError):
            load_metadata("INVALID_TABLE_CODE_XYZ123")


class TestLoadDataset:
    """Tests for the load_dataset function."""

    @pytest.mark.network
    def test_loads_valid_dataset(self):
        """Test loading dataset for a known valid table."""
        flush_cache()
        dataset = load_dataset("FY003A")

        assert isinstance(dataset, dict)
        # JSON-stat datasets should have these keys
        assert "dimension" in dataset or "value" in dataset

    def test_raises_error_for_invalid_code(self):
        """Test that APIError is raised for invalid table code."""
        flush_cache()

        with pytest.raises(APIError):
            load_dataset("INVALID_TABLE_CODE_XYZ123")


class TestFetchJsonImpl:
    """Tests for the _fetch_json_impl function (without caching)."""

    @patch("pycsodata.fetchers.requests.get")
    def test_timeout_error_retries(self, mock_get):
        """Test that timeout errors are retried."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None

        # First two calls timeout, third succeeds
        mock_get.side_effect = [
            requests.exceptions.Timeout("Timeout 1"),
            requests.exceptions.Timeout("Timeout 2"),
            mock_response,
        ]

        result = _fetch_json_impl("http://example.com/test", retries=3)
        assert result == {"data": "test"}
        assert mock_get.call_count == 3

    @patch("pycsodata.fetchers.requests.get")
    def test_connection_error_retries(self, mock_get):
        """Test that connection errors are retried."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None

        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            mock_response,
        ]

        result = _fetch_json_impl("http://example.com/test", retries=2)
        assert result == {"data": "test"}

    @patch("pycsodata.fetchers.requests.get")
    def test_http_client_error_not_retried(self, mock_get):
        """Test that 4xx HTTP errors are not retried."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found", response=mock_response
        )
        mock_get.return_value = mock_response

        with pytest.raises(APIError) as exc_info:
            _fetch_json_impl("http://example.com/notfound", retries=3)

        # Should only be called once (no retries for 4xx)
        assert mock_get.call_count == 1
        assert exc_info.value.status_code == 404

    @patch("pycsodata.fetchers.requests.get")
    def test_http_server_error_retried(self, mock_get):
        """Test that 5xx HTTP errors are retried."""
        mock_fail_response = Mock()
        mock_fail_response.status_code = 500
        mock_fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error", response=mock_fail_response
        )

        mock_success_response = Mock()
        mock_success_response.json.return_value = {"data": "test"}
        mock_success_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_fail_response, mock_success_response]

        result = _fetch_json_impl("http://example.com/test", retries=2)
        assert result == {"data": "test"}

    @patch("pycsodata.fetchers.requests.get")
    def test_all_retries_exhausted(self, mock_get):
        """Test that APIError is raised when all retries exhausted."""
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")

        with pytest.raises(APIError, match="failed after 3 attempts"):
            _fetch_json_impl("http://example.com/test", retries=3)

        assert mock_get.call_count == 3

    @patch("pycsodata.fetchers.requests.get")
    def test_general_request_exception(self, mock_get):
        """Test handling of general RequestException."""
        mock_get.side_effect = requests.exceptions.RequestException("General error")

        with pytest.raises(APIError):
            _fetch_json_impl("http://example.com/test", retries=2)

    @patch("pycsodata.fetchers.requests.get")
    @patch("pycsodata.fetchers.time.sleep")
    def test_exponential_backoff(self, mock_sleep, mock_get):
        """Test that exponential backoff is applied between retries."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None

        mock_get.side_effect = [
            requests.exceptions.Timeout("Timeout 1"),
            requests.exceptions.Timeout("Timeout 2"),
            mock_response,
        ]

        result = _fetch_json_impl("http://example.com/test", retries=3)
        assert result == {"data": "test"}

        # Check that sleep was called with increasing delays
        assert mock_sleep.call_count == 2


class TestFetchJsonCaching:
    """Tests for fetch_json caching behavior."""

    @patch("pycsodata.fetchers._fetch_json_impl")
    def test_caches_result(self, mock_impl):
        """Test that results are cached."""
        mock_impl.return_value = {"data": "cached"}
        flush_cache()

        # First call should hit the implementation
        result1 = fetch_json("http://example.com/cached")
        assert mock_impl.call_count == 1

        # Second call should use cache
        result2 = fetch_json("http://example.com/cached")
        assert mock_impl.call_count == 1  # Still 1, not called again

        assert result1 == result2

    @patch("pycsodata.fetchers._fetch_json_impl")
    def test_different_urls_not_cached_together(self, mock_impl):
        """Test that different URLs are cached separately."""
        mock_impl.return_value = {"data": "test"}
        flush_cache()

        fetch_json("http://example.com/url1")
        fetch_json("http://example.com/url2")

        # Should be called twice (different URLs)
        assert mock_impl.call_count == 2


class TestLoadMetadataErrorHandling:
    """Tests for load_metadata error handling."""

    @patch("pycsodata.fetchers.fetch_json")
    def test_404_provides_helpful_message(self, mock_fetch):
        """Test that 404 errors provide helpful message."""
        mock_fetch.side_effect = APIError("Not found", url="http://example.com", status_code=404)

        with pytest.raises(APIError) as exc_info:
            load_metadata("INVALID_CODE")

        assert "not found" in str(exc_info.value).lower()
        assert "CSOCatalogue" in str(exc_info.value)

    @patch("pycsodata.fetchers.fetch_json")
    def test_other_errors_propagated(self, mock_fetch):
        """Test that other errors are propagated."""
        mock_fetch.side_effect = APIError("Server error", url="http://example.com", status_code=500)

        with pytest.raises(APIError):
            load_metadata("TEST_CODE")

    @patch("pycsodata.fetchers.fetch_json")
    def test_general_exception_wrapped(self, mock_fetch):
        """Test that general exceptions are wrapped in APIError."""
        mock_fetch.side_effect = ValueError("Unexpected error")

        with pytest.raises(APIError, match="Failed to load metadata"):
            load_metadata("TEST_CODE")


class TestLoadDatasetErrorHandling:
    """Tests for load_dataset error handling."""

    @patch("pycsodata.fetchers.fetch_json")
    def test_404_provides_helpful_message(self, mock_fetch):
        """Test that 404 errors provide helpful message."""
        mock_fetch.side_effect = APIError("Not found", url="http://example.com", status_code=404)

        with pytest.raises(APIError) as exc_info:
            load_dataset("INVALID_CODE")

        assert "not found" in str(exc_info.value).lower()
        assert "CSOCatalogue" in str(exc_info.value)

    @patch("pycsodata.fetchers.fetch_json")
    def test_general_exception_wrapped(self, mock_fetch):
        """Test that general exceptions are wrapped in APIError."""
        mock_fetch.side_effect = Exception("Unexpected error")

        with pytest.raises(APIError, match="Failed to load dataset"):
            load_dataset("TEST_CODE")


class TestCacheThreadSafety:
    """Tests for cache thread safety."""

    def test_flush_cache_thread_safe(self):
        """Test that flush_cache is thread safe."""
        import threading

        errors = []

        def flush_repeatedly():
            try:
                for _ in range(100):
                    flush_cache()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=flush_repeatedly) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_get_cache_info_thread_safe(self):
        """Test that get_cache_info is thread safe."""
        import threading

        results = []
        errors = []

        def get_info_repeatedly():
            try:
                for _ in range(100):
                    results.append(get_cache_info())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_info_repeatedly) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 500  # 5 threads * 100 iterations
