"""Tests for the public API exposed by __init__.py."""

from pycsodata import (
    APIError,
    CacheInfo,
    CSOCache,
    CSOCatalogue,
    CSODataset,
    DataError,
    IncludeIDs,
    IncludeIDsSpec,
    PivotFormat,
    SpatialError,
    ValidationError,
)


class TestPublicImports:
    """Test that all public API items are importable."""

    def test_csodataset_importable(self):
        """Test that CSODataset is importable."""
        assert CSODataset is not None

    def test_csocatalogue_importable(self):
        """Test that CSOCatalogue is importable."""
        assert CSOCatalogue is not None

    def test_csocache_importable(self):
        """Test that CSOCache is importable."""
        assert CSOCache is not None

    def test_cacheinfo_importable(self):
        """Test that CacheInfo is importable."""
        assert CacheInfo is not None

    def test_includeids_importable(self):
        """Test that IncludeIDs is importable."""
        assert IncludeIDs is not None
        assert IncludeIDs.ALL is not None
        assert IncludeIDs.SPATIAL_ONLY is not None
        assert IncludeIDs.NONE is not None

    def test_pivotformat_importable(self):
        """Test that PivotFormat is importable."""
        assert PivotFormat is not None
        assert PivotFormat.LONG is not None
        assert PivotFormat.WIDE is not None
        assert PivotFormat.TIDY is not None

    def test_exceptions_importable(self):
        """Test that exceptions are importable."""
        assert DataError is not None
        assert APIError is not None
        assert SpatialError is not None
        assert ValidationError is not None


class TestCSOCacheAPI:
    """Tests for the CSOCache API exposed at module level."""

    def test_csocache_flush(self):
        """Test that CSOCache.flush() works."""
        cache = CSOCache()
        cache.flush()
        info = cache.info()
        assert info.size == 0

    def test_csocache_info_returns_cacheinfo(self):
        """Test that CSOCache.info() returns CacheInfo."""
        cache = CSOCache()
        cache.flush()
        info = cache.info()
        assert isinstance(info, CacheInfo)

    def test_csocache_info_has_expected_attributes(self):
        """Test that CacheInfo has expected attributes."""
        cache = CSOCache()
        cache.flush()
        info = cache.info()
        assert hasattr(info, "size")
        assert hasattr(info, "maxsize")
        assert hasattr(info, "ttl_seconds")
        assert hasattr(info, "hit_rate")


class TestIncludeIDsSpec:
    """Tests for IncludeIDsSpec type alias."""

    def test_includeids_spec_is_exported(self):
        """Test that IncludeIDsSpec is exported."""
        assert IncludeIDsSpec is not None


class TestVersion:
    """Tests for version information."""

    def test_version_exists(self):
        """Test that __version__ is defined."""
        import pycsodata

        assert hasattr(pycsodata, "__version__")
        assert isinstance(pycsodata.__version__, str)

    def test_version_format(self):
        """Test that version follows semver format."""
        import pycsodata

        version = pycsodata.__version__

        parts = version.split(".")
        assert len(parts) >= 2
        # Major and minor should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()


class TestModuleAllExports:
    """Tests for __all__ exports."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        import pycsodata

        expected_exports = [
            "APIError",
            "CacheInfo",
            "CSOCache",
            "CSOCatalogue",
            "CSODataset",
            "DataError",
            "IncludeIDs",
            "IncludeIDsSpec",
            "PivotFormat",
            "SpatialError",
            "ValidationError",
        ]

        for export in expected_exports:
            assert hasattr(pycsodata, export), f"Missing export: {export}"

    def test_all_list_complete(self):
        """Test that __all__ list contains expected items."""
        import pycsodata

        assert hasattr(pycsodata, "__all__")
        assert "CSODataset" in pycsodata.__all__
        assert "CSOCatalogue" in pycsodata.__all__
        assert "CSOCache" in pycsodata.__all__
        assert "CacheInfo" in pycsodata.__all__


class TestModuleDocstring:
    """Tests for module docstring."""

    def test_module_has_docstring(self):
        """Test that module has a docstring."""
        import pycsodata

        assert pycsodata.__doc__ is not None
        assert len(pycsodata.__doc__) > 0

    def test_docstring_contains_example(self):
        """Test that docstring contains example usage."""
        import pycsodata

        assert pycsodata.__doc__ is not None
        assert "CSODataset" in pycsodata.__doc__
        assert "example" in pycsodata.__doc__.lower() or "Example" in pycsodata.__doc__
