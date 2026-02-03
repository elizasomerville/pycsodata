"""Tests for the _types module."""

import pytest

from pycsodata._types import (
    IncludeIDs,
    PivotFormat,
    SpatialInfo,
)


class TestIncludeIDsEnum:
    """Tests for the IncludeIDs enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert IncludeIDs.ALL.value == "all"
        assert IncludeIDs.SPATIAL_ONLY.value == "spatial_only"
        assert IncludeIDs.NONE.value == "none"

    def test_enum_is_string(self):
        """Test that enum inherits from str for comparison."""
        assert isinstance(IncludeIDs.ALL, str)
        assert IncludeIDs.ALL == "all"

    def test_enum_membership(self):
        """Test membership checking."""
        assert IncludeIDs.ALL in IncludeIDs
        assert "invalid" not in [e.value for e in IncludeIDs]


class TestPivotFormatEnum:
    """Tests for the PivotFormat enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert PivotFormat.LONG.value == "long"
        assert PivotFormat.WIDE.value == "wide"
        assert PivotFormat.TIDY.value == "tidy"

    def test_enum_is_string(self):
        """Test that enum inherits from str."""
        assert isinstance(PivotFormat.LONG, str)
        assert PivotFormat.WIDE == "wide"


class TestSpatialInfo:
    """Tests for the SpatialInfo dataclass."""

    def test_default_values(self):
        """Test that defaults are None."""
        info = SpatialInfo()
        assert info.url is None
        assert info.key is None

    def test_is_available_false_when_empty(self):
        """Test is_available is False when no data."""
        info = SpatialInfo()
        assert info.is_available is False

    def test_is_available_false_when_partial(self):
        """Test is_available is False with only URL."""
        info = SpatialInfo(url="http://example.com")
        assert info.is_available is False

    def test_is_available_true_when_complete(self):
        """Test is_available is True with both fields."""
        info = SpatialInfo(url="http://example.com", key="County")
        assert info.is_available is True

    def test_immutability(self):
        """Test that the dataclass is frozen."""
        info = SpatialInfo(url="http://example.com", key="County")
        with pytest.raises(AttributeError):
            info.url = "http://other.com"  # type: ignore
