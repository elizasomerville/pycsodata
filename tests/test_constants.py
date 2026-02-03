"""Tests for the constants module."""

from pycsodata.constants import (
    CACHE_TTL_SECONDS,
    CSO_BASE_URL,
    DEFAULT_CRS,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    ID_COLUMN_SUFFIX,
    MISENCODED_CHARACTER_MAP,
    NATIONAL_AREA_CODE,
    NATIONAL_AREA_LABELS,
    ROI_GEOMETRY_URL,
    STATISTIC_LABELS,
)


class TestApiConstants:
    """Tests for API-related constants."""

    def test_cso_base_url_is_https(self):
        """Test that CSO base URL uses HTTPS."""
        assert CSO_BASE_URL.startswith("https://")

    def test_timeout_is_positive(self):
        """Test that timeout is a positive integer."""
        assert isinstance(DEFAULT_TIMEOUT, int)
        assert DEFAULT_TIMEOUT > 0

    def test_retries_is_positive(self):
        """Test that retries is a positive integer."""
        assert isinstance(DEFAULT_RETRIES, int)
        assert DEFAULT_RETRIES > 0

    def test_cache_ttl_is_24_hours(self):
        """Test that cache TTL is approximately 24 hours."""
        assert CACHE_TTL_SECONDS == 24 * 60 * 60


class TestSpatialConstants:
    """Tests for spatial-related constants."""

    def test_default_crs_is_wgs84(self):
        """Test that default CRS is WGS84."""
        assert DEFAULT_CRS == "EPSG:4326"

    def test_roi_geometry_url_is_valid(self):
        """Test that ROI geometry URL is a valid HTTPS URL."""
        assert ROI_GEOMETRY_URL.startswith("https://")
        assert "geojson" in ROI_GEOMETRY_URL.lower()


class TestDataConstants:
    """Tests for data processing constants."""

    def test_national_area_code(self):
        """Test national area code value."""
        assert NATIONAL_AREA_CODE == "IE0"

    def test_national_area_labels_contains_ireland(self):
        """Test that national labels include Ireland."""
        assert "Ireland" in NATIONAL_AREA_LABELS
        assert "State" in NATIONAL_AREA_LABELS

    def test_id_column_suffix(self):
        """Test ID column suffix."""
        assert ID_COLUMN_SUFFIX == " ID"

    def test_statistic_labels(self):
        """Test statistic dimension labels."""
        assert "Statistic" in STATISTIC_LABELS
        assert "STATISTIC" in STATISTIC_LABELS


class TestMisencodedCharacterMap:
    """Tests for the misencoded character map."""

    def test_map_has_entries(self):
        """Test that the map has entries."""
        assert len(MISENCODED_CHARACTER_MAP) > 0

    def test_map_contains_common_irish_characters(self):
        """Test that map covers common Irish fada characters."""
        # These are the target characters (correct Irish)
        target_chars = set(MISENCODED_CHARACTER_MAP.values())

        # Should include uppercase and lowercase á, é, í, ó, ú
        assert "Á" in target_chars
        assert "á" in target_chars
        assert "É" in target_chars
        assert "é" in target_chars
        assert "í" in target_chars
        assert "Ó" in target_chars
        assert "ó" in target_chars
        assert "ú" in target_chars

    def test_map_keys_are_single_characters(self):
        """Test that map keys are single characters."""
        for key in MISENCODED_CHARACTER_MAP:
            assert len(key) == 1

    def test_map_values_are_single_characters(self):
        """Test that map values are single characters."""
        for value in MISENCODED_CHARACTER_MAP.values():
            assert len(value) == 1
