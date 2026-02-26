"""Tests for the constants module."""

from pycsodata.constants import (
    CACHE_TTL_SECONDS,
    CSO_BASE_URL,
    DEFAULT_CRS,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    ID_COLUMN_SUFFIX,
    MET_EIREANN_SPATIAL_KEY,
    MET_EIREANN_TABLE_PREFIX,
    MISENCODED_CHARACTER_MAP,
    NATIONAL_AREA_CODE,
    NATIONAL_AREA_LABELS,
    ROI_GEOMETRY_URL,
    STATISTIC_LABELS,
    WEATHER_STATIONS,
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


class TestMetEireannConstants:
    """Tests for Met Éireann weather station constants."""

    def test_met_eireann_table_prefix(self):
        """Test that Met Éireann table prefix is correct."""
        assert MET_EIREANN_TABLE_PREFIX == "MTM"

    def test_met_eireann_spatial_key(self):
        """Test that Met Éireann spatial key is correct."""
        assert MET_EIREANN_SPATIAL_KEY == "Meteorological Weather Station"

    def test_weather_stations_is_nonempty_string(self):
        """Test that weather stations data is a non-empty string."""
        assert isinstance(WEATHER_STATIONS, str)
        assert len(WEATHER_STATIONS.strip()) > 0

    def test_weather_stations_has_header(self):
        """Test that weather stations CSV has expected header."""
        lines = WEATHER_STATIONS.strip().splitlines()
        header = lines[0].strip()
        assert "station_id" in header
        assert "Latitude" in header
        assert "Longitude" in header
        assert "Elevation" in header

    def test_weather_stations_has_data_rows(self):
        """Test that weather stations CSV has data rows."""
        lines = WEATHER_STATIONS.strip().splitlines()
        # Header + at least one data row
        assert len(lines) > 1

    def test_weather_stations_row_format(self):
        """Test that each weather station row has 4 comma-separated fields."""
        lines = WEATHER_STATIONS.strip().splitlines()
        for line in lines[1:]:  # Skip header
            fields = line.strip().split(",")
            assert len(fields) == 4, f"Expected 4 fields, got {len(fields)}: {line}"

    def test_weather_stations_names_in_title_case(self):
        """Test that weather station names are in Title Case."""
        lines = WEATHER_STATIONS.strip().splitlines()
        for line in lines[1:]:
            name = line.strip().split(",")[0]
            assert name == name.strip()
            assert name[0].isupper(), f"Station name not Title Case: {name}"

    def test_weather_stations_valid_coordinates(self):
        """Test that weather station coordinates are valid for Ireland."""
        lines = WEATHER_STATIONS.strip().splitlines()
        for line in lines[1:]:
            fields = line.strip().split(",")
            lat = float(fields[1])
            lon = float(fields[2])
            # Ireland approximate bounding box
            assert 51.0 <= lat <= 56.0, f"Latitude {lat} out of range for {fields[0]}"
            assert -11.0 <= lon <= -5.0, f"Longitude {lon} out of range for {fields[0]}"

    def test_weather_stations_known_stations_present(self):
        """Test that well-known weather stations are present."""
        station_names = set()
        lines = WEATHER_STATIONS.strip().splitlines()
        for line in lines[1:]:
            station_names.add(line.strip().split(",")[0])

        expected = {"Dublin Airport", "Cork Airport", "Shannon Airport", "Valentia Observatory"}
        assert expected.issubset(station_names)
