"""Tests for the parsers module."""

import pandas as pd

from pycsodata._types import SpatialInfo
from pycsodata.parsers import (
    _build_tags,
    _extract_time_variable,
    _extract_units,
    _get_statistic_dimension,
    _process_notes,
    extract_id_mapping,
    extract_spatial_info,
    parse_metadata,
    parse_temporal_column,
    repair_json,
    repair_text,
)


class TestRepairText:
    """Tests for the repair_text function."""

    def test_repairs_single_character(self):
        """Test repairing a single misencoded character."""
        assert repair_text("┴ras") == "Áras"

    def test_repairs_multiple_characters(self):
        """Test repairing multiple misencoded characters."""
        assert repair_text("╔ire") == "Éire"
        assert repair_text("Θire") == "éire"

    def test_repairs_acute_accents(self):
        """Test all acute accent repairs."""
        assert repair_text("┴") == "Á"
        assert repair_text("ß") == "á"
        assert repair_text("╔") == "É"
        assert repair_text("Θ") == "é"
        assert repair_text("φ") == "í"
        assert repair_text("╙") == "Ó"
        assert repair_text("≤") == "ó"
        assert repair_text("·") == "ú"

    def test_preserves_correct_text(self):
        """Test that correctly encoded text is unchanged."""
        assert repair_text("Dublin") == "Dublin"
        assert repair_text("Áras") == "Áras"

    def test_handles_empty_string(self):
        """Test handling of empty string."""
        assert repair_text("") == ""

    def test_handles_mixed_content(self):
        """Test text with both correct and incorrect encoding."""
        assert repair_text("Baile ┴tha Cliath") == "Baile Átha Cliath"


class TestRepairJson:
    """Tests for the repair_json function."""

    def test_repairs_dict_values(self):
        """Test repairing values in a dictionary."""
        data = {"name": "╔ire", "code": "IE"}
        fixed = repair_json(data)
        assert fixed["name"] == "Éire"
        assert fixed["code"] == "IE"

    def test_repairs_list_items(self):
        """Test repairing items in a list."""
        data = ["╔ire", "Θire", "Dublin"]
        fixed = repair_json(data)
        assert fixed == ["Éire", "éire", "Dublin"]

    def test_repairs_nested_structures(self):
        """Test repairing nested dict/list structures."""
        data = {
            "name": "╔ire",
            "places": ["Θire", {"city": "Ceann·igh"}],
        }
        fixed = repair_json(data)
        assert fixed["name"] == "Éire"
        assert fixed["places"][0] == "éire"
        assert fixed["places"][1]["city"] == "Ceannúigh"

    def test_preserves_non_string_types(self):
        """Test that non-string types are unchanged."""
        data = {"count": 42, "active": True, "ratio": 3.14, "empty": None}
        fixed = repair_json(data)
        assert fixed == data

    def test_handles_empty_structures(self):
        """Test handling of empty structures."""
        assert repair_json({}) == {}
        assert repair_json([]) == []


class TestExtractSpatialInfo:
    """Tests for the extract_spatial_info function."""

    def test_extracts_spatial_info_when_present(self):
        """Test extraction when spatial data is present."""
        metadata = {
            "dimension": {
                "County": {
                    "label": "County",
                    "link": {"enclosure": [{"href": "http://example.com/counties.geojson"}]},
                }
            }
        }
        info = extract_spatial_info(metadata)
        assert isinstance(info, SpatialInfo)
        assert info.url == "http://example.com/counties.geojson"
        assert info.key == "County"
        assert info.is_available is True

    def test_returns_empty_when_no_spatial(self):
        """Test that empty SpatialInfo is returned when no spatial data."""
        metadata = {"dimension": {"Year": {"label": "Year"}}}
        info = extract_spatial_info(metadata)
        assert info.url is None
        assert info.key is None
        assert info.is_available is False

    def test_handles_missing_dimension_key(self):
        """Test handling of missing dimension key."""
        metadata = {}
        info = extract_spatial_info(metadata)
        assert info.is_available is False


class TestExtractIdMapping:
    """Tests for the extract_id_mapping function."""

    def test_creates_label_to_id_mapping(self):
        """Test creation of label-to-ID mapping."""
        dim_info = {
            "category": {
                "label": {
                    "IE061": "Dublin",
                    "IE062": "Cork",
                }
            }
        }
        mapping = extract_id_mapping(dim_info)
        assert mapping["Dublin"] == "IE061"
        assert mapping["Cork"] == "IE062"

    def test_returns_empty_when_no_category(self):
        """Test that empty dict is returned when no category."""
        dim_info = {}
        mapping = extract_id_mapping(dim_info)
        assert mapping == {}

    def test_returns_empty_when_no_label(self):
        """Test that empty dict is returned when no label."""
        dim_info = {"category": {}}
        mapping = extract_id_mapping(dim_info)
        assert mapping == {}


class TestParseMetadata:
    """Tests for the parse_metadata function."""

    def test_parses_basic_fields(self):
        """Test parsing of basic metadata fields."""
        raw_metadata = {
            "label": "Population Dataset",
            "extension": {
                "matrix": "FY003A",
                "official": True,
            },
            "dimension": {},
        }
        result = parse_metadata(raw_metadata)

        assert result["table_code"] == "FY003A"  # type: ignore
        assert result["title"] == "Population Dataset"  # type: ignore
        assert result["official"] is True  # type: ignore

    def test_parses_tags(self):
        """Test that tags are generated correctly."""
        raw_metadata = {
            "extension": {
                "experimental": True,
                "official": False,
            },
            "dimension": {},
        }
        result = parse_metadata(raw_metadata)
        assert "Experimental Statistics" in result["tags"]  # type: ignore
        assert "Official Statistics" not in result["tags"]  # type: ignore


class TestParseTemporalColumn:
    """Tests for the parse_temporal_column function."""

    def test_parses_year_format(self):
        """Test parsing of year-only format."""
        df = pd.DataFrame({"Year": ["2020", "2021", "2022"], "value": [1, 2, 3]})
        result = parse_temporal_column(df, "Year")
        assert result["Year"].tolist() == [2020, 2021, 2022]

    def test_handles_missing_time_variable(self):
        """Test handling when time variable is not in DataFrame."""
        df = pd.DataFrame({"County": ["Dublin", "Cork"], "value": [1, 2]})
        result = parse_temporal_column(df, "Year")
        assert "Year" not in result.columns

    def test_handles_none_time_variable(self):
        """Test handling of None time variable."""
        df = pd.DataFrame({"Year": ["2020", "2021"], "value": [1, 2]})
        result = parse_temporal_column(df, None)
        assert result["Year"].tolist() == ["2020", "2021"]  # Unchanged

    def test_handles_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = parse_temporal_column(df, "Year")
        assert result.empty


class TestParseTemporalColumnFormats:
    """Tests for various temporal column formats."""

    def test_parses_monthly_format_with_m(self):
        """Test parsing of monthly format (2022M01)."""
        df = pd.DataFrame({"Month": ["2022M01", "2022M02", "2022M03"], "value": [1, 2, 3]})
        result = parse_temporal_column(df, "Month")

        # Should be Period type with monthly frequency
        assert hasattr(result["Month"].iloc[0], "freqstr") or str(result["Month"].dtype).startswith(
            "period"
        )

    def test_parses_monthly_format_mixed(self):
        """Test parsing of monthly format with mixed format."""
        df = pd.DataFrame(
            {"Month": ["January 2022", "February 2022", "March 2022"], "value": [1, 2, 3]}
        )
        result = parse_temporal_column(df, "Month")

        # Should be parsed as periods
        assert len(result) == 3

    def test_parses_quarterly_format(self):
        """Test parsing of quarterly format."""
        df = pd.DataFrame({"Quarter": ["2022Q1", "2022Q2", "2022Q3"], "value": [1, 2, 3]})
        result = parse_temporal_column(df, "Quarter")

        # Should be Period type with quarterly frequency
        assert len(result) == 3

    def test_parses_weekly_format(self):
        """Test parsing of weekly format."""
        df = pd.DataFrame({"Week": ["2022-01-01", "2022-01-08", "2022-01-15"], "value": [1, 2, 3]})
        result = parse_temporal_column(df, "Week")

        # Should be date type
        assert len(result) == 3

    def test_skips_influenza_season(self):
        """Test that influenza season is skipped."""
        df = pd.DataFrame({"Influenza Season": ["2021/2022", "2022/2023"], "value": [1, 2]})
        result = parse_temporal_column(df, "Influenza Season")

        # Should be unchanged (strings)
        assert result["Influenza Season"].tolist() == ["2021/2022", "2022/2023"]

    def test_skips_academic_year(self):
        """Test that academic year is skipped."""
        df = pd.DataFrame({"Academic Year": ["2021/2022", "2022/2023"], "value": [1, 2]})
        result = parse_temporal_column(df, "Academic Year")

        # Should be unchanged
        assert result["Academic Year"].tolist() == ["2021/2022", "2022/2023"]

    def test_skips_halfyear(self):
        """Test that halfyear is skipped."""
        df = pd.DataFrame({"Halfyear Period": ["2022H1", "2022H2"], "value": [1, 2]})
        result = parse_temporal_column(df, "Halfyear Period")

        # Should be unchanged
        assert result["Halfyear Period"].tolist() == ["2022H1", "2022H2"]

    def test_parses_date_without_time(self):
        """Test parsing of date without time component."""
        df = pd.DataFrame({"Date": ["2022-01-15", "2022-02-15", "2022-03-15"], "value": [1, 2, 3]})
        result = parse_temporal_column(df, "Date")

        # Should be date type
        assert len(result) == 3

    def test_handles_none_dataframe(self):
        """Test handling of None DataFrame."""
        result = parse_temporal_column(None, "Year")  # type: ignore
        assert result is None


class TestGetStatisticDimension:
    """Tests for _get_statistic_dimension function."""

    def test_finds_statistic_dimension(self):
        """Test finding STATISTIC dimension."""
        dimensions = {
            "STATISTIC": {"label": "Statistic", "category": {"label": {"pop": "Population"}}},
            "Year": {"label": "Year"},
        }
        result = _get_statistic_dimension(dimensions)
        assert result.get("label") == "Statistic"

    def test_finds_statistic_with_different_case(self):
        """Test finding Statistic dimension (different case)."""
        dimensions = {
            "Statistic": {"label": "Statistic", "category": {"label": {"pop": "Population"}}},
            "Year": {"label": "Year"},
        }
        result = _get_statistic_dimension(dimensions)
        assert result.get("label") == "Statistic"

    def test_returns_empty_dict_when_not_found(self):
        """Test that empty dict is returned when not found."""
        dimensions = {"Year": {"label": "Year"}, "County": {"label": "County"}}
        result = _get_statistic_dimension(dimensions)
        assert result == {}

    def test_returns_empty_when_statistic_not_dict(self):
        """Test that empty dict is returned when STATISTIC is not a dict."""
        dimensions = {"STATISTIC": "not a dict", "Year": {"label": "Year"}}
        result = _get_statistic_dimension(dimensions)
        assert result == {}


class TestExtractUnits:
    """Tests for _extract_units function."""

    def test_extracts_units_from_category(self):
        """Test extracting units from category."""
        statistic_dim = {
            "category": {"unit": {"pop": {"label": "Number"}, "rate": {"label": "Percentage"}}}
        }
        result = _extract_units(statistic_dim)
        assert "Number" in result
        assert "Percentage" in result

    def test_returns_empty_when_no_category(self):
        """Test returns empty list when no category."""
        statistic_dim = {}
        result = _extract_units(statistic_dim)
        assert result == []

    def test_returns_empty_when_no_unit(self):
        """Test returns empty list when no unit."""
        statistic_dim = {"category": {}}
        result = _extract_units(statistic_dim)
        assert result == []

    def test_skips_entries_without_label(self):
        """Test that entries without label are skipped."""
        statistic_dim = {"category": {"unit": {"pop": {"label": "Number"}, "no_label": {}}}}
        result = _extract_units(statistic_dim)
        assert result == ["Number"]


class TestExtractTimeVariable:
    """Tests for _extract_time_variable function."""

    def test_extracts_time_variable(self):
        """Test extracting time variable from role."""
        metadata = {"role": {"time": ["TLIST(A1)"]}}
        dimensions = {"TLIST(A1)": {"label": "Year"}}
        result = _extract_time_variable(metadata, dimensions)
        assert result == "Year"

    def test_returns_none_when_no_time_role(self):
        """Test returns None when no time role."""
        metadata = {"role": {}}
        dimensions = {"Year": {"label": "Year"}}
        result = _extract_time_variable(metadata, dimensions)
        assert result is None

    def test_returns_none_when_no_role(self):
        """Test returns None when no role."""
        metadata = {}
        dimensions = {"Year": {"label": "Year"}}
        result = _extract_time_variable(metadata, dimensions)
        assert result is None

    def test_returns_none_when_time_dim_not_found(self):
        """Test returns None when time dimension not found."""
        metadata = {"role": {"time": ["MISSING"]}}
        dimensions = {"Year": {"label": "Year"}}
        result = _extract_time_variable(metadata, dimensions)
        assert result is None

    def test_handles_none_label(self):
        """Test handling of None label."""
        metadata = {"role": {"time": ["TLIST(A1)"]}}
        dimensions = {"TLIST(A1)": {"label": None}}
        result = _extract_time_variable(metadata, dimensions)
        assert result is None


class TestProcessNotes:
    """Tests for _process_notes function."""

    def test_removes_formatting_tags(self):
        """Test removal of [i] and [/i] tags."""
        notes = ["This is [i]italic[/i] text"]
        result = _process_notes(notes)
        assert result[0] == "This is italic text"

    def test_normalises_whitespace(self):
        """Test normalisation of whitespace."""
        notes = ["Multiple   spaces   here"]
        result = _process_notes(notes)
        assert "  " not in result[0]

    def test_replaces_newlines(self):
        """Test replacement of newlines."""
        notes = ["Line one\nLine two"]
        result = _process_notes(notes)
        assert "\n" not in result[0]

    def test_converts_url_tags(self):
        """Test conversion of [url=...] tags."""
        notes = ["See [url=http://example.com]this link[/url] for details"]
        result = _process_notes(notes)
        assert "this link" in result[0]
        assert "http://example.com" in result[0]

    def test_skips_empty_notes(self):
        """Test that empty notes are skipped."""
        notes = ["", "Valid note", None, "Another note"]
        result = _process_notes(notes)
        assert len(result) == 2
        assert "Valid note" in result
        assert "Another note" in result

    def test_handles_empty_list(self):
        """Test handling of empty list."""
        result = _process_notes([])
        assert result == []


class TestBuildTags:
    """Tests for _build_tags function."""

    def test_adds_experimental_tag(self):
        """Test adding Experimental Statistics tag."""
        extension = {"experimental": True}
        result = _build_tags(extension, False)
        assert "Experimental Statistics" in result

    def test_adds_reservation_tag(self):
        """Test adding Reservation Statistics tag."""
        extension = {"reservation": True}
        result = _build_tags(extension, False)
        assert "Reservation Statistics" in result

    def test_adds_archive_tag(self):
        """Test adding Archive Statistics tag."""
        extension = {"archive": True}
        result = _build_tags(extension, False)
        assert "Archive Statistics" in result

    def test_adds_analytical_tag(self):
        """Test adding Analytical Statistics tag."""
        extension = {"analytical": True}
        result = _build_tags(extension, False)
        assert "Analytical Statistics" in result

    def test_adds_official_tag(self):
        """Test adding Official Statistics tag."""
        extension = {"official": True}
        result = _build_tags(extension, False)
        assert "Official Statistics" in result

    def test_adds_geographic_tag(self):
        """Test adding Geographic Data tag when spatial available."""
        extension = {}
        result = _build_tags(extension, True)
        assert "Geographic Data" in result

    def test_multiple_tags(self):
        """Test adding multiple tags."""
        extension = {"experimental": True, "official": True}
        result = _build_tags(extension, True)
        assert "Experimental Statistics" in result
        assert "Official Statistics" in result
        assert "Geographic Data" in result

    def test_no_tags_when_all_false(self):
        """Test empty list when all flags are False."""
        extension = {"experimental": False, "official": False}
        result = _build_tags(extension, False)
        assert result == []


class TestParseMetadataEdgeCases:
    """Additional tests for parse_metadata function."""

    def test_parses_full_metadata(self):
        """Test parsing of complete metadata."""
        raw_metadata = {
            "label": "Test Dataset",
            "updated": "2023-06-15T10:00:00",
            "extension": {
                "matrix": "TEST01",
                "official": True,
                "experimental": False,
                "reasons": ["Monthly update"],
                "contact": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "phone": "+353 1 234 5678",
                },
                "copyright": {"name": "CSO", "href": "https://cso.ie"},
            },
            "dimension": {
                "STATISTIC": {
                    "label": "Statistic",
                    "category": {
                        "label": {"pop": "Population"},
                        "unit": {"pop": {"label": "Number"}},
                    },
                },
                "Year": {"label": "Year"},
            },
            "role": {"time": ["Year"]},
            "note": ["This is a test note"],
        }

        result = parse_metadata(raw_metadata)

        assert result.get("table_code") == "TEST01"
        assert result.get("title") == "Test Dataset"
        assert result.get("official") is True
        assert result.get("contact_name") == "John Doe"
        assert result.get("copyright_name") == "CSO"

    def test_handles_missing_extension(self):
        """Test handling of missing extension."""
        raw_metadata = {"dimension": {}}
        result = parse_metadata(raw_metadata)
        assert result.get("table_code") is None

    def test_handles_missing_updated(self):
        """Test handling of missing updated field."""
        raw_metadata = {"extension": {"matrix": "TEST01"}, "dimension": {}}
        result = parse_metadata(raw_metadata)
        assert result.get("last_updated") is None
