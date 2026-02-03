"""Tests for the sanitise module."""

from pycsodata.sanitise import (
    create_reverse_mapping,
    create_sanitisation_mapping,
    sanitise_dict_keys,
    sanitise_dict_values,
    sanitise_list,
    sanitise_string,
)


class TestSanitiseString:
    """Tests for sanitise_string function."""

    def test_replace_ampersand(self):
        """Test that '&' is replaced with 'and'."""
        assert sanitise_string("Counties & Cities") == "County and City"
        assert sanitise_string("A & B") == "A and B"

    def test_replace_slash_with_spaces(self):
        """Test that ' / ' is replaced with '/'."""
        assert sanitise_string("A / B") == "A/B"
        assert sanitise_string("A /B") == "A/B"
        assert sanitise_string("A/ B") == "A/B"

    def test_replace_multiple_spaces(self):
        """Test that multiple spaces are replaced with single space."""
        assert sanitise_string("Multiple   spaces") == "Multiple spaces"
        assert sanitise_string("A    B     C") == "A B C"

    def test_strip_edge_whitespace(self):
        """Test that edge whitespace is stripped."""
        assert sanitise_string("  trimmed  ") == "trimmed"
        assert sanitise_string("\n\t trimmed \t\n") == "trimmed"

    def test_sanitisation_dict_mappings(self):
        """Test that SANITISATION_DICT mappings are applied."""
        # These are from constants.py
        assert sanitise_string("Counties") == "County"
        assert sanitise_string("CensusYear") == "Census Year"
        assert sanitise_string("NUTS 3 Regions") == "NUTS 3 Region"
        assert sanitise_string("Electoral Divisions") == "Electoral Division"

    def test_combined_transformations(self):
        """Test combined transformations."""
        assert sanitise_string("  Counties & Cities  ") == "County and City"
        assert sanitise_string("A  &  B / C") == "A and B/C"

    def test_non_string_passthrough(self):
        """Test that non-strings are passed through unchanged."""
        assert sanitise_string(123) == 123  # type: ignore
        assert sanitise_string(None) is None  # type: ignore


class TestSanitiseList:
    """Tests for sanitise_list function."""

    def test_sanitise_list_of_strings(self):
        """Test sanitising a list of strings."""
        result = sanitise_list(["Counties", "Electoral Divisions", "Normal"])
        assert result == ["County", "Electoral Division", "Normal"]

    def test_sanitise_empty_list(self):
        """Test sanitising an empty list."""
        assert sanitise_list([]) == []

    def test_mixed_types_in_list(self):
        """Test that non-strings in list are passed through."""
        result = sanitise_list(["Counties", 123, "Normal"])  # type: ignore
        assert result == ["County", 123, "Normal"]


class TestSanitiseDictKeys:
    """Tests for sanitise_dict_keys function."""

    def test_sanitise_keys(self):
        """Test sanitising dictionary keys."""
        result = sanitise_dict_keys({"Counties": 1, "Normal": 2})
        assert result == {"County": 1, "Normal": 2}

    def test_empty_dict(self):
        """Test sanitising empty dictionary."""
        assert sanitise_dict_keys({}) == {}


class TestSanitiseDictValues:
    """Tests for sanitise_dict_values function."""

    def test_sanitise_string_values(self):
        """Test sanitising string values in dictionary."""
        result = sanitise_dict_values({"key": "Counties"})
        assert result == {"key": "County"}

    def test_sanitise_list_values(self):
        """Test sanitising list values in dictionary."""
        result = sanitise_dict_values({"key": ["Counties", "Electoral Divisions"]})
        assert result == {"key": ["County", "Electoral Division"]}

    def test_non_string_values_passthrough(self):
        """Test that non-string values pass through."""
        result = sanitise_dict_values({"num": 123, "bool": True})
        assert result == {"num": 123, "bool": True}


class TestCreateSanitisationMapping:
    """Tests for create_sanitisation_mapping function."""

    def test_create_mapping(self):
        """Test creating original-to-sanitised mapping."""
        names = ["Counties", "Normal", "CensusYear"]
        mapping = create_sanitisation_mapping(names)
        assert mapping == {
            "Counties": "County",
            "Normal": "Normal",
            "CensusYear": "Census Year",
        }


class TestCreateReverseMapping:
    """Tests for create_reverse_mapping function."""

    def test_create_reverse_mapping(self):
        """Test creating sanitised-to-original mapping."""
        names = ["Counties", "Normal"]
        mapping = create_reverse_mapping(names)
        assert mapping == {
            "County": "Counties",
            "Normal": "Normal",
        }
