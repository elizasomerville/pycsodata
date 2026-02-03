"""Tests for the search module."""

from datetime import date

from pycsodata.search import (
    _parse_primary,
    _parse_string_primary,
    _tokenise_expression,
    adjust_date_to_period_end,
    count_matching_terms,
    date_in_date_range,
    date_range_overlaps,
    extract_search_terms,
    parse_date_input,
    parse_date_range_tuple,
    parse_search_expression,
    parse_string_search_expression,
)


class TestParseSearchExpression:
    """Tests for the parse_search_expression function."""

    def test_empty_query_matches_everything(self):
        """Test that an empty query matches everything."""
        matcher = parse_search_expression("")
        assert matcher(["foo", "bar"]) is True
        assert matcher([]) is True

    def test_simple_term_match(self):
        """Test matching a simple term."""
        matcher = parse_search_expression("foo")
        assert matcher(["foo bar", "baz"]) is True
        assert matcher(["bar", "baz"]) is False

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        matcher = parse_search_expression("FOO")
        assert matcher(["foo bar", "baz"]) is True

    def test_and_expression(self):
        """Test AND expression matching."""
        matcher = parse_search_expression("foo AND bar")
        assert matcher(["foo bar", "baz"]) is True
        assert matcher(["foo", "baz"]) is False
        assert matcher(["bar", "baz"]) is False

    def test_or_expression(self):
        """Test OR expression matching."""
        matcher = parse_search_expression("foo OR baz")
        assert matcher(["foo", "bar"]) is True
        assert matcher(["baz", "qux"]) is True
        assert matcher(["bar", "qux"]) is False

    def test_not_expression(self):
        """Test NOT expression matching."""
        matcher = parse_search_expression("NOT foo")
        assert matcher(["bar", "baz"]) is True
        assert matcher(["foo", "bar"]) is False

    def test_parentheses(self):
        """Test parenthesized expressions."""
        matcher = parse_search_expression("(foo OR bar) AND baz")
        assert matcher(["foo baz", "qux"]) is True
        assert matcher(["bar baz", "qux"]) is True
        assert matcher(["foo", "bar"]) is False

    def test_complex_expression(self):
        """Test complex nested expressions."""
        matcher = parse_search_expression("(County OR City) AND Population")
        assert matcher(["County Population", "Other"]) is True
        assert matcher(["City Population", "Other"]) is True
        assert matcher(["County", "Other"]) is False


class TestParseStringSearchExpression:
    """Tests for the parse_string_search_expression function."""

    def test_empty_query_matches_everything(self):
        """Test that an empty query matches everything."""
        matcher = parse_string_search_expression("")
        assert matcher("any string") is True

    def test_simple_term_match(self):
        """Test matching a simple term."""
        matcher = parse_string_search_expression("population")
        assert matcher("Population by county") is True
        assert matcher("Census data") is False

    def test_and_expression(self):
        """Test AND expression matching."""
        matcher = parse_string_search_expression("population AND county")
        assert matcher("Population by county") is True
        assert matcher("Population data") is False

    def test_or_expression(self):
        """Test OR expression matching."""
        matcher = parse_string_search_expression("population OR census")
        assert matcher("Population data") is True
        assert matcher("Census data") is True
        assert matcher("Economic data") is False

    def test_not_expression(self):
        """Test NOT expression matching."""
        matcher = parse_string_search_expression("population AND NOT census")
        assert matcher("Population by county") is True
        assert matcher("Census population data") is False

    def test_handles_none_text(self):
        """Test that None text returns False."""
        matcher = parse_string_search_expression("population")
        assert matcher(None) is False  # type: ignore


class TestParseDateInput:
    """Tests for the parse_date_input function."""

    def test_year_only(self):
        """Test parsing year only."""
        result, granularity = parse_date_input("2023")
        assert result == date(2023, 1, 1)
        assert granularity == "year"

    def test_quarter_format_year_first(self):
        """Test parsing quarter format with year first."""
        result, granularity = parse_date_input("2023Q1")
        assert result == date(2023, 1, 1)
        assert granularity == "quarter"

    def test_quarter_format_q_first(self):
        """Test parsing quarter format with Q first."""
        result, granularity = parse_date_input("Q2 2023")
        assert result == date(2023, 4, 1)
        assert granularity == "quarter"

    def test_month_year_format(self):
        """Test parsing month-year format."""
        result, granularity = parse_date_input("January 2023")
        assert result == date(2023, 1, 1)
        assert granularity == "month"

    def test_year_month_format(self):
        """Test parsing year-month format."""
        result, granularity = parse_date_input("2023 January")
        assert result == date(2023, 1, 1)
        assert granularity == "month"

    def test_iso_month_format(self):
        """Test parsing ISO month format."""
        result, granularity = parse_date_input("2023-01")
        assert result == date(2023, 1, 1)
        assert granularity == "month"

    def test_iso_date_format(self):
        """Test parsing ISO date format."""
        result, granularity = parse_date_input("2023-01-15")
        assert result == date(2023, 1, 15)
        assert granularity == "day"

    def test_invalid_format_returns_none(self):
        """Test that invalid format returns None."""
        result, granularity = parse_date_input("not a date")
        assert result is None
        assert granularity == ""


class TestParseDateRangeTuple:
    """Tests for the parse_date_range_tuple function."""

    def test_valid_range_tuple(self):
        """Test parsing a valid range tuple."""
        result = parse_date_range_tuple("(2020, 2023)")
        assert result == ("2020", "2023")

    def test_range_with_spaces(self):
        """Test parsing range with extra spaces."""
        result = parse_date_range_tuple("( January 2020, December 2023 )")
        assert result == ("January 2020", "December 2023")

    def test_invalid_format_returns_none(self):
        """Test that invalid format returns None."""
        assert parse_date_range_tuple("2020") is None
        assert parse_date_range_tuple("2020-2023") is None

    def test_too_many_commas_returns_none(self):
        """Test that too many commas returns None."""
        assert parse_date_range_tuple("(2020, 2021, 2022)") is None


class TestAdjustDateToPeriodEnd:
    """Tests for the adjust_date_to_period_end function."""

    def test_year_adjustment(self):
        """Test year end adjustment."""
        result = adjust_date_to_period_end(date(2023, 1, 1), "year")
        assert result == date(2023, 12, 31)

    def test_quarter_adjustment(self):
        """Test quarter end adjustment."""
        # Q1
        result = adjust_date_to_period_end(date(2023, 1, 1), "quarter")
        assert result == date(2023, 3, 31)
        # Q2
        result = adjust_date_to_period_end(date(2023, 4, 1), "quarter")
        assert result == date(2023, 6, 30)

    def test_month_adjustment(self):
        """Test month end adjustment."""
        result = adjust_date_to_period_end(date(2023, 1, 1), "month")
        assert result == date(2023, 1, 31)

    def test_day_no_adjustment(self):
        """Test that day granularity doesn't adjust."""
        result = adjust_date_to_period_end(date(2023, 1, 15), "day")
        assert result == date(2023, 1, 15)


class TestDateInDateRange:
    """Tests for the date_in_date_range function."""

    def test_year_in_range(self):
        """Test year within date range."""
        assert date_in_date_range(date(2020, 1, 1), "year", "2015 - 2024") is True
        assert date_in_date_range(date(2025, 1, 1), "year", "2015 - 2024") is False

    def test_single_value_range(self):
        """Test single value date range."""
        assert date_in_date_range(date(2020, 1, 1), "year", "2020") is True
        assert date_in_date_range(date(2021, 1, 1), "year", "2020") is False

    def test_invalid_range_returns_false(self):
        """Test that invalid range returns False."""
        assert date_in_date_range(date(2020, 1, 1), "year", "") is False
        assert date_in_date_range(date(2020, 1, 1), "year", None) is False  # type: ignore


class TestDateRangeOverlaps:
    """Tests for the date_range_overlaps function."""

    def test_overlapping_ranges(self):
        """Test overlapping date ranges."""
        assert (
            date_range_overlaps(date(2018, 1, 1), date(2022, 1, 1), "year", "year", "2015 - 2024")
            is True
        )

    def test_non_overlapping_ranges(self):
        """Test non-overlapping date ranges."""
        assert (
            date_range_overlaps(date(2025, 1, 1), date(2026, 1, 1), "year", "year", "2015 - 2024")
            is False
        )

    def test_invalid_range_returns_false(self):
        """Test that invalid range returns False."""
        assert date_range_overlaps(date(2020, 1, 1), date(2022, 1, 1), "year", "year", "") is False


class TestTokeniseExpression:
    """Tests for _tokenise_expression function."""

    def test_tokenise_simple_terms(self):
        """Test tokenising simple terms."""
        tokens = _tokenise_expression("foo bar")
        assert tokens == ["foo", "bar"]

    def test_tokenise_operators(self):
        """Test tokenising with operators."""
        tokens = _tokenise_expression("foo AND bar")
        assert tokens == ["foo", "AND", "bar"]

    def test_tokenise_parentheses(self):
        """Test tokenising with parentheses."""
        tokens = _tokenise_expression("(foo OR bar)")
        assert tokens == ["(", "foo", "OR", "bar", ")"]

    def test_tokenise_double_quoted_string(self):
        """Test tokenising double-quoted strings."""
        tokens = _tokenise_expression('"exact phrase" OR other')
        assert tokens == ["exact phrase", "OR", "other"]

    def test_tokenise_single_quoted_string(self):
        """Test tokenising single-quoted strings."""
        tokens = _tokenise_expression("'exact phrase' AND term")
        assert tokens == ["exact phrase", "AND", "term"]

    def test_tokenise_empty_string(self):
        """Test tokenising empty string."""
        tokens = _tokenise_expression("")
        assert tokens == []

    def test_tokenise_whitespace_only(self):
        """Test tokenising whitespace-only string."""
        tokens = _tokenise_expression("   ")
        assert tokens == []

    def test_tokenise_complex_expression(self):
        """Test tokenising complex expression."""
        tokens = _tokenise_expression('(County OR "City Region") AND Population')
        assert tokens == ["(", "County", "OR", "City Region", ")", "AND", "Population"]


class TestParsePrimary:
    """Tests for _parse_primary function."""

    def test_parse_empty_tokens(self):
        """Test parsing empty tokens."""
        tokens = []
        pos = [0]
        matcher = _parse_primary(tokens, pos)
        assert matcher(["anything"]) is True

    def test_parse_operator_at_start(self):
        """Test parsing when operator is at start."""
        tokens = ["AND", "foo"]
        pos = [0]
        matcher = _parse_primary(tokens, pos)
        # Should skip AND and match "foo"
        assert matcher(["foo"]) is True


class TestParseStringPrimary:
    """Tests for _parse_string_primary function."""

    def test_parse_empty_tokens(self):
        """Test parsing empty tokens."""
        tokens = []
        pos = [0]
        matcher = _parse_string_primary(tokens, pos)
        assert matcher("anything") is True

    def test_parse_operator_at_start(self):
        """Test parsing when operator is at start."""
        tokens = ["OR", "foo"]
        pos = [0]
        matcher = _parse_string_primary(tokens, pos)
        # Should skip OR and match "foo"
        assert matcher("foo bar") is True

    def test_parse_parentheses(self):
        """Test parsing parenthesized expression."""
        tokens = ["(", "foo", ")"]
        pos = [0]
        matcher = _parse_string_primary(tokens, pos)
        assert matcher("foo bar") is True

    def test_parse_unclosed_parenthesis(self):
        """Test parsing unclosed parenthesis."""
        tokens = ["(", "foo"]
        pos = [0]
        matcher = _parse_string_primary(tokens, pos)
        # Should still work, just no closing paren
        assert matcher("foo bar") is True


class TestParseDateInputEdgeCases:
    """Edge case tests for parse_date_input."""

    def test_abbreviated_month_names(self):
        """Test abbreviated month names."""
        months = [
            ("Jan 2023", 1),
            ("Feb 2023", 2),
            ("Mar 2023", 3),
            ("Apr 2023", 4),
            ("May 2023", 5),
            ("Jun 2023", 6),
            ("Jul 2023", 7),
            ("Aug 2023", 8),
            ("Sep 2023", 9),
            ("Oct 2023", 10),
            ("Nov 2023", 11),
            ("Dec 2023", 12),
        ]
        for date_str, expected_month in months:
            result, granularity = parse_date_input(date_str)
            assert result is not None, f"Failed for {date_str}"
            assert result.month == expected_month, f"Failed for {date_str}"
            assert granularity == "month"

    def test_sept_abbreviation(self):
        """Test 'sept' abbreviation for September."""
        result, granularity = parse_date_input("Sept 2023")
        assert result is not None
        assert result.month == 9
        assert granularity == "month"

    def test_slash_date_format(self):
        """Test slash date format (month/year)."""
        result, granularity = parse_date_input("01/2023")
        assert result == date(2023, 1, 1)
        assert granularity == "month"

    def test_year_slash_month_format(self):
        """Test year/month format."""
        result, granularity = parse_date_input("2023/01")
        assert result == date(2023, 1, 1)
        assert granularity == "month"

    def test_invalid_month_number(self):
        """Test invalid month number returns None."""
        result, _granularity = parse_date_input("2023-13")
        assert result is None

    def test_european_date_format(self):
        """Test European date format (day first)."""
        result, granularity = parse_date_input("15/01/2023")
        assert result == date(2023, 1, 15)
        assert granularity == "day"

    def test_quarter_with_space(self):
        """Test quarter format with space."""
        result, granularity = parse_date_input("2023 Q3")
        assert result == date(2023, 7, 1)
        assert granularity == "quarter"

    def test_quarter_format_all_quarters(self):
        """Test all quarter formats."""
        quarters = [("2023Q1", 1), ("2023Q2", 4), ("2023Q3", 7), ("2023Q4", 10)]
        for date_str, expected_month in quarters:
            result, granularity = parse_date_input(date_str)
            assert result is not None, f"Failed for {date_str}"
            assert result.month == expected_month, f"Failed for {date_str}"
            assert granularity == "quarter"


class TestAdjustDateToPeriodEndEdgeCases:
    """Edge case tests for adjust_date_to_period_end."""

    def test_quarter_q2(self):
        """Test Q2 adjustment."""
        result = adjust_date_to_period_end(date(2023, 4, 1), "quarter")
        assert result == date(2023, 6, 30)

    def test_quarter_q3(self):
        """Test Q3 adjustment."""
        result = adjust_date_to_period_end(date(2023, 7, 1), "quarter")
        assert result == date(2023, 9, 30)

    def test_quarter_q4(self):
        """Test Q4 adjustment."""
        result = adjust_date_to_period_end(date(2023, 10, 1), "quarter")
        assert result == date(2023, 12, 31)

    def test_month_december(self):
        """Test December month adjustment."""
        result = adjust_date_to_period_end(date(2023, 12, 1), "month")
        assert result == date(2023, 12, 31)

    def test_month_february(self):
        """Test February month adjustment."""
        result = adjust_date_to_period_end(date(2023, 2, 1), "month")
        assert result == date(2023, 2, 28)

    def test_month_february_leap_year(self):
        """Test February in leap year."""
        result = adjust_date_to_period_end(date(2024, 2, 1), "month")
        assert result == date(2024, 2, 29)

    def test_day_granularity_unchanged(self):
        """Test day granularity returns same date."""
        result = adjust_date_to_period_end(date(2023, 6, 15), "day")
        assert result == date(2023, 6, 15)


class TestDateInDateRangeEdgeCases:
    """Edge case tests for date_in_date_range."""

    def test_none_date_range_returns_false(self):
        """Test that None date range returns False."""
        result = date_in_date_range(date(2023, 1, 1), "year", None)  # type: ignore
        assert result is False

    def test_nan_date_range_returns_false(self):
        """Test that NaN date range returns False."""
        result = date_in_date_range(date(2023, 1, 1), "year", float("nan"))  # type: ignore
        assert result is False

    def test_single_value_range(self):
        """Test single value (not a range)."""
        result = date_in_date_range(date(2023, 1, 1), "year", "2023")
        assert result is True

    def test_single_value_range_no_match(self):
        """Test single value that doesn't match."""
        result = date_in_date_range(date(2020, 1, 1), "year", "2023")
        assert result is False

    def test_multiple_dashes_returns_false(self):
        """Test that multiple dashes returns False."""
        result = date_in_date_range(date(2023, 1, 1), "year", "2020 - 2022 - 2024")
        assert result is False

    def test_unparseable_dates_returns_false(self):
        """Test that unparseable dates return False."""
        result = date_in_date_range(date(2023, 1, 1), "year", "abc - xyz")
        assert result is False

    def test_quarter_granularity_matching(self):
        """Test quarter granularity matching."""
        # Q2 2023 should match range including Q2
        result = date_in_date_range(date(2023, 4, 1), "quarter", "2022 - 2024")
        assert result is True


class TestDateRangeOverlapsEdgeCases:
    """Edge case tests for date_range_overlaps."""

    def test_none_date_range_returns_false(self):
        """Test that None date range returns False."""
        result = date_range_overlaps(
            date(2020, 1, 1),
            date(2023, 1, 1),
            "year",
            "year",
            None,  # type: ignore
        )
        assert result is False

    def test_nan_date_range_returns_false(self):
        """Test that NaN date range returns False."""
        result = date_range_overlaps(
            date(2020, 1, 1),
            date(2023, 1, 1),
            "year",
            "year",
            float("nan"),  # type: ignore
        )
        assert result is False

    def test_single_value_range(self):
        """Test single value (not a range)."""
        result = date_range_overlaps(date(2023, 1, 1), date(2023, 12, 31), "year", "year", "2023")
        assert result is True

    def test_no_overlap(self):
        """Test when ranges don't overlap."""
        result = date_range_overlaps(
            date(2010, 1, 1), date(2015, 12, 31), "year", "year", "2020 - 2024"
        )
        assert result is False

    def test_partial_overlap(self):
        """Test partial overlap."""
        result = date_range_overlaps(
            date(2018, 1, 1), date(2022, 12, 31), "year", "year", "2020 - 2024"
        )
        assert result is True

    def test_unparseable_dates_returns_false(self):
        """Test that unparseable dates return False."""
        result = date_range_overlaps(
            date(2020, 1, 1), date(2023, 1, 1), "year", "year", "abc - xyz"
        )
        assert result is False


class TestSearchExpressionPrecedence:
    """Tests for operator precedence in search expressions."""

    def test_and_binds_tighter_than_or(self):
        """Test that AND binds tighter than OR."""
        matcher = parse_search_expression("a OR b AND c")
        # Should be parsed as: a OR (b AND c)
        assert matcher(["a"]) is True
        assert matcher(["b and c"]) is True
        assert matcher(["b"]) is False

    def test_not_binds_tightest(self):
        """Test that NOT binds tightest."""
        matcher = parse_search_expression("a AND NOT b")
        assert matcher(["a"]) is True
        assert matcher(["a", "b"]) is False


class TestStringSearchExpressionPrecedence:
    """Tests for operator precedence in string search expressions."""

    def test_and_binds_tighter_than_or(self):
        """Test that AND binds tighter than OR."""
        matcher = parse_string_search_expression("population OR county AND data")
        # Should be parsed as: population OR (county AND data)
        assert matcher("population") is True
        assert matcher("county data") is True
        assert matcher("county") is False

    def test_nested_not(self):
        """Test nested NOT expressions."""
        matcher = parse_string_search_expression("NOT (a AND b)")
        assert matcher("a") is True
        assert matcher("a b") is False


class TestSearchExpressionQuotedStrings:
    """Tests for quoted string handling in search expressions."""

    def test_exact_phrase_match(self):
        """Test exact phrase matching with quotes."""
        matcher = parse_search_expression('"population data"')
        assert matcher(["population data"]) is True
        assert matcher(["population", "data"]) is False  # Not as one item

    def test_quoted_with_operators(self):
        """Test quoted strings with operators."""
        matcher = parse_search_expression('"exact phrase" AND other')
        assert matcher(["exact phrase", "other"]) is True
        assert matcher(["exact phrase"]) is False


class TestStringSearchExpressionQuotedStrings:
    """Tests for quoted string handling in string search expressions."""

    def test_exact_phrase_match(self):
        """Test exact phrase matching with quotes."""
        matcher = parse_string_search_expression('"population data"')
        assert matcher("The population data is here") is True
        assert matcher("population and data") is False


class TestExtractSearchTerms:
    """Tests for the extract_search_terms function."""

    def test_simple_terms(self):
        """Test extracting simple space-separated terms."""
        terms = extract_search_terms("electoral division")
        assert terms == ["electoral", "division"]

    def test_with_and_operator(self):
        """Test extracting terms with AND operator."""
        terms = extract_search_terms("population AND county")
        assert terms == ["population", "county"]

    def test_with_or_operator(self):
        """Test extracting terms with OR operator."""
        terms = extract_search_terms("cork OR dublin")
        assert terms == ["cork", "dublin"]

    def test_with_not_operator(self):
        """Test that NOT terms are excluded."""
        terms = extract_search_terms("population NOT census")
        assert terms == ["population"]
        assert "census" not in terms

    def test_complex_expression(self):
        """Test extracting terms from complex expression."""
        terms = extract_search_terms("(population OR census) AND county NOT electoral")
        assert "population" in terms
        assert "census" in terms
        assert "county" in terms
        assert "electoral" not in terms

    def test_empty_query(self):
        """Test extracting terms from empty query."""
        terms = extract_search_terms("")
        assert terms == []

    def test_lowercase_conversion(self):
        """Test that terms are lowercased."""
        terms = extract_search_terms("Electoral Division")
        assert terms == ["electoral", "division"]


class TestCountMatchingTerms:
    """Tests for the count_matching_terms function."""

    def test_all_terms_match(self):
        """Test counting when all terms match."""
        count = count_matching_terms("Electoral Division Population", ["electoral", "division"])
        assert count == 2

    def test_some_terms_match(self):
        """Test counting when some terms match."""
        count = count_matching_terms(
            "Population by County", ["electoral", "division", "population"]
        )
        assert count == 1

    def test_no_terms_match(self):
        """Test counting when no terms match."""
        count = count_matching_terms("Population by County", ["electoral", "division"])
        assert count == 0

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        count = count_matching_terms("ELECTORAL DIVISION", ["electoral", "division"])
        assert count == 2

    def test_empty_text(self):
        """Test counting with empty text."""
        count = count_matching_terms("", ["electoral", "division"])
        assert count == 0

    def test_empty_terms(self):
        """Test counting with empty terms list."""
        count = count_matching_terms("Electoral Division", [])
        assert count == 0

    def test_none_text(self):
        """Test counting with None text."""
        count = count_matching_terms(None, ["electoral", "division"])  # type: ignore
        assert count == 0
