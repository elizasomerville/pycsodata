"""Boolean search expression parsing for catalogue searching.

This module provides parsing and evaluation of boolean search expressions
with AND, OR, NOT operators and parentheses. It supports both list-based
matching (for variables) and string matching (for titles).

Public Functions:
    parse_search_expression: Parse expression for list matching.
    parse_string_search_expression: Parse expression for string matching.
    extract_search_terms: Extract positive search terms from a query.
    count_matching_terms: Count how many search terms match a text.
    parse_date_input: Parse flexible date formats.
    parse_date_range_tuple: Parse date range tuples.
    date_in_date_range: Check if a date falls within a range.
    date_range_overlaps: Check if two date ranges overlap.
    adjust_date_to_period_end: Adjust a date to its period end.

Examples:
    >>> from pycsodata.search import parse_search_expression
    >>> matcher = parse_search_expression("Cork AND Population")
    >>> matcher(["Cork Population", "Dublin Data"])  # True
    >>> matcher(["Dublin Population"])  # False
"""

from __future__ import annotations

import re
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# Boolean Search Expression Parser (for list matching)
# =============================================================================


def parse_search_expression(
    query: str,
) -> Callable[[list[str]], bool]:
    """Parse a search expression with AND/OR operators into a match function.

    Supports boolean expressions with AND, OR operators and parentheses.
    Uses standard boolean precedence: AND binds tighter than OR.

    Args:
        query: A search expression like "County AND Population" or
               "(Cork OR Dublin) AND Population".

    Returns:
        A function that takes a list of strings and returns True if
        the expression matches any item in the list.

    Examples:
        >>> matcher = parse_search_expression("Cork AND Population")
        >>> matcher(["Cork Population", "Dublin Data"])  # True
        >>> matcher(["Dublin Population"])  # False
    """
    # Tokenise the query
    tokens = _tokenise_expression(query)
    if not tokens:
        # Empty query matches everything
        return lambda _items: True

    # Parse and return the matcher function
    pos = [0]  # Use list to allow modification in nested function
    return _parse_or_expression(tokens, pos)


def _tokenise_expression(query: str) -> list[str]:
    """Tokenise a search expression into tokens.

    Handles AND, OR, NOT operators, parentheses, and quoted strings.
    Quoted strings are preserved as single tokens.

    Args:
        query: The search expression to tokenise.

    Returns:
        A list of tokens.
    """
    tokens: list[str] = []
    i = 0
    query = query.strip()

    while i < len(query):
        # Skip whitespace
        if query[i].isspace():
            i += 1
            continue

        # Parentheses
        if query[i] == "(":
            tokens.append("(")
            i += 1
            continue
        if query[i] == ")":
            tokens.append(")")
            i += 1
            continue

        # Quoted string
        if query[i] in ('"', "'"):
            quote_char = query[i]
            i += 1
            start = i
            while i < len(query) and query[i] != quote_char:
                i += 1
            tokens.append(query[start:i])
            if i < len(query):
                i += 1  # Skip closing quote
            continue

        # Word (including AND/OR operators)
        start = i
        while i < len(query) and not query[i].isspace() and query[i] not in "()'\"":
            i += 1
        word = query[start:i]
        if word:
            tokens.append(word)

    return tokens


def _parse_or_expression(tokens: list[str], pos: list[int]) -> Callable[[list[str]], bool]:
    """Parse an OR expression (lowest precedence).

    Args:
        tokens: The list of tokens to parse.
        pos: A single-element list containing the current position
            (used as a mutable reference).

    Returns:
        A matcher function that evaluates the OR expression.
    """
    left: Callable[[list[str]], bool] = _parse_and_expression(tokens, pos)

    while pos[0] < len(tokens) and tokens[pos[0]].upper() == "OR":
        pos[0] += 1  # Skip OR
        right = _parse_and_expression(tokens, pos)
        left_func = left
        right_func = right

        def or_matcher(
            items: list[str],
            lf: Callable[[list[str]], bool] = left_func,
            rf: Callable[[list[str]], bool] = right_func,
        ) -> bool:
            return lf(items) or rf(items)

        left = or_matcher

    return left


def _parse_and_expression(tokens: list[str], pos: list[int]) -> Callable[[list[str]], bool]:
    """Parse an AND expression (higher precedence than OR).

    Args:
        tokens: The list of tokens to parse.
        pos: A single-element list containing the current position.

    Returns:
        A matcher function that evaluates the AND expression.
    """
    left: Callable[[list[str]], bool] = _parse_not_expression(tokens, pos)

    while pos[0] < len(tokens) and tokens[pos[0]].upper() == "AND":
        pos[0] += 1  # Skip AND
        right = _parse_not_expression(tokens, pos)
        left_func = left
        right_func = right

        def and_matcher(
            items: list[str],
            lf: Callable[[list[str]], bool] = left_func,
            rf: Callable[[list[str]], bool] = right_func,
        ) -> bool:
            return lf(items) and rf(items)

        left = and_matcher

    return left


def _parse_not_expression(tokens: list[str], pos: list[int]) -> Callable[[list[str]], bool]:
    """Parse a NOT expression (higher precedence than AND).

    Args:
        tokens: The list of tokens to parse.
        pos: A single-element list containing the current position.

    Returns:
        A matcher function that evaluates the NOT expression.
    """
    if pos[0] < len(tokens) and tokens[pos[0]].upper() == "NOT":
        pos[0] += 1  # Skip NOT
        operand = _parse_primary(tokens, pos)

        def not_matcher(items: list[str], op: Callable[[list[str]], bool] = operand) -> bool:
            return not op(items)

        return not_matcher

    return _parse_primary(tokens, pos)


def _parse_primary(tokens: list[str], pos: list[int]) -> Callable[[list[str]], bool]:
    """Parse a primary expression (term or parenthesised expression).

    A primary is either a search term or a parenthesised sub-expression.

    Args:
        tokens: The list of tokens to parse.
        pos: A single-element list containing the current position.

    Returns:
        A matcher function for the primary expression.
    """
    if pos[0] >= len(tokens):
        return lambda _items: True

    token = tokens[pos[0]]

    if token == "(":  # nosec B105 - parsing token, not password
        pos[0] += 1  # Skip (
        result = _parse_or_expression(tokens, pos)
        if pos[0] < len(tokens) and tokens[pos[0]] == ")":
            pos[0] += 1  # Skip )
        return result

    # Skip operators that might appear at start
    if token.upper() in ("AND", "OR", "NOT"):
        pos[0] += 1
        return _parse_primary(tokens, pos)

    # It's a search term
    pos[0] += 1
    term = token.lower()

    def term_matcher(items: list[str], t: str = term) -> bool:
        return any(t in item.lower() for item in items)

    return term_matcher


# =============================================================================
# Single String Search Expression Parser (for title, time_variable)
# =============================================================================


def parse_string_search_expression(
    query: str,
) -> Callable[[str], bool]:
    """Parse a search expression with AND/OR/NOT operators for single string matching.

    Supports boolean expressions with AND, OR, NOT operators and parentheses.
    Quoted strings are matched exactly as phrases.
    Uses standard boolean precedence: NOT > AND > OR.

    Args:
        query: A search expression like "population AND county" or
               '"exact phrase" OR alternative'.

    Returns:
        A function that takes a string and returns True if
        the expression matches the string.

    Examples:
        >>> matcher = parse_string_search_expression("population AND NOT census")
        >>> matcher("Population by county")  # True
        >>> matcher("Census population data")  # False
    """
    tokens = _tokenise_expression(query)
    if not tokens:
        return lambda _text: True

    pos = [0]
    return _parse_string_or_expression(tokens, pos)


def _parse_string_or_expression(tokens: list[str], pos: list[int]) -> Callable[[str], bool]:
    """Parse an OR expression for string matching.

    Args:
        tokens: The list of tokens to parse.
        pos: A single-element list containing the current position.

    Returns:
        A matcher function that evaluates the OR expression on a string.
    """
    left: Callable[[str], bool] = _parse_string_and_expression(tokens, pos)

    while pos[0] < len(tokens) and tokens[pos[0]].upper() == "OR":
        pos[0] += 1
        right = _parse_string_and_expression(tokens, pos)
        left_func = left
        right_func = right

        def or_matcher(
            text: str, lf: Callable[[str], bool] = left_func, rf: Callable[[str], bool] = right_func
        ) -> bool:
            return lf(text) or rf(text)

        left = or_matcher

    return left


def _parse_string_and_expression(tokens: list[str], pos: list[int]) -> Callable[[str], bool]:
    """Parse an AND expression for string matching.

    Args:
        tokens: The list of tokens to parse.
        pos: A single-element list containing the current position.

    Returns:
        A matcher function that evaluates the AND expression on a string.
    """
    left: Callable[[str], bool] = _parse_string_not_expression(tokens, pos)

    while pos[0] < len(tokens) and tokens[pos[0]].upper() == "AND":
        pos[0] += 1
        right = _parse_string_not_expression(tokens, pos)
        left_func = left
        right_func = right

        def and_matcher(
            text: str, lf: Callable[[str], bool] = left_func, rf: Callable[[str], bool] = right_func
        ) -> bool:
            return lf(text) and rf(text)

        left = and_matcher

    return left


def _parse_string_not_expression(tokens: list[str], pos: list[int]) -> Callable[[str], bool]:
    """Parse a NOT expression for string matching.

    Args:
        tokens: The list of tokens to parse.
        pos: A single-element list containing the current position.

    Returns:
        A matcher function that evaluates the NOT expression on a string.
    """
    if pos[0] < len(tokens) and tokens[pos[0]].upper() == "NOT":
        pos[0] += 1
        operand = _parse_string_primary(tokens, pos)

        def not_matcher(text: str, op: Callable[[str], bool] = operand) -> bool:
            return not op(text)

        return not_matcher

    return _parse_string_primary(tokens, pos)


def _parse_string_primary(tokens: list[str], pos: list[int]) -> Callable[[str], bool]:
    """Parse a primary expression for string matching.

    A primary is either a search term or a parenthesised sub-expression.

    Args:
        tokens: The list of tokens to parse.
        pos: A single-element list containing the current position.

    Returns:
        A matcher function for the primary expression.
    """
    if pos[0] >= len(tokens):
        return lambda _text: True

    token = tokens[pos[0]]

    if token == "(":  # nosec B105 - parsing token, not password
        pos[0] += 1
        result = _parse_string_or_expression(tokens, pos)
        if pos[0] < len(tokens) and tokens[pos[0]] == ")":
            pos[0] += 1
        return result

    if token.upper() in ("AND", "OR", "NOT"):
        pos[0] += 1
        return _parse_string_primary(tokens, pos)

    pos[0] += 1
    term = token.lower()

    def term_matcher(text: str, t: str = term) -> bool:
        return t in text.lower() if text else False

    return term_matcher


# =============================================================================
# Search Term Extraction and Counting
# =============================================================================


def extract_search_terms(query: str) -> list[str]:
    """Extract positive search terms from a query expression.

    Extracts all terms that contribute positively to a match,
    excluding terms preceded by NOT and excluding operators.

    Args:
        query: A search expression like "electoral division" or
               "population AND county NOT census".

    Returns:
        A list of lowercase search terms (excluding NOT terms and operators).

    Examples:
        >>> extract_search_terms("electoral division")
        ['electoral', 'division']
        >>> extract_search_terms("population AND county NOT census")
        ['population', 'county']
    """
    tokens = _tokenise_expression(query)
    terms: list[str] = []
    skip_next = False

    for token in tokens:
        upper_token = token.upper()
        if upper_token == "NOT":  # nosec B105 - parsing token, not password
            skip_next = True
            continue
        if upper_token in ("AND", "OR", "(", ")"):
            continue
        if skip_next:
            skip_next = False
            continue
        terms.append(token.lower())

    return terms


def count_matching_terms(text: str, terms: list[str]) -> int:
    """Count how many search terms match within a text.

    Args:
        text: The text to search within.
        terms: A list of lowercase search terms.

    Returns:
        The count of terms that appear in the text (case-insensitive).

    Examples:
        >>> count_matching_terms("Electoral Division Population", ["electoral", "division"])
        2
        >>> count_matching_terms("Population by County", ["electoral", "division"])
        0
    """
    if not text or not terms:
        return 0
    text_lower = text.lower()
    return sum(1 for term in terms if term in text_lower)


# =============================================================================
# Date Range Parsing and Matching
# =============================================================================


def parse_date_range_tuple(query: str) -> tuple[str, str] | None:
    """Parse a date range tuple from query string.

    Accepts format: "(date1, date2)" where date1 and date2 can be any
    valid date format (year, month-year, quarter, full date).

    Args:
        query: A string like "(2020, 2023)" or "(January 2020, December 2023)".

    Returns:
        A tuple of (start_date_str, end_date_str) or None if not a valid range format.
    """
    query = query.strip()
    if not query.startswith("(") or not query.endswith(")"):
        return None

    inner = query[1:-1]
    # Split on comma, but be careful with date formats that might contain commas
    parts = inner.split(",")
    if len(parts) != 2:
        return None

    return parts[0].strip(), parts[1].strip()


def parse_date_input(date_str: str) -> tuple[date | None, str]:
    """Parse a flexible date input into a date object and granularity.

    Accepts:
    - Full date: "2023-01-15", "15/01/2023", "January 15, 2023"
    - Month and year: "2023-01", "January 2023", "01/2023"
    - Year only: "2023"
    - Quarter: "2023Q1", "Q1 2023", "2023 Q1"

    Returns:
        A tuple of (parsed_date, granularity) where granularity is one of
        "year", "quarter", "month", or "day". Returns (None, "") if parsing fails.
    """
    date_str = date_str.strip()

    # Try year only (e.g., "2023")
    if re.match(r"^\d{4}$", date_str):
        return date(int(date_str), 1, 1), "year"

    # Try quarter format (e.g., "2023Q1", "Q1 2023", "2023 Q1", "1999Q1")
    quarter_match = re.match(
        r"^(?:(\d{4})\s*Q([1-4])|Q([1-4])\s*(\d{4}))$", date_str, re.IGNORECASE
    )
    if quarter_match:
        if quarter_match.group(1):
            year = int(quarter_match.group(1))
            quarter = int(quarter_match.group(2))
        else:
            year = int(quarter_match.group(4))
            quarter = int(quarter_match.group(3))
        # Quarter start month: Q1=1, Q2=4, Q3=7, Q4=10
        month = (quarter - 1) * 3 + 1
        return date(year, month, 1), "quarter"

    # Try month-year formats
    month_names = {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "sept": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    }

    # "January 2023" or "Jan 2023"
    month_year_match = re.match(r"^([a-zA-Z]+)\s+(\d{4})$", date_str)
    if month_year_match:
        month_name = month_year_match.group(1).lower()
        if month_name in month_names:
            return date(int(month_year_match.group(2)), month_names[month_name], 1), "month"

    # "2023 January" or "2023 Jan"
    year_month_match = re.match(r"^(\d{4})\s+([a-zA-Z]+)$", date_str)
    if year_month_match:
        month_name = year_month_match.group(2).lower()
        if month_name in month_names:
            return date(int(year_month_match.group(1)), month_names[month_name], 1), "month"

    # "2023-01" or "01/2023" or "2023/01"
    month_num_match = re.match(r"^(\d{4})[-/](\d{1,2})$|^(\d{1,2})[-/](\d{4})$", date_str)
    if month_num_match:
        if month_num_match.group(1):
            year = int(month_num_match.group(1))
            month = int(month_num_match.group(2))
        else:
            month = int(month_num_match.group(3))
            year = int(month_num_match.group(4))
        if 1 <= month <= 12:
            return date(year, month, 1), "month"

    # Try ISO format (YYYY-MM-DD) first without dayfirst
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        try:
            parsed = pd.to_datetime(date_str, format="%Y-%m-%d")
            return parsed.date(), "day"
        except (ValueError, TypeError):
            pass

    # Try full date formats using pandas for flexibility (dayfirst for European dates)
    try:
        parsed = pd.to_datetime(date_str, dayfirst=True, format="mixed")
        return parsed.date(), "day"
    except (ValueError, TypeError):
        pass

    return None, ""


def adjust_date_to_period_end(d: date, granularity: str) -> date:
    """Adjust a date to the end of its period based on granularity.

    For year granularity, returns 31 December of that year.
    For quarter granularity, returns the last day of the quarter.
    For month granularity, returns the last day of the month.
    For day granularity, returns the date unchanged.

    Args:
        d: The date to adjust.
        granularity: One of "year", "quarter", "month", or "day".

    Returns:
        The adjusted date at the end of the period.

    Examples:
        >>> from datetime import date
        >>> adjust_date_to_period_end(date(2023, 1, 1), "year")
        datetime.date(2023, 12, 31)
    """
    if granularity == "year":
        return date(d.year, 12, 31)
    elif granularity == "quarter":
        quarter_end_month = d.month + 2
        if quarter_end_month <= 3:
            return date(d.year, 3, 31)
        elif quarter_end_month <= 6:
            return date(d.year, 6, 30)
        elif quarter_end_month <= 9:
            return date(d.year, 9, 30)
        else:
            return date(d.year, 12, 31)
    elif granularity == "month":
        if d.month == 12:
            return date(d.year, 12, 31)
        else:
            next_month = date(d.year, d.month + 1, 1)
            end = next_month - pd.Timedelta(days=1)
            return end.date() if hasattr(end, "date") else end  # type: ignore
    return d


def date_range_overlaps(
    query_start: date,
    query_end: date,
    _query_start_gran: str,
    query_end_gran: str,
    date_range_str: str,
) -> bool:
    """Check if a query date range overlaps with a date range string.

    Args:
        query_start: Start of the query range.
        query_end: End of the query range.
        query_start_gran: Granularity of the start date.
        query_end_gran: Granularity of the end date.
        date_range_str: The date range string (e.g., "2015 - 2024").

    Returns:
        True if the ranges overlap.
    """
    if not date_range_str or pd.isna(date_range_str):
        return False

    parts = date_range_str.split(" - ")
    if len(parts) == 1:
        start_str = end_str = parts[0].strip()
    elif len(parts) == 2:
        start_str = parts[0].strip()
        end_str = parts[1].strip()
    else:
        return False

    start_date, _start_gran = parse_date_input(start_str)
    end_date, end_gran = parse_date_input(end_str)

    if start_date is None or end_date is None:
        return False

    # Adjust end date to end of period
    end_date = adjust_date_to_period_end(end_date, end_gran)
    query_end = adjust_date_to_period_end(query_end, query_end_gran)

    # Check overlap: ranges overlap if start1 <= end2 AND start2 <= end1
    return query_start <= end_date and start_date <= query_end


def date_in_date_range(query_date: date, granularity: str, date_range_str: str) -> bool:
    """Check if a query date falls within a date range string.

    The date range string is typically in format "start - end" where
    start and end can be years, months, quarters, etc.

    Args:
        query_date: The date to check.
        granularity: The granularity of the query ("year", "quarter", "month", "day").
        date_range_str: The date range string (e.g., "2015 - 2024", "2022 January - 2025 December").

    Returns:
        True if the query date falls within the date range.
    """
    if not date_range_str or pd.isna(date_range_str):
        return False

    # Split on " - " to get start and end
    parts = date_range_str.split(" - ")
    if len(parts) == 1:
        # Single value, check if it matches
        start_str = end_str = parts[0].strip()
    elif len(parts) == 2:
        start_str = parts[0].strip()
        end_str = parts[1].strip()
    else:
        return False

    # Parse start and end dates
    start_date, _start_gran = parse_date_input(start_str)
    end_date, end_gran = parse_date_input(end_str)

    if start_date is None or end_date is None:
        return False

    # Adjust end date to end of period based on granularity
    end_date = adjust_date_to_period_end(end_date, end_gran)

    # For query date, we also need to consider its range based on granularity
    query_end = adjust_date_to_period_end(query_date, granularity)

    # Check if ranges overlap
    # Two ranges overlap if: start1 <= end2 AND start2 <= end1
    return query_date <= end_date and start_date <= query_end
