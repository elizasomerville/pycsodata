"""Catalogue of available CSO datasets.

This module provides functionality to browse and search the CSO's
collection of statistical datasets through the CSOCatalogue class.

Examples:
    >>> from pycsodata import CSOCatalogue
    >>> catalogue = CSOCatalogue()
    >>> toc = catalogue.toc()
    >>> results = catalogue.search(title="population AND county")
"""

from __future__ import annotations

from urllib.parse import quote

import pandas as pd

from pycsodata.constants import (
    CSO_BASE_URL,
    STATISTIC_LABELS,
)
from pycsodata.fetchers import fetch_json
from pycsodata.sanitise import sanitise_list, sanitise_string
from pycsodata.search import (
    count_matching_terms,
    date_in_date_range,
    date_range_overlaps,
    extract_search_terms,
    parse_date_input,
    parse_date_range_tuple,
    parse_search_expression,
    parse_string_search_expression,
)


class CSOCatalogue:
    """Browse and search available CSO datasets.

    The catalogue provides access to the CSO's table of contents and
    allows searching by various criteria. It also provides hierarchical
    navigation through subjects, products, and datasets.

    Args:
        cache: Whether to cache API responses. Defaults to True.
        sanitise: Whether to sanitise variable names for consistency.
            When True, applies standardised transformations to variable names.

    Methods:
        toc: Get the full table of contents as a DataFrame.
        search: Search the catalogue by various criteria.

    Examples:
        >>> catalogue = CSOCatalogue()
        >>> toc = catalogue.toc()
        >>> results = catalogue.search(title="population")
        >>> print(results.head())
    """

    def __init__(self, cache: bool = True, sanitise: bool = False) -> None:
        self._cache_enabled = cache
        self._sanitise = sanitise
        self._toc_cache: dict[str, pd.DataFrame] = {}

    # =========================================================================
    # Table of Contents Methods
    # =========================================================================

    def toc(self, from_date: str | None = None) -> pd.DataFrame:
        """Get the table of contents for all CSO datasets.

        Args:
            from_date: Only return tables modified after this date (YYYY-MM-DD).
                Defaults to 2000-01-01 to include all datasets.

        Returns:
            A DataFrame with columns: Code, Title, Variables, Time Variable,
                Date Range, Updated, Organisation, Exceptional.

        Examples:
            >>> catalogue = CSOCatalogue()
            >>> toc = catalogue.toc(from_date="2023-01-01")
            >>> print(len(toc))
        """
        if from_date is None:
            from_date = "2000-01-01"  # Default to a very early date to get all datasets

        # Check instance cache
        if self._cache_enabled and from_date in self._toc_cache:
            return self._toc_cache[from_date].copy()

        # Fetch data from API
        url = f"{CSO_BASE_URL}.ReadCollection/{quote(from_date)}/en"

        data = fetch_json(url, cache=self._cache_enabled)
        items = data.get("link", {}).get("item", [])

        # Parse response into structured data
        records = []
        for item in items:
            record = self._parse_toc_item(item)
            if record:
                # Apply sanitisation if enabled
                if self._sanitise:
                    record = self._sanitise_toc_record(record)
                records.append(record)

        toc_df = pd.DataFrame(records)

        if not toc_df.empty:
            toc_df = toc_df.sort_values("Updated", ascending=False).reset_index(drop=True)

        # Cache result
        if self._cache_enabled:
            self._toc_cache[from_date] = toc_df

        return toc_df.copy()

    @staticmethod
    def _sanitise_toc_record(record: dict) -> dict:
        """Apply sanitisation to a table of contents record.

        Sanitises variable names and time variable labels for consistency.

        Args:
            record: The raw table of contents record dictionary.

        Returns:
            The sanitised record dictionary.
        """
        sanitised = dict(record)
        if sanitised.get("Variables"):
            sanitised["Variables"] = sanitise_list(sanitised["Variables"])
        if sanitised.get("Time Variable"):
            sanitised["Time Variable"] = sanitise_string(sanitised["Time Variable"])
        return sanitised

    def search(
        self,
        *,
        code: str | None = None,
        title: str | None = None,
        variables: str | None = None,
        time_variable: str | None = None,
        time_range: str | None = None,
        from_date: str | None = None,
        organisation: str | None = None,
        exceptional: bool | None = None,
    ) -> pd.DataFrame:
        """Search the table of contents for datasets matching criteria.

        All criteria are combined with AND logic. Text searches support
        boolean expressions with AND, OR, NOT operators and parentheses.
        Use quotation marks for exact phrase matching.

        Args:
            code: Filter by table code (substring match).
            title: Filter by title. Supports boolean expressions:
                - "population" - titles containing "population"
                - "population AND county" - must contain both
                - "population OR census" - must contain either
                - "population NOT census" - contains population but not census
                - '"exact phrase"' - matches exact phrase
            variables: Filter by variable names. Supports boolean expressions
                with AND/OR/NOT operators and parentheses. Examples:
                - "County" - datasets with a variable containing "County"
                - "County AND Year" - must have both
                - "Cork OR Dublin" - must have either
                - "County AND NOT Electoral" - has County but not Electoral
                - "(Cork OR Dublin) AND Population" - complex expressions
            time_variable: Filter by time variable label. Supports boolean expressions
                like title.
            time_range: Filter by time range. Accepts:
                - Single date: "2023", "January 2023", "2023Q1", "2023-01-15"
                - Date range tuple: "(2020, 2023)" or "(January 2020, December 2023)"
                Returns datasets whose date range overlaps with the specified date/range.
            from_date: Only include datasets updated on or after this date.
            organisation: Filter by organisation name (substring match).
            exceptional: Filter by exceptional release status.

        Returns:
            A DataFrame containing matching datasets.

        Examples:
            >>> catalogue = CSOCatalogue()
            >>> results = catalogue.search(title="census AND population")
            >>> results = catalogue.search(variables="County AND NOT Electoral")
            >>> # Find datasets covering 2020-2023
            >>> results = catalogue.search(time_range="(2020, 2023)")
        """
        toc_df = self.toc(from_date=from_date)

        if toc_df.empty:
            return toc_df

        mask = pd.Series([True] * len(toc_df), index=toc_df.index)

        if code:
            mask &= self._text_contains(toc_df["Code"], code)

        if title:
            mask &= self._text_matches_expression(toc_df["Title"], title)

        if variables:
            mask &= self._list_contains_expression(toc_df["Variables"], variables)

        if time_variable:
            mask &= self._text_matches_expression(toc_df["Time Variable"], time_variable)

        if time_range:
            mask &= self._date_range_filter(toc_df["Date Range"], time_range)

        if from_date:
            target_date = pd.to_datetime(from_date).date()
            mask &= toc_df["Updated"] >= target_date

        if organisation:
            mask &= self._text_contains(toc_df["Organisation"], organisation)

        if exceptional is not None:
            mask &= toc_df["Exceptional"] == exceptional

        result_df = toc_df[mask].copy()

        if result_df.empty:
            return result_df.reset_index(drop=True)

        # Calculate relevance score based on search terms
        result_df["_relevance"] = self._calculate_relevance(
            result_df, title=title, variables=variables
        )

        # Sort by relevance (descending), then by date updated (descending)
        result_df = result_df.sort_values(
            ["_relevance", "Updated"], ascending=[False, False]
        ).reset_index(drop=True)

        # Remove the temporary relevance column
        result_df = result_df.drop(columns=["_relevance"])

        return result_df

    # =========================================================================
    # Private Methods
    # =========================================================================

    @staticmethod
    def _calculate_relevance(
        df: pd.DataFrame,
        *,
        title: str | None = None,
        variables: str | None = None,
    ) -> pd.Series:
        """Calculate relevance scores for search results.

        The relevance score is based on how many search terms match in the
        title and variables columns. Results matching more terms are ranked
        higher.

        Args:
            df: The DataFrame of search results.
            title: The title search query (if provided).
            variables: The variables search query (if provided).

        Returns:
            A Series of relevance scores (higher = more relevant).
        """
        scores = pd.Series(0, index=df.index)

        # Calculate title relevance
        if title:
            title_terms = extract_search_terms(title)
            if title_terms:
                scores += df["Title"].apply(lambda text: count_matching_terms(text, title_terms))

        # Calculate relevance of variables
        if variables:
            var_terms = extract_search_terms(variables)
            if var_terms:

                def count_var_matches(var_list: list[str]) -> int:
                    if not var_list:
                        return 0
                    combined = " ".join(var_list)
                    return count_matching_terms(combined, var_terms)

                scores += df["Variables"].apply(count_var_matches)

        return scores

    @staticmethod
    def _parse_toc_item(item: dict) -> dict | None:
        """Parse a single table of contents item from the API response.

        Extracts relevant fields from the raw API item and transforms
        them into a structured dictionary.

        Args:
            item: The raw item dictionary from the API response.

        Returns:
            A dictionary with parsed fields, or None if parsing fails.
        """
        try:
            extension = item.get("extension", {})
            code = extension.get("matrix")

            if not code:
                return None

            # Extract variables (excluding Statistic)
            dimensions = item.get("dimension", {})
            variables = [
                dim_info.get("label", dim_name)
                for dim_name, dim_info in dimensions.items()
                if dim_info.get("label") not in STATISTIC_LABELS
            ]

            # Extract time variable
            time_variable = None
            role = item.get("role", {})
            time_dim = role.get("time", [])[0]
            if time_dim and time_dim in dimensions:
                time_variable = dimensions[time_dim].get("label")

            # Extract range of time variable
            date_range = None
            if time_dim and time_dim in dimensions:
                time_index = list(
                    dimensions[time_dim].get("category", {}).get("label", {}).values()
                )
                if time_index:
                    if len(time_index) == 0:
                        date_range = None
                    elif len(time_index) == 1:
                        date_range = str(time_index[0])
                    else:
                        start_date = time_index[0]
                        end_date = time_index[-1]
                        date_range = f"{start_date} - {end_date}"
            # Extract organisation name
            org_name = extension.get("copyright", {}).get("name", "")

            # Parse update date
            updated_str = item.get("updated")
            updated = None
            if updated_str:
                updated = pd.to_datetime(updated_str, format="mixed").date()

            return {
                "Code": code,
                "Title": item.get("label", ""),
                "Variables": variables,
                "Time Variable": time_variable,
                "Date Range": date_range,
                "Updated": updated,
                "Organisation": org_name,
                "Exceptional": extension.get("exceptional", False),
            }

        except Exception:
            return None

    @staticmethod
    def _text_contains(series: pd.Series, query: str) -> pd.Series:
        """Check if a text series contains a substring (case-insensitive).

        Args:
            series: A pandas Series of strings to search.
            query: The substring to search for.

        Returns:
            A boolean Series indicating which rows contain the query.
        """
        return series.str.contains(query, case=False, na=False)

    @staticmethod
    def _text_matches_expression(series: pd.Series, query: str) -> pd.Series:
        """Check if a text series matches a boolean search expression.

        Supports AND, OR, NOT operators, parentheses, and quoted exact phrases.

        Args:
            series: A Series of strings.
            query: A search expression (e.g., "population AND county", '"exact phrase"').

        Returns:
            A boolean Series indicating which rows match the expression.
        """
        matcher = parse_string_search_expression(query)
        return series.apply(lambda text: matcher(text) if pd.notna(text) else False)

    @staticmethod
    def _list_contains_expression(series: pd.Series, query: str) -> pd.Series:
        """Check if any item in a list column matches a boolean search expression.

        Supports AND, OR, NOT operators and parentheses for complex expressions.

        Args:
            series: A Series where each element is a list of strings.
            query: A search expression (e.g., "County AND Year", "Cork OR Dublin").

        Returns:
            A boolean Series indicating which rows match the expression.
        """
        # Parse the expression into a matcher function
        matcher = parse_search_expression(query)

        def match_items(items: list[str]) -> bool:
            if not items:
                return False
            return matcher([item.lower() for item in items])

        return series.apply(match_items)

    @staticmethod
    def _date_range_filter(series: pd.Series, query: str) -> pd.Series:
        """Filter by date range, supporting both single dates and date range tuples.

        Args:
            series: A Series of date range strings (e.g., "2015 - 2024").
            query: Either a single date string or a range tuple "(date1, date2)".

        Returns:
            A boolean Series indicating which rows' date ranges match the query.
        """
        # Check if query is a range tuple format
        range_tuple = parse_date_range_tuple(query)

        if range_tuple:
            # Parse both dates in the range
            start_str, end_str = range_tuple
            query_start, start_gran = parse_date_input(start_str)
            query_end, end_gran = parse_date_input(end_str)

            if query_start is None or query_end is None:
                # Fall back to substring match if parsing fails
                return series.str.contains(query, case=False, na=False)

            return series.apply(
                lambda dr: date_range_overlaps(query_start, query_end, start_gran, end_gran, dr)
            )
        else:
            # Single date - use existing logic
            query_date, granularity = parse_date_input(query)
            if query_date is None:
                return series.str.contains(query, case=False, na=False)

            return series.apply(lambda dr: date_in_date_range(query_date, granularity, dr))
