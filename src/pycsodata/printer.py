"""Metadata printing utilities for CSO datasets.

This module provides formatted printing of dataset metadata information,
separating presentation concerns from the core dataset functionality.

Classes:
    MetadataPrinter: Formats and prints dataset metadata to the console.

Examples:
    >>> from pycsodata import CSODataset
    >>> dataset = CSODataset("FY003A")
    >>> dataset.describe()  # Uses MetadataPrinter internally
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pycsodata._types import DatasetMetadata, FilterSpec


class MetadataPrinter:
    """Helper class for formatting and printing dataset metadata.

    This class handles the display formatting of CSO dataset metadata,
    providing a clean separation between data access and presentation.

    Args:
        metadata: The structured metadata dictionary for the dataset.
        filters: Optional filters that were applied to the dataset.
        drop_filtered_cols: Whether filtered columns are being dropped.
    """

    WIDTH = 85
    LABEL_WIDTH = 20

    def __init__(
        self,
        metadata: DatasetMetadata,
        filters: FilterSpec | None = None,
        drop_filtered_cols: bool = False,
    ) -> None:
        self.meta = metadata
        # normalise filter keys (STATISTIC -> Statistic)
        self.filters = self._normalise_filter_keys(filters) if filters else {}
        self.drop_filtered_cols = drop_filtered_cols

    @staticmethod
    def _normalise_filter_keys(filters: FilterSpec) -> FilterSpec:
        """Normalise filter keys, particularly STATISTIC -> Statistic.

        Args:
            filters: The original filter specification.

        Returns:
            A new filter specification with normalised keys.
        """
        normalised: dict = {}
        for key, value in filters.items():
            if key.upper() == "STATISTIC":
                normalised_key = "Statistic"
            elif key.upper() == "STATISTIC ID":
                normalised_key = "Statistic ID"
            else:
                normalised_key = key
            normalised[normalised_key] = value
        return normalised

    def print_all(self) -> None:
        """Print all metadata sections to the console.

        Outputs a formatted summary including header, variables,
        tags, time and spatial info, update date, notes, contact
        information, and copyright details.
        """
        self._print_header()
        self._print_variables()
        self._print_tags()
        self._print_time_and_spatial()
        self._print_updated()
        self._print_notes()
        self._print_contact()
        self._print_copyright()

    def _print_header(self) -> None:
        """Print the dataset code and title."""
        self._print_line("Code:", self.meta.get("table_code", "Unknown"))

        title = self.meta.get("title")
        if title:
            self._print_line("Title:", title)

        print()

    def _print_variables(self) -> None:
        """Print variables, statistics, and their units."""
        variables = self.meta.get("variables", [])
        statistics = self.meta.get("statistics", [])
        units = self.meta.get("units", [])

        # Use filtered statistics and units if available
        if "Statistic" in self.filters:
            filtered_stats = self.filters["Statistic"]
            if isinstance(filtered_stats, (list | tuple)):
                statistics = [str(s) for s in filtered_stats]
                units = [
                    self.meta.get("units", [])[self.meta.get("statistics", []).index(s)]
                    for s in statistics
                    if s in self.meta.get("statistics", [])
                ]

        # If drop_filtered_cols is True, remove filtered variables
        if self.drop_filtered_cols:
            variables = [var for var in variables if var not in self.filters]

        for i, var in enumerate(variables, start=1):
            if str(var).upper() == "STATISTIC" and statistics:
                if i == 1:
                    print(f"{'Variables:':<{self.LABEL_WIDTH}} [{i}] Statistic")
                else:
                    print(f"{'':<{self.LABEL_WIDTH}} [{i}] Statistic")
                for j, stat in enumerate(statistics, start=1):
                    print(f"{'':<24}({j}) {stat}")
                    unit = units[j - 1] if j - 1 < len(units) else "N/A"
                    print(f"{'':<28}Unit: {unit}")
            else:
                label = "Variables:" if i == 1 else ""
                print(f"{label:<{self.LABEL_WIDTH}} [{i}] {var}")

        print()

    def _print_tags(self) -> None:
        """Print dataset classification tags."""
        tags = self.meta.get("tags", [])
        tag_str = ", ".join(tags) if tags else "None"
        print(f"{'Tags:':<{self.LABEL_WIDTH}} {tag_str}")

    def _print_time_and_spatial(self) -> None:
        """Print time variable and geographic variable information."""
        time_var = self.meta.get("time_variable")
        if time_var:
            print(f"{'Time Variable:':<{self.LABEL_WIDTH}} {time_var}")

        if self.meta.get("geographic"):
            spatial_key = self.meta.get("spatial_key")
            print(f"{'Geographic Variable:':<{self.LABEL_WIDTH}} {spatial_key}")

        print()

    def _print_updated(self) -> None:
        """Print the last updated date and reason for release."""
        updated = self.meta.get("last_updated")
        if updated:
            print(f"{'Last Updated:':<{self.LABEL_WIDTH}} {updated.strftime('%Y-%m-%d')}")

        reasons = self.meta.get("reasons", [])
        if reasons:
            print(f"{'Reason for Release:':<{self.LABEL_WIDTH}} {', '.join(reasons)}")

        print()

    def _print_notes(self) -> None:
        """Print dataset notes with text wrapping."""
        notes = self.meta.get("notes", [])

        for i, note in enumerate(notes):
            label = "Notes:" if i == 0 else ""
            self._print_wrapped(
                f"* {note}",
                initial_indent=f"{label:<{self.LABEL_WIDTH - 1}}"
                if i == 0
                else " " * (self.LABEL_WIDTH - 1),
            )
        if notes:
            print()

    def _print_contact(self) -> None:
        """Print contact name, email, and phone number."""
        contact_name = self.meta.get("contact_name")
        contact_email = self.meta.get("contact_email")
        contact_phone = self.meta.get("contact_phone")

        if contact_name:
            print(f"{'Contact Name:':<{self.LABEL_WIDTH}} {contact_name}")
        if contact_email:
            print(f"{'Contact Email:':<{self.LABEL_WIDTH}} {contact_email}")
        if contact_phone:
            print(f"{'Contact Phone:':<{self.LABEL_WIDTH}} {contact_phone}")

    def _print_copyright(self) -> None:
        """Print copyright name and URL."""
        name = self.meta.get("copyright_name")
        href = self.meta.get("copyright_href")

        if name:
            if href:
                print(f"{'Copyright:':<{self.LABEL_WIDTH}} {name} ({href})")
            else:
                print(f"{'Copyright:':<{self.LABEL_WIDTH}} {name}")
            print()

    def _print_line(self, label: str, value: str) -> None:
        """Print a single labelled line with consistent formatting.

        Args:
            label: The label text (e.g., "Code:").
            value: The value to display.
        """
        print(f"{label:<{self.LABEL_WIDTH}} {value}")

    def _print_wrapped(self, text: str, initial_indent: str) -> None:
        """Print wrapped text with indentation.

        Args:
            text: The text to wrap and print.
            initial_indent: The indentation for the first line.
        """
        wrapper = textwrap.TextWrapper(
            width=self.WIDTH,
            initial_indent=initial_indent,
            subsequent_indent=" " * (self.LABEL_WIDTH + 1),
            break_long_words=False,
            break_on_hyphens=False,
        )
        print(wrapper.fill(text))
