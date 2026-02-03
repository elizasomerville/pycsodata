"""Column name sanitisation utilities for pycsodata.

This module provides functions to sanitise column names consistently across
the package. Sanitisation helps ensure consistent naming when working with
multiple CSO datasets that may use different naming conventions.

Sanitisation includes:
    - Replacing '&' with 'and'
    - Replacing ' / ' or ' /' with '/'
    - Replacing multiple spaces with single spaces
    - Stripping edge whitespace
    - Removing trailing full stops
    - Applying SANITISATION_DICT mappings from constants.py

Examples:
    >>> from pycsodata.sanitise import sanitise_string, sanitise_list
    >>> sanitise_string("Counties & Cities")
    'County and City'
    >>> sanitise_list(["NUTS 3 Regions", "Counties"])
    ['NUTS 3 Region', 'County']
"""

from __future__ import annotations

import re
from typing import Any

from pycsodata.constants import SANITISATION_DICT


def sanitise_string(value: str) -> str:
    """Sanitise a single string value.

    Applies the following transformations in order:
    1. Replace '&' with 'and'
    2. Replace ' / ' or ' /' with '/'
    3. Replace multiple spaces with single spaces
    4. Strip edge whitespace
    5. Remove trailing full stops
    6. Apply SANITISATION_DICT mappings

    Args:
        value: The string to sanitise.

    Returns:
        The sanitised string.

    Examples:
        >>> sanitise_string("Counties & Cities")
        'County and City'
        >>> sanitise_string("  Multiple   spaces  ")
        'Multiple spaces'
    """
    if not isinstance(value, str):
        return value

    # Step 1: Replace '&' with 'and'
    result = value.replace("&", "and")

    # Step 2: Replace ' / ' or ' /' with '/'
    result = re.sub(r"\s*/\s*", "/", result)

    # Step 3: Replace multiple spaces with single spaces
    result = re.sub(r"\s+", " ", result)

    # Step 4: Strip edge whitespace
    result = result.strip()

    # Step 5: If ends with a full stop, remove it
    if result.endswith("."):
        result = result[:-1].rstrip()

    # Step 6: Apply SANITISATION_DICT mappings
    if result in SANITISATION_DICT:
        result = SANITISATION_DICT[result]

    return result


def sanitise_list(values: list[str]) -> list[str]:
    """Sanitise a list of strings.

    Applies sanitise_string to each element in the list.

    Args:
        values: List of strings to sanitise.

    Returns:
        List of sanitised strings.

    Examples:
        >>> sanitise_list(["Counties", "NUTS 3 Regions"])
        ['County', 'NUTS 3 Region']
    """
    return [sanitise_string(v) if isinstance(v, str) else v for v in values]


def sanitise_dict_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Sanitise dictionary keys.

    Creates a new dictionary with sanitised keys whilst preserving values.

    Args:
        d: Dictionary with string keys to sanitise.

    Returns:
        New dictionary with sanitised keys.

    Examples:
        >>> sanitise_dict_keys({"Counties": ["Dublin", "Cork"]})
        {'County': ['Dublin', 'Cork']}
    """
    return {sanitise_string(k): v for k, v in d.items()}


def sanitise_dict_values(d: dict[str, Any]) -> dict[str, Any]:
    """Sanitise string values in a dictionary (not recursively).

    Creates a new dictionary with sanitised string values. Lists of
    strings are also sanitised. Non-string values are preserved unchanged.

    Args:
        d: Dictionary with potentially string values.

    Returns:
        New dictionary with sanitised string values.

    Examples:
        >>> sanitise_dict_values({"name": "Counties", "count": 32})
        {'name': 'County', 'count': 32}
    """
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            result[k] = sanitise_string(v)
        elif isinstance(v, list):
            result[k] = sanitise_list(v)
        else:
            result[k] = v
    return result


def create_sanitisation_mapping(original_names: list[str]) -> dict[str, str]:
    """Create a mapping from original names to sanitised names.

    This is useful for looking up the sanitised version of a name and for
    maintaining backwards compatibility with original API names.

    Args:
        original_names: List of original column names.

    Returns:
        Dictionary mapping original names to sanitised names.

    Examples:
        >>> create_sanitisation_mapping(["Counties", "NUTS 3 Regions"])
        {'Counties': 'County', 'NUTS 3 Regions': 'NUTS 3 Region'}
    """
    return {name: sanitise_string(name) for name in original_names}


def create_reverse_mapping(original_names: list[str]) -> dict[str, str]:
    """Create a mapping from sanitised names back to original names.

    Useful for translating user-provided sanitised names back to the
    original API names when making requests.

    Args:
        original_names: List of original column names.

    Returns:
        Dictionary mapping sanitised names to original names.

    Examples:
        >>> create_reverse_mapping(["Counties", "NUTS 3 Regions"])
        {'County': 'Counties', 'NUTS 3 Region': 'NUTS 3 Regions'}
    """
    return {sanitise_string(name): name for name in original_names}
