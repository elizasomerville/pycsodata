"""Custom exceptions for pycsodata.

This module defines the exception hierarchy used throughout the package.
All exceptions inherit from DataError, making it easy to catch all
package-specific errors.

Exception Hierarchy:
    DataError (base)
    ├── APIError: Network and API communication errors.
    ├── SpatialError: Spatial data and geometry errors.
    └── ValidationError: Input validation errors.

Examples:
    >>> from pycsodata import CSODataset
    >>> from pycsodata.exceptions import APIError, DataError
    >>> try:
    ...     dataset = CSODataset("INVALID_CODE")
    ... except APIError as e:
    ...     print(f"API error: {e}")
    ... except DataError as e:
    ...     print(f"Data error: {e}")
"""

from __future__ import annotations


class DataError(Exception):
    """Base exception for all pycsodata errors.

    This is the root of the exception hierarchy. Catching this exception
    will catch all package-specific errors.

    Attributes:
        message: Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


class APIError(DataError):
    """Raised when API requests fail.

    This exception is raised when:
    - Network requests to the CSO API fail after retries
    - The API returns an unexpected response format
    - Required data is missing from the API response

    Attributes:
        message: Human-readable error description.
        url: The URL that failed, if available.
        status_code: HTTP status code, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.url = url
        self.status_code = status_code
        super().__init__(message)


class SpatialError(DataError):
    """Raised when spatial operations fail.

    This exception is raised when:
    - Spatial data is not available for a dataset
    - Spatial merge operations fail
    - Geometry validation fails

    Attributes:
        message: Human-readable error description.
        table_code: The table code that failed, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        table_code: str | None = None,
    ) -> None:
        self.table_code = table_code
        super().__init__(message)


class ValidationError(DataError):
    """Raised when input validation fails.

    This exception is raised when:
    - Invalid parameter values are provided
    - Required parameters are missing
    - Filter values don't match available dimensions

    Attributes:
        message: Human-readable error description.
        parameter: The parameter that failed validation, if available.
        value: The invalid value, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        parameter: str | None = None,
        value: object = None,
    ) -> None:
        self.parameter = parameter
        self.value = value
        super().__init__(message)
