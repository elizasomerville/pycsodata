"""Tests for the exceptions module."""

import pytest

from pycsodata.exceptions import (
    APIError,
    DataError,
    SpatialError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Tests for the exception class hierarchy."""

    def test_api_error_inherits_from_data_error(self):
        """Test APIError is a subclass of DataError."""
        assert issubclass(APIError, DataError)

    def test_spatial_error_inherits_from_data_error(self):
        """Test SpatialError is a subclass of DataError."""
        assert issubclass(SpatialError, DataError)

    def test_validation_error_inherits_from_data_error(self):
        """Test ValidationError is a subclass of DataError."""
        assert issubclass(ValidationError, DataError)

    def test_all_inherit_from_exception(self):
        """Test all exceptions inherit from Exception."""
        assert issubclass(DataError, Exception)
        assert issubclass(APIError, Exception)
        assert issubclass(SpatialError, Exception)
        assert issubclass(ValidationError, Exception)


class TestDataError:
    """Tests for the DataError base exception."""

    def test_message_stored(self):
        """Test that message is stored as attribute."""
        err = DataError("Test message")
        assert err.message == "Test message"

    def test_str_representation(self):
        """Test string representation."""
        err = DataError("Test message")
        assert str(err) == "Test message"

    def test_can_be_raised_and_caught(self):
        """Test exception can be raised and caught."""
        with pytest.raises(DataError) as exc_info:
            raise DataError("Test error")
        assert "Test error" in str(exc_info.value)


class TestAPIError:
    """Tests for the APIError exception."""

    def test_basic_creation(self):
        """Test basic error creation."""
        err = APIError("Request failed")
        assert err.message == "Request failed"
        assert err.url is None
        assert err.status_code is None

    def test_with_url(self):
        """Test error with URL."""
        err = APIError("Request failed", url="http://example.com")
        assert err.url == "http://example.com"
        assert err.status_code is None

    def test_with_status_code(self):
        """Test error with status code."""
        err = APIError("Not found", url="http://example.com", status_code=404)
        assert err.url == "http://example.com"
        assert err.status_code == 404

    def test_caught_as_data_error(self):
        """Test that APIError can be caught as DataError."""
        with pytest.raises(DataError):
            raise APIError("API failed")


class TestSpatialError:
    """Tests for the SpatialError exception."""

    def test_basic_creation(self):
        """Test basic error creation."""
        err = SpatialError("Merge failed")
        assert err.message == "Merge failed"
        assert err.table_code is None

    def test_with_table_code(self):
        """Test error with table code."""
        err = SpatialError("No spatial data", table_code="FY003A")
        assert err.table_code == "FY003A"

    def test_caught_as_data_error(self):
        """Test that SpatialError can be caught as DataError."""
        with pytest.raises(DataError):
            raise SpatialError("Spatial operation failed")


class TestValidationError:
    """Tests for the ValidationError exception."""

    def test_basic_creation(self):
        """Test basic error creation."""
        err = ValidationError("Invalid value")
        assert err.message == "Invalid value"
        assert err.parameter is None
        assert err.value is None

    def test_with_parameter(self):
        """Test error with parameter name."""
        err = ValidationError("Invalid format", parameter="date_from")
        assert err.parameter == "date_from"

    def test_with_value(self):
        """Test error with invalid value."""
        err = ValidationError(
            "Must be positive",
            parameter="count",
            value=-5,
        )
        assert err.parameter == "count"
        assert err.value == -5

    def test_caught_as_data_error(self):
        """Test that ValidationError can be caught as DataError."""
        with pytest.raises(DataError):
            raise ValidationError("Validation failed")
