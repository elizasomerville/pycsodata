"""Tests for the printer module."""

from datetime import datetime

from pycsodata._types import DatasetMetadata
from pycsodata.printer import MetadataPrinter


class TestMetadataPrinter:
    """Tests for the MetadataPrinter class."""

    def test_initialisation(self):
        """Test MetadataPrinter initialises correctly."""
        metadata = DatasetMetadata(
            table_code="FY003A",
            title="Test Dataset",
        )
        printer = MetadataPrinter(metadata)
        assert printer.meta == metadata
        assert printer.filters == {}
        assert printer.drop_filtered_cols is False

    def test_initialisation_with_filters(self):
        """Test MetadataPrinter initialises with filters."""
        metadata = DatasetMetadata(table_code="FY003A")
        filters: dict = {"County": ["Dublin", "Cork"]}
        printer = MetadataPrinter(metadata, filters)
        assert printer.filters == filters

    def test_normalise_filter_keys_statistic(self):
        """Test that STATISTIC is normalised to Statistic."""
        filters: dict = {"STATISTIC": ["Population"]}
        normalised = MetadataPrinter._normalise_filter_keys(filters)
        assert "Statistic" in normalised
        assert "STATISTIC" not in normalised

    def test_normalise_filter_keys_statistic_id(self):
        """Test that STATISTIC ID is normalised to Statistic ID."""
        filters: dict = {"STATISTIC ID": ["ABC123"]}
        normalised = MetadataPrinter._normalise_filter_keys(filters)
        assert "Statistic ID" in normalised

    def test_normalise_filter_keys_other_keys_unchanged(self):
        """Test that other filter keys are unchanged."""
        filters: dict = {"County": ["Dublin"]}
        normalised = MetadataPrinter._normalise_filter_keys(filters)
        assert "County" in normalised

    def test_print_all_outputs_to_stdout(self, capsys):
        """Test that print_all outputs to stdout."""
        metadata = DatasetMetadata(
            table_code="FY003A",
            title="Test Dataset",
            variables=["County", "Year"],
            statistics=[],
            tags=["census", "population"],
        )
        printer = MetadataPrinter(metadata)
        printer.print_all()

        captured = capsys.readouterr()
        assert "FY003A" in captured.out
        assert "Test Dataset" in captured.out

    def test_print_header_outputs_code_and_title(self, capsys):
        """Test that header prints code and title."""
        metadata = DatasetMetadata(
            table_code="FY003A",
            title="Test Dataset",
        )
        printer = MetadataPrinter(metadata)
        printer._print_header()

        captured = capsys.readouterr()
        assert "Code:" in captured.out
        assert "FY003A" in captured.out
        assert "Title:" in captured.out
        assert "Test Dataset" in captured.out

    def test_print_variables_outputs_variables(self, capsys):
        """Test that variables are printed."""
        metadata = DatasetMetadata(
            variables=["County", "Year", "Sex"],
            statistics=[],
            units=[],
        )
        printer = MetadataPrinter(metadata)
        printer._print_variables()

        captured = capsys.readouterr()
        assert "Variables:" in captured.out
        assert "County" in captured.out
        assert "Year" in captured.out
        assert "Sex" in captured.out

    def test_print_tags_outputs_tags(self, capsys):
        """Test that tags are printed."""
        metadata = DatasetMetadata(
            tags=["census", "population", "demographic"],
        )
        printer = MetadataPrinter(metadata)
        printer._print_tags()

        captured = capsys.readouterr()
        assert "Tags:" in captured.out
        assert "census" in captured.out
        assert "population" in captured.out

    def test_print_tags_handles_empty_tags(self, capsys):
        """Test that empty tags shows None."""
        metadata = DatasetMetadata(tags=[])
        printer = MetadataPrinter(metadata)
        printer._print_tags()

        captured = capsys.readouterr()
        assert "None" in captured.out

    def test_print_time_and_spatial_outputs_time_variable(self, capsys):
        """Test that time variable is printed."""
        metadata = DatasetMetadata(
            time_variable="Census Year",
        )
        printer = MetadataPrinter(metadata)
        printer._print_time_and_spatial()

        captured = capsys.readouterr()
        assert "Time Variable:" in captured.out
        assert "Census Year" in captured.out

    def test_print_updated_outputs_date(self, capsys):
        """Test that last updated date is printed."""
        metadata = DatasetMetadata(
            last_updated=datetime(2023, 6, 15),
        )
        printer = MetadataPrinter(metadata)
        printer._print_updated()

        captured = capsys.readouterr()
        assert "Last Updated:" in captured.out
        assert "2023-06-15" in captured.out

    def test_print_contact_outputs_contact_info(self, capsys):
        """Test that contact information is printed."""
        metadata = DatasetMetadata(
            contact_name="John Doe",
            contact_email="john@example.com",
            contact_phone="+353 1 234 5678",
        )
        printer = MetadataPrinter(metadata)
        printer._print_contact()

        captured = capsys.readouterr()
        assert "Contact Name:" in captured.out
        assert "John Doe" in captured.out
        assert "Contact Email:" in captured.out
        assert "john@example.com" in captured.out

    def test_print_copyright_outputs_copyright_info(self, capsys):
        """Test that copyright information is printed."""
        metadata = DatasetMetadata(
            copyright_name="Central Statistics Office",
            copyright_href="https://www.cso.ie",
        )
        printer = MetadataPrinter(metadata)
        printer._print_copyright()

        captured = capsys.readouterr()
        assert "Copyright:" in captured.out
        assert "Central Statistics Office" in captured.out
        assert "https://www.cso.ie" in captured.out

    def test_drop_filtered_cols_removes_variables(self, capsys):
        """Test that filtered variables are removed when drop_filtered_cols is True."""
        metadata = DatasetMetadata(
            variables=["County", "Year", "Sex"],
            statistics=[],
            units=[],
        )
        filters: dict = {"County": ["Dublin"]}
        printer = MetadataPrinter(metadata, filters=filters, drop_filtered_cols=True)
        printer._print_variables()

        captured = capsys.readouterr()
        assert "County" not in captured.out
        assert "Year" in captured.out
        assert "Sex" in captured.out


class TestMetadataPrinterFormatting:
    """Tests for MetadataPrinter formatting constants."""

    def test_width_constant(self):
        """Test that WIDTH constant is set."""
        assert MetadataPrinter.WIDTH == 85

    def test_label_width_constant(self):
        """Test that LABEL_WIDTH constant is set."""
        assert MetadataPrinter.LABEL_WIDTH == 20


class TestMetadataPrinterNotes:
    """Tests for notes printing."""

    def test_print_notes_single_note(self, capsys):
        """Test printing a single note."""
        metadata = DatasetMetadata(notes=["This is a test note about the dataset."])
        printer = MetadataPrinter(metadata)
        printer._print_notes()

        captured = capsys.readouterr()
        assert "Notes:" in captured.out
        assert "This is a test note" in captured.out

    def test_print_notes_multiple_notes(self, capsys):
        """Test printing multiple notes."""
        metadata = DatasetMetadata(notes=["First note.", "Second note.", "Third note."])
        printer = MetadataPrinter(metadata)
        printer._print_notes()

        captured = capsys.readouterr()
        assert "First note." in captured.out
        assert "Second note." in captured.out
        assert "Third note." in captured.out

    def test_print_notes_empty(self, capsys):
        """Test printing when no notes."""
        metadata = DatasetMetadata(notes=[])
        printer = MetadataPrinter(metadata)
        printer._print_notes()

        captured = capsys.readouterr()
        # Should not output anything for empty notes
        assert "Notes:" not in captured.out

    def test_print_notes_long_text_wraps(self, capsys):
        """Test that long notes are wrapped."""
        long_note = "This is a very long note " * 20
        metadata = DatasetMetadata(notes=[long_note])
        printer = MetadataPrinter(metadata)
        printer._print_notes()

        captured = capsys.readouterr()
        # Output should contain the note text
        assert "very long note" in captured.out


class TestMetadataPrinterStatistics:
    """Tests for statistics printing within variables."""

    def test_print_variables_with_statistics(self, capsys):
        """Test printing variables that include statistics."""
        metadata = DatasetMetadata(
            variables=["County", "STATISTIC", "Year"],
            statistics=["Population", "Birth Rate"],
            units=["Number", "Rate per 1000"],
        )
        printer = MetadataPrinter(metadata)
        printer._print_variables()

        captured = capsys.readouterr()
        assert "Statistic" in captured.out
        assert "Population" in captured.out
        assert "Birth Rate" in captured.out
        assert "Unit:" in captured.out
        assert "Number" in captured.out

    def test_print_variables_with_filtered_statistics(self, capsys):
        """Test printing variables with filtered statistics."""
        metadata = DatasetMetadata(
            variables=["County", "Statistic", "Year"],
            statistics=["Population", "Birth Rate", "Death Rate"],
            units=["Number", "Rate per 1000", "Rate per 1000"],
        )
        filters: dict = {"Statistic": ["Population"]}
        printer = MetadataPrinter(metadata, filters=filters)
        printer._print_variables()

        captured = capsys.readouterr()
        # Should show only filtered statistics
        assert "Population" in captured.out

    def test_print_variables_statistic_not_first(self, capsys):
        """Test printing when Statistic is not the first variable."""
        metadata = DatasetMetadata(
            variables=["County", "Year", "STATISTIC"],
            statistics=["Population"],
            units=["Number"],
        )
        printer = MetadataPrinter(metadata)
        printer._print_variables()

        captured = capsys.readouterr()
        assert "County" in captured.out
        assert "Year" in captured.out


class TestMetadataPrinterSpatial:
    """Tests for spatial information printing."""

    def test_print_spatial_info(self, capsys):
        """Test printing geographic variable info."""
        metadata = DatasetMetadata(geographic=True, spatial_key="County")
        printer = MetadataPrinter(metadata)
        printer._print_time_and_spatial()

        captured = capsys.readouterr()
        assert "Geographic Variable:" in captured.out
        assert "County" in captured.out

    def test_print_no_spatial_info(self, capsys):
        """Test printing when no geographic data."""
        metadata = DatasetMetadata(geographic=False)
        printer = MetadataPrinter(metadata)
        printer._print_time_and_spatial()

        captured = capsys.readouterr()
        assert "Geographic Variable:" not in captured.out


class TestMetadataPrinterUpdated:
    """Tests for update information printing."""

    def test_print_updated_with_reasons(self, capsys):
        """Test printing update info with reasons."""
        metadata = DatasetMetadata(
            last_updated=datetime(2023, 6, 15), reasons=["Monthly update", "Data revision"]
        )
        printer = MetadataPrinter(metadata)
        printer._print_updated()

        captured = capsys.readouterr()
        assert "2023-06-15" in captured.out
        assert "Reason for Release:" in captured.out
        assert "Monthly update" in captured.out

    def test_print_updated_no_date(self, capsys):
        """Test printing when no update date."""
        metadata = DatasetMetadata(last_updated=None)
        printer = MetadataPrinter(metadata)
        printer._print_updated()

        captured = capsys.readouterr()
        assert "Last Updated:" not in captured.out


class TestMetadataPrinterContactExtended:
    """Additional tests for contact information printing."""

    def test_print_contact_partial(self, capsys):
        """Test printing partial contact info."""
        metadata = DatasetMetadata(contact_name="John Doe", contact_email=None, contact_phone=None)
        printer = MetadataPrinter(metadata)
        printer._print_contact()

        captured = capsys.readouterr()
        assert "Contact Name:" in captured.out
        assert "John Doe" in captured.out
        assert "Contact Email:" not in captured.out

    def test_print_contact_with_phone(self, capsys):
        """Test printing contact info with phone."""
        metadata = DatasetMetadata(contact_name="John Doe", contact_phone="+353 1 234 5678")
        printer = MetadataPrinter(metadata)
        printer._print_contact()

        captured = capsys.readouterr()
        assert "Contact Phone:" in captured.out
        assert "+353 1 234 5678" in captured.out


class TestMetadataPrinterCopyrightExtended:
    """Additional tests for copyright information printing."""

    def test_print_copyright_without_href(self, capsys):
        """Test printing copyright without URL."""
        metadata = DatasetMetadata(copyright_name="Central Statistics Office", copyright_href=None)
        printer = MetadataPrinter(metadata)
        printer._print_copyright()

        captured = capsys.readouterr()
        assert "Copyright:" in captured.out
        assert "Central Statistics Office" in captured.out
        assert "(" not in captured.out  # No URL parentheses

    def test_print_copyright_empty(self, capsys):
        """Test printing when no copyright info."""
        metadata = DatasetMetadata(copyright_name=None, copyright_href=None)
        printer = MetadataPrinter(metadata)
        printer._print_copyright()

        captured = capsys.readouterr()
        assert "Copyright:" not in captured.out


class TestMetadataPrinterPrintLine:
    """Tests for _print_line helper method."""

    def test_print_line_alignment(self, capsys):
        """Test that print_line aligns correctly."""
        metadata = DatasetMetadata()
        printer = MetadataPrinter(metadata)
        printer._print_line("Label:", "Value")

        captured = capsys.readouterr()
        assert "Label:" in captured.out
        assert "Value" in captured.out


class TestMetadataPrinterPrintWrapped:
    """Tests for _print_wrapped helper method."""

    def test_print_wrapped_short_text(self, capsys):
        """Test wrapping short text."""
        metadata = DatasetMetadata()
        printer = MetadataPrinter(metadata)
        printer._print_wrapped("Short text", initial_indent=" " * 10)

        captured = capsys.readouterr()
        assert "Short text" in captured.out

    def test_print_wrapped_long_text(self, capsys):
        """Test wrapping long text."""
        metadata = DatasetMetadata()
        printer = MetadataPrinter(metadata)
        long_text = "This is a very long text that should be wrapped across multiple lines " * 5
        printer._print_wrapped(long_text, initial_indent=" " * 10)

        captured = capsys.readouterr()
        assert "very long text" in captured.out


class TestMetadataPrinterEdgeCases:
    """Edge case tests for MetadataPrinter."""

    def test_print_all_with_minimal_metadata(self, capsys):
        """Test print_all with minimal metadata."""
        metadata = DatasetMetadata()
        printer = MetadataPrinter(metadata)
        printer.print_all()

        captured = capsys.readouterr()
        # Should not raise, even with minimal metadata
        assert "Code:" in captured.out

    def test_print_header_unknown_code(self, capsys):
        """Test header with None table code."""
        metadata = DatasetMetadata(table_code=None)  # type: ignore
        printer = MetadataPrinter(metadata)
        printer._print_header()

        captured = capsys.readouterr()
        # None is printed when table_code is None
        assert "None" in captured.out or "Code:" in captured.out

    def test_drop_filtered_cols_with_non_matching_filter(self, capsys):
        """Test drop_filtered_cols when filter doesn't match a variable."""
        metadata = DatasetMetadata(
            variables=["County", "Year"],
            statistics=[],
            units=[],
        )
        filters: dict = {"NonExistent": ["Value"]}
        printer = MetadataPrinter(metadata, filters=filters, drop_filtered_cols=True)
        printer._print_variables()

        captured = capsys.readouterr()
        # Should still print all variables
        assert "County" in captured.out
        assert "Year" in captured.out

    def test_statistics_index_out_of_range(self, capsys):
        """Test handling when statistics index is out of range for units."""
        metadata = DatasetMetadata(
            variables=["STATISTIC"],
            statistics=["Pop", "Rate", "Count"],
            units=["Number"],  # Only one unit, but 3 statistics
        )
        printer = MetadataPrinter(metadata)
        printer._print_variables()

        captured = capsys.readouterr()
        # Should handle gracefully with "N/A" for missing units
        assert "Pop" in captured.out
