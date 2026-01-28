"""Tests for the progress module."""

from unittest.mock import patch

import pytest

from src.progress import (
    DocumentProcessingProgress,
    MultiStageProgress,
    create_progress,
    create_simple_progress,
    create_spinner_progress,
    iterate_with_progress,
    print_error,
    print_info,
    print_stats,
    print_success,
    print_warning,
    progress_bar,
    spinner,
)


class TestProgressFactories:
    """Tests for progress bar factory functions."""

    def test_create_progress(self):
        """Test creating a standard progress bar."""
        progress = create_progress()
        assert progress is not None
        # Check that it has the expected columns
        assert len(progress.columns) > 0

    def test_create_simple_progress(self):
        """Test creating a simple progress bar."""
        progress = create_simple_progress()
        assert progress is not None

    def test_create_spinner_progress(self):
        """Test creating a spinner progress."""
        progress = create_spinner_progress()
        assert progress is not None


class TestSpinnerContextManager:
    """Tests for the spinner context manager."""

    def test_spinner_basic(self):
        """Test basic spinner usage."""
        with spinner("Loading..."):
            # Just verify it doesn't raise
            pass

    def test_spinner_completes(self):
        """Test that spinner completes without error."""
        completed = False
        with spinner("Processing"):
            completed = True
        assert completed


class TestProgressBarContextManager:
    """Tests for the progress_bar context manager."""

    def test_progress_bar_basic(self):
        """Test basic progress bar usage."""
        with progress_bar("Processing", total=10) as advance:
            for _ in range(10):
                advance()

    def test_progress_bar_advance_custom_amount(self):
        """Test advancing by custom amounts."""
        with progress_bar("Processing", total=100) as advance:
            advance(50)
            advance(30)
            advance(20)


class TestIterateWithProgress:
    """Tests for iterate_with_progress function."""

    def test_iterate_basic(self):
        """Test basic iteration with progress."""
        items = [1, 2, 3, 4, 5]
        result = list(iterate_with_progress(items, "Processing"))
        assert result == items

    def test_iterate_empty_list(self):
        """Test iteration with empty list."""
        items = []
        result = list(iterate_with_progress(items, "Processing"))
        assert result == []

    def test_iterate_with_show_item(self):
        """Test iteration showing current item."""
        items = ["a", "b", "c"]
        result = list(iterate_with_progress(items, "Processing", show_item=True))
        assert result == items

    def test_iterate_preserves_order(self):
        """Test that iteration preserves item order."""
        items = list(range(100))
        result = list(iterate_with_progress(items, "Processing"))
        assert result == items


class TestDocumentProcessingProgress:
    """Tests for DocumentProcessingProgress class."""

    def test_basic_usage(self):
        """Test basic progress tracking."""
        progress = DocumentProcessingProgress()
        progress.start(total_files=5)
        for i in range(5):
            progress.update(f"File {i + 1}")
            progress.advance()
        progress.finish()

    def test_context_manager(self):
        """Test using as context manager."""
        with DocumentProcessingProgress() as progress:
            progress.start(total_files=3)
            for i in range(3):
                progress.advance()

    def test_finish_without_start(self):
        """Test finishing without starting (should not error)."""
        progress = DocumentProcessingProgress()
        progress.finish()  # Should not raise

    def test_update_without_start(self):
        """Test updating without starting (should not error)."""
        progress = DocumentProcessingProgress()
        progress.update("test")  # Should not raise
        progress.advance()  # Should not raise


class TestMultiStageProgress:
    """Tests for MultiStageProgress class."""

    def test_basic_usage(self):
        """Test basic multi-stage progress."""
        stages = ["Load", "Chunk", "Embed", "Store"]
        with MultiStageProgress(stages) as progress:
            for stage in stages:
                progress.update_stage(stage, 50)
                progress.complete_stage(stage)

    def test_update_stage_progress(self):
        """Test updating stage progress."""
        stages = ["Step 1", "Step 2"]
        progress = MultiStageProgress(stages)
        progress.start()
        progress.update_stage("Step 1", 25)
        progress.update_stage("Step 1", 50)
        progress.update_stage("Step 1", 75)
        progress.complete_stage("Step 1")
        progress.finish()

    def test_finish_without_start(self):
        """Test finishing without starting."""
        progress = MultiStageProgress(["Step 1"])
        progress.finish()  # Should not raise


class TestPrintFunctions:
    """Tests for print helper functions."""

    @patch("src.progress.console")
    def test_print_success(self, mock_console):
        """Test print_success function."""
        print_success("Operation completed")
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "Operation completed" in call_args
        assert "green" in call_args

    @patch("src.progress.console")
    def test_print_error(self, mock_console):
        """Test print_error function."""
        print_error("Something failed")
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "Something failed" in call_args
        assert "red" in call_args

    @patch("src.progress.console")
    def test_print_warning(self, mock_console):
        """Test print_warning function."""
        print_warning("Be careful")
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "Be careful" in call_args
        assert "yellow" in call_args

    @patch("src.progress.console")
    def test_print_info(self, mock_console):
        """Test print_info function."""
        print_info("FYI")
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "FYI" in call_args
        assert "blue" in call_args

    @patch("src.progress.console")
    def test_print_stats(self, mock_console):
        """Test print_stats function."""
        stats = {"Documents": 10, "Chunks": 50, "Size": "1.2 MB"}
        print_stats(stats)
        mock_console.print.assert_called_once()


class TestProgressIntegration:
    """Integration tests for progress indicators."""

    def test_nested_progress_operations(self):
        """Test that progress operations can be nested."""
        with spinner("Outer operation"):
            items = [1, 2, 3]
            result = list(iterate_with_progress(items, "Inner operation"))
            assert result == items

    def test_progress_with_exception(self):
        """Test that progress handles exceptions gracefully."""
        with pytest.raises(ValueError):
            with progress_bar("Processing", total=10) as advance:
                advance(5)
                raise ValueError("Simulated error")

    def test_spinner_with_exception(self):
        """Test that spinner handles exceptions gracefully."""
        with pytest.raises(RuntimeError):
            with spinner("Loading"):
                raise RuntimeError("Simulated error")
