"""Tests for the HTML documentation scraper."""

# Import the scraper module
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from html_scraper import HTMLScraper, ScraperConfig, ScraperStats, load_config


class TestScraperConfig:
    """Tests for ScraperConfig validation."""

    def test_default_values(self):
        """Test that config has sensible defaults."""
        config = ScraperConfig(sources=["https://example.com/toc.htm"])
        assert config.output_dir == "data/documents/html/"
        assert config.delay_seconds == 1.0
        assert "toc.htm" in config.skip_patterns

    def test_custom_values(self):
        """Test config with custom values."""
        config = ScraperConfig(
            output_dir="/custom/path/",
            delay_seconds=2.5,
            sources=["https://example.com/docs/toc.htm"],
            skip_patterns=["index.html"],
        )
        assert config.output_dir == "/custom/path/"
        assert config.delay_seconds == 2.5
        assert config.skip_patterns == ["index.html"]

    def test_empty_sources_raises_error(self):
        """Test that empty sources list raises validation error."""
        with pytest.raises(ValueError, match="At least one source URL is required"):
            ScraperConfig(sources=[])

    def test_negative_delay_raises_error(self):
        """Test that negative delay raises validation error."""
        with pytest.raises(ValueError):
            ScraperConfig(sources=["https://example.com/toc.htm"], delay_seconds=-1.0)


class TestHTMLScraper:
    """Tests for HTMLScraper class."""

    @pytest.fixture
    def config(self):
        """Create a basic config for testing."""
        return ScraperConfig(
            output_dir="/tmp/test_output",
            delay_seconds=0,  # No delay for tests
            sources=["https://docs.example.com/guide/toc.htm"],
            skip_patterns=["index.htm", "toc.htm", "copyright"],
        )

    @pytest.fixture
    def scraper(self, config):
        """Create a scraper instance for testing."""
        return HTMLScraper(config)

    def test_sanitize_filename_simple(self, scraper):
        """Test filename sanitization with simple URL."""
        url = "https://example.com/docs/guide.html"
        filename = scraper.sanitize_filename(url)
        assert filename == "docs_guide.html"

    def test_sanitize_filename_nested_path(self, scraper):
        """Test filename sanitization with nested path."""
        url = "https://example.com/en/docs/v2/intro.html"
        filename = scraper.sanitize_filename(url)
        assert filename == "en_docs_v2_intro.html"

    def test_sanitize_filename_root_path(self, scraper):
        """Test filename sanitization with root path."""
        url = "https://example.com/index.html"
        filename = scraper.sanitize_filename(url)
        assert filename == "index.html"

    def test_extract_urls_from_toc(self, scraper):
        """Test URL extraction from TOC HTML."""
        mock_html = """
        <html>
        <body>
            <a href="intro.html">Introduction</a>
            <a href="chapter1.html#section1">Chapter 1</a>
            <a href="chapter2.html">Chapter 2</a>
            <a href="toc.htm">Table of Contents</a>
            <a href="index.htm">Index</a>
            <a href="https://external.com/page.html">External</a>
            <a href="copyright.html">Copyright</a>
        </body>
        </html>
        """

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.content = mock_html.encode()
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            urls = scraper.extract_urls_from_toc(
                "https://docs.example.com/guide/toc.htm"
            )

        # Should include intro, chapter1, chapter2
        # Should exclude: toc.htm, index.htm, external link, copyright
        assert len(urls) == 3
        assert "https://docs.example.com/guide/intro.html" in urls
        assert "https://docs.example.com/guide/chapter1.html" in urls
        assert "https://docs.example.com/guide/chapter2.html" in urls

    def test_extract_urls_removes_anchors(self, scraper):
        """Test that URL anchors/fragments are removed."""
        mock_html = """
        <html>
        <body>
            <a href="page.html#section1">Section 1</a>
            <a href="page.html#section2">Section 2</a>
        </body>
        </html>
        """

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.content = mock_html.encode()
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            urls = scraper.extract_urls_from_toc("https://docs.example.com/toc.htm")

        # Both links point to same page (different anchors), should dedupe
        assert len(urls) == 1
        assert urls[0] == "https://docs.example.com/page.html"

    def test_extract_urls_only_html_files(self, scraper):
        """Test that only .html files are extracted."""
        mock_html = """
        <html>
        <body>
            <a href="page.html">HTML Page</a>
            <a href="document.pdf">PDF Document</a>
            <a href="image.png">Image</a>
            <a href="styles.css">Stylesheet</a>
        </body>
        </html>
        """

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.content = mock_html.encode()
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            urls = scraper.extract_urls_from_toc("https://docs.example.com/toc.htm")

        assert len(urls) == 1
        assert urls[0].endswith(".html")

    def test_download_html_success(self, scraper):
        """Test successful HTML download."""
        mock_html = "<html><body>Test content</body></html>"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.text = mock_html
                mock_response.raise_for_status = MagicMock()
                mock_get.return_value = mock_response

                result = scraper.download_html(
                    "https://example.com/docs/page.html", output_dir
                )

            assert result is True
            saved_file = output_dir / "docs_page.html"
            assert saved_file.exists()
            assert saved_file.read_text() == mock_html

    def test_download_html_failure(self, scraper):
        """Test HTML download failure handling."""
        import requests

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch("requests.get") as mock_get:
                mock_get.side_effect = requests.RequestException("Connection failed")

                result = scraper.download_html(
                    "https://example.com/docs/page.html", output_dir
                )

            assert result is False
            assert len(scraper.stats.errors) == 1
            assert "Connection failed" in scraper.stats.errors[0]


class TestScraperRun:
    """Tests for the scraper run method."""

    @pytest.fixture
    def config(self):
        """Create a config for testing."""
        return ScraperConfig(
            output_dir="/tmp/test_output",
            delay_seconds=0,
            sources=["https://docs.example.com/toc.htm"],
            skip_patterns=["toc.htm"],
        )

    def test_dry_run_does_not_download(self, config):
        """Test that dry run mode doesn't download files."""
        scraper = HTMLScraper(config)

        mock_toc_html = """
        <html><body>
            <a href="page1.html">Page 1</a>
            <a href="page2.html">Page 2</a>
        </body></html>
        """

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.content = mock_toc_html.encode()
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            stats = scraper.run(dry_run=True)

        # Should only call get once (for TOC), not for individual pages
        assert mock_get.call_count == 1
        assert stats.urls_found == 2
        assert stats.urls_downloaded == 0

    def test_run_downloads_files(self, config):
        """Test that run downloads files when not in dry run mode."""
        mock_toc_html = """
        <html><body>
            <a href="page1.html">Page 1</a>
        </body></html>
        """
        mock_page_html = "<html><body>Page content</body></html>"

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            scraper = HTMLScraper(config)

            with patch("requests.get") as mock_get:
                # First call returns TOC, subsequent calls return page content
                mock_toc_response = MagicMock()
                mock_toc_response.content = mock_toc_html.encode()
                mock_toc_response.raise_for_status = MagicMock()

                mock_page_response = MagicMock()
                mock_page_response.text = mock_page_html
                mock_page_response.raise_for_status = MagicMock()

                mock_get.side_effect = [mock_toc_response, mock_page_response]

                stats = scraper.run(dry_run=False)

            assert stats.urls_found == 1
            assert stats.urls_downloaded == 1
            assert stats.urls_failed == 0

            # Check file was created
            output_path = Path(tmpdir)
            files = list(output_path.glob("*.html"))
            assert len(files) == 1


class TestLoadConfig:
    """Tests for config file loading."""

    def test_load_valid_config(self):
        """Test loading a valid YAML config file."""
        config_content = """
output_dir: /custom/output/
delay_seconds: 2.0
sources:
  - https://example.com/docs/toc.htm
skip_patterns:
  - index.htm
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            config = load_config(config_path)
            assert config.output_dir == "/custom/output/"
            assert config.delay_seconds == 2.0
            assert len(config.sources) == 1
        finally:
            config_path.unlink()

    def test_load_missing_config_raises_error(self):
        """Test that missing config file raises error."""
        import click

        with pytest.raises(click.ClickException, match="Config file not found"):
            load_config(Path("/nonexistent/config.yaml"))


class TestScraperStats:
    """Tests for ScraperStats dataclass."""

    def test_default_stats(self):
        """Test default stats values."""
        stats = ScraperStats()
        assert stats.tocs_processed == 0
        assert stats.urls_found == 0
        assert stats.urls_downloaded == 0
        assert stats.urls_failed == 0
        assert stats.urls_skipped == 0
        assert stats.errors == []

    def test_stats_mutation(self):
        """Test that stats can be updated."""
        stats = ScraperStats()
        stats.tocs_processed = 2
        stats.urls_found = 10
        stats.errors.append("Test error")

        assert stats.tocs_processed == 2
        assert stats.urls_found == 10
        assert len(stats.errors) == 1
