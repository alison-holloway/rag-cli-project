#!/usr/bin/env python3
"""
HTML Documentation Scraper

Downloads HTML documentation files from TOC (Table of Contents) pages
for use with the RAG indexing system.

Usage:
    python tools/html_scraper.py                    # Use default config
    python tools/html_scraper.py --dry-run          # Show what would be downloaded
    python tools/html_scraper.py --verbose          # Show detailed progress
    python tools/html_scraper.py --config custom.yaml  # Use custom config
"""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import click
import requests
import yaml
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Default config location
DEFAULT_CONFIG_PATH = Path("config/html_scraper.yaml")

console = Console()


class ScraperConfig(BaseModel):
    """Configuration for the HTML scraper."""

    output_dir: str = Field(default="data/documents/html/")
    delay_seconds: float = Field(default=1.0, ge=0.0)
    sources: list[str] = Field(default_factory=list)
    skip_patterns: list[str] = Field(
        default_factory=lambda: ["index.htm", "toc.htm", "lot.htm", "copyright"]
    )

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one source URL is required")
        return v


@dataclass
class ScraperStats:
    """Statistics from a scraping run."""

    tocs_processed: int = 0
    urls_found: int = 0
    urls_downloaded: int = 0
    urls_failed: int = 0
    urls_skipped: int = 0
    errors: list[str] = field(default_factory=list)


class HTMLScraper:
    """Scrapes HTML documentation from TOC pages."""

    def __init__(self, config: ScraperConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.stats = ScraperStats()

    def extract_urls_from_toc(self, toc_url: str) -> list[str]:
        """Extract all content URLs from a documentation TOC page."""
        if self.verbose:
            console.print(f"  Fetching TOC: {toc_url}")

        response = requests.get(toc_url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a", href=True)

        # Get base URL for resolving relative paths
        base_url = toc_url.rsplit("/", 1)[0] + "/"

        content_urls = []
        for link in links:
            href = link["href"]

            # Skip external links
            if href.startswith("http"):
                continue

            # Skip patterns from config
            if any(skip.lower() in href.lower() for skip in self.config.skip_patterns):
                continue

            # Make absolute URL
            absolute_url = urljoin(base_url, href)

            # Remove anchor/fragment
            parsed = urlparse(absolute_url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            # Only include .html files
            if clean_url.endswith(".html"):
                content_urls.append(clean_url)

        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in content_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    def sanitize_filename(self, url: str) -> str:
        """Convert URL to a safe filename."""
        parsed = urlparse(url)
        path = parsed.path.lstrip("/")
        filename = path.replace("/", "_")
        return filename

    def download_html(self, url: str, output_dir: Path) -> bool:
        """Download HTML from URL and save to output directory."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            filename = self.sanitize_filename(url)
            filepath = output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)

            return True

        except requests.RequestException as e:
            self.stats.errors.append(f"{url}: {e}")
            return False

    def run(self, dry_run: bool = False) -> ScraperStats:
        """Run the scraper on all configured TOC sources."""
        output_dir = Path(self.config.output_dir)

        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Extract URLs from all TOCs
        all_urls: list[str] = []

        console.print("\n[bold]Phase 1:[/bold] Extracting URLs from TOC pages\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing TOCs", total=len(self.config.sources))

            for toc_url in self.config.sources:
                try:
                    urls = self.extract_urls_from_toc(toc_url)
                    all_urls.extend(urls)
                    self.stats.tocs_processed += 1

                    if self.verbose:
                        console.print(f"    Found {len(urls)} URLs")

                except requests.RequestException as e:
                    self.stats.errors.append(f"TOC {toc_url}: {e}")
                    console.print(f"  [red]Error:[/red] {e}")

                progress.advance(task)

        # Deduplicate URLs across all TOCs
        seen = set()
        unique_urls = []
        for url in all_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        self.stats.urls_found = len(unique_urls)
        self.stats.urls_skipped = len(all_urls) - len(unique_urls)

        console.print(f"\nFound {len(unique_urls)} unique URLs", end="")
        if self.stats.urls_skipped > 0:
            console.print(f" ({self.stats.urls_skipped} duplicates skipped)")
        else:
            console.print()

        if dry_run:
            console.print(
                "\n[yellow]Dry run - URLs that would be downloaded:[/yellow]\n"
            )
            for url in unique_urls:
                console.print(f"  {url}")
            return self.stats

        # Phase 2: Download HTML files
        console.print("\n[bold]Phase 2:[/bold] Downloading HTML files\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading", total=len(unique_urls))

            for url in unique_urls:
                if self.download_html(url, output_dir):
                    self.stats.urls_downloaded += 1
                else:
                    self.stats.urls_failed += 1

                progress.advance(task)

                # Respectful delay between requests
                if self.config.delay_seconds > 0:
                    import time

                    time.sleep(self.config.delay_seconds)

        return self.stats


def load_config(config_path: Path) -> ScraperConfig:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return ScraperConfig(**data)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=False, path_type=Path),
    default=DEFAULT_CONFIG_PATH,
    help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH})",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output directory from config",
)
@click.option(
    "--delay",
    type=float,
    default=None,
    help="Override delay between requests (seconds)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be downloaded without downloading",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed progress information",
)
def main(
    config_path: Path,
    output_dir: Path | None,
    delay: float | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Download HTML documentation files from TOC pages.

    Extracts URLs from configured TOC (Table of Contents) pages and downloads
    the HTML content for RAG indexing.
    """
    console.print("[bold]HTML Documentation Scraper[/bold]\n")

    # Load config
    try:
        config = load_config(config_path)
    except Exception as e:
        raise click.ClickException(str(e))

    # Apply CLI overrides
    if output_dir is not None:
        config.output_dir = str(output_dir)
    if delay is not None:
        config.delay_seconds = delay

    if verbose:
        console.print(f"Config: {config_path}")
        console.print(f"Output: {config.output_dir}")
        console.print(f"Delay: {config.delay_seconds}s")
        console.print(f"Sources: {len(config.sources)}")

    # Run scraper
    scraper = HTMLScraper(config, verbose=verbose)
    stats = scraper.run(dry_run=dry_run)

    # Print summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Summary[/bold]")
    console.print("=" * 60)
    console.print(f"TOCs processed:  {stats.tocs_processed}")
    console.print(f"URLs found:      {stats.urls_found}")

    if not dry_run:
        console.print(f"Downloaded:      {stats.urls_downloaded}")
        console.print(f"Failed:          {stats.urls_failed}")
        console.print(f"Output:          {config.output_dir}")

    if stats.errors:
        console.print("\n[red]Errors:[/red]")
        for error in stats.errors:
            console.print(f"  - {error}")


if __name__ == "__main__":
    main()
