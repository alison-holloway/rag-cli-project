#!/usr/bin/env python3
"""
DITA Documentation Ingestion Tool

Parses DITA-generated HTML files and performs semantic chunking based on
document type (task, concept, reference, topic). Stores chunks in ChromaDB
with rich metadata for improved RAG retrieval.

Usage:
    python tools/ingest_dita_docs.py                    # Use default config
    python tools/ingest_dita_docs.py --dry-run          # Preview chunks without storing
    python tools/ingest_dita_docs.py --verbose          # Detailed logging
    python tools/ingest_dita_docs.py --clear-first      # Clear existing docs before ingesting
    python tools/ingest_dita_docs.py --input-dir ./docs # Override input directory
"""

import hashlib
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import click
import yaml
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import Chunk
from src.vector_store import VectorStore

# Default config location
DEFAULT_CONFIG_PATH = Path("config/dita_chunker.yaml")

console = Console()


# =============================================================================
# Configuration
# =============================================================================


class DITAChunkerConfig(BaseModel):
    """Configuration for the DITA chunker."""

    input_dir: str = Field(default="data/documents/html/")
    collection_name: str = Field(default="rag_documents")
    min_chunk_size: int = Field(default=200)
    max_chunk_size: int = Field(default=4000)
    target_chunk_size: int = Field(default=1500)
    verbose: bool = Field(default=False)


def load_config(config_path: Path) -> DITAChunkerConfig:
    """Load configuration from YAML file."""
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return DITAChunkerConfig(**data)
    return DITAChunkerConfig()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DITAStep:
    """A step in a task procedure."""

    number: int
    command: str  # The step instruction (in <span>)
    details: str  # Additional content (in <div>)
    code_blocks: list[str] = field(default_factory=list)


@dataclass
class DITAExample:
    """An example block."""

    title: str | None
    description: str
    code: str | None


@dataclass
class DITASection:
    """A section within a DITA document."""

    id: str | None
    title: str | None
    content: str  # Text content
    html: str  # Original HTML


@dataclass
class DITADocument:
    """Represents a parsed DITA HTML document."""

    file_path: Path
    dita_type: str  # task, concept, reference, topic
    title: str
    abstract: str | None
    content_text: str  # Full text content
    metadata: dict[str, str]  # Extracted meta tags
    sections: list[DITASection] = field(default_factory=list)
    steps: list[DITAStep] | None = None  # For task docs only
    examples: list[DITAExample] = field(default_factory=list)
    document_hash: str = ""

    def __post_init__(self):
        """Calculate document hash after initialization."""
        if not self.document_hash:
            content = self.file_path.read_bytes()
            self.document_hash = hashlib.md5(content).hexdigest()


# =============================================================================
# DITA HTML Parser
# =============================================================================


class DITAHTMLParser:
    """Parses DITA-generated HTML into structured components."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.soup: BeautifulSoup | None = None

    def parse(self, file_path: Path) -> DITADocument:
        """Parse a DITA HTML file into structured document."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        self.soup = BeautifulSoup(content, "lxml")

        # Extract document type
        dita_type = self._extract_dita_type()

        # Extract metadata
        metadata = self._extract_metadata()

        # Extract title
        title = self._extract_title()

        # Extract abstract
        abstract = self._extract_abstract()

        # Extract main content
        content_text = self._extract_content_text()

        # Parse sections
        sections = self._parse_sections()

        # Parse steps (for task documents)
        steps = self._parse_steps() if dita_type == "task" else None

        # Parse examples
        examples = self._parse_examples()

        return DITADocument(
            file_path=file_path,
            dita_type=dita_type,
            title=title,
            abstract=abstract,
            content_text=content_text,
            metadata=metadata,
            sections=sections,
            steps=steps,
            examples=examples,
        )

    def _extract_dita_type(self) -> str:
        """Extract DC.Type from meta tags."""
        meta = self.soup.find("meta", attrs={"name": "DC.Type"})
        if meta and meta.get("content"):
            return meta["content"].lower()
        return "topic"  # Default

    def _extract_metadata(self) -> dict[str, str]:
        """Extract metadata from meta tags."""
        metadata = {}

        # Map of meta names to our metadata keys
        meta_map = {
            "DC.Title": "dc_title",
            "DC.Identifier": "dc_identifier",
            "abstract": "abstract",
            "dcterms.created": "created",
        }

        for meta_name, key in meta_map.items():
            meta = self.soup.find("meta", attrs={"name": meta_name})
            if meta and meta.get("content"):
                metadata[key] = meta["content"]

        return metadata

    def _extract_title(self) -> str:
        """Extract page title."""
        # Try h2 with sect2 class first (DITA pattern)
        h2 = self.soup.find("h2", class_="sect2")
        if h2:
            return h2.get_text(strip=True)

        # Fall back to title tag
        title_tag = self.soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)

        return "Untitled"

    def _extract_abstract(self) -> str | None:
        """Extract abstract from meta tag."""
        meta = self.soup.find("meta", attrs={"name": "abstract"})
        if meta and meta.get("content"):
            return meta["content"]
        return None

    def _extract_content_text(self) -> str:
        """Extract main content as plain text."""
        # Find the main article content
        article = self.soup.find("article")
        if not article:
            # Fall back to body
            article = self.soup.find("body")

        if not article:
            return ""

        # Remove script and style tags
        for tag in article.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()

        # Get text with some structure preserved
        return article.get_text(separator="\n", strip=True)

    def _parse_sections(self) -> list[DITASection]:
        """Parse <div class='section'> elements."""
        sections = []
        section_divs = self.soup.find_all("div", class_="section")

        for div in section_divs:
            section_id = div.get("id")

            # Try to find a title within the section
            title = None
            title_elem = div.find(["h3", "h4", "p"], class_=lambda x: x and "title" in x.lower() if x else False)
            if title_elem:
                title = title_elem.get_text(strip=True)

            content = div.get_text(separator="\n", strip=True)
            html = str(div)

            sections.append(DITASection(id=section_id, title=title, content=content, html=html))

        return sections

    def _parse_steps(self) -> list[DITAStep] | None:
        """Parse task steps from <ol>/<li class='stepexpand'>."""
        steps = []

        # Find all stepexpand list items
        step_items = self.soup.find_all("li", class_="stepexpand")

        if not step_items:
            return None

        for i, li in enumerate(step_items, 1):
            # Command is usually in the first <span>
            span = li.find("span", recursive=False)
            command = span.get_text(strip=True) if span else ""

            # Details are in <div> elements after the span
            details_parts = []
            code_blocks = []

            for child in li.children:
                if isinstance(child, Tag):
                    if child.name == "div":
                        # Get text from div, excluding code blocks
                        div_text = []
                        for elem in child.descendants:
                            if isinstance(elem, Tag) and elem.name == "pre":
                                code_blocks.append(elem.get_text(strip=True))
                            elif isinstance(elem, str) and elem.strip():
                                div_text.append(elem.strip())
                        if div_text:
                            details_parts.append(" ".join(div_text))

            details = "\n".join(details_parts)

            steps.append(DITAStep(number=i, command=command, details=details, code_blocks=code_blocks))

        return steps if steps else None

    def _parse_examples(self) -> list[DITAExample]:
        """Parse <div class='example'> blocks."""
        examples = []
        example_divs = self.soup.find_all("div", class_="example")

        for div in example_divs:
            # Title is in <p class="titleinexample">
            title_p = div.find("p", class_="titleinexample")
            title = title_p.get_text(strip=True) if title_p else None

            # Code is in <pre class="pre codeblock">
            code_pre = div.find("pre", class_="codeblock")
            if not code_pre:
                code_pre = div.find("pre")
            code = code_pre.get_text(strip=True) if code_pre else None

            # Description is other text content
            desc_parts = []
            for elem in div.children:
                if isinstance(elem, Tag):
                    if elem.name == "p" and "titleinexample" not in (elem.get("class") or []):
                        desc_parts.append(elem.get_text(strip=True))

            description = "\n".join(desc_parts)

            examples.append(DITAExample(title=title, description=description, code=code))

        return examples


# =============================================================================
# Semantic Chunkers (Strategy Pattern)
# =============================================================================


class DITAChunker(ABC):
    """Base class for DITA document chunkers."""

    def __init__(self, config: DITAChunkerConfig):
        self.config = config

    @abstractmethod
    def chunk(self, document: DITADocument) -> list[Chunk]:
        """Convert DITA document to chunks."""
        pass

    def _build_metadata(self, document: DITADocument, **extra) -> dict:
        """Build chunk metadata from document metadata."""
        base_meta = {
            "source_file": document.file_path.name,
            "source_path": str(document.file_path.absolute()),
            "file_type": "dita_html",
            "dita_type": document.dita_type,
            "document_hash": document.document_hash,
            "title": document.title,
        }

        if document.abstract:
            # Truncate abstract for ChromaDB
            base_meta["abstract"] = document.abstract[:500]

        # Add extracted metadata
        if "dc_identifier" in document.metadata:
            base_meta["dc_identifier"] = document.metadata["dc_identifier"]
        if "created" in document.metadata:
            base_meta["created"] = document.metadata["created"]

        base_meta.update(extra)
        return base_meta

    def _create_chunk(
        self, content: str, document: DITADocument, chunk_index: int, **extra_metadata
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        metadata = self._build_metadata(document, chunk_index=chunk_index, **extra_metadata)
        return Chunk(
            content=content,
            chunk_index=chunk_index,
            start_char=0,
            end_char=len(content),
            metadata=metadata,
        )

    def _is_too_large(self, text: str) -> bool:
        """Check if text exceeds max chunk size."""
        return len(text) > self.config.max_chunk_size

    def _is_too_small(self, text: str) -> bool:
        """Check if text is below min chunk size."""
        return len(text) < self.config.min_chunk_size


class TaskChunker(DITAChunker):
    """Chunks task documents - keeps entire procedures together."""

    def chunk(self, document: DITADocument) -> list[Chunk]:
        """Keep entire task procedure as one chunk."""
        chunks = []

        # Build content: title + abstract + all steps
        content_parts = []

        # Add title
        content_parts.append(f"# {document.title}")

        # Add abstract if present
        if document.abstract:
            content_parts.append(f"\n{document.abstract}\n")

        # Add steps
        if document.steps:
            content_parts.append("\n## Steps\n")
            for step in document.steps:
                content_parts.append(f"{step.number}. {step.command}")
                if step.details:
                    content_parts.append(f"   {step.details}")
                for code in step.code_blocks:
                    content_parts.append(f"   ```\n   {code}\n   ```")

        # Add examples
        if document.examples:
            content_parts.append("\n## Examples\n")
            for example in document.examples:
                if example.title:
                    content_parts.append(f"### {example.title}")
                if example.description:
                    content_parts.append(example.description)
                if example.code:
                    content_parts.append(f"```\n{example.code}\n```")

        content = "\n".join(content_parts)

        # If content is too large, split at step boundaries
        if self._is_too_large(content) and document.steps and len(document.steps) > 3:
            return self._split_large_task(document)

        # Create single chunk for entire task
        chunk = self._create_chunk(content, document, chunk_index=0, content_type="procedure")
        chunks.append(chunk)

        return chunks

    def _split_large_task(self, document: DITADocument) -> list[Chunk]:
        """Split a large task at step boundaries."""
        chunks = []
        chunk_index = 0

        # Header for all chunks
        header = f"# {document.title}\n"
        if document.abstract:
            header += f"{document.abstract}\n"

        # Group steps into chunks
        current_content = [header, "\n## Steps\n"]
        current_step_start = 1

        for step in document.steps:
            step_content = [f"{step.number}. {step.command}"]
            if step.details:
                step_content.append(f"   {step.details}")
            for code in step.code_blocks:
                step_content.append(f"   ```\n   {code}\n   ```")

            step_text = "\n".join(step_content)

            # Check if adding this step would exceed max size
            test_content = "\n".join(current_content) + "\n" + step_text
            if self._is_too_large(test_content) and len(current_content) > 2:
                # Save current chunk
                content = "\n".join(current_content)
                chunk = self._create_chunk(
                    content,
                    document,
                    chunk_index,
                    content_type="procedure",
                    step_range=f"{current_step_start}-{step.number - 1}",
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with header
                current_content = [header, "\n## Steps (continued)\n"]
                current_step_start = step.number

            current_content.append(step_text)

        # Save final chunk
        if current_content:
            content = "\n".join(current_content)
            if not self._is_too_small(content):
                chunk = self._create_chunk(
                    content,
                    document,
                    chunk_index,
                    content_type="procedure",
                    step_range=f"{current_step_start}-{len(document.steps)}",
                )
                chunks.append(chunk)

        return chunks


class ConceptChunker(DITAChunker):
    """Chunks concept documents - typically kept whole."""

    def chunk(self, document: DITADocument) -> list[Chunk]:
        """Keep concept as single chunk unless too large."""
        chunks = []

        # Build content
        content_parts = [f"# {document.title}"]

        if document.abstract:
            content_parts.append(f"\n{document.abstract}\n")

        # Add main content (excluding duplicated abstract)
        main_content = document.content_text
        if document.abstract and main_content.startswith(document.abstract):
            main_content = main_content[len(document.abstract) :].strip()

        content_parts.append(main_content)

        content = "\n".join(content_parts)

        # If too large, split at section boundaries
        if self._is_too_large(content) and document.sections:
            return self._split_at_sections(document)

        chunk = self._create_chunk(content, document, chunk_index=0, content_type="concept")
        chunks.append(chunk)

        return chunks

    def _split_at_sections(self, document: DITADocument) -> list[Chunk]:
        """Split concept at section boundaries."""
        chunks = []

        header = f"# {document.title}\n"
        if document.abstract:
            header += f"{document.abstract}\n"

        for i, section in enumerate(document.sections):
            content = header + "\n" + section.content
            chunk = self._create_chunk(
                content,
                document,
                chunk_index=i,
                content_type="concept",
                section_id=section.id,
                section_title=section.title,
            )
            chunks.append(chunk)

        return chunks


class ReferenceChunker(DITAChunker):
    """Chunks reference documents - splits at semantic boundaries."""

    def chunk(self, document: DITADocument) -> list[Chunk]:
        """Split reference doc at logical boundaries."""
        chunks = []
        chunk_index = 0

        # Header with title and abstract
        header = f"# {document.title}\n"
        if document.abstract:
            header += f"{document.abstract}\n"

        # Main content (syntax + params) as first chunk
        main_content = document.content_text

        # Remove examples from main content for separate chunking
        example_texts = [ex.code or "" for ex in document.examples]
        for ex_text in example_texts:
            if ex_text and ex_text in main_content:
                main_content = main_content.replace(ex_text, "")

        # Create main content chunk if not empty
        if main_content.strip() and not self._is_too_small(main_content):
            chunk = self._create_chunk(
                header + main_content.strip(), document, chunk_index, content_type="reference"
            )
            chunks.append(chunk)
            chunk_index += 1

        # Create separate chunks for examples if there are multiple
        if len(document.examples) > 1:
            for example in document.examples:
                example_content = header + "## Example\n"
                if example.title:
                    example_content += f"### {example.title}\n"
                if example.description:
                    example_content += f"{example.description}\n"
                if example.code:
                    example_content += f"```\n{example.code}\n```"

                if not self._is_too_small(example_content):
                    chunk = self._create_chunk(
                        example_content,
                        document,
                        chunk_index,
                        content_type="example",
                        example_title=example.title,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

        # If no chunks created, use full document
        if not chunks:
            content = header + document.content_text
            chunk = self._create_chunk(content, document, chunk_index=0, content_type="reference")
            chunks.append(chunk)

        return chunks


class TopicChunker(DITAChunker):
    """Chunks generic topic documents."""

    def chunk(self, document: DITADocument) -> list[Chunk]:
        """Section-based chunking for general topics."""
        chunks = []

        header = f"# {document.title}\n"
        if document.abstract:
            header += f"{document.abstract}\n"

        # If we have sections, use them
        if document.sections:
            for i, section in enumerate(document.sections):
                content = header + "\n" + section.content

                # Skip if too small
                if self._is_too_small(content):
                    continue

                # Split if too large
                if self._is_too_large(content):
                    # Just truncate for now - complex splitting would need more logic
                    content = content[: self.config.max_chunk_size]

                chunk = self._create_chunk(
                    content,
                    document,
                    chunk_index=i,
                    content_type="topic",
                    section_id=section.id,
                    section_title=section.title,
                )
                chunks.append(chunk)
        else:
            # No sections - use full document
            content = header + document.content_text
            chunk = self._create_chunk(content, document, chunk_index=0, content_type="topic")
            chunks.append(chunk)

        return chunks


def get_chunker(dita_type: str, config: DITAChunkerConfig) -> DITAChunker:
    """Factory to get appropriate chunker for document type."""
    chunkers = {
        "task": TaskChunker,
        "concept": ConceptChunker,
        "reference": ReferenceChunker,
        "topic": TopicChunker,
    }
    chunker_class = chunkers.get(dita_type, TopicChunker)
    return chunker_class(config)


# =============================================================================
# Ingestion Pipeline
# =============================================================================


@dataclass
class IngestionStats:
    """Statistics from ingestion run."""

    files_processed: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    by_type: dict = field(default_factory=lambda: defaultdict(int))
    chunks_by_type: dict = field(default_factory=lambda: defaultdict(int))
    errors: list = field(default_factory=list)


class DITAIngestionPipeline:
    """Orchestrates DITA document ingestion."""

    def __init__(self, config: DITAChunkerConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.parser = DITAHTMLParser(verbose=config.verbose)
        self.stats = IngestionStats()
        self.vector_store: VectorStore | None = None

        if not dry_run:
            self.vector_store = VectorStore()

    def run(self, input_dir: Path) -> IngestionStats:
        """Process all DITA HTML files in directory."""
        html_files = sorted(input_dir.glob("*.html"))

        if not html_files:
            console.print(f"[yellow]No HTML files found in {input_dir}[/yellow]")
            return self.stats

        console.print(f"\nProcessing {len(html_files)} HTML files from {input_dir}\n")

        all_chunks: list[Chunk] = []

        # Progress bar for parsing
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing documents", total=len(html_files))

            for file_path in html_files:
                try:
                    # Parse
                    document = self.parser.parse(file_path)
                    self.stats.by_type[document.dita_type] += 1

                    # Chunk
                    chunker = get_chunker(document.dita_type, self.config)
                    chunks = chunker.chunk(document)
                    all_chunks.extend(chunks)

                    self.stats.files_processed += 1
                    self.stats.chunks_created += len(chunks)
                    self.stats.chunks_by_type[document.dita_type] += len(chunks)

                    if self.config.verbose:
                        console.print(
                            f"  [dim]{file_path.name}[/dim] -> {document.dita_type} -> {len(chunks)} chunks"
                        )

                except Exception as e:
                    self.stats.files_failed += 1
                    self.stats.errors.append(f"{file_path.name}: {e}")
                    if self.config.verbose:
                        console.print(f"  [red]Error: {file_path.name}: {e}[/red]")

                progress.update(task, advance=1)

        # Store chunks
        if not self.dry_run and all_chunks and self.vector_store:
            console.print(f"\nStoring {len(all_chunks)} chunks in ChromaDB...")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Generating embeddings & storing", total=len(all_chunks))

                # Add chunks in batches
                batch_size = 50
                for i in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[i : i + batch_size]
                    self.vector_store.add_chunks(batch)
                    progress.update(task, advance=len(batch))

        return self.stats

    def clear_collection(self):
        """Clear all documents from the collection."""
        if self.vector_store:
            self.vector_store.clear()
            console.print("[yellow]Cleared existing documents from collection[/yellow]")


def print_stats(stats: IngestionStats):
    """Print ingestion statistics."""
    console.print("\n[bold]Ingestion Summary[/bold]")
    console.print("=" * 50)

    # Files table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Document Type")
    table.add_column("Files", justify="right")
    table.add_column("Chunks", justify="right")

    for dtype in sorted(stats.by_type.keys()):
        table.add_row(dtype, str(stats.by_type[dtype]), str(stats.chunks_by_type[dtype]))

    table.add_row("", "", "")
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{stats.files_processed}[/bold]",
        f"[bold]{stats.chunks_created}[/bold]",
    )

    console.print(table)

    if stats.files_failed > 0:
        console.print(f"\n[red]Failed: {stats.files_failed} files[/red]")
        for error in stats.errors[:5]:
            console.print(f"  [dim]{error}[/dim]")
        if len(stats.errors) > 5:
            console.print(f"  [dim]... and {len(stats.errors) - 5} more[/dim]")


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=False, path_type=Path),
    default=DEFAULT_CONFIG_PATH,
    help="Path to configuration file",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Override input directory from config",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Parse and chunk without storing to ChromaDB",
)
@click.option(
    "--clear-first",
    is_flag=True,
    help="Clear existing documents before ingesting",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed progress",
)
def main(config_path: Path, input_dir: Path | None, dry_run: bool, clear_first: bool, verbose: bool):
    """Ingest DITA HTML documentation into RAG vector store.

    Parses DITA-generated HTML files and performs semantic chunking based on
    document type (task, concept, reference, topic).
    """
    console.print("[bold]DITA Documentation Ingestion Tool[/bold]")
    console.print("=" * 50)

    # Load config
    config = load_config(config_path)

    # Apply CLI overrides
    if verbose:
        config.verbose = True
    if input_dir:
        config.input_dir = str(input_dir)

    # Resolve input directory
    input_path = Path(config.input_dir)
    if not input_path.exists():
        console.print(f"[red]Error: Input directory not found: {input_path}[/red]")
        raise SystemExit(1)

    # Show configuration
    console.print(f"Config: {config_path}")
    console.print(f"Input:  {input_path}")
    console.print(f"Mode:   {'[yellow]DRY RUN[/yellow]' if dry_run else 'Live'}")

    # Create pipeline
    pipeline = DITAIngestionPipeline(config, dry_run=dry_run)

    # Clear if requested
    if clear_first and not dry_run:
        pipeline.clear_collection()

    # Run ingestion
    stats = pipeline.run(input_path)

    # Print results
    print_stats(stats)

    if dry_run:
        console.print("\n[yellow]Dry run complete - no data stored[/yellow]")
    else:
        console.print(f"\n[green]Successfully ingested {stats.chunks_created} chunks[/green]")


if __name__ == "__main__":
    main()
