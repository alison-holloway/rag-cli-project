#!/usr/bin/env python3
"""
Re-ingestion Tool for Embedding Model Migration

Clears the existing ChromaDB collection and re-indexes all documents
using a new embedding model. Essential when changing embedding models
due to dimension incompatibility.

IMPORTANT: This is a destructive operation! Use --backup to create a backup first.

Usage:
    python tools/reingest_with_new_embeddings.py                    # Use model from .env
    python tools/reingest_with_new_embeddings.py --model BAAI/bge-small-en-v1.5
    python tools/reingest_with_new_embeddings.py --dry-run          # Preview without changes
    python tools/reingest_with_new_embeddings.py --backup           # Create backup first
"""

import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm
from rich.table import Table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings, PROJECT_ROOT
from src.embedder import Embedder
from src.vector_store import VectorStore

# Import DITA parsing from ingest_dita_docs
from tools.ingest_dita_docs import (
    DITAChunkerConfig,
    DITAIngestionPipeline,
    load_config as load_dita_config,
)

console = Console()

# Default paths
DEFAULT_DITA_CONFIG = Path("config/dita_chunker.yaml")
DEFAULT_BACKUP_DIR = Path("data/backups")


@dataclass
class ReingestionStats:
    """Statistics from re-ingestion."""

    old_chunk_count: int = 0
    new_chunk_count: int = 0
    files_processed: int = 0
    embedding_model: str = ""
    embedding_dimension: int = 0
    backup_path: Path | None = None
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class ReingestionPipeline:
    """Pipeline for re-ingesting documents with a new embedding model."""

    def __init__(
        self,
        embedding_model: str,
        dita_config: DITAChunkerConfig,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        self.embedding_model = embedding_model
        self.dita_config = dita_config
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = ReingestionStats(embedding_model=embedding_model)

    def create_backup(self, backup_dir: Path) -> Path | None:
        """Create a timestamped backup of the vector database."""
        settings = get_settings()
        vector_db_path = settings.vector_store.persist_path

        if not vector_db_path.exists():
            console.print("[yellow]No existing vector database to backup[/yellow]")
            return None

        # Create backup directory
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Timestamped backup name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"vector_db_backup_{timestamp}"

        console.print(f"Creating backup: {backup_path}")

        if not self.dry_run:
            shutil.copytree(vector_db_path, backup_path)
            console.print(f"[green]Backup created: {backup_path}[/green]")

        return backup_path

    def get_current_stats(self) -> dict:
        """Get statistics about current vector store."""
        try:
            store = VectorStore(collection_name=self.dita_config.collection_name)
            stats = store.get_stats()
            store.close()
            return stats
        except Exception:
            return {"total_chunks": 0, "total_documents": 0}

    def run(self, input_dir: Path, backup_dir: Path | None = None) -> ReingestionStats:
        """Run the re-ingestion pipeline."""
        start_time = time.perf_counter()

        # Get current stats before clearing
        current_stats = self.get_current_stats()
        self.stats.old_chunk_count = current_stats.get("total_chunks", 0)

        console.print(f"\n[bold]Current state:[/bold]")
        console.print(f"  Chunks: {self.stats.old_chunk_count}")
        console.print(f"  Documents: {current_stats.get('total_documents', 0)}")

        # Create backup if requested
        if backup_dir:
            self.stats.backup_path = self.create_backup(backup_dir)

        if self.dry_run:
            console.print("\n[yellow]DRY RUN - No changes will be made[/yellow]")

        # Create embedder with specified model
        console.print(f"\n[bold]Loading embedding model: {self.embedding_model}[/bold]")

        if not self.dry_run:
            embedder = Embedder(model_name=self.embedding_model, verbose=self.verbose)
            self.stats.embedding_dimension = embedder.embedding_dimension
            console.print(f"  Dimension: {self.stats.embedding_dimension}")
        else:
            console.print("  [dim](Skipped in dry run)[/dim]")

        # Create DITA ingestion pipeline
        # We reuse the DITA pipeline which handles parsing, chunking, and storage
        console.print(f"\n[bold]Processing documents from: {input_dir}[/bold]")

        if not self.dry_run:
            # Create a new vector store with the new embedder
            store = VectorStore(
                collection_name=self.dita_config.collection_name,
                embedder=embedder,
            )

            # Clear existing data
            console.print("\n[bold red]Clearing existing collection...[/bold red]")
            cleared = store.clear()
            console.print(f"  Cleared {cleared} chunks")

            # Close this store - the DITA pipeline will create its own
            store.close()

        # Run DITA ingestion with the new embedding model
        # We need to temporarily set the embedding model in the environment
        import os

        old_model = os.environ.get("EMBEDDING_MODEL")
        os.environ["EMBEDDING_MODEL"] = self.embedding_model

        try:
            # Create pipeline with clear_first=False since we already cleared
            pipeline = DITAIngestionPipeline(self.dita_config, dry_run=self.dry_run)

            # We need to ensure the pipeline uses our embedder
            # Reset the global embedder singleton
            import src.embedder as embedder_module

            embedder_module._embedder = None  # Reset singleton

            # Run ingestion
            dita_stats = pipeline.run(input_dir)

            self.stats.files_processed = dita_stats.files_processed
            self.stats.new_chunk_count = dita_stats.chunks_created
            self.stats.errors = dita_stats.errors

        finally:
            # Restore original environment
            if old_model:
                os.environ["EMBEDDING_MODEL"] = old_model
            elif "EMBEDDING_MODEL" in os.environ:
                del os.environ["EMBEDDING_MODEL"]

        self.stats.duration_seconds = time.perf_counter() - start_time
        return self.stats


def print_stats(stats: ReingestionStats) -> None:
    """Print re-ingestion statistics."""
    table = Table(title="Re-ingestion Results", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Embedding Model", stats.embedding_model)
    table.add_row("Embedding Dimension", str(stats.embedding_dimension))
    table.add_row("Files Processed", str(stats.files_processed))
    table.add_row("Old Chunk Count", str(stats.old_chunk_count))
    table.add_row("New Chunk Count", str(stats.new_chunk_count))
    table.add_row("Duration", f"{stats.duration_seconds:.1f}s")

    if stats.backup_path:
        table.add_row("Backup Location", str(stats.backup_path))

    console.print(table)

    if stats.errors:
        console.print(f"\n[yellow]Warnings: {len(stats.errors)} files had issues[/yellow]")


@click.command()
@click.option(
    "--model",
    default=None,
    help="Embedding model to use (overrides .env EMBEDDING_MODEL)",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Input directory for documents (overrides config)",
)
@click.option(
    "--dita-config",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_DITA_CONFIG,
    help="Path to DITA chunker configuration",
)
@click.option(
    "--backup",
    is_flag=True,
    help="Create backup of existing vector database before clearing",
)
@click.option(
    "--backup-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_BACKUP_DIR,
    help="Directory to store backups",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview without making any changes",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompt",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed progress",
)
def main(
    model: str | None,
    input_dir: Path | None,
    dita_config: Path,
    backup: bool,
    backup_dir: Path,
    dry_run: bool,
    yes: bool,
    verbose: bool,
) -> None:
    """Re-ingest documents with a new embedding model.

    This tool clears the existing ChromaDB collection and re-indexes all
    documents using the specified embedding model. Use this when migrating
    to a new embedding model.

    WARNING: This is a destructive operation! Use --backup to save a copy
    of your current vector database first.
    """
    console.print("[bold]Re-ingestion Tool for Embedding Model Migration[/bold]")
    console.print("=" * 55)

    # Load settings and config
    settings = get_settings()
    config = load_dita_config(dita_config)

    # Determine embedding model
    if model is None:
        model = settings.embedding.embedding_model

    # Determine input directory
    if input_dir is None:
        input_dir = PROJECT_ROOT / config.input_dir

    if not input_dir.exists():
        console.print(f"[red]Error: Input directory not found: {input_dir}[/red]")
        raise SystemExit(1)

    # Show configuration
    console.print(f"\nConfiguration:")
    console.print(f"  Embedding model: [bold]{model}[/bold]")
    console.print(f"  Input directory: {input_dir}")
    console.print(f"  Collection: {config.collection_name}")
    console.print(f"  Backup: {'Yes' if backup else 'No'}")
    console.print(f"  Mode: {'[yellow]DRY RUN[/yellow]' if dry_run else '[red]LIVE[/red]'}")

    # Warn about destructive operation
    if not dry_run:
        console.print(
            Panel(
                "[bold red]WARNING[/bold red]: This will DELETE all existing embeddings "
                "and re-create them with the new model.\n\n"
                "Make sure you have a backup or use --backup flag!",
                title="Destructive Operation",
                border_style="red",
            )
        )

        if not yes:
            if not Confirm.ask("Do you want to continue?"):
                console.print("[yellow]Aborted.[/yellow]")
                raise SystemExit(0)

    # Run re-ingestion
    pipeline = ReingestionPipeline(
        embedding_model=model,
        dita_config=config,
        dry_run=dry_run,
        verbose=verbose,
    )

    backup_path = backup_dir if backup else None
    stats = pipeline.run(input_dir, backup_dir=backup_path)

    # Print results
    console.print("\n")
    print_stats(stats)

    if dry_run:
        console.print("\n[yellow]Dry run complete - no changes were made[/yellow]")
    else:
        console.print(f"\n[green]Successfully re-ingested {stats.new_chunk_count} chunks[/green]")
        console.print(f"[green]Using embedding model: {model}[/green]")


if __name__ == "__main__":
    main()
