#!/usr/bin/env python3
"""
Embedding Model Comparison Tool

Compares retrieval results between two embedding models (or two ChromaDB collections)
for the same set of queries. Useful for validating model migration.

Usage:
    python tools/compare_embeddings.py --old all-MiniLM-L6-v2 --new BAAI/bge-small-en-v1.5
    python tools/compare_embeddings.py --queries queries.txt --output comparison.md
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import Embedder

console = Console()

# Default test queries
DEFAULT_QUERIES = [
    "How do I install the CLI?",
    "What are the providers?",
    "How do I configure a Kubernetes cluster?",
]

DEFAULT_DOCS_DIR = Path("data/documents/html")


@dataclass
class RetrievalResult:
    """Result of a single retrieval for a query."""

    rank: int
    doc_title: str
    similarity: float
    doc_preview: str = ""


@dataclass
class QueryComparison:
    """Comparison of retrieval results for a single query."""

    query: str
    old_results: list[RetrievalResult]
    new_results: list[RetrievalResult]
    old_query_time_ms: float
    new_query_time_ms: float

    @property
    def overlap_count(self) -> int:
        """Number of documents appearing in both result sets."""
        old_titles = {r.doc_title for r in self.old_results}
        new_titles = {r.doc_title for r in self.new_results}
        return len(old_titles & new_titles)

    @property
    def overlap_percentage(self) -> float:
        """Percentage of overlap between result sets."""
        if not self.old_results or not self.new_results:
            return 0.0
        max_possible = min(len(self.old_results), len(self.new_results))
        return (self.overlap_count / max_possible) * 100


@dataclass
class ComparisonReport:
    """Full comparison report between two models."""

    old_model: str
    new_model: str
    num_documents: int
    comparisons: list[QueryComparison] = field(default_factory=list)

    @property
    def avg_old_similarity(self) -> float:
        """Average similarity for old model."""
        all_scores = []
        for c in self.comparisons:
            all_scores.extend([r.similarity for r in c.old_results])
        return float(np.mean(all_scores)) if all_scores else 0.0

    @property
    def avg_new_similarity(self) -> float:
        """Average similarity for new model."""
        all_scores = []
        for c in self.comparisons:
            all_scores.extend([r.similarity for r in c.new_results])
        return float(np.mean(all_scores)) if all_scores else 0.0

    @property
    def avg_old_query_time(self) -> float:
        """Average query time for old model."""
        if not self.comparisons:
            return 0.0
        return float(np.mean([c.old_query_time_ms for c in self.comparisons]))

    @property
    def avg_new_query_time(self) -> float:
        """Average query time for new model."""
        if not self.comparisons:
            return 0.0
        return float(np.mean([c.new_query_time_ms for c in self.comparisons]))

    @property
    def avg_overlap(self) -> float:
        """Average overlap percentage."""
        if not self.comparisons:
            return 0.0
        return float(np.mean([c.overlap_percentage for c in self.comparisons]))


class EmbeddingComparator:
    """Compares retrieval quality between two embedding models."""

    def __init__(
        self,
        old_model: str,
        new_model: str,
        docs_dir: Path,
        max_docs: int = 50,
        top_k: int = 5,
        verbose: bool = False,
    ):
        self.old_model = old_model
        self.new_model = new_model
        self.docs_dir = docs_dir
        self.max_docs = max_docs
        self.top_k = top_k
        self.verbose = verbose

        self.documents: list[tuple[str, str, str]] = []  # (filename, title, content)
        self.old_embedder: Embedder | None = None
        self.new_embedder: Embedder | None = None
        self.old_doc_embeddings: np.ndarray | None = None
        self.new_doc_embeddings: np.ndarray | None = None

    def load_documents(self) -> int:
        """Load HTML documents."""
        html_files = list(self.docs_dir.glob("*.html"))[:self.max_docs]

        for html_file in html_files:
            try:
                with open(html_file, encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")

                title_tag = soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else html_file.stem

                for tag in soup.find_all(["script", "style", "nav"]):
                    tag.decompose()

                body = soup.find("body")
                if body:
                    content = body.get_text(separator=" ", strip=True)
                    content = " ".join(content.split())[:3000]
                    self.documents.append((html_file.name, title, content))
            except Exception:
                pass

        return len(self.documents)

    def setup_models(self) -> None:
        """Load both embedding models and pre-compute document embeddings."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load old model
            task = progress.add_task(f"Loading {self.old_model}...", total=None)
            self.old_embedder = Embedder(model_name=self.old_model, verbose=False)
            progress.remove_task(task)

            # Load new model
            task = progress.add_task(f"Loading {self.new_model}...", total=None)
            self.new_embedder = Embedder(model_name=self.new_model, verbose=False)
            progress.remove_task(task)

            # Embed documents with old model
            task = progress.add_task(
                f"Embedding docs with {self.old_model}...", total=None
            )
            doc_texts = [d[2] for d in self.documents]
            self.old_doc_embeddings = self.old_embedder.embed_texts(doc_texts)
            progress.remove_task(task)

            # Embed documents with new model
            task = progress.add_task(
                f"Embedding docs with {self.new_model}...", total=None
            )
            self.new_doc_embeddings = self.new_embedder.embed_texts(doc_texts)
            progress.remove_task(task)

    def compare_query(self, query: str) -> QueryComparison:
        """Compare retrieval results for a single query."""
        # Query with old model
        start = time.perf_counter()
        old_query_emb = self.old_embedder.embed_query(query)
        old_similarities = []
        for i, doc_emb in enumerate(self.old_doc_embeddings):
            sim = self.old_embedder.similarity(old_query_emb, doc_emb)
            old_similarities.append((sim, i))
        old_similarities.sort(reverse=True)
        old_time = (time.perf_counter() - start) * 1000

        # Query with new model
        start = time.perf_counter()
        new_query_emb = self.new_embedder.embed_query(query)
        new_similarities = []
        for i, doc_emb in enumerate(self.new_doc_embeddings):
            sim = self.new_embedder.similarity(new_query_emb, doc_emb)
            new_similarities.append((sim, i))
        new_similarities.sort(reverse=True)
        new_time = (time.perf_counter() - start) * 1000

        # Build results
        old_results = []
        for rank, (sim, idx) in enumerate(old_similarities[:self.top_k], 1):
            _, title, content = self.documents[idx]
            old_results.append(
                RetrievalResult(
                    rank=rank,
                    doc_title=title,
                    similarity=sim,
                    doc_preview=content[:100] + "...",
                )
            )

        new_results = []
        for rank, (sim, idx) in enumerate(new_similarities[:self.top_k], 1):
            _, title, content = self.documents[idx]
            new_results.append(
                RetrievalResult(
                    rank=rank,
                    doc_title=title,
                    similarity=sim,
                    doc_preview=content[:100] + "...",
                )
            )

        return QueryComparison(
            query=query,
            old_results=old_results,
            new_results=new_results,
            old_query_time_ms=old_time,
            new_query_time_ms=new_time,
        )

    def run_comparison(self, queries: list[str]) -> ComparisonReport:
        """Run full comparison on all queries."""
        report = ComparisonReport(
            old_model=self.old_model,
            new_model=self.new_model,
            num_documents=len(self.documents),
        )

        for query in queries:
            comparison = self.compare_query(query)
            report.comparisons.append(comparison)

        return report


def generate_markdown_report(report: ComparisonReport) -> str:
    """Generate markdown comparison report."""
    lines = [
        "# Embedding Model Comparison Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Old Model:** {report.old_model}",
        f"**New Model:** {report.new_model}",
        f"**Documents:** {report.num_documents}",
        "",
        "## Summary",
        "",
        "| Metric | Old Model | New Model | Change |",
        "|--------|-----------|-----------|--------|",
    ]

    # Calculate changes
    sim_change = report.avg_new_similarity - report.avg_old_similarity
    sim_pct = (sim_change / report.avg_old_similarity * 100) if report.avg_old_similarity else 0
    time_change = report.avg_new_query_time - report.avg_old_query_time
    time_pct = (time_change / report.avg_old_query_time * 100) if report.avg_old_query_time else 0

    lines.append(
        f"| Avg Similarity | {report.avg_old_similarity:.3f} | "
        f"{report.avg_new_similarity:.3f} | {sim_change:+.3f} ({sim_pct:+.1f}%) |"
    )
    lines.append(
        f"| Avg Query Time | {report.avg_old_query_time:.1f}ms | "
        f"{report.avg_new_query_time:.1f}ms | {time_change:+.1f}ms ({time_pct:+.1f}%) |"
    )
    lines.append(f"| Avg Overlap | - | - | {report.avg_overlap:.1f}% |")
    lines.append("")

    # Per-query details
    for comp in report.comparisons:
        lines.append(f"## Query: \"{comp.query}\"")
        lines.append("")
        lines.append(
            f"Overlap: {comp.overlap_count}/{len(comp.old_results)} documents "
            f"({comp.overlap_percentage:.0f}%)"
        )
        lines.append("")

        lines.append(f"### {report.old_model}")
        lines.append(f"Query time: {comp.old_query_time_ms:.1f}ms")
        lines.append("")
        lines.append("| Rank | Similarity | Document |")
        lines.append("|------|------------|----------|")
        for r in comp.old_results:
            title = r.doc_title[:50] + "..." if len(r.doc_title) > 50 else r.doc_title
            lines.append(f"| {r.rank} | {r.similarity:.3f} | {title} |")
        lines.append("")

        lines.append(f"### {report.new_model}")
        lines.append(f"Query time: {comp.new_query_time_ms:.1f}ms")
        lines.append("")
        lines.append("| Rank | Similarity | Document |")
        lines.append("|------|------------|----------|")
        for r in comp.new_results:
            title = r.doc_title[:50] + "..." if len(r.doc_title) > 50 else r.doc_title
            lines.append(f"| {r.rank} | {r.similarity:.3f} | {title} |")
        lines.append("")

    return "\n".join(lines)


def print_summary(report: ComparisonReport) -> None:
    """Print comparison summary table."""
    table = Table(title="Embedding Model Comparison", show_header=True)
    table.add_column("Metric")
    table.add_column(report.old_model, justify="right")
    table.add_column(report.new_model, justify="right")
    table.add_column("Change", justify="right")

    # Similarity row
    sim_change = report.avg_new_similarity - report.avg_old_similarity
    sim_pct = (sim_change / report.avg_old_similarity * 100) if report.avg_old_similarity else 0
    sim_style = "green" if sim_change > 0 else "red"
    table.add_row(
        "Avg Similarity",
        f"{report.avg_old_similarity:.3f}",
        f"{report.avg_new_similarity:.3f}",
        f"[{sim_style}]{sim_change:+.3f} ({sim_pct:+.1f}%)[/{sim_style}]",
    )

    # Query time row
    time_change = report.avg_new_query_time - report.avg_old_query_time
    time_pct = (time_change / report.avg_old_query_time * 100) if report.avg_old_query_time else 0
    time_style = "green" if time_change < 0 else "yellow"
    table.add_row(
        "Avg Query Time",
        f"{report.avg_old_query_time:.1f}ms",
        f"{report.avg_new_query_time:.1f}ms",
        f"[{time_style}]{time_change:+.1f}ms ({time_pct:+.1f}%)[/{time_style}]",
    )

    # Overlap row
    table.add_row(
        "Result Overlap",
        "-",
        "-",
        f"{report.avg_overlap:.1f}%",
    )

    console.print(table)


@click.command()
@click.option(
    "--old",
    "old_model",
    default="all-MiniLM-L6-v2",
    help="Old embedding model name",
)
@click.option(
    "--new",
    "new_model",
    default="BAAI/bge-small-en-v1.5",
    help="New embedding model name",
)
@click.option(
    "--docs-dir",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_DOCS_DIR,
    help="Directory containing HTML documents",
)
@click.option(
    "--queries",
    type=click.Path(exists=True, path_type=Path),
    help="File with queries (one per line). Uses defaults if not provided.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output markdown file for comparison report",
)
@click.option(
    "--max-docs",
    type=int,
    default=50,
    help="Maximum documents to use",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of top results to compare",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed output",
)
def main(
    old_model: str,
    new_model: str,
    docs_dir: Path,
    queries: Path | None,
    output: Path | None,
    max_docs: int,
    top_k: int,
    verbose: bool,
) -> None:
    """Compare retrieval quality between two embedding models.

    Runs the same queries against both models and compares similarity scores,
    retrieved documents, and query latency.
    """
    console.print("[bold]Embedding Model Comparison Tool[/bold]")
    console.print("=" * 50)

    # Load queries
    if queries:
        query_list = [
            line.strip()
            for line in queries.read_text().splitlines()
            if line.strip()
        ]
    else:
        query_list = DEFAULT_QUERIES

    console.print(f"\nComparing models:")
    console.print(f"  Old: {old_model}")
    console.print(f"  New: {new_model}")
    console.print(f"\nQueries: {len(query_list)}")
    for q in query_list:
        console.print(f"  - {q}")

    # Create comparator
    comparator = EmbeddingComparator(
        old_model=old_model,
        new_model=new_model,
        docs_dir=docs_dir,
        max_docs=max_docs,
        top_k=top_k,
        verbose=verbose,
    )

    # Load documents
    console.print(f"\nLoading documents from: {docs_dir}")
    num_docs = comparator.load_documents()
    console.print(f"Loaded {num_docs} documents")

    if num_docs == 0:
        console.print("[red]No documents loaded. Exiting.[/red]")
        raise SystemExit(1)

    # Setup models
    console.print("\nSetting up models...")
    comparator.setup_models()

    # Run comparison
    console.print("\nRunning comparison...")
    report = comparator.run_comparison(query_list)

    # Print summary
    console.print("\n")
    print_summary(report)

    # Print per-query details if verbose
    if verbose:
        for comp in report.comparisons:
            console.print(f"\n[bold]Query: \"{comp.query}\"[/bold]")
            console.print(f"Overlap: {comp.overlap_count}/{top_k} ({comp.overlap_percentage:.0f}%)")

            table = Table(show_header=True)
            table.add_column("Rank")
            table.add_column(f"{old_model}", justify="left")
            table.add_column("Sim", justify="right")
            table.add_column(f"{new_model}", justify="left")
            table.add_column("Sim", justify="right")

            for i in range(top_k):
                old_r = comp.old_results[i] if i < len(comp.old_results) else None
                new_r = comp.new_results[i] if i < len(comp.new_results) else None

                old_title = old_r.doc_title[:30] if old_r else "-"
                old_sim = f"{old_r.similarity:.3f}" if old_r else "-"
                new_title = new_r.doc_title[:30] if new_r else "-"
                new_sim = f"{new_r.similarity:.3f}" if new_r else "-"

                table.add_row(str(i + 1), old_title, old_sim, new_title, new_sim)

            console.print(table)

    # Highlight improvement
    sim_change = report.avg_new_similarity - report.avg_old_similarity
    if sim_change > 0:
        pct = (sim_change / report.avg_old_similarity * 100) if report.avg_old_similarity else 0
        console.print(f"\n[green bold]{new_model} shows {pct:.1f}% improvement in similarity[/green bold]")
    else:
        console.print(f"\n[yellow]No improvement in similarity scores[/yellow]")

    # Write output if requested
    if output:
        markdown = generate_markdown_report(report)
        output.write_text(markdown)
        console.print(f"\n[green]Report saved to: {output}[/green]")


if __name__ == "__main__":
    main()
