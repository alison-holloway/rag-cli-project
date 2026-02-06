#!/usr/bin/env python3
"""
Embedding Model Benchmark Tool

Compares multiple sentence-transformer embedding models on technical documentation
retrieval quality. Tests similarity scores, query latency, and model load times.

Usage:
    python tools/benchmark_embeddings.py                    # Run full benchmark
    python tools/benchmark_embeddings.py --output results.md
    python tools/benchmark_embeddings.py --verbose
    python tools/benchmark_embeddings.py --models 2          # Test only first 2 models
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
from bs4 import BeautifulSoup
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

from src.embedder import Embedder

console = Console()

# Models to benchmark (in order of increasing quality/size)
MODELS = [
    ("all-MiniLM-L6-v2", 384, "Baseline - fast, small"),
    ("all-mpnet-base-v2", 768, "Higher quality, 2x dimensions"),
    ("BAAI/bge-small-en-v1.5", 384, "BGE family, same dimensions"),
    ("BAAI/bge-base-en-v1.5", 768, "Best quality, largest"),
]

# Test queries from requirements
BENCHMARK_QUERIES = [
    "How do I install the CLI?",
    "What are the providers?",
    "How do I configure a Kubernetes cluster?",
]

# Default document directory
DEFAULT_DOCS_DIR = Path("data/documents/html")


@dataclass
class DocumentSample:
    """A sample document for benchmarking."""

    filename: str
    title: str
    content: str


@dataclass
class QueryResult:
    """Results for a single query against a model."""

    query: str
    top_k_scores: list[float]  # Similarity scores for top-k documents
    top_k_docs: list[str]  # Document titles for top-k
    query_time_ms: float


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a single model."""

    model_name: str
    dimensions: int
    description: str
    model_load_time_s: float
    query_results: list[QueryResult] = field(default_factory=list)

    @property
    def avg_similarity(self) -> float:
        """Average similarity score across all queries."""
        all_scores = []
        for qr in self.query_results:
            all_scores.extend(qr.top_k_scores)
        return float(np.mean(all_scores)) if all_scores else 0.0

    @property
    def max_similarity(self) -> float:
        """Maximum similarity score across all queries."""
        all_scores = []
        for qr in self.query_results:
            all_scores.extend(qr.top_k_scores)
        return float(np.max(all_scores)) if all_scores else 0.0

    @property
    def min_similarity(self) -> float:
        """Minimum similarity score across all queries."""
        all_scores = []
        for qr in self.query_results:
            all_scores.extend(qr.top_k_scores)
        return float(np.min(all_scores)) if all_scores else 0.0

    @property
    def avg_query_time_ms(self) -> float:
        """Average query time in milliseconds."""
        if not self.query_results:
            return 0.0
        return float(np.mean([qr.query_time_ms for qr in self.query_results]))


class EmbeddingBenchmark:
    """Benchmarks embedding models on document retrieval."""

    def __init__(
        self,
        docs_dir: Path,
        top_k: int = 5,
        max_docs: int = 50,
        verbose: bool = False,
    ):
        self.docs_dir = docs_dir
        self.top_k = top_k
        self.max_docs = max_docs
        self.verbose = verbose
        self.documents: list[DocumentSample] = []

    def load_documents(self) -> int:
        """Load HTML documents and extract text content."""
        html_files = list(self.docs_dir.glob("*.html"))

        if not html_files:
            console.print(f"[red]No HTML files found in {self.docs_dir}[/red]")
            return 0

        # Limit number of documents for benchmark speed
        html_files = html_files[: self.max_docs]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            disable=not self.verbose,
        ) as progress:
            task = progress.add_task("Loading documents", total=len(html_files))

            for html_file in html_files:
                try:
                    doc = self._parse_html(html_file)
                    if doc and len(doc.content) > 100:  # Skip very short documents
                        self.documents.append(doc)
                except Exception as e:
                    if self.verbose:
                        console.print(f"[yellow]Warning: {html_file.name}: {e}[/yellow]")

                progress.update(task, advance=1)

        return len(self.documents)

    def _parse_html(self, filepath: Path) -> DocumentSample | None:
        """Parse HTML file and extract text content."""
        with open(filepath, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else filepath.stem

        # Extract main content (remove scripts, styles, nav)
        for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        # Get body text
        body = soup.find("body")
        if not body:
            return None

        content = body.get_text(separator=" ", strip=True)

        # Clean up whitespace
        content = " ".join(content.split())

        return DocumentSample(
            filename=filepath.name,
            title=title,
            content=content[:3000],  # Truncate for embedding
        )

    def benchmark_model(
        self,
        model_name: str,
        dimensions: int,
        description: str,
    ) -> BenchmarkResult:
        """Run benchmark for a single model."""
        # Time model loading
        start = time.perf_counter()
        embedder = Embedder(model_name=model_name, verbose=False)
        # Force model load by accessing it
        _ = embedder.embedding_dimension
        load_time = time.perf_counter() - start

        result = BenchmarkResult(
            model_name=model_name,
            dimensions=dimensions,
            description=description,
            model_load_time_s=load_time,
        )

        # Embed all documents
        doc_texts = [doc.content for doc in self.documents]
        doc_embeddings = embedder.embed_texts(doc_texts, batch_size=16)

        # Run each query
        for query in BENCHMARK_QUERIES:
            start = time.perf_counter()

            # Embed query
            query_embedding = embedder.embed_query(query)

            # Calculate similarities
            similarities = []
            for i, doc_emb in enumerate(doc_embeddings):
                sim = embedder.similarity(query_embedding, doc_emb)
                similarities.append((sim, i))

            # Sort by similarity (descending)
            similarities.sort(reverse=True)
            top_k = similarities[: self.top_k]

            query_time = (time.perf_counter() - start) * 1000  # Convert to ms

            result.query_results.append(
                QueryResult(
                    query=query,
                    top_k_scores=[s[0] for s in top_k],
                    top_k_docs=[self.documents[s[1]].title for s in top_k],
                    query_time_ms=query_time,
                )
            )

        return result

    def run_benchmark(self, model_count: int | None = None) -> list[BenchmarkResult]:
        """Run benchmark on all models."""
        models_to_test = MODELS[:model_count] if model_count else MODELS
        results = []

        console.print(f"\nBenchmarking {len(models_to_test)} models...")

        for model_name, dimensions, description in models_to_test:
            console.print(f"\n[bold]Testing: {model_name}[/bold]")

            try:
                result = self.benchmark_model(model_name, dimensions, description)
                results.append(result)

                console.print(f"  Load time: {result.model_load_time_s:.1f}s")
                console.print(f"  Avg similarity: {result.avg_similarity:.3f}")
                console.print(f"  Avg query time: {result.avg_query_time_ms:.1f}ms")

            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

        return results


def generate_markdown_report(
    results: list[BenchmarkResult],
    num_docs: int,
) -> str:
    """Generate a markdown report of benchmark results."""
    lines = [
        "# Embedding Model Benchmark Results",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Documents:** {num_docs} HTML files from technical documentation",
        f"**Queries:** {len(BENCHMARK_QUERIES)} test queries",
        "",
        "## Summary Table",
        "",
        "| Model | Dims | Avg Sim | Max Sim | Avg Query (ms) | Load (s) | Notes |",
        "|-------|------|---------|---------|----------------|----------|-------|",
    ]

    # Find best model by avg similarity
    best_idx = max(range(len(results)), key=lambda i: results[i].avg_similarity)

    for i, r in enumerate(results):
        note = r.description
        if i == best_idx:
            note = "**Best**"
        elif i == 0:
            note = "Baseline"

        lines.append(
            f"| {r.model_name} | {r.dimensions} | {r.avg_similarity:.3f} | "
            f"{r.max_similarity:.3f} | {r.avg_query_time_ms:.1f} | "
            f"{r.model_load_time_s:.1f} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Detailed Results by Query",
            "",
        ]
    )

    for query_idx, query in enumerate(BENCHMARK_QUERIES):
        lines.append(f"### Query: \"{query}\"")
        lines.append("")

        for r in results:
            qr = r.query_results[query_idx]
            lines.append(f"**{r.model_name}** (query time: {qr.query_time_ms:.1f}ms)")
            lines.append("")
            lines.append("| Rank | Similarity | Document |")
            lines.append("|------|------------|----------|")

            for rank, (score, doc_title) in enumerate(
                zip(qr.top_k_scores, qr.top_k_docs), 1
            ):
                # Truncate long titles
                title = doc_title[:50] + "..." if len(doc_title) > 50 else doc_title
                lines.append(f"| {rank} | {score:.3f} | {title} |")

            lines.append("")

    lines.extend(
        [
            "## Recommendation",
            "",
        ]
    )

    best = results[best_idx]
    baseline = results[0]
    improvement = (
        (best.avg_similarity - baseline.avg_similarity) / baseline.avg_similarity * 100
    )

    lines.extend(
        [
            f"**Recommended model:** `{best.model_name}`",
            "",
            f"- **{improvement:.1f}% improvement** in average similarity over baseline",
            f"- Average similarity: {best.avg_similarity:.3f} (vs {baseline.avg_similarity:.3f} baseline)",
            f"- Query latency: {best.avg_query_time_ms:.1f}ms",
            f"- Model load time: {best.model_load_time_s:.1f}s",
            "",
        ]
    )

    return "\n".join(lines)


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print results as a Rich table."""
    table = Table(title="Embedding Model Benchmark Results", show_header=True)

    table.add_column("Model", style="bold")
    table.add_column("Dims", justify="right")
    table.add_column("Avg Sim", justify="right")
    table.add_column("Max Sim", justify="right")
    table.add_column("Query (ms)", justify="right")
    table.add_column("Load (s)", justify="right")

    # Find best model
    best_idx = max(range(len(results)), key=lambda i: results[i].avg_similarity)

    for i, r in enumerate(results):
        style = "green bold" if i == best_idx else None
        table.add_row(
            r.model_name,
            str(r.dimensions),
            f"{r.avg_similarity:.3f}",
            f"{r.max_similarity:.3f}",
            f"{r.avg_query_time_ms:.1f}",
            f"{r.model_load_time_s:.1f}",
            style=style,
        )

    console.print(table)


@click.command()
@click.option(
    "--docs-dir",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_DOCS_DIR,
    help="Directory containing HTML documents",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output markdown file for results",
)
@click.option(
    "--models",
    type=int,
    default=None,
    help="Number of models to test (1-4, default: all)",
)
@click.option(
    "--max-docs",
    type=int,
    default=50,
    help="Maximum documents to use for benchmark",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of top results to track per query",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed progress",
)
def main(
    docs_dir: Path,
    output: Path | None,
    models: int | None,
    max_docs: int,
    top_k: int,
    verbose: bool,
) -> None:
    """Benchmark embedding models on document retrieval.

    Tests multiple sentence-transformer models to find the best one for
    technical documentation retrieval based on similarity scores and latency.
    """
    console.print("[bold]Embedding Model Benchmark Tool[/bold]")
    console.print("=" * 50)

    # Validate model count
    if models is not None and (models < 1 or models > len(MODELS)):
        console.print(f"[red]Error: --models must be between 1 and {len(MODELS)}[/red]")
        raise SystemExit(1)

    # Create benchmark
    benchmark = EmbeddingBenchmark(
        docs_dir=docs_dir,
        top_k=top_k,
        max_docs=max_docs,
        verbose=verbose,
    )

    # Load documents
    console.print(f"\nLoading documents from: {docs_dir}")
    num_docs = benchmark.load_documents()

    if num_docs == 0:
        console.print("[red]No documents loaded. Exiting.[/red]")
        raise SystemExit(1)

    console.print(f"Loaded {num_docs} documents")

    # Run benchmark
    results = benchmark.run_benchmark(model_count=models)

    if not results:
        console.print("[red]No benchmark results. Exiting.[/red]")
        raise SystemExit(1)

    # Print results table
    console.print("\n")
    print_results_table(results)

    # Find and announce winner
    best_idx = max(range(len(results)), key=lambda i: results[i].avg_similarity)
    best = results[best_idx]
    baseline = results[0]

    console.print(f"\n[green bold]Recommended: {best.model_name}[/green bold]")
    if best_idx > 0:
        improvement = (
            (best.avg_similarity - baseline.avg_similarity)
            / baseline.avg_similarity
            * 100
        )
        console.print(f"  {improvement:.1f}% improvement over baseline")

    # Write output file if requested
    if output:
        report = generate_markdown_report(results, num_docs)
        output.write_text(report)
        console.print(f"\n[green]Results saved to: {output}[/green]")


if __name__ == "__main__":
    main()
