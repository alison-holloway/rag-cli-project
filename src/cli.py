"""Main CLI interface for RAG CLI."""

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import get_settings
from .logger import get_logger, set_log_level, setup_logging
from .progress import (
    print_error,
    print_info,
    print_success,
    print_warning,
    progress_bar,
    spinner,
)

console = Console()
logger = get_logger(__name__)


def _setup_ollama(model: str = "llama3.1:8b") -> bool:
    """Set up Ollama service and pull the required model.

    Args:
        model: The model to pull if not available.

    Returns:
        True if Ollama is ready to use, False otherwise.
    """
    import shutil
    import subprocess
    import time

    console.print("\n[bold]Setting up Ollama...[/bold]")

    # Step 1: Check if Ollama is installed
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        print_warning("Ollama is not installed.")
        print_info("Install Ollama from: https://ollama.ai")
        print_info("  macOS: brew install ollama")
        print_info("  Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        return False

    print_success("Ollama is installed")

    # Step 2: Check if Ollama is running
    def is_ollama_running() -> bool:
        try:
            import ollama

            client = ollama.Client()
            client.list()
            return True
        except Exception:
            return False

    if not is_ollama_running():
        print_info("Starting Ollama service...")
        try:
            # Start ollama serve in the background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # Wait for it to start (up to 10 seconds)
            for i in range(20):
                time.sleep(0.5)
                if is_ollama_running():
                    break
            if is_ollama_running():
                print_success("Ollama service started")
            else:
                print_warning("Could not start Ollama service")
                print_info("Try running 'ollama serve' manually")
                return False
        except Exception as e:
            print_warning(f"Failed to start Ollama: {e}")
            print_info("Try running 'ollama serve' manually")
            return False
    else:
        print_success("Ollama service is running")

    # Step 3: Check if model is available
    try:
        import ollama

        client = ollama.Client()
        response = client.list()

        if hasattr(response, "models"):
            model_names = [m.model for m in response.models]
        else:
            model_names = [m["name"] for m in response.get("models", [])]

        # Check if model is already pulled (handle both 'llama3.1:8b' and 'llama3.1')
        base_model = model.split(":")[0]
        model_available = any(base_model in name for name in model_names)

        if model_available:
            print_success(f"Model '{model}' is available")
            return True

    except Exception as e:
        print_warning(f"Could not check models: {e}")
        return False

    # Step 4: Pull the model
    print_info(f"Pulling model '{model}' (this may take a few minutes)...")
    console.print("[dim]Model size: ~4.7GB for llama3.1:8b[/dim]")

    try:
        import ollama

        client = ollama.Client()

        # Use streaming to show progress
        current_status = ""
        with console.status(f"[bold blue]Downloading {model}...") as status:
            for progress in client.pull(model, stream=True):
                if hasattr(progress, "status"):
                    new_status = progress.status
                else:
                    new_status = progress.get("status", "")

                if new_status != current_status:
                    current_status = new_status
                    status.update(f"[bold blue]{current_status}...")

                # Show download progress if available
                if hasattr(progress, "completed") and hasattr(progress, "total"):
                    completed = progress.completed or 0
                    total = progress.total or 0
                    if total > 0:
                        pct = (completed / total) * 100
                        size_gb = total / (1024**3)
                        done_gb = completed / (1024**3)
                        status.update(
                            f"[bold blue]{current_status}: "
                            f"{done_gb:.1f}/{size_gb:.1f}GB ({pct:.0f}%)"
                        )

        print_success(f"Model '{model}' pulled successfully")
        return True

    except Exception as e:
        print_warning(f"Failed to pull model: {e}")
        print_info(f"Try running 'ollama pull {model}' manually")
        return False


@click.group()
@click.version_option(version=__version__, prog_name="rag-cli")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output (INFO logs)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging (even more detailed)",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """RAG CLI - A command-line RAG system for document question-answering.

    Process documents (PDF, Markdown, HTML) and enable intelligent
    question-answering using local embeddings and LLM integration.
    """
    import logging

    ctx.ensure_object(dict)

    # Initialize logging with appropriate verbosity
    # Default: quiet (WARNING only)
    # --verbose: show INFO messages
    # --debug: show DEBUG messages (implies verbose)
    log = setup_logging(verbose=verbose or debug)

    if debug:
        set_log_level(logging.DEBUG)
    elif verbose:
        set_log_level(logging.INFO)

    # Store settings and logger in context
    ctx.obj["settings"] = get_settings()
    ctx.obj["logger"] = log
    ctx.obj["debug"] = debug
    ctx.obj["verbose"] = verbose or debug

    log.debug("RAG CLI initialized")


@cli.command()
@click.argument("project_name", default="rag-project")
@click.option(
    "--skip-ollama",
    is_flag=True,
    help="Skip automatic Ollama setup",
)
@click.pass_context
def init(ctx: click.Context, project_name: str, skip_ollama: bool) -> None:
    """Initialize a new RAG project.

    Creates the necessary directory structure, initializes the vector store,
    and sets up Ollama with the required model.
    """
    settings = ctx.obj["settings"]

    console.print(
        Panel(
            f"Initializing RAG project: [bold cyan]{project_name}[/bold cyan]",
            title="RAG CLI",
        )
    )

    # Create directories
    dirs_to_create = [
        settings.vector_store.persist_path,
        Path("data/documents"),
        Path("logs"),
    ]

    for dir_path in dirs_to_create:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print_success(f"Created directory: {dir_path}")
        else:
            print_info(f"Directory exists: {dir_path}")

    # Initialize vector store
    with spinner("Initializing vector store..."):
        from .vector_store import VectorStore

        store = VectorStore()
        count = store.count()

    print_success(f"Vector store ready ({count} chunks)")

    # Set up Ollama (unless skipped)
    if not skip_ollama:
        ollama_model = settings.llm.ollama_model
        ollama_ready = _setup_ollama(model=ollama_model)
        if not ollama_ready:
            print_warning("Ollama setup incomplete - queries may not work")
            print_info("Run 'ollama serve' and 'ollama pull llama3.1:8b' manually")
    else:
        print_info("Skipped Ollama setup")

    console.print()
    print_success("Project initialized successfully!")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--recursive", "-r",
    is_flag=True,
    help="Recursively process directories",
)
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Re-index documents even if already indexed",
)
@click.pass_context
def add(ctx: click.Context, path: str, recursive: bool, force: bool) -> None:
    """Add documents to the knowledge base.

    PATH can be a single file or a directory containing documents.
    Supported formats: PDF, Markdown, HTML, TXT.
    """
    from .chunker import TextChunker
    from .document_loader import DocumentLoader
    from .vector_store import VectorStore

    path = Path(path)
    debug = ctx.obj.get("debug", False)

    console.print(f"Adding documents from: [cyan]{path}[/cyan]")

    # Initialize components
    loader = DocumentLoader()
    chunker = TextChunker()
    store = VectorStore()

    # Get list of existing documents to check for duplicates
    if not force:
        docs_list = store.list_documents()
        existing_docs = {d.get("source_file", "") for d in docs_list}
    else:
        existing_docs = set()

    # Collect files to process
    if path.is_file():
        files = [path]
    else:
        # Load from directory
        supported_extensions = loader.supported_extensions()
        files = []
        if recursive:
            for ext in supported_extensions:
                files.extend(path.rglob(f"*{ext}"))
        else:
            for ext in supported_extensions:
                files.extend(path.glob(f"*{ext}"))

    if not files:
        print_warning(f"No supported documents found in {path}")
        print_info(f"Supported formats: {', '.join(loader.supported_extensions())}")
        return

    console.print(f"Found [cyan]{len(files)}[/cyan] document(s) to process")

    # Process documents
    total_chunks = 0
    processed_files = 0
    skipped_files = 0
    failed_files = 0

    with progress_bar("Processing documents", total=len(files)) as advance:
        for file_path in files:
            try:
                # Check if already indexed
                if file_path.name in existing_docs:
                    if debug:
                        console.print(f"  [dim]Skipping: {file_path.name}[/dim]")
                    skipped_files += 1
                    advance()
                    continue

                # Load document
                if debug:
                    console.print(f"  [dim]Loading: {file_path}[/dim]")

                doc = loader.load(str(file_path))

                if not doc.content.strip():
                    print_warning(f"Empty document: {file_path.name}")
                    skipped_files += 1
                    advance()
                    continue

                # Chunk document
                chunks = chunker.chunk_document(doc)

                if debug:
                    console.print(f"  [dim]Created {len(chunks)} chunks[/dim]")

                if not chunks:
                    print_warning(f"No chunks created from: {file_path.name}")
                    skipped_files += 1
                    advance()
                    continue

                # Add to vector store
                chunk_ids = store.add_chunks(chunks)

                if debug:
                    console.print(f"  [dim]Stored {len(chunk_ids)} chunks[/dim]")

                total_chunks += len(chunk_ids)
                processed_files += 1

            except Exception as e:
                print_error(f"Failed to process {file_path.name}: {e}")
                if debug:
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
                failed_files += 1

            advance()

    # Summary
    console.print()
    if processed_files > 0:
        msg = f"Processed {processed_files} document(s), {total_chunks} chunks added"
        print_success(msg)
    if skipped_files > 0:
        print_info(f"Skipped {skipped_files} document(s)")
    if failed_files > 0:
        print_error(f"Failed to process {failed_files} document(s)")

    # Show total
    total_in_store = store.count()
    console.print(f"\nTotal chunks in knowledge base: [bold cyan]{total_in_store}[/]")


@cli.command()
@click.argument("question")
@click.option(
    "--llm", "-l",
    type=click.Choice(["ollama", "claude"]),
    default="ollama",
    help="LLM provider to use (default: ollama - free and local)",
)
@click.option(
    "-k", "--top-k",
    default=5,
    help="Number of relevant chunks to retrieve",
)
@click.option(
    "--temperature", "-t",
    default=None,
    type=float,
    help="LLM temperature (0.0-1.0)",
)
@click.option(
    "--show-sources", "-s",
    is_flag=True,
    help="Show source documents used",
)
@click.pass_context
def query(
    ctx: click.Context,
    question: str,
    llm: str,
    top_k: int,
    temperature: float | None,
    show_sources: bool,
) -> None:
    """Query the knowledge base.

    Ask a question and get an answer based on the indexed documents.
    """
    from .llm_client import LLMClient
    from .pipeline import RAGPipeline
    from .retriever import Retriever
    from .vector_store import VectorStore

    debug = ctx.obj.get("debug", False)

    # Check if we have documents
    store = VectorStore()
    chunk_count = store.count()

    if chunk_count == 0:
        print_error("No documents indexed yet.")
        print_info("Add documents with: rag-cli add <path>")
        return

    console.print(f"Query: [cyan]{question}[/cyan]")
    console.print(f"Using LLM: [yellow]{llm}[/yellow], retrieving top {top_k} chunks")
    console.print()

    try:
        # Create pipeline
        retriever = Retriever(vector_store=store, top_k=top_k)
        llm_client = LLMClient(provider=llm)
        pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client)

        # Check if LLM is available
        if not llm_client.is_available():
            if llm == "ollama":
                print_error("Ollama is not running.")
                print_info("Start Ollama with: ollama serve")
                print_info("Then pull a model: ollama pull llama3.1:8b")
            else:
                print_error("Claude API is not available.")
                print_info("Set your API key: export ANTHROPIC_API_KEY=your_key")
            return

        # Execute query
        with spinner("Searching and generating answer..."):
            result = pipeline.query(
                question=question,
                top_k=top_k,
                temperature=temperature,
            )

        # Display answer
        console.print(Panel(result.answer, title="Answer", border_style="green"))

        # Show sources if requested
        if show_sources and result.sources:
            console.print("\n[bold]Sources:[/bold]")
            for source in result.sources:
                console.print(f"  • {source}")

        # Debug info
        if debug:
            console.print(f"\n[dim]Retrieved {result.retrieval.total_chunks} chunks")
            console.print(f"Average similarity: {result.retrieval.avg_similarity:.3f}")
            console.print(f"Template used: {result.template_used}[/dim]")

    except Exception as e:
        print_error(f"Query failed: {e}")
        if debug:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


@cli.command()
@click.option(
    "--llm", "-l",
    type=click.Choice(["ollama", "claude"]),
    default="ollama",
    help="LLM provider to use (default: ollama - free and local)",
)
@click.option(
    "--stream", "-s",
    is_flag=True,
    help="Stream responses as they are generated",
)
@click.pass_context
def chat(ctx: click.Context, llm: str, stream: bool) -> None:
    """Start an interactive chat session.

    Enter a conversation loop to ask multiple questions.
    """
    from .llm_client import LLMClient
    from .pipeline import RAGPipeline
    from .retriever import Retriever
    from .vector_store import VectorStore

    # Check if we have documents
    store = VectorStore()
    chunk_count = store.count()

    if chunk_count == 0:
        print_error("No documents indexed yet.")
        print_info("Add documents with: rag-cli add <path>")
        return

    console.print(
        Panel(
            f"Interactive chat mode (using [yellow]{llm}[/yellow])\n"
            f"Knowledge base: {chunk_count} chunks\n"
            "Type 'exit' or 'quit' to end the session.",
            title="RAG CLI Chat",
        )
    )

    try:
        # Create pipeline
        retriever = Retriever(vector_store=store)
        llm_client = LLMClient(provider=llm)
        pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client)

        # Check if LLM is available
        if not llm_client.is_available():
            if llm == "ollama":
                print_error("Ollama is not running.")
                print_info("Start Ollama with: ollama serve")
            else:
                print_error("Claude API is not available.")
            return

        # Chat loop
        while True:
            try:
                question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()

                if not question:
                    continue

                if question.lower() in ("exit", "quit", "q"):
                    console.print("[dim]Goodbye![/dim]")
                    break

                console.print("[bold green]Assistant:[/bold green] ", end="")

                if stream:
                    for chunk in pipeline.query_stream(question):
                        console.print(chunk, end="")
                    console.print()
                else:
                    with spinner("Thinking..."):
                        result = pipeline.query(question)
                    console.print(result.answer)

            except KeyboardInterrupt:
                console.print("\n[dim]Goodbye![/dim]")
                break

    except Exception as e:
        print_error(f"Chat failed: {e}")


@cli.command("list")
@click.pass_context
def list_documents(ctx: click.Context) -> None:
    """List all indexed documents."""
    from .vector_store import VectorStore

    store = VectorStore()
    documents = store.list_documents()

    if not documents:
        print_info("No documents indexed yet.")
        print_info("Add documents with: rag-cli add <path>")
        return

    console.print(f"[bold]Indexed Documents ({len(documents)}):[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("Document", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Chunks", justify="right")

    sorted_docs = sorted(documents, key=lambda d: d.get("source_file", ""))
    for i, doc in enumerate(sorted_docs, 1):
        table.add_row(
            str(i),
            doc.get("source_file", "unknown"),
            doc.get("file_type", "unknown"),
            str(doc.get("chunk_count", 0)),
        )

    console.print(table)

    # Show total chunks
    total_chunks = store.count()
    console.print(f"\nTotal chunks: [bold]{total_chunks}[/bold]")


@cli.command()
@click.argument("document_name")
@click.pass_context
def remove(ctx: click.Context, document_name: str) -> None:
    """Remove a document from the knowledge base."""
    from .vector_store import VectorStore

    store = VectorStore()

    # Check if document exists
    documents = store.list_documents()
    doc_names = [d.get("source_file", "") for d in documents]
    if document_name not in doc_names:
        print_error(f"Document not found: {document_name}")
        print_info("Use 'rag-cli list' to see indexed documents")
        return

    console.print(f"Removing document: [cyan]{document_name}[/cyan]")

    with spinner("Removing..."):
        deleted_count = store.delete_document(document_name)

    print_success(f"Removed {deleted_count} chunks from '{document_name}'")

    # Show remaining
    remaining = store.count()
    console.print(f"Remaining chunks: [bold]{remaining}[/bold]")


@cli.command()
@click.option("--confirm", is_flag=True, help="Confirm clearing the database")
@click.pass_context
def clear(ctx: click.Context, confirm: bool) -> None:
    """Clear the entire knowledge base."""
    from .vector_store import VectorStore

    store = VectorStore()
    current_count = store.count()

    if current_count == 0:
        print_info("Knowledge base is already empty.")
        return

    if not confirm:
        print_warning(
            f"This will delete all {current_count} chunks from the knowledge base."
        )
        console.print("Use --confirm flag to proceed.")
        return

    with spinner("Clearing knowledge base..."):
        store.clear()

    print_success("Knowledge base cleared.")


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show system statistics."""
    from .config import get_settings
    from .vector_store import VectorStore

    settings = get_settings()
    store = VectorStore()

    stats = store.get_stats()
    documents = store.list_documents()

    console.print(Panel("[bold]RAG CLI Statistics[/bold]", title="Stats"))

    table = Table(show_header=False, box=None)
    table.add_column("Stat", style="dim")
    table.add_column("Value", style="bold cyan")

    table.add_row("Documents indexed", str(len(documents)))
    table.add_row("Total chunks", str(stats.get("total_chunks", 0)))
    table.add_row("Collection", stats.get("collection_name", "N/A"))

    table.add_row("", "")
    table.add_row("Embedding model", settings.embedding.embedding_model)
    table.add_row("Chunk size", str(settings.chunking.chunk_size))
    table.add_row("Chunk overlap", str(settings.chunking.chunk_overlap))
    table.add_row("LLM provider", settings.llm.default_llm_provider)

    console.print(table)


@cli.group()
def config() -> None:
    """Manage configuration settings."""
    pass


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value."""
    console.print(f"Setting [cyan]{key}[/cyan] = [green]{value}[/green]")
    print_warning("Runtime configuration changes not yet implemented.")
    print_info("Edit .env file or set environment variables instead.")


@config.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Get a configuration value."""
    settings = get_settings()

    # Navigate nested settings
    parts = key.split(".")
    value = settings

    try:
        for part in parts:
            value = getattr(value, part)
        console.print(f"[cyan]{key}[/cyan] = [green]{value}[/green]")
    except AttributeError:
        print_error(f"Unknown setting: {key}")


@config.command("list")
@click.pass_context
def config_list(ctx: click.Context) -> None:
    """List all configuration values."""
    settings = ctx.obj.get("settings") or get_settings()

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # LLM settings
    table.add_row("llm.default_provider", settings.llm.default_llm_provider)
    table.add_row("llm.ollama_url", settings.llm.ollama_base_url)
    table.add_row("llm.ollama_model", settings.llm.ollama_model)
    table.add_row("llm.temperature", str(settings.llm.llm_temperature))
    table.add_row("llm.max_tokens", str(settings.llm.max_tokens))

    # Embedding settings
    table.add_row("embedding.model", settings.embedding.embedding_model)

    # Chunking settings
    table.add_row("chunking.chunk_size", str(settings.chunking.chunk_size))
    table.add_row("chunking.chunk_overlap", str(settings.chunking.chunk_overlap))

    # Retrieval settings
    table.add_row("retrieval.top_k", str(settings.retrieval.top_k_results))

    # Vector store settings
    table.add_row("vector_store.path", str(settings.vector_store.persist_path))

    # Logging settings
    table.add_row("logging.level", settings.logging.log_level)

    console.print(table)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
