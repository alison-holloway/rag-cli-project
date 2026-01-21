"""Main CLI interface for RAG CLI."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import get_settings
from .logger import setup_logging

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="rag-cli")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """RAG CLI - A command-line RAG system for document question-answering.

    Process documents (PDF, Markdown, HTML) and enable intelligent
    question-answering using local embeddings and LLM integration.
    """
    ctx.ensure_object(dict)

    # Initialize logging
    logger = setup_logging()
    if debug:
        import logging

        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    # Store settings and logger in context
    ctx.obj["settings"] = get_settings()
    ctx.obj["logger"] = logger

    logger.debug("RAG CLI initialized")


@cli.command()
@click.argument("project_name", default="rag-project")
def init(project_name: str) -> None:
    """Initialize a new RAG project.

    Creates the necessary directory structure and configuration files.
    """
    console.print(
        Panel(
            f"Initializing RAG project: [bold cyan]{project_name}[/bold cyan]",
            title="RAG CLI",
        )
    )
    # TODO: Implement project initialization
    console.print("[green]Project initialized successfully![/green]")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def add(path: str) -> None:
    """Add documents to the knowledge base.

    PATH can be a single file or a directory containing documents.
    Supported formats: PDF, Markdown, HTML.
    """
    console.print(f"Adding documents from: [cyan]{path}[/cyan]")
    # TODO: Implement document addition
    console.print("[green]Documents added successfully![/green]")


@cli.command()
@click.argument("question")
@click.option(
    "--llm",
    type=click.Choice(["ollama", "claude"]),
    default="ollama",
    help="LLM provider to use (default: ollama - free and local)",
)
@click.option(
    "-k",
    "--top-k",
    default=5,
    help="Number of relevant chunks to retrieve",
)
def query(question: str, llm: str, top_k: int) -> None:
    """Query the knowledge base.

    Ask a question and get an answer based on the indexed documents.
    """
    console.print(f"Query: [cyan]{question}[/cyan]")
    console.print(f"Using LLM: [yellow]{llm}[/yellow]")
    # TODO: Implement query functionality
    console.print("[dim]No documents indexed yet. Use 'rag-cli add' first.[/dim]")


@cli.command()
@click.option(
    "--llm",
    type=click.Choice(["ollama", "claude"]),
    default="ollama",
    help="LLM provider to use (default: ollama - free and local)",
)
def chat(llm: str) -> None:
    """Start an interactive chat session.

    Enter a conversation loop to ask multiple questions.
    """
    console.print(
        Panel(
            f"Interactive chat mode (using [yellow]{llm}[/yellow])\n"
            "Type 'exit' or 'quit' to end the session.",
            title="RAG CLI Chat",
        )
    )
    # TODO: Implement chat loop
    console.print("[dim]Chat mode not yet implemented.[/dim]")


@cli.command("list")
def list_documents() -> None:
    """List all indexed documents."""
    console.print("[bold]Indexed Documents:[/bold]")
    # TODO: Implement document listing
    console.print("[dim]No documents indexed yet.[/dim]")


@cli.command()
@click.argument("document_id")
def remove(document_id: str) -> None:
    """Remove a document from the knowledge base."""
    console.print(f"Removing document: [cyan]{document_id}[/cyan]")
    # TODO: Implement document removal
    console.print("[green]Document removed successfully![/green]")


@cli.command()
@click.option("--confirm", is_flag=True, help="Confirm clearing the database")
def clear(confirm: bool) -> None:
    """Clear the entire knowledge base."""
    if not confirm:
        console.print(
            "[yellow]Warning:[/yellow] This will delete all indexed documents.\n"
            "Use --confirm flag to proceed."
        )
        return
    # TODO: Implement database clearing
    console.print("[green]Knowledge base cleared.[/green]")


@cli.command()
def stats() -> None:
    """Show system statistics."""
    console.print(Panel("[bold]RAG CLI Statistics[/bold]", title="Stats"))
    # TODO: Implement statistics display
    console.print("Documents indexed: [cyan]0[/cyan]")
    console.print("Total chunks: [cyan]0[/cyan]")
    console.print("Vector store size: [cyan]0 MB[/cyan]")


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
    # TODO: Implement configuration setting
    console.print("[green]Configuration updated.[/green]")


@config.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Get a configuration value."""
    # TODO: Implement configuration getting
    console.print(f"[cyan]{key}[/cyan] = [dim]not set[/dim]")


@config.command("list")
@click.pass_context
def config_list(ctx: click.Context) -> None:
    """List all configuration values."""
    settings = ctx.obj.get("settings") or get_settings()

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # LLM settings
    table.add_row("LLM Provider", settings.llm.default_llm_provider)
    table.add_row("Ollama URL", settings.llm.ollama_base_url)
    table.add_row("Ollama Model", settings.llm.ollama_model)
    table.add_row("Temperature", str(settings.llm.llm_temperature))
    table.add_row("Max Tokens", str(settings.llm.max_tokens))

    # Embedding settings
    table.add_row("Embedding Model", settings.embedding.embedding_model)

    # Chunking settings
    table.add_row("Chunk Size", str(settings.chunking.chunk_size))
    table.add_row("Chunk Overlap", str(settings.chunking.chunk_overlap))

    # Retrieval settings
    table.add_row("Top K Results", str(settings.retrieval.top_k_results))

    # Logging settings
    table.add_row("Log Level", settings.logging.log_level)

    console.print(table)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
