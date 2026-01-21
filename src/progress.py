"""Progress indicators for RAG CLI.

Provides rich progress bars and spinners for long-running operations.
"""

from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


def create_progress() -> Progress:
    """Create a standard progress bar for file operations.

    Returns:
        Progress instance configured for file processing.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


def create_simple_progress() -> Progress:
    """Create a simple progress bar without time estimates.

    Returns:
        Progress instance with minimal columns.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    )


def create_spinner_progress() -> Progress:
    """Create a spinner-only progress indicator.

    Returns:
        Progress instance with just a spinner.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


@contextmanager
def spinner(description: str):
    """Context manager for a simple spinner.

    Args:
        description: Text to show next to the spinner.

    Yields:
        None
    """
    with create_spinner_progress() as progress:
        progress.add_task(description, total=None)
        yield


@contextmanager
def progress_bar(description: str, total: int):
    """Context manager for a progress bar.

    Args:
        description: Text to show with the progress bar.
        total: Total number of items to process.

    Yields:
        Function to call to advance the progress bar.
    """
    with create_progress() as progress:
        task = progress.add_task(description, total=total)

        def advance(amount: int = 1):
            progress.update(task, advance=amount)

        yield advance


def iterate_with_progress(
    items: list,
    description: str,
    show_item: bool = False,
) -> Iterator:
    """Iterate over items with a progress bar.

    Args:
        items: Items to iterate over.
        description: Description for the progress bar.
        show_item: If True, show the current item in the description.

    Yields:
        Items from the input list.
    """
    with create_progress() as progress:
        task = progress.add_task(description, total=len(items))

        for item in items:
            if show_item:
                # Truncate item string if too long
                item_str = str(item)
                if len(item_str) > 40:
                    item_str = item_str[:37] + "..."
                progress.update(task, description=f"{description}: {item_str}")

            yield item
            progress.update(task, advance=1)


class DocumentProcessingProgress:
    """Progress tracker for document processing operations.

    Tracks loading, chunking, and embedding stages.
    """

    def __init__(self, console_output: Console | None = None):
        """Initialize the progress tracker.

        Args:
            console_output: Console to use for output.
        """
        self.console = console_output or console
        self._progress: Progress | None = None
        self._task_id = None

    def start(self, total_files: int) -> None:
        """Start tracking document processing.

        Args:
            total_files: Total number of files to process.
        """
        self._progress = create_progress()
        self._progress.start()
        self._task_id = self._progress.add_task(
            "Processing documents", total=total_files
        )

    def update(self, description: str) -> None:
        """Update the progress description.

        Args:
            description: New description to show.
        """
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, description=description)

    def advance(self, amount: int = 1) -> None:
        """Advance the progress bar.

        Args:
            amount: Amount to advance by.
        """
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=amount)

    def finish(self) -> None:
        """Finish the progress tracking."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
        return False


class MultiStageProgress:
    """Progress tracker for multi-stage operations.

    Useful for operations like: Load -> Chunk -> Embed -> Store
    """

    def __init__(self, stages: list[str], console_output: Console | None = None):
        """Initialize multi-stage progress.

        Args:
            stages: List of stage names.
            console_output: Console for output.
        """
        self.stages = stages
        self.console = console_output or console
        self.current_stage = 0
        self._progress: Progress | None = None
        self._tasks: dict[str, int] = {}

    def start(self) -> None:
        """Start the multi-stage progress display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        self._progress.start()

        # Create a task for each stage
        for i, stage in enumerate(self.stages):
            prefix = "[ ]" if i > 0 else "[*]"
            task_id = self._progress.add_task(
                f"{prefix} {stage}",
                total=100,
                visible=True,
            )
            self._tasks[stage] = task_id

    def update_stage(self, stage: str, progress: float, description: str = "") -> None:
        """Update progress for a specific stage.

        Args:
            stage: Stage name.
            progress: Progress percentage (0-100).
            description: Optional description update.
        """
        if self._progress and stage in self._tasks:
            task_id = self._tasks[stage]
            update_kwargs = {"completed": progress}
            if description:
                prefix = "[*]" if progress < 100 else "[+]"
                update_kwargs["description"] = f"{prefix} {description}"
            self._progress.update(task_id, **update_kwargs)

    def complete_stage(self, stage: str) -> None:
        """Mark a stage as complete.

        Args:
            stage: Stage name to mark complete.
        """
        if self._progress and stage in self._tasks:
            task_id = self._tasks[stage]
            self._progress.update(
                task_id,
                completed=100,
                description=f"[+] {stage}",
            )

            # Start next stage if there is one
            stage_idx = self.stages.index(stage)
            if stage_idx + 1 < len(self.stages):
                next_stage = self.stages[stage_idx + 1]
                next_task_id = self._tasks[next_stage]
                self._progress.update(
                    next_task_id,
                    description=f"[*] {next_stage}",
                )

    def finish(self) -> None:
        """Finish the multi-stage progress."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
        return False


# Convenience functions for common operations


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Success message to display.
    """
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: Error message to display.
    """
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Warning message to display.
    """
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message.

    Args:
        message: Info message to display.
    """
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def print_stats(stats: dict) -> None:
    """Print statistics in a formatted way.

    Args:
        stats: Dictionary of stat names to values.
    """
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Stat", style="dim")
    table.add_column("Value", style="bold")

    for key, value in stats.items():
        table.add_row(key, str(value))

    console.print(table)
