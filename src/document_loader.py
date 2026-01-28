"""Document loader module for RAG CLI.

Handles loading and text extraction from PDF, Markdown, and HTML files.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import markdown
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata associated with a document."""

    filename: str
    file_path: str
    file_type: str
    file_size: int
    created_at: datetime
    modified_at: datetime
    document_hash: str
    title: str | None = None
    author: str | None = None
    page_count: int | None = None
    word_count: int | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for storage."""
        return {
            "filename": self.filename,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "document_hash": self.document_hash,
            "title": self.title,
            "author": self.author,
            "page_count": self.page_count,
            "word_count": self.word_count,
            **self.extra,
        }


@dataclass
class Document:
    """Represents a loaded document with content and metadata."""

    content: str
    metadata: DocumentMetadata
    pages: list[str] | None = None  # For PDFs, content per page

    @property
    def word_count(self) -> int:
        """Count words in the document."""
        return len(self.content.split())


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    supported_extensions: list[str] = []

    @abstractmethod
    def load(self, file_path: Path) -> Document:
        """Load a document from the given path.

        Args:
            file_path: Path to the document file.

        Returns:
            Document object with content and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file type is not supported.
        """
        pass

    @classmethod
    def can_handle(cls, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return file_path.suffix.lower() in cls.supported_extensions

    def _get_file_metadata(self, file_path: Path, file_type: str) -> dict:
        """Get basic file metadata."""
        stat = file_path.stat()

        # Calculate file hash
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        return {
            "filename": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_type": file_type,
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_birthtime)
            if hasattr(stat, "st_birthtime")
            else datetime.fromtimestamp(stat.st_ctime),
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
            "document_hash": file_hash,
        }


class PDFLoader(BaseLoader):
    """Loader for PDF documents."""

    supported_extensions = [".pdf"]

    def load(self, file_path: Path) -> Document:
        """Load a PDF document.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Document with extracted text and metadata.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.can_handle(file_path):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        logger.debug(f"Loading PDF: {file_path}")

        reader = PdfReader(file_path)

        # Extract text from each page
        pages = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(text)
            logger.debug(f"Extracted {len(text)} chars from page {page_num + 1}")

        # Combine all text
        full_text = "\n\n".join(pages)

        # Get PDF metadata
        pdf_info = reader.metadata or {}
        title = pdf_info.get("/Title") if pdf_info else None
        author = pdf_info.get("/Author") if pdf_info else None

        # Clean up title/author if they're PyPDF objects
        if title and hasattr(title, "__str__"):
            title = str(title)
        if author and hasattr(author, "__str__"):
            author = str(author)

        # Build metadata
        base_meta = self._get_file_metadata(file_path, "pdf")
        metadata = DocumentMetadata(
            **base_meta,
            title=title,
            author=author,
            page_count=len(reader.pages),
            word_count=len(full_text.split()),
        )

        logger.info(
            f"Loaded PDF: {file_path.name} "
            f"({metadata.page_count} pages, {metadata.word_count} words)"
        )

        return Document(content=full_text, metadata=metadata, pages=pages)


class MarkdownLoader(BaseLoader):
    """Loader for Markdown documents."""

    supported_extensions = [".md", ".markdown"]

    def load(self, file_path: Path) -> Document:
        """Load a Markdown document.

        Args:
            file_path: Path to the Markdown file.

        Returns:
            Document with extracted text (HTML tags stripped) and metadata.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.can_handle(file_path):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        logger.debug(f"Loading Markdown: {file_path}")

        # Read the raw markdown content
        raw_content = file_path.read_text(encoding="utf-8")

        # Convert to HTML then extract plain text
        html_content = markdown.markdown(
            raw_content,
            extensions=["tables", "fenced_code", "toc"],
        )

        # Extract plain text from HTML
        soup = BeautifulSoup(html_content, "lxml")
        plain_text = soup.get_text(separator="\n")

        # Clean up excessive whitespace
        plain_text = re.sub(r"\n{3,}", "\n\n", plain_text)
        plain_text = plain_text.strip()

        # Try to extract title from first H1
        title = None
        h1_match = re.match(r"^#\s+(.+)$", raw_content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()

        # Build metadata
        base_meta = self._get_file_metadata(file_path, "markdown")
        metadata = DocumentMetadata(
            **base_meta,
            title=title,
            word_count=len(plain_text.split()),
            extra={"raw_markdown_length": len(raw_content)},
        )

        logger.info(f"Loaded Markdown: {file_path.name} ({metadata.word_count} words)")

        return Document(content=plain_text, metadata=metadata)


class HTMLLoader(BaseLoader):
    """Loader for HTML documents."""

    supported_extensions = [".html", ".htm"]

    def __init__(self, preserve_links: bool = False):
        """Initialize HTML loader.

        Args:
            preserve_links: If True, preserve link URLs in the text.
        """
        self.preserve_links = preserve_links

    def load(self, file_path: Path) -> Document:
        """Load an HTML document.

        Args:
            file_path: Path to the HTML file.

        Returns:
            Document with extracted text (tags stripped) and metadata.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.can_handle(file_path):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        logger.debug(f"Loading HTML: {file_path}")

        # Read the raw HTML content
        raw_content = file_path.read_text(encoding="utf-8")

        # Parse with BeautifulSoup
        soup = BeautifulSoup(raw_content, "lxml")

        # Extract title from <title> tag
        title = None
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()

        # Extract author from meta tags
        author = None
        author_meta = soup.find("meta", {"name": "author"})
        if author_meta:
            author = author_meta.get("content", "").strip() or None

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Handle links if preserving
        if self.preserve_links:
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                text = a_tag.get_text()
                if href.startswith(("http://", "https://")):
                    a_tag.replace_with(f"{text} ({href})")

        # Extract text
        plain_text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = [line.strip() for line in plain_text.splitlines()]
        plain_text = "\n".join(line for line in lines if line)
        plain_text = re.sub(r"\n{3,}", "\n\n", plain_text)

        # Build metadata
        base_meta = self._get_file_metadata(file_path, "html")
        metadata = DocumentMetadata(
            **base_meta,
            title=title,
            author=author,
            word_count=len(plain_text.split()),
            extra={"raw_html_length": len(raw_content)},
        )

        logger.info(f"Loaded HTML: {file_path.name} ({metadata.word_count} words)")

        return Document(content=plain_text, metadata=metadata)


class DocumentLoader:
    """Main document loader that delegates to appropriate format-specific loaders."""

    def __init__(self):
        """Initialize the document loader with all available format loaders."""
        self._loaders: list[BaseLoader] = [
            PDFLoader(),
            MarkdownLoader(),
            HTMLLoader(),
        ]
        self._extension_map: dict[str, BaseLoader] = {}

        # Build extension map for quick lookup
        for loader in self._loaders:
            for ext in loader.supported_extensions:
                self._extension_map[ext] = loader

    @property
    def supported_extensions(self) -> list[str]:
        """Get list of all supported file extensions."""
        return list(self._extension_map.keys())

    def can_load(self, file_path: Path | str) -> bool:
        """Check if the file can be loaded.

        Args:
            file_path: Path to check.

        Returns:
            True if the file type is supported.
        """
        path = Path(file_path)
        return path.suffix.lower() in self._extension_map

    def load(self, file_path: Path | str) -> Document:
        """Load a document from the given path.

        Args:
            file_path: Path to the document.

        Returns:
            Document object with content and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file type is not supported.
        """
        path = Path(file_path)

        # Check extension first to give clearer error messages
        ext = path.suffix.lower()
        if ext not in self._extension_map:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported types: {', '.join(self.supported_extensions)}"
            )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        loader = self._extension_map[ext]
        return loader.load(path)

    def load_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
    ) -> Iterator[Document]:
        """Load all supported documents from a directory.

        Args:
            directory: Path to the directory.
            recursive: If True, search subdirectories.

        Yields:
            Document objects for each successfully loaded file.
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        # Find all files with supported extensions
        pattern = "**/*" if recursive else "*"
        files = list(dir_path.glob(pattern))

        loaded_count = 0
        error_count = 0

        for file_path in files:
            if not file_path.is_file():
                continue

            if not self.can_load(file_path):
                continue

            try:
                yield self.load(file_path)
                loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                error_count += 1

        logger.info(
            f"Loaded {loaded_count} documents from {dir_path} ({error_count} errors)"
        )
