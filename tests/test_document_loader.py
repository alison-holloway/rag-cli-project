"""Tests for document loader module."""

from pathlib import Path

import pytest

from src.document_loader import (
    Document,
    DocumentLoader,
    DocumentMetadata,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
)

# Get the fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""

    def test_to_dict(self) -> None:
        """Test metadata serialization to dictionary."""
        from datetime import datetime

        metadata = DocumentMetadata(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_type="pdf",
            file_size=1024,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            modified_at=datetime(2024, 1, 2, 12, 0, 0),
            document_hash="abc123",
            title="Test Document",
            author="Test Author",
            page_count=5,
            word_count=500,
        )

        result = metadata.to_dict()

        assert result["filename"] == "test.pdf"
        assert result["file_type"] == "pdf"
        assert result["title"] == "Test Document"
        assert result["author"] == "Test Author"
        assert result["page_count"] == 5
        assert result["word_count"] == 500
        assert "created_at" in result
        assert "modified_at" in result


class TestPDFLoader:
    """Tests for PDF document loading."""

    @pytest.fixture
    def loader(self) -> PDFLoader:
        """Create a PDF loader instance."""
        return PDFLoader()

    @pytest.fixture
    def sample_pdf(self) -> Path:
        """Get path to sample PDF."""
        return FIXTURES_DIR / "sample.pdf"

    def test_can_handle_pdf(self, loader: PDFLoader) -> None:
        """Test that loader recognizes PDF files."""
        assert loader.can_handle(Path("test.pdf"))
        assert loader.can_handle(Path("test.PDF"))
        assert not loader.can_handle(Path("test.txt"))
        assert not loader.can_handle(Path("test.md"))

    def test_load_pdf(self, loader: PDFLoader, sample_pdf: Path) -> None:
        """Test loading a PDF document."""
        if not sample_pdf.exists():
            pytest.skip("Sample PDF not found")

        doc = loader.load(sample_pdf)

        assert isinstance(doc, Document)
        assert isinstance(doc.metadata, DocumentMetadata)
        assert doc.metadata.file_type == "pdf"
        assert doc.metadata.filename == "sample.pdf"
        assert doc.metadata.page_count is not None
        assert doc.metadata.page_count >= 1
        assert len(doc.content) > 0

    def test_load_pdf_extracts_text(self, loader: PDFLoader, sample_pdf: Path) -> None:
        """Test that PDF text extraction works."""
        if not sample_pdf.exists():
            pytest.skip("Sample PDF not found")

        doc = loader.load(sample_pdf)

        # Check that some expected content is present
        assert "RAG" in doc.content or "PDF" in doc.content or "Document" in doc.content

    def test_load_pdf_has_pages(self, loader: PDFLoader, sample_pdf: Path) -> None:
        """Test that PDF loader extracts pages separately."""
        if not sample_pdf.exists():
            pytest.skip("Sample PDF not found")

        doc = loader.load(sample_pdf)

        assert doc.pages is not None
        assert len(doc.pages) == doc.metadata.page_count

    def test_load_nonexistent_pdf_raises(self, loader: PDFLoader) -> None:
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            loader.load(Path("nonexistent.pdf"))

    def test_load_wrong_extension_raises(self, loader: PDFLoader) -> None:
        """Test that loading wrong file type raises error."""
        with pytest.raises(ValueError):
            loader.load(FIXTURES_DIR / "sample.md")


class TestMarkdownLoader:
    """Tests for Markdown document loading."""

    @pytest.fixture
    def loader(self) -> MarkdownLoader:
        """Create a Markdown loader instance."""
        return MarkdownLoader()

    @pytest.fixture
    def sample_md(self) -> Path:
        """Get path to sample Markdown file."""
        return FIXTURES_DIR / "sample.md"

    def test_can_handle_markdown(self, loader: MarkdownLoader) -> None:
        """Test that loader recognizes Markdown files."""
        assert loader.can_handle(Path("test.md"))
        assert loader.can_handle(Path("test.markdown"))
        assert loader.can_handle(Path("test.MD"))
        assert not loader.can_handle(Path("test.txt"))
        assert not loader.can_handle(Path("test.pdf"))

    def test_load_markdown(self, loader: MarkdownLoader, sample_md: Path) -> None:
        """Test loading a Markdown document."""
        if not sample_md.exists():
            pytest.skip("Sample Markdown not found")

        doc = loader.load(sample_md)

        assert isinstance(doc, Document)
        assert isinstance(doc.metadata, DocumentMetadata)
        assert doc.metadata.file_type == "markdown"
        assert doc.metadata.filename == "sample.md"
        assert len(doc.content) > 0

    def test_load_markdown_extracts_title(
        self, loader: MarkdownLoader, sample_md: Path
    ) -> None:
        """Test that Markdown title extraction works."""
        if not sample_md.exists():
            pytest.skip("Sample Markdown not found")

        doc = loader.load(sample_md)

        assert doc.metadata.title is not None
        assert "Sample Markdown Document" in doc.metadata.title

    def test_load_markdown_strips_html(
        self, loader: MarkdownLoader, sample_md: Path
    ) -> None:
        """Test that HTML tags are stripped from Markdown output."""
        if not sample_md.exists():
            pytest.skip("Sample Markdown not found")

        doc = loader.load(sample_md)

        # Content should not contain HTML tags
        assert "<h1>" not in doc.content
        assert "<p>" not in doc.content
        assert "<ul>" not in doc.content

    def test_load_markdown_preserves_text(
        self, loader: MarkdownLoader, sample_md: Path
    ) -> None:
        """Test that text content is preserved."""
        if not sample_md.exists():
            pytest.skip("Sample Markdown not found")

        doc = loader.load(sample_md)

        assert "RAG" in doc.content
        assert "Document Processing" in doc.content

    def test_load_nonexistent_markdown_raises(self, loader: MarkdownLoader) -> None:
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            loader.load(Path("nonexistent.md"))


class TestHTMLLoader:
    """Tests for HTML document loading."""

    @pytest.fixture
    def loader(self) -> HTMLLoader:
        """Create an HTML loader instance."""
        return HTMLLoader()

    @pytest.fixture
    def sample_html(self) -> Path:
        """Get path to sample HTML file."""
        return FIXTURES_DIR / "sample.html"

    def test_can_handle_html(self, loader: HTMLLoader) -> None:
        """Test that loader recognizes HTML files."""
        assert loader.can_handle(Path("test.html"))
        assert loader.can_handle(Path("test.htm"))
        assert loader.can_handle(Path("test.HTML"))
        assert not loader.can_handle(Path("test.txt"))
        assert not loader.can_handle(Path("test.pdf"))

    def test_load_html(self, loader: HTMLLoader, sample_html: Path) -> None:
        """Test loading an HTML document."""
        if not sample_html.exists():
            pytest.skip("Sample HTML not found")

        doc = loader.load(sample_html)

        assert isinstance(doc, Document)
        assert isinstance(doc.metadata, DocumentMetadata)
        assert doc.metadata.file_type == "html"
        assert doc.metadata.filename == "sample.html"
        assert len(doc.content) > 0

    def test_load_html_extracts_title(
        self, loader: HTMLLoader, sample_html: Path
    ) -> None:
        """Test that HTML title extraction works."""
        if not sample_html.exists():
            pytest.skip("Sample HTML not found")

        doc = loader.load(sample_html)

        assert doc.metadata.title is not None
        assert "Sample HTML Document" in doc.metadata.title

    def test_load_html_extracts_author(
        self, loader: HTMLLoader, sample_html: Path
    ) -> None:
        """Test that HTML author extraction works."""
        if not sample_html.exists():
            pytest.skip("Sample HTML not found")

        doc = loader.load(sample_html)

        assert doc.metadata.author is not None
        assert "RAG CLI Team" in doc.metadata.author

    def test_load_html_removes_scripts(
        self, loader: HTMLLoader, sample_html: Path
    ) -> None:
        """Test that script content is removed."""
        if not sample_html.exists():
            pytest.skip("Sample HTML not found")

        doc = loader.load(sample_html)

        assert "console.log" not in doc.content
        assert "This script should be ignored" not in doc.content

    def test_load_html_removes_styles(
        self, loader: HTMLLoader, sample_html: Path
    ) -> None:
        """Test that style content is removed."""
        if not sample_html.exists():
            pytest.skip("Sample HTML not found")

        doc = loader.load(sample_html)

        assert "font-family" not in doc.content
        assert "background-color" not in doc.content

    def test_load_html_preserves_text(
        self, loader: HTMLLoader, sample_html: Path
    ) -> None:
        """Test that text content is preserved."""
        if not sample_html.exists():
            pytest.skip("Sample HTML not found")

        doc = loader.load(sample_html)

        assert "RAG CLI" in doc.content
        assert "ChromaDB" in doc.content

    def test_load_nonexistent_html_raises(self, loader: HTMLLoader) -> None:
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            loader.load(Path("nonexistent.html"))


class TestDocumentLoader:
    """Tests for main DocumentLoader class."""

    @pytest.fixture
    def loader(self) -> DocumentLoader:
        """Create a DocumentLoader instance."""
        return DocumentLoader()

    def test_supported_extensions(self, loader: DocumentLoader) -> None:
        """Test that all expected extensions are supported."""
        extensions = loader.supported_extensions

        assert ".pdf" in extensions
        assert ".md" in extensions
        assert ".markdown" in extensions
        assert ".html" in extensions
        assert ".htm" in extensions

    def test_can_load(self, loader: DocumentLoader) -> None:
        """Test can_load method."""
        assert loader.can_load(Path("test.pdf"))
        assert loader.can_load(Path("test.md"))
        assert loader.can_load(Path("test.html"))
        assert loader.can_load("test.pdf")  # String path
        assert not loader.can_load(Path("test.txt"))
        assert not loader.can_load(Path("test.docx"))

    def test_load_pdf(self, loader: DocumentLoader) -> None:
        """Test loading PDF through main loader."""
        sample_pdf = FIXTURES_DIR / "sample.pdf"
        if not sample_pdf.exists():
            pytest.skip("Sample PDF not found")

        doc = loader.load(sample_pdf)

        assert doc.metadata.file_type == "pdf"

    def test_load_markdown(self, loader: DocumentLoader) -> None:
        """Test loading Markdown through main loader."""
        sample_md = FIXTURES_DIR / "sample.md"
        if not sample_md.exists():
            pytest.skip("Sample Markdown not found")

        doc = loader.load(sample_md)

        assert doc.metadata.file_type == "markdown"

    def test_load_html(self, loader: DocumentLoader) -> None:
        """Test loading HTML through main loader."""
        sample_html = FIXTURES_DIR / "sample.html"
        if not sample_html.exists():
            pytest.skip("Sample HTML not found")

        doc = loader.load(sample_html)

        assert doc.metadata.file_type == "html"

    def test_load_unsupported_raises(self, loader: DocumentLoader) -> None:
        """Test that loading unsupported file type raises error."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load(Path("test.xyz"))

    def test_load_nonexistent_raises(self, loader: DocumentLoader) -> None:
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            loader.load(Path("nonexistent.pdf"))

    def test_load_directory(self, loader: DocumentLoader) -> None:
        """Test loading all documents from a directory."""
        if not FIXTURES_DIR.exists():
            pytest.skip("Fixtures directory not found")

        docs = list(loader.load_directory(FIXTURES_DIR))

        # Should load all 3 sample files
        assert len(docs) >= 3

        file_types = {doc.metadata.file_type for doc in docs}
        assert "pdf" in file_types
        assert "markdown" in file_types
        assert "html" in file_types

    def test_load_directory_nonexistent_raises(self, loader: DocumentLoader) -> None:
        """Test that loading from non-existent directory raises error."""
        with pytest.raises(FileNotFoundError):
            list(loader.load_directory(Path("nonexistent_dir")))


class TestDocument:
    """Tests for Document class."""

    def test_word_count_property(self) -> None:
        """Test word count calculation."""
        from datetime import datetime

        metadata = DocumentMetadata(
            filename="test.txt",
            file_path="/test.txt",
            file_type="text",
            file_size=100,
            created_at=datetime.now(),
            modified_at=datetime.now(),
            document_hash="abc",
        )

        doc = Document(content="Hello world this is a test", metadata=metadata)

        assert doc.word_count == 6
