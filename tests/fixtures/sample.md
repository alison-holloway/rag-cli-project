# Sample Markdown Document

This is a sample markdown document for testing the RAG CLI document loader.

## Introduction

The RAG (Retrieval-Augmented Generation) system combines the power of large language models with document retrieval to provide accurate, context-aware responses.

## Key Features

- **Document Processing**: Supports PDF, Markdown, and HTML files
- **Local Embeddings**: Uses sentence-transformers for privacy
- **Vector Storage**: ChromaDB for efficient similarity search
- **LLM Integration**: Supports Ollama (free) and Claude (optional)

## Code Example

```python
from rag_cli import DocumentLoader

loader = DocumentLoader()
doc = loader.load("example.pdf")
print(doc.content)
```

## Table Example

| Feature | Status |
|---------|--------|
| PDF Support | Done |
| Markdown | Done |
| HTML | Done |

## Conclusion

This document serves as a test fixture for the document loading functionality.
