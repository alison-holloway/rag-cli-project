# ADR 0001: Python Version Compatibility

**Date:** 2026-02-04

**Status:** Accepted

**Deciders:** Development team

## Context

During development and user testing, we discovered that the RAG CLI project has Python version compatibility issues stemming from its dependency chain:

1. **ChromaDB** requires `onnxruntime>=1.14.1` for its default embedding functions
2. **onnxruntime** does not publish pre-built wheels for all Python version and platform combinations
3. Specifically, `onnxruntime` lacks Python 3.13 wheels for macOS x86_64 (Intel Macs)

This caused installation failures for users on Intel Macs trying to use Python 3.13:

```
ERROR: ResolutionImpossible ... onnxruntime
Additionally, some packages in these conflicts have no matching distributions
available for your environment: onnxruntime
```

### Affected Platforms

| Platform | Python 3.12 | Python 3.13 |
|----------|-------------|-------------|
| macOS Apple Silicon (arm64) | ✅ Works | ✅ Works |
| macOS Intel (x86_64) | ✅ Works | ❌ Fails |
| Linux x86_64 | ✅ Works | ✅ Works |
| Windows x86_64 | ✅ Works | ✅ Works |

## Decision

**Recommend Python 3.12 as the primary supported version** across all documentation and installation guides.

- Python 3.12 provides the broadest platform compatibility
- Python 3.13 remains supported for users on compatible platforms (Apple Silicon, Linux, Windows)
- Python 3.14 is explicitly not supported (ChromaDB compatibility)

### Changes Made

1. Updated all documentation to recommend Python 3.12
2. Added platform-specific notes for Intel Mac users
3. Added troubleshooting section for onnxruntime installation failures
4. Updated `requirements.txt` to require `chromadb>=0.5.0` for pydantic 2.x compatibility

## Consequences

### Positive

- Installation works reliably across all major platforms
- Clear documentation prevents user frustration
- No code changes required - purely documentation fix

### Negative

- Users on Apple Silicon/Linux may want to use Python 3.13 features but documentation defaults to 3.12
- Need to maintain awareness of onnxruntime wheel availability for future Python versions

## Future Considerations

### Container-Based Deployment

To eliminate platform-specific dependency issues entirely, consider providing containerized deployment options:

```yaml
# Example docker-compose.yml
services:
  rag-cli:
    image: rag-cli:latest
    volumes:
      - ./data:/app/data
      - ./documents:/app/documents
    ports:
      - "8000:8000"
```

**Benefits:**
- Eliminates Python version and platform compatibility issues
- Consistent environment across all users
- Easier deployment and updates
- Could bundle Ollama in the container

**Considerations:**
- Docker requires additional setup for users
- Performance overhead (minimal for this use case)
- Volume mounting for documents and data persistence

### Alternative Vector Databases

ChromaDB's dependency on onnxruntime (for default embeddings) creates friction. Since this project uses sentence-transformers for embeddings anyway, consider alternatives:

| Database | Pros | Cons |
|----------|------|------|
| **Weaviate** | Production-ready, good Python client, no onnxruntime dep | Heavier, typically run as service |
| **Qdrant** | Rust-based, fast, good Python client | Requires separate server process |
| **Milvus** | Highly scalable, mature | Complex setup, overkill for local use |
| **LanceDB** | Embedded (like ChromaDB), no onnxruntime | Newer, less mature ecosystem |
| **pgvector** | Leverages PostgreSQL, SQL interface | Requires PostgreSQL |

**Recommendation:** If ChromaDB continues to cause issues, evaluate **LanceDB** (similar embedded use case, pure Python/Rust) or **Qdrant** (if moving to client-server architecture).

### Monitoring

- Watch onnxruntime releases for Python 3.13+ macOS x86_64 wheels
- Track ChromaDB updates that might make onnxruntime optional
- Re-evaluate when Python 3.14 approaches mainstream adoption

## References

- [onnxruntime PyPI](https://pypi.org/project/onnxruntime/#files) - wheel availability
- [ChromaDB GitHub](https://github.com/chroma-core/chroma) - dependency discussions
- [LanceDB](https://lancedb.com/) - potential alternative
- [Weaviate](https://weaviate.io/) - potential alternative
