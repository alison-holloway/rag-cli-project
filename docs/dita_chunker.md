# DITA Documentation Chunker

A semantic chunking tool for DITA-generated HTML documentation that respects document structure for improved RAG retrieval quality.

## Overview

The standard RAG chunking approach splits documents at fixed character boundaries, which can break up procedural steps and lose semantic context. This tool performs **semantic chunking** based on DITA document types:

- **Task documents**: Keeps entire procedures together (steps + examples)
- **Concept documents**: Preserves complete explanations
- **Reference documents**: Splits at logical boundaries (syntax/params vs examples)
- **Topic documents**: Section-based chunking

## Usage

```bash
# Basic usage with default configuration
python tools/ingest_dita_docs.py

# Preview without storing (dry run)
python tools/ingest_dita_docs.py --dry-run

# Show detailed progress
python tools/ingest_dita_docs.py --verbose

# Clear existing documents before ingesting
python tools/ingest_dita_docs.py --clear-first

# Override input directory
python tools/ingest_dita_docs.py --input-dir ./my-docs/

# Use custom config file
python tools/ingest_dita_docs.py --config ./my-config.yaml
```

## Configuration

Configuration file: `config/dita_chunker.yaml`

```yaml
# Input/output settings
input_dir: "data/documents/html/"
collection_name: "rag_documents"

# Chunk size constraints (characters)
min_chunk_size: 200
max_chunk_size: 4000
target_chunk_size: 1500

# Logging
verbose: false
```

## Chunking Strategies

### Task Documents

Task documents contain step-by-step procedures. The chunker keeps all steps together as a single retrieval unit:

```
# Installing the CLI

Install the Oracle Cloud Native Environment Command Line Interface...

## Steps

1. Set up the Oracle Linux Yum Server Repository.
   If the system uses the Oracle Linux Yum Server...
   ```
   sudo dnf install -y oracle-ocne-release-el9
   ```

2. Set up ULN.
   If the system is registered to use ULN...

3. Install the CLI.
   ```
   sudo dnf install -y ocne
   ```
```

**Result**: Complete procedure in one chunk, making retrieval return actionable instructions.

### Concept Documents

Concept documents are explanatory content kept as single units:

```
# Pods

Describes Kubernetes pods.

Kubernetes introduces the concept of pods, which are groupings of one or more
containers and their shared storage...
```

### Reference Documents

Reference documents (CLI commands, APIs) are split at semantic boundaries:

- **Chunk 1**: Command description + syntax + parameters
- **Chunk 2+**: Examples (one per chunk if multiple)

### Topic Documents

General topic pages are split at `<div class="section">` boundaries.

## Metadata

Each chunk includes rich metadata for filtering and source tracking:

| Field | Description |
|-------|-------------|
| `source_file` | Original HTML filename |
| `source_path` | Full path to source file |
| `file_type` | Always `dita_html` |
| `dita_type` | Document type: task, concept, reference, topic |
| `title` | Page/section title |
| `abstract` | Document abstract (truncated to 500 chars) |
| `dc_identifier` | DITA document identifier |
| `created` | Document creation timestamp |
| `content_type` | Chunk content type: procedure, concept, reference, example, topic |
| `document_hash` | MD5 hash for deduplication |

## Chunk Count Comparison

| Strategy | Files | Chunks | Avg Chunks/File |
|----------|-------|--------|-----------------|
| **Naive (800 char)** | 233 | ~1,200+ | ~5+ |
| **Semantic DITA** | 233 | 351 | 1.5 |

The semantic approach produces **~70% fewer chunks** while maintaining complete, coherent retrieval units.

## Benefits for RAG

1. **Complete Procedures**: Task queries return full step-by-step instructions
2. **Coherent Context**: Concepts aren't split mid-explanation
3. **Better Relevance**: Fewer chunks = less noise in retrieval
4. **Rich Metadata**: Filter by document type, search by identifier

## Example Query

```bash
rag-cli query "How do I install the CLI?" --show-sources
```

Returns the complete installation procedure from the task document, not fragments spread across multiple chunks.

## Integration

The tool integrates with the existing RAG CLI infrastructure:

- Uses the same ChromaDB collection (`rag_documents`)
- Same embedding model (`all-MiniLM-L6-v2`)
- Documents appear in `rag-cli list` and `rag-cli stats`
- Query with `rag-cli query` or the Web UI
