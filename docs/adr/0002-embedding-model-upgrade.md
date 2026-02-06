# ADR 0002: Embedding Model Upgrade

**Date:** 2026-02-06

**Status:** Accepted

**Deciders:** Development team

## Context

The RAG CLI system uses embedding models to convert text into vectors for semantic search. During evaluation, we observed:

1. **Low similarity scores** - Average cosine similarity of ~0.37 for relevant queries with the original model
2. **User expectation** - Target similarity scores of 0.6+ for confident retrieval
3. **Domain specificity** - Technical documentation (Oracle OLCNE) may benefit from specialized models

### Benchmark Results

We benchmarked 4 embedding models on 50 HTML documents from our technical documentation corpus using 3 representative queries:

| Model | Dims | Avg Sim | Max Sim | Avg Query (ms) | Load (s) |
|-------|------|---------|---------|----------------|----------|
| all-MiniLM-L6-v2 | 384 | 0.368 | 0.704 | 33.6 | 7.4 |
| all-mpnet-base-v2 | 768 | 0.407 | 0.753 | 596.4 | 166.8 |
| BAAI/bge-small-en-v1.5 | 384 | **0.680** | **0.801** | **8.1** | 58.1 |
| BAAI/bge-base-en-v1.5 | 768 | 0.635 | 0.754 | 42.4 | 128.9 |

**Test Queries:**
- "How do I install the CLI?"
- "What are the providers?"
- "How do I configure a Kubernetes cluster?"

## Decision

**Upgrade to `BAAI/bge-small-en-v1.5`** as the default embedding model.

### Rationale

1. **84.8% improvement** in average similarity scores (0.368 -> 0.680)
2. **Same dimensions** as baseline (384), no storage increase required
3. **Fastest query time** at 8.1ms (75% faster than baseline!)
4. **Best quality/speed tradeoff** - Higher similarity than larger models with lower latency
5. **BGE model family** - Optimized for retrieval tasks, well-maintained by BAAI

### Why not BAAI/bge-base-en-v1.5?

While bge-base has slightly higher max similarity (0.754 vs 0.801 for bge-small, interestingly bge-small is higher here), the smaller model actually outperforms it on average similarity (0.680 vs 0.635) while being 5x faster and using half the dimensions.

### Migration Steps

1. **Backup existing data:**
   ```bash
   cp -r data/vector_db data/vector_db_backup_$(date +%Y%m%d)
   ```

2. **Update .env:**
   ```bash
   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   ```

3. **Re-ingest documents:**
   ```bash
   python tools/reingest_with_new_embeddings.py --backup --verbose
   ```

4. **Verify retrieval quality:**
   ```bash
   python tools/compare_embeddings.py --old all-MiniLM-L6-v2 --new BAAI/bge-small-en-v1.5
   ```

## Consequences

### Positive

- **Significantly improved retrieval quality** - 84.8% higher similarity scores
- **Faster queries** - 75% reduction in query latency (33.6ms -> 8.1ms)
- **No storage increase** - Same 384 dimensions as previous model
- **User queries more likely to find relevant content**
- **Better semantic understanding** for technical documentation

### Negative

- **One-time re-indexing required** - All documents must be re-embedded
- **Initial model download** - ~130MB download for new model
- **Model load time** - 58.1s on first load (one-time cost, cached after)

### Mitigations

- The `reingest_with_new_embeddings.py` tool automates the migration
- The `--backup` flag protects against data loss
- Model is cached after first download

## Rollback Instructions

If issues arise, rollback to previous model:

1. **Restore backup:**
   ```bash
   rm -rf data/vector_db
   mv data/vector_db_backup_YYYYMMDD data/vector_db
   ```

2. **Revert .env:**
   ```bash
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

3. **Restart services:**
   ```bash
   ./stop-web.sh && ./start-web.sh
   ```

## Tools Created

As part of this migration, we created three new tools in `tools/`:

1. **benchmark_embeddings.py** - Compare embedding model performance
2. **reingest_with_new_embeddings.py** - Re-index documents with new model
3. **compare_embeddings.py** - Compare retrieval results between models

## Future Considerations

- **Query prefixes** - BGE models support optional query prefixes for asymmetric retrieval
- **Fine-tuning** - Consider fine-tuning on domain-specific corpus for further improvement
- **Hybrid retrieval** - Combine dense (embedding) with sparse (BM25) retrieval

## References

- [BGE Embeddings on HuggingFace](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Sentence Transformers Pretrained Models](https://www.sbert.net/docs/pretrained_models.html)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Benchmark Results](../embedding_benchmark_results.md)
