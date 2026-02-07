# Embedding Model Benchmark Results

**Date:** 2026-02-06
**Documents:** 50 HTML files from technical documentation
**Queries:** 3 test queries

## Summary Table

| Model | Dims | Avg Sim | Max Sim | Avg Query (ms) | Load (s) | Notes |
|-------|------|---------|---------|----------------|----------|-------|
| all-MiniLM-L6-v2 | 384 | 0.368 | 0.704 | 33.6 | 7.4 | Baseline |
| all-mpnet-base-v2 | 768 | 0.407 | 0.753 | 596.4 | 166.8 | Higher quality, 2x dimensions |
| BAAI/bge-small-en-v1.5 | 384 | 0.680 | 0.801 | 8.1 | 58.1 | **Best** |
| BAAI/bge-base-en-v1.5 | 768 | 0.635 | 0.754 | 42.4 | 128.9 | Best quality, largest |

## Detailed Results by Query

### Query: "How do I install the CLI?"

**all-MiniLM-L6-v2** (query time: 43.1ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.356 | Installing the CLI |
| 2 | 0.304 | Preface |
| 3 | 0.291 | Preface |
| 4 | 0.283 | Installing an Application |
| 5 | 0.274 | Release 2.0.0 |

**all-mpnet-base-v2** (query time: 1247.8ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.388 | Installing an Application |
| 2 | 0.339 | Installing the CLI |
| 3 | 0.330 | Installing an Application from a Template |
| 4 | 0.305 | Preface |
| 5 | 0.299 | Release 2.0.0 |

**BAAI/bge-small-en-v1.5** (query time: 8.5ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.742 | Installing an Application |
| 2 | 0.732 | Installing the CLI |
| 3 | 0.707 | Installing an Application from a Template |
| 4 | 0.702 | Preface |
| 5 | 0.694 | Release 2.0.0 |

**BAAI/bge-base-en-v1.5** (query time: 50.2ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.691 | Installing an Application |
| 2 | 0.677 | Installing the CLI |
| 3 | 0.663 | Installing an Application from a Template |
| 4 | 0.653 | Preface |
| 5 | 0.649 | Managing Applications |

### Query: "What are the providers?"

**all-MiniLM-L6-v2** (query time: 23.1ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.216 | Kubernetes Components |
| 2 | 0.163 | Linux Virtualization Manager Provider |
| 3 | 0.139 | Creating an Access Token |
| 4 | 0.131 | Creating a Bring Your Own Cluster |
| 5 | 0.131 | Release 2.0.4 |

**all-mpnet-base-v2** (query time: 397.8ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.199 | Kubernetes Components |
| 2 | 0.177 | Preface |
| 3 | 0.163 | Pods |
| 4 | 0.159 | Release 2.0.4 |
| 5 | 0.152 | ocne completion |

**BAAI/bge-small-en-v1.5** (query time: 7.5ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.548 | Release 2.0.4 |
| 2 | 0.548 | Linux Virtualization Manager Provider |
| 3 | 0.525 | Adding the UI and Application Catalogs into a Clus... |
| 4 | 0.521 | Setting a Proxy Server for the UI |
| 5 | 0.518 | Creating a Bring Your Own Cluster |

**BAAI/bge-base-en-v1.5** (query time: 28.5ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.518 | Kubernetes Components |
| 2 | 0.517 | Release 2.0.4 |
| 3 | 0.513 | Linux Virtualization Manager Provider |
| 4 | 0.497 | Getting a Catalog |
| 5 | 0.495 | Setting a Proxy Server for the UI |

### Query: "How do I configure a Kubernetes cluster?"

**all-MiniLM-L6-v2** (query time: 34.5ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.704 | Creating a Bring Your Own Cluster |
| 2 | 0.652 | Creating a libvirt Cluster |
| 3 | 0.632 | Upgrade a Bring Your Own Cluster |
| 4 | 0.629 | Adding the UI and Application Catalogs into a Clus... |
| 5 | 0.611 | Preface |

**all-mpnet-base-v2** (query time: 143.6ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.753 | Creating a Bring Your Own Cluster |
| 2 | 0.751 | Connecting to a Cluster |
| 3 | 0.730 | Creating a libvirt Cluster |
| 4 | 0.702 | Preface |
| 5 | 0.657 | Linux Virtualization Manager Provider |

**BAAI/bge-small-en-v1.5** (query time: 8.1ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.801 | Creating a Bring Your Own Cluster |
| 2 | 0.797 | Creating a libvirt Cluster |
| 3 | 0.789 | Connecting to a Cluster |
| 4 | 0.789 | Connecting to a Cluster |
| 5 | 0.781 | Connecting to a Cluster |

**BAAI/bge-base-en-v1.5** (query time: 48.5ms)

| Rank | Similarity | Document |
|------|------------|----------|
| 1 | 0.754 | Creating a libvirt Cluster |
| 2 | 0.733 | Connecting to a Cluster |
| 3 | 0.733 | Connecting to a Cluster |
| 4 | 0.732 | Creating a Bring Your Own Cluster |
| 5 | 0.705 | Connecting to a Cluster |

## Recommendation

**Recommended model:** `BAAI/bge-small-en-v1.5`

- **84.8% improvement** in average similarity over baseline
- Average similarity: 0.680 (vs 0.368 baseline)
- Query latency: 8.1ms
- Model load time: 58.1s
