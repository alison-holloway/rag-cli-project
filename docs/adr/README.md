# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the RAG CLI project.

ADRs document significant architectural decisions made during development, including the context, decision rationale, and consequences.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-python-version-compatibility.md) | Python Version Compatibility | Accepted | 2026-02-04 |
| [0002](0002-embedding-model-upgrade.md) | Embedding Model Upgrade | Accepted | 2026-02-06 |

## ADR Format

Each ADR follows this structure:

- **Title**: Short descriptive name
- **Date**: When the decision was made
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: The issue motivating the decision
- **Decision**: What we decided to do
- **Consequences**: The resulting impact (positive and negative)
- **Future Considerations**: Related decisions that may be needed

## Creating New ADRs

1. Copy the template or an existing ADR
2. Use the next sequential number (e.g., `0002-*.md`)
3. Fill in all sections
4. Update this README's index
5. Submit for review

## References

- [ADR GitHub Organization](https://adr.github.io/)
- [Michael Nygard's ADR article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
