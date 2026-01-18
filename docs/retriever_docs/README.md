# Retriever Documentation

This directory contains detailed documentation for each retrieval method used in WildFrostRAG.

## Available Retriever Guides

### Completed
- **[Full-Text Search](fulltext_search.md)** - Neo4j Lucene-based keyword retrieval

### Coming Soon
- **Vector Search** - Semantic similarity using embeddings
- **BM25** - Probabilistic keyword ranking
- **Hybrid Retrieval** - Combining multiple methods with RRF
- **Text2Cypher** - Structured query generation (WIP)
- **Graph RAG** - Community-based graph traversal (WIP)

## Quick Comparison

| Retriever | Type | Best For | Speed | Semantic Understanding |
|-----------|------|----------|-------|----------------------|
| **Vector** | Semantic | Concepts, synonyms | Fast | ✓✓✓ |
| **Full-Text** | Lexical | Exact terms, names | Very Fast | ✗ |
| **BM25** | Lexical | Keyword ranking | Slower | ✗ |
| **Hybrid** | Combined | Best of both worlds | Fast | ✓✓ |

## Testing Retrievers

Use the test script to try different retrievers:

```bash
# Test full-text search
poetry run python -m scripts.test_neo4j_retrieval --retriever fulltext "Frost Guardian"

# Test vector search
poetry run python -m scripts.test_neo4j_retrieval --retriever vector "healing cards"

# Test hybrid
poetry run python -m scripts.test_neo4j_retrieval --retriever fulltext_vector "attack damage"
```

## Evaluation

Run full evaluation on all queries:

```bash
poetry run python -m scripts.evaluate_retrievers --run-num 1 --retriever fulltext --chunking no
```

## Contributing

When adding a new retriever, create a corresponding guide following the template in `fulltext_search.md`.

Include:
1. What it is (overview)
2. How it works (technical details)
3. When to use it (use cases)
4. How to use it (examples)
5. Troubleshooting (common issues)

---

*For general documentation best practices, see [../documentation_guide.md](../documentation_guide.md)*
