# Retriever Documentation

This directory contains detailed documentation for each retrieval method used in WildFrostRAG.

## Available Retriever Guides

- **[Vector Search](vector_search.md)** - Semantic similarity using embeddings
- **[Full-Text Search](fulltext_search.md)** - Neo4j Lucene-based keyword retrieval
- **[BM25 Search](bm25_search.md)** - Probabilistic keyword ranking (deprecated)
- **[Hybrid Retrieval](hybrid_retrieval.md)** - Combining multiple methods with RRF
- **[Graph-Aware Retrievers](graph_aware_retrievers.md)** - VectorThenCypher + Text2CypherVectorHybrid

## Quick Comparison

| Retriever | Type | Best For | Speed | Graph Enrichment |
|-----------|------|----------|-------|------------------|
| **Vector** | Semantic | Concepts, synonyms | Fast | No |
| **Full-Text** | Lexical | Exact terms, names | Very Fast | No |
| **BM25** | Lexical | Keyword ranking | Slower* | No |
| **Hybrid (RRF)** | Combined | Best of both worlds | Fast | No |
| **Text2Cypher** | LLM-generated | Structured queries | Medium | Yes |
| **VectorThenCypher** | Graph-aware | Enriched card context | Fast | Yes |
| **Text2CypherVectorHybrid** | Graph-aware | Precise + fallback | Medium | Yes |

*BM25 loads all documents into memory - deprecated for large datasets

## Testing Retrievers

Use the test script to try different retrievers:

```bash
# Test full-text search
python -m scripts.test_neo4j_retrieval --retriever fulltext "Frost Guardian"

# Test vector search
python -m scripts.test_neo4j_retrieval --retriever vector "healing cards"

# Test hybrid
python -m scripts.test_neo4j_retrieval --retriever fulltext_vector "attack damage"

# Test graph-aware (vector + graph traversal)
python -m scripts.test_neo4j_retrieval --retriever vector_then_cypher "What tribe is Snoffel in?"
```

## Evaluation

Run full evaluation on all queries:

```bash
# Basic retrievers
python -m scripts.evaluate_retrievers --run-num 1 --retriever fulltext --chunking no
python -m scripts.evaluate_retrievers --run-num 1 --retriever vector --chunking no

# Graph-aware retrievers
python -m scripts.evaluate_retrievers --run-num 1 --retriever vector_then_cypher --chunking no
python -m scripts.evaluate_retrievers --run-num 1 --retriever text2cypher_vector --text2cypher-prompt TEXT2CYPHER_PROMPT_V1 --chunking no
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
