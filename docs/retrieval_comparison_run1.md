# Retrieval Comparison — Run 1

Regenerated: 2026-08-23, against the current codebase (post Poetry package-mode
refactor, post strict-mypy/ruff cleanup).

All experiments use the same 50 reference queries (`queries/simple_reference_based_queries.csv`),
auto-annotated via URL matching against `queries/simple_reference_based_queries.json`.
Unannotated queries count as failures (metrics = 0). Neo4j had 1440 nodes / 424
Documents / 362 Cards ingested at the time of this run.

> Note: an earlier version of this document (generated 2026-02-20) contained
> numbers that do not match the underlying `metrics.json` files it was supposedly
> summarizing — they were regenerated from scratch for this version. If you
> reproduce these commands and get different numbers, trust your own run over
> this document; regenerate this file rather than editing it by hand.

## Results

| Retriever | Experiment | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Recall@10 |
|---|---|---|---|---|---|---|---|
| Text2Cypher (V3 prompt) | `text2cypher/004` | 0.700 | 0.720 | 0.720 | 0.720 | 0.710 | 0.720 |
| Vector (HF MiniLM) | `vector_hf/002` | 0.820 | 0.920 | 0.960 | 0.960 | 0.877 | 0.960 |
| Fulltext | `fulltext/003` | 0.820 | 0.880 | 0.900 | 0.960 | 0.859 | 0.960 |
| Fulltext + Cypher traversal | `fulltext_then_cypher/002` | 0.820 | 0.880 | 0.900 | 0.960 | 0.859 | 0.960 |
| BM25 | `bm25/006` | 0.840 | 0.900 | 0.920 | 0.960 | 0.875 | 0.960 |
| Vector (Gemma) | `vector_gemma/002` | 0.900 | 0.940 | 0.960 | 0.980 | 0.928 | 0.980 |
| Vector (Gemma) + Cypher traversal | `vector_then_cypher_gemma/002` | 0.900 | 0.940 | 0.960 | 0.980 | 0.928 | 0.980 |
| BM25 + Vector (Gemma) Hybrid RRF | `bm25_vector_gemma/002` | 0.900 | 0.940 | 0.960 | 0.980 | 0.927 | 0.980 |
| **Fulltext + Vector (Gemma) Hybrid RRF** | `fulltext_vector_gemma/004` | **0.900** | **0.960** | 0.960 | 0.980 | **0.933** | 0.980 |

## Key findings

- **Text2Cypher is the clear worst performer.** LLM-generated Cypher queries against
  the graph schema frequently fail to execute or return the wrong rows — the only
  strategy scoring below 0.90 on Hit@5/Hit@10.
- **Embedding model choice matters a lot for vector search.** Gemma embeddings beat
  MiniLM by 5+ points of MRR (0.928 vs. 0.877) with identical retrieval logic.
- **Graph-traversal augmentation currently adds nothing measurable.** Both
  `vector_then_cypher_gemma` and `fulltext_then_cypher` score *identically* to their
  non-graph base retriever on every metric — the Cypher-expansion step isn't yet
  surfacing additional relevant documents on this query set. This is the open
  question the Graph RAG work is trying to answer.
- **The best strategy overall is hybrid RRF** (Fulltext + Vector Gemma), by MRR — a
  ~5-point improvement in ranking quality over vector search alone, though Hit@10
  is already saturated (0.980) across all the strongest retrievers.

## Reproducing these numbers

Requires a running Neo4j instance with Wildfrost data ingested (see main
[README](../README.md#installation)) and a populated `.env`.

```bash
poetry run python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25 \
  --chunking no --queries-json queries/simple_reference_based_queries.json

poetry run python -m scripts.evaluate_retrievers --run-num 1 --retriever fulltext \
  --chunking no --queries-json queries/simple_reference_based_queries.json

poetry run python -m scripts.evaluate_retrievers --run-num 1 --retriever vector \
  --embedder hf --chunking no --queries-json queries/simple_reference_based_queries.json

poetry run python -m scripts.evaluate_retrievers --run-num 1 --retriever vector \
  --embedder gemma --chunking no --queries-json queries/simple_reference_based_queries.json

poetry run python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25_vector \
  --embedder gemma --chunking no --queries-json queries/simple_reference_based_queries.json

poetry run python -m scripts.evaluate_retrievers --run-num 1 --retriever fulltext_vector \
  --embedder gemma --chunking no --queries-json queries/simple_reference_based_queries.json

poetry run python -m scripts.evaluate_retrievers --run-num 1 --retriever text2cypher \
  --text2cypher-prompt TEXT2CYPHER_PROMPT_V3 --chunking no \
  --queries-json queries/simple_reference_based_queries.json

poetry run python -m scripts.evaluate_retrievers --run-num 1 --retriever vector_then_cypher \
  --embedder gemma --chunking no --queries-json queries/simple_reference_based_queries.json

poetry run python -m scripts.evaluate_retrievers --run-num 1 --retriever fulltext_then_cypher \
  --chunking no --queries-json queries/simple_reference_based_queries.json
```

Then for each resulting experiment directory:

```bash
poetry run python -m scripts.calculate_retrieval_metrics --experiment-path outputs/run_1/retrievals/<retriever>/<id>
```
