# Hybrid Retrieval (RRF)

## Overview

Hybrid retrieval combines multiple retrieval methods using **Reciprocal Rank Fusion (RRF)** to get the best of both lexical and semantic search. It's often the best choice when you want both keyword precision and semantic understanding.

---

## Table of Contents

1. [What is Hybrid Retrieval?](#what-is-hybrid-retrieval)
2. [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf)
3. [Available Hybrid Retrievers](#available-hybrid-retrievers)
4. [Usage Guide](#usage-guide)
5. [When to Use Hybrid Retrieval](#when-to-use-hybrid-retrieval)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## What is Hybrid Retrieval?

Hybrid retrieval runs **multiple retrieval methods** on the same query and **fuses their results** into a single ranked list.

### Example

**Query:** "Frost Guardian healing"

```
┌─────────────────────────────────────────────────────────────┐
│  Query: "Frost Guardian healing"                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Fulltext Search     │  │  Vector Search       │
│                      │  │                      │
│  Finds exact matches │  │  Finds semantic      │
│  for "Frost Guardian"│  │  matches for healing │
│                      │  │                      │
│  1. Frost Guardian   │  │  1. Restoration card │
│  2. Guardian of Frost│  │  2. Frost Guardian   │
│  3. Frostbite card   │  │  3. Health recovery  │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           └───────────┬─────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Reciprocal Rank Fusion                                     │
│                                                             │
│  "Frost Guardian" appears in BOTH lists → boosted!          │
│                                                             │
│  Final ranking:                                             │
│  1. Frost Guardian (score: 0.032) ← in both lists           │
│  2. Restoration card (score: 0.016)                         │
│  3. Guardian of Frost (score: 0.015)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Reciprocal Rank Fusion (RRF)

### The Formula

For each document across all retriever results:

```
RRF_score = Σ (weight / (k + rank))
```

Where:
- `k` = smoothing constant (default: 60)
- `rank` = position in the retriever's result list (1-indexed)
- `weight` = retriever weight (default: 1.0 for all)

### Why RRF Works

1. **Rank-based** - Doesn't require normalized scores across methods
2. **Handles duplicates** - Documents in multiple lists get boosted
3. **Simple** - No training required, just fusion

### Example Calculation

Document "Frost Guardian":
- Fulltext rank: 1 → score = 1/(60+1) = 0.0164
- Vector rank: 2 → score = 1/(60+2) = 0.0161
- **RRF score = 0.0164 + 0.0161 = 0.0325**

Document "Restoration card":
- Fulltext rank: not present → score = 0
- Vector rank: 1 → score = 1/(60+1) = 0.0164
- **RRF score = 0.0164**

Result: "Frost Guardian" ranks higher despite being #2 in vector search.

---

## Available Hybrid Retrievers

### 1. FulltextVectorHybridRetriever

Combines **Fulltext + Vector** search.

```python
from src.rag.retrievers import FulltextVectorHybridRetriever

retriever = FulltextVectorHybridRetriever(driver)
results = retriever.search("Frost Guardian healing", k=5)
```

**Best for:** General-purpose queries needing both keyword and semantic matching.

### 2. BM25VectorHybridRetriever

Combines **BM25 + Vector** search.

```python
from src.rag.retrievers import BM25VectorHybridRetriever

retriever = BM25VectorHybridRetriever(driver)  # BM25 deprecation warning
results = retriever.search("attack damage", k=5)
```

**Best for:** Research comparing true BM25 vs Lucene.

### 3. BM25FulltextVectorHybridRetriever

Combines **all three**: BM25 + Fulltext + Vector.

```python
from src.rag.retrievers import BM25FulltextVectorHybridRetriever

retriever = BM25FulltextVectorHybridRetriever(driver)
results = retriever.search("healing cards", k=5)
```

**Best for:** Maximum recall, research experiments.

### 4. Text2CypherVectorHybridRetriever

Combines **Text2Cypher + Vector** with fallback.

See [Graph-Aware Retrievers](graph_aware_retrievers.md) for details.

---

## Usage Guide

### CLI Usage

```bash
# Fulltext + Vector hybrid
python -m scripts.evaluate_retrievers --run-num 1 --retriever fulltext_vector --chunking no

# BM25 + Vector hybrid
python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25_vector --chunking no

# All three combined
python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25_fulltext_vector --chunking no
```

### Test Queries

```bash
python -m scripts.test_neo4j_retrieval --retriever fulltext_vector "Frost Guardian"
```

### Programmatic Usage

```python
from src.rag.retrievers import FulltextVectorHybridRetriever
from neo4j import GraphDatabase
from src.utils.config import settings

driver = GraphDatabase.driver(
    settings.neo4j_uri.get_secret_value(),
    auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value())
)

retriever = FulltextVectorHybridRetriever(driver)
results = retriever.search("healing cards", k=5)

# Access individual retriever results
print(retriever.last_individual_results)  # {'fulltext': [...], 'vector': [...]}
```

---

## When to Use Hybrid Retrieval

### Use Hybrid When:

1. **Queries mix keywords and concepts**
   - "Frost Guardian healing" → keyword match + semantic

2. **Maximize recall**
   - Don't want to miss relevant documents

3. **Uncertain query type**
   - Don't know if user will use exact terms or descriptions

4. **Research/evaluation**
   - Compare individual vs hybrid performance

### Hybrid Often Outperforms Individual Methods

Research consistently shows hybrid retrieval beats single methods:
- Catches keyword matches fulltext would find
- Catches semantic matches vector would find
- Documents in both lists get boosted (likely more relevant)

---

## Configuration

### RRF Parameters

Located in `src/utils/config.py`:

```python
rrf_k1: int = 60  # Smoothing constant for RRF
```

### Weights

Currently all retrievers use equal weights (1.0). To customize:

```python
from src.rag.retrievers.hybrid_retrievers import HybridRetriever

# Custom hybrid with different weights
retriever = HybridRetriever(
    retrievers=[fulltext_retriever, vector_retriever],
    retriever_names=['fulltext', 'vector'],
    weights=[1.5, 1.0],  # Weight fulltext higher
    k1=60
)
```

---

## Result Structure

Hybrid results include extra metadata:

```json
{
  "text": "Frost Guardian protects allies...",
  "source_file": "cards/frost_guardian.html",
  "score": 0.0325,
  "rrf_score": 0.0325,
  "search_type": "hybrid_rrf",
  "source_retriever": "fulltext",
  "retriever_scores": {
    "fulltext": 12.5,
    "vector": 0.87
  },
  "rag_context": "..."
}
```

- `score` / `rrf_score`: The fused RRF score
- `source_retriever`: Which retriever found this result first
- `retriever_scores`: Original scores from each retriever (for analysis)

---

## Troubleshooting

### Problem: Results seem biased toward one retriever

**Cause:** One retriever returns many more results

**Note:** RRF is rank-based, not score-based, so this is usually okay. If needed, adjust weights.

### Problem: Duplicate detection failing

**Cause:** Same document has slightly different text representations

**Current approach:** Uses first 50 chars of text + source_file as identifier.

### Problem: Slow performance

**Cause:** Running multiple retrievers sequentially

**Note:** Hybrid retrievers run each sub-retriever with `k*2` to allow for fusion, then return top `k`. This is by design.

---

## Related Files

- **Implementation:** `src/rag/retrievers/hybrid_retrievers.py`
- **Base class:** `src/rag/retrievers/hybrid_retrievers.py::HybridRetriever`
- **Configuration:** `src/utils/config.py`

---

*Last Updated: 2026-01-25*
