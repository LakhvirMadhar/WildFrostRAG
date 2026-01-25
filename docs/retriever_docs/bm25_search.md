# BM25 Search

> **Deprecation Warning:** BM25Retriever loads all documents into memory. For production use with large datasets, prefer `Neo4jFullTextSearch` which uses Lucene's on-disk BM25-style scoring.

## Overview

BM25 (Best Matching 25) is a **probabilistic lexical retrieval** algorithm that ranks documents based on keyword relevance. It's an improvement over TF-IDF with better handling of document length and term saturation.

---

## Table of Contents

1. [What is BM25?](#what-is-bm25)
2. [How It Works](#how-it-works)
3. [Implementation in WildFrostRAG](#implementation-in-wildfrostrag)
4. [Usage Guide](#usage-guide)
5. [When to Use BM25](#when-to-use-bm25)
6. [BM25 vs Full-Text Search](#bm25-vs-full-text-search)
7. [Troubleshooting](#troubleshooting)

---

## What is BM25?

BM25 is a **bag-of-words** retrieval function that scores documents based on:
- How often query terms appear in the document (term frequency)
- How rare the terms are across all documents (inverse document frequency)
- Document length normalization

### Example

**Query:** "Frost Guardian attack"

**BM25 calculates:**
- "Frost" appears 3 times in doc A, 1 time in doc B
- "Guardian" is rare (high IDF) → weights heavily
- Doc A is longer → normalized down
- Final ranking based on combined scores

---

## How It Works

### The BM25 Formula

```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))
```

Where:
- `f(qi, D)` = term frequency of query term in document
- `|D|` = document length
- `avgdl` = average document length
- `k1` = term frequency saturation parameter (default: 1.5)
- `b` = length normalization parameter (default: 0.75)

### Key Improvements Over TF-IDF

1. **Term frequency saturation** - Diminishing returns for repeated terms
2. **Length normalization** - Fairer comparison between short and long documents
3. **Tunable parameters** - k1 and b can be adjusted for different corpora

---

## Implementation in WildFrostRAG

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  First Search Call                                          │
│  - Load ALL documents from Neo4j into memory                │
│  - Tokenize and preprocess text                             │
│  - Build BM25 index (rank_bm25 library)                     │
│  - Cache at class level                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Subsequent Searches (cached)                               │
│  - Tokenize query                                           │
│  - Score against BM25 index                                 │
│  - Return top-k results                                     │
└─────────────────────────────────────────────────────────────┘
```

### Code Structure

**Retriever Class:** `src/rag/retrievers/bm25_retriever.py`

```python
class BM25Retriever(BaseNeo4jRetriever):
    # Class-level cache
    _shared_cache = {
        'documents': None,
        'node_data': None,
        'bm25_model': None,
        'cache_key': None
    }

    def search(self, query: str, k: int = 5):
        if self.bm25_model is None:
            self._load_documents_from_neo4j()  # Loads ALL docs

        query_tokens = self._preprocess_text(query)
        scores = self.bm25_model.get_scores(query_tokens)
        # Return top-k by score
```

### Preprocessing

Uses NLTK for tokenization and stop word removal:

```python
def _preprocess_text(self, text: str) -> List[str]:
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t.isalpha() and t not in stop_words]
```

---

## Usage Guide

### Test Queries

```bash
python -m scripts.test_neo4j_retrieval --retriever bm25 "Frost Guardian"
```

### Run Evaluation

```bash
python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25 --chunking no
```

### Programmatic Usage

```python
from src.rag.retrievers import BM25Retriever
from neo4j import GraphDatabase
from src.utils.config import settings

driver = GraphDatabase.driver(
    settings.neo4j_uri.get_secret_value(),
    auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value())
)

retriever = BM25Retriever(driver)  # Warning will be shown
results = retriever.search("attack damage", k=5)
```

---

## When to Use BM25

### Use BM25 When:

1. **Research comparison** - Comparing true BM25 vs Lucene's implementation
2. **Small datasets** - Memory usage is acceptable
3. **Offline evaluation** - Not production workloads

### Don't Use BM25 When:

1. **Large datasets** - Will crash with 10k+ documents
2. **Production systems** - Use Neo4jFullTextSearch instead
3. **Memory-constrained environments**

---

## BM25 vs Full-Text Search

| Aspect | BM25Retriever | Neo4jFullTextSearch |
|--------|---------------|---------------------|
| **Algorithm** | True BM25 (Okapi) | Lucene TF-IDF variants |
| **Storage** | In-memory (Python) | On-disk (Neo4j) |
| **Scalability** | Poor (memory-bound) | Excellent |
| **Speed** | Fast after loading | Very fast |
| **First query** | Slow (loads all docs) | Fast |
| **Ranking quality** | Often better | Good |

### Why Keep Both?

For research purposes:
- True BM25 may rank differently than Lucene
- Useful for ablation studies
- Academic comparisons

---

## Troubleshooting

### Problem: Memory error / crash

**Cause:** Too many documents loaded into memory

**Solution:** Use `Neo4jFullTextSearch` instead:
```python
from src.rag.retrievers import Neo4jFullTextSearch
retriever = Neo4jFullTextSearch(driver)  # On-disk, scales well
```

### Problem: Slow first query

**Cause:** Loading all documents from Neo4j

**Note:** Subsequent queries are fast due to class-level caching.

### Problem: NLTK download errors

**Cause:** Missing NLTK data

**Solution:** The retriever auto-downloads, but you can manually:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Related Files

- **Implementation:** `src/rag/retrievers/bm25_retriever.py`
- **Alternative:** `src/rag/retrievers/neo4j_fulltext_search.py`
- **Configuration:** `src/utils/config.py`

---

*Last Updated: 2026-01-25*
