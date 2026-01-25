# Vector Search

## Overview

Vector search uses **semantic similarity** to find documents based on meaning rather than exact keywords. It converts text into numerical vectors (embeddings) and finds documents whose vectors are closest to the query vector.

---

## Table of Contents

1. [What is Vector Search?](#what-is-vector-search)
2. [How It Works](#how-it-works)
3. [Implementation in WildFrostRAG](#implementation-in-wildfrostrag)
4. [Usage Guide](#usage-guide)
5. [When to Use Vector Search](#when-to-use-vector-search)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## What is Vector Search?

Vector search is a **semantic retrieval** method that understands the meaning behind words, not just the literal text.

### Example

**Query:** "cards that heal allies"

**Vector search finds:**
- Documents about restoration, health recovery, support cards
- Matches concepts even if exact words differ

**It CAN:**
- Understand synonyms ("heal" ~ "restore" ~ "recover")
- Find semantically similar content
- Handle paraphrased queries

**It does NOT:**
- Match exact keywords reliably (use fulltext for that)
- Understand game-specific jargon without training data

---

## How It Works

### 1. Embedding Generation

Text is converted to dense vectors using a pre-trained model:

```
"Snoffel heals allies" → [0.12, -0.34, 0.56, ..., 0.89]  (384 dimensions)
```

### 2. Cosine Similarity

Query and document vectors are compared using cosine similarity:

```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

- Score of 1.0 = identical meaning
- Score of 0.0 = unrelated
- Higher scores = more relevant

### 3. Vector Index

Neo4j stores embeddings in a vector index for fast approximate nearest neighbor (ANN) search.

```
┌─────────────────────────────────────────────────────────────┐
│  Query: "healing cards"                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Embed Query                                                │
│  "healing cards" → [0.23, -0.45, 0.67, ...]                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Neo4j Vector Index (ANN Search)                            │
│  Find k nearest neighbors by cosine similarity              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Results:                                                   │
│  1. "Snoffel restores health..." (score: 0.89)              │
│  2. "Companion that heals..." (score: 0.85)                 │
│  3. "Support card for allies..." (score: 0.82)              │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation in WildFrostRAG

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Data Ingestion (scripts/ingest_data.py)                    │
│  - Load documents                                           │
│  - Generate embeddings with sentence-transformers           │
│  - Store in Neo4j with vector index                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Neo4j Database                                             │
│  - Document nodes with 'embedding' property                 │
│  - Vector index for fast similarity search                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Retrieval (Neo4jVectorSearch)                              │
│  - Embed query using same model                             │
│  - Query vector index                                       │
│  - Return top-k similar documents                           │
└─────────────────────────────────────────────────────────────┘
```

### Code Structure

**Retriever Class:** `src/rag/retrievers/neo4j_vector_search.py`

```python
class Neo4jVectorSearch(BaseNeo4jRetriever):
    def search(self, query: str, k: int = 5):
        # Embed the query
        model = self.get_embedding_model()
        query_embedding = model.encode(query).tolist()

        # Query Neo4j vector index
        search_query = """
        CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
        YIELD node, score
        RETURN node, score
        """
        return self._execute_query(search_query, params)
```

**Embedding Model:** Cached at class level for efficiency

```python
@classmethod
def get_embedding_model(cls):
    if cls._embedding_model is None:
        cls._embedding_model = SentenceTransformer(settings.embedding_model)
    return cls._embedding_model
```

---

## Usage Guide

### 1. Verify Index Exists

```cypher
SHOW INDEXES
WHERE type = 'VECTOR'
```

### 2. Test Queries

```bash
# Single query test
python -m scripts.test_neo4j_retrieval --retriever vector "healing cards"

# With custom k
python -m scripts.test_neo4j_retrieval --retriever vector "attack damage" --k 10
```

### 3. Run Evaluation

```bash
python -m scripts.evaluate_retrievers --run-num 1 --retriever vector --chunking no
```

### 4. Direct Cypher Query

```cypher
// First, you'd need the embedding vector - this is just for illustration
CALL db.index.vector.queryNodes('document-embeddings', 5, $query_embedding)
YIELD node, score
RETURN node.text AS text, score
ORDER BY score DESC
```

---

## When to Use Vector Search

### Use Vector Search When:

1. **Semantic understanding matters**
   - Query: "cards that protect allies" → finds "Frost Guardian shields..."

2. **Synonyms and paraphrasing**
   - Query: "cold damage" → finds "frost attack", "ice effect"

3. **Concept-based queries**
   - Query: "defensive strategy" → finds cards about blocking, health, shields

4. **User doesn't know exact terminology**
   - Query: "make my team stronger" → finds buff cards

### Don't Use Vector Search When:

1. **Exact keyword matching needed**
   - Query: "Azul Candle" → use fulltext instead

2. **Proper nouns and specific terms**
   - Game-specific jargon may not embed well

3. **Speed is critical**
   - Fulltext is faster for simple keyword lookups

---

## Configuration

Located in `src/utils/config.py`:

```python
# Embedding model
embedding_model: str = "all-MiniLM-L6-v2"  # 384 dimensions

# Vector index settings
vector_index_name: str = "document-embeddings"

# Multiple embedding configs supported
embedding_configs: dict = {
    "hf": {
        "model": "all-MiniLM-L6-v2",
        "index_name": "document-embeddings"
    },
    "openai": {
        "model": "text-embedding-3-small",
        "index_name": "document-embeddings-openai"
    }
}
```

### Switching Embedders

```bash
# Use HuggingFace (default)
python -m scripts.evaluate_retrievers --retriever vector --embedder hf

# Use OpenAI
python -m scripts.evaluate_retrievers --retriever vector --embedder openai
```

---

## Troubleshooting

### Problem: No results returned

**Possible causes:**
1. Vector index doesn't exist
2. Documents don't have embeddings
3. Query embedding failed

**Debug:**
```cypher
// Check if embeddings exist
MATCH (d:Document)
WHERE d.embedding IS NOT NULL
RETURN count(d) as docs_with_embeddings

// Check vector index
SHOW INDEXES WHERE type = 'VECTOR'
```

### Problem: Poor relevance

**Possible causes:**
1. Embedding model doesn't understand domain
2. Query too short or ambiguous

**Solutions:**
1. Try hybrid retrieval (vector + fulltext)
2. Use more specific queries
3. Consider fine-tuning embeddings (advanced)

### Problem: Slow queries

**Possible causes:**
1. Large number of documents
2. High k value

**Solutions:**
1. Reduce k
2. Use approximate search (default in Neo4j)

---

## Related Files

- **Implementation:** `src/rag/retrievers/neo4j_vector_search.py`
- **Embeddings:** `src/embeddings/`
- **Vector store:** `src/neo4j_kg/vector_store.py`
- **Configuration:** `src/utils/config.py`

---

*Last Updated: 2026-01-25*
