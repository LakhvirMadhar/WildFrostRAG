# Neo4j Lucene Full-Text Search

## Overview

Neo4j's full-text search uses **Apache Lucene** under the hood to provide fast, keyword-based retrieval. This document explains how it works, when to use it, and how it's implemented in WildFrostRAG.

---

## Table of Contents

1. [What is Full-Text Search?](#what-is-full-text-search)
2. [How Lucene Works (Under the Hood)](#how-lucene-works-under-the-hood)
3. [Implementation in WildFrostRAG](#implementation-in-wildfrostrag)
4. [Usage Guide](#usage-guide)
5. [When to Use Full-Text Search](#when-to-use-full-text-search)
6. [Performance Characteristics](#performance-characteristics)
7. [Troubleshooting](#troubleshooting)
8. [Comparison with Other Retrievers](#comparison-with-other-retrievers)

---

## What is Full-Text Search?

Full-text search is a **lexical retrieval** method that finds documents based on exact or partial keyword matches. Unlike semantic search (which understands meaning), full-text search looks for literal word occurrences.

### Example

**Query:** "Frost Guardian"

**Full-text search finds:**
- Documents containing "Frost" AND/OR "Guardian"
- Exact matches score higher than partial matches
- Word order and proximity can affect scoring

**It does NOT:**
- Understand synonyms ("cold" vs "frost")
- Understand semantics ("ice protector" as similar to "Frost Guardian")
- For that, use vector search instead!

---

## How Lucene Works (Under the Hood)

### The Inverted Index

Lucene builds an **inverted index** - a data structure that maps words to the documents containing them.

**Example:**

```
Documents:
1. "Frost Guardian is a powerful card"
2. "Guardian of the Frost protects allies"
3. "Fire Elemental deals damage"

Inverted Index:
"frost"     → [Doc 1, Doc 2]
"guardian"  → [Doc 1, Doc 2]
"powerful"  → [Doc 1]
"card"      → [Doc 1]
"protects"  → [Doc 2]
"allies"    → [Doc 2]
"fire"      → [Doc 3]
"elemental" → [Doc 3]
...
```

### Query Processing

When you search for "Frost Guardian":

1. **Tokenization**: Query split into ["frost", "guardian"]
2. **Lookup**: Find documents containing each word
   - "frost" → [1, 2]
   - "guardian" → [1, 2]
3. **Scoring**: Calculate relevance for each document
   - TF-IDF scoring (Term Frequency × Inverse Document Frequency)
   - Document 1: High score (both words present, close together)
   - Document 2: Medium score (both words present, farther apart)
   - Document 3: No score (neither word present)
4. **Ranking**: Return results sorted by score

### TF-IDF Scoring

**Term Frequency (TF):**
- How often does the word appear in this document?
- More occurrences = higher score

**Inverse Document Frequency (IDF):**
- How rare is this word across all documents?
- Rare words (e.g., "Azul") → high weight
- Common words (e.g., "the", "a") → low weight

**Formula (simplified):**
```
Score = TF × IDF
```

**Example:**
- "Azul Candle" - "Azul" is rare → high IDF → high score for relevant docs
- "the card" - "the" is common → low IDF → low impact on score

---

## Implementation in WildFrostRAG

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Data Ingestion                       │
│  (scripts/ingest_data.py - Stage 4)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Create Fulltext Index                           │
│  src/neo4j_kg/vector_store.py::create_fulltext_index()  │
│                                                          │
│  CREATE FULLTEXT INDEX `document-fulltext`              │
│  FOR (n:Document) ON EACH [n.text]                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Neo4j Database                             │
│  ┌──────────────────────────────────────┐              │
│  │  Lucene Inverted Index               │              │
│  │  - Tokenized text from all Documents │              │
│  │  - Pre-computed scores               │              │
│  │  - Fast keyword lookup               │              │
│  └──────────────────────────────────────┘              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Retrieval (Query Time)                        │
│  src/rag/retrievers/neo4j_fulltext_search.py            │
│                                                          │
│  CALL db.index.fulltext.queryNodes(                     │
│      'document-fulltext',                               │
│      'Frost Guardian'                                   │
│  )                                                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
              Ranked Results
              (with Lucene scores)
```

### Code Structure

**Index Creation:** `src/neo4j_kg/vector_store.py`
```python
def create_fulltext_index(
    index_name: str,
    node_label: str = "Document",
    text_property: str = "text"
) -> None:
    """Creates Lucene-based fulltext index in Neo4j."""
    # Uses Neo4j 5.x syntax
    create_query = f"""
    CREATE FULLTEXT INDEX `{index_name}` IF NOT EXISTS
    FOR (n:{node_label}) ON EACH [n.{text_property}]
    """
```

**Retriever Class:** `src/rag/retrievers/neo4j_fulltext_search.py`
```python
class Neo4jFullTextSearch(BaseNeo4jRetriever):
    def search(self, query: str, k: int = 5):
        """Query the Lucene fulltext index."""
        search_query = """
        CALL db.index.fulltext.queryNodes($index_name, $query)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT $k
        """
```

### Configuration

Located in `src/utils/config.py`:
```python
fulltext_index_name: str = "document-fulltext"  # Index name in Neo4j
```

---

## Usage Guide

### 1. Create the Index (One-Time Setup)

Run the data ingestion pipeline:
```bash
poetry run python -m scripts.ingest_data --no-chunking
```

This automatically creates the fulltext index in Stage 4.

### 2. Verify Index Exists

In Neo4j Browser:
```cypher
SHOW FULLTEXT INDEXES
```

Expected output:
```
name: "document-fulltext"
type: "FULLTEXT"
state: "ONLINE"
entityType: "NODE"
labelsOrTypes: ["Document"]
properties: ["text"]
```

### 3. Test Queries (Interactive)

**Single query test:**
```bash
poetry run python -m scripts.test_neo4j_retrieval \
    --retriever fulltext \
    "Frost Guardian"
```

**With custom k:**
```bash
poetry run python -m scripts.test_neo4j_retrieval \
    --retriever fulltext \
    "attack damage" \
    --k 10
```

### 4. Run Evaluation (All Queries)

Evaluate against your query dataset:
```bash
poetry run python -m scripts.evaluate_retrievers \
    --run-num 1 \
    --retriever fulltext \
    --chunking no
```

### 5. Direct Cypher Query (Neo4j Browser)

```cypher
CALL db.index.fulltext.queryNodes('document-fulltext', 'Frost Guardian')
YIELD node, score
RETURN node.text AS text, node.source_file AS source, score
ORDER BY score DESC
LIMIT 5
```

---

## When to Use Full-Text Search

### ✅ Use Full-Text Search When:

1. **Exact keyword matching is important**
   - Query: "Azul Candle" → Want exact card name matches
   - Proper nouns, specific terms

2. **User knows specific terminology**
   - Query: "Overburn" → Want cards mentioning this game mechanic
   - Technical jargon, game-specific terms

3. **Fast retrieval is critical**
   - Fulltext search is very fast (pre-indexed)
   - Good for real-time applications

4. **Hybrid retrieval**
   - Combine with vector search for best of both worlds
   - Fulltext finds keyword matches, vector finds semantic matches

### ❌ Don't Use Full-Text Search When:

1. **Synonyms matter**
   - Query: "cold cards" won't match "Frost Guardian"
   - Use vector search instead

2. **Semantic understanding needed**
   - Query: "cards that heal allies" requires understanding
   - Fulltext only matches literal words

3. **Query has typos**
   - Query: "Forst Guardian" won't match "Frost Guardian"
   - BM25 might be more forgiving

---

## Performance Characteristics

### Speed
- **Index Creation**: One-time, happens during ingestion (~5 seconds)
- **Query Time**: Very fast (milliseconds for most queries)
- **Memory**: Index stored on disk by Neo4j, minimal RAM usage

### Scalability
- Scales well to millions of documents
- Index size proportional to unique words (not document count)

### Trade-offs
| Aspect | Full-Text Search | Vector Search | BM25 |
|--------|------------------|---------------|------|
| **Speed** | Very Fast | Fast | Slower |
| **Semantic Understanding** | None | Excellent | None |
| **Exact Matches** | Excellent | Poor | Good |
| **Memory Usage** | Low (disk-based) | High (in-memory) | High (in-memory) |
| **Setup** | One-time index | One-time index | Loads on each use* |

*Note: BM25 now uses class-level caching, so only loads once per session

---

## Troubleshooting

### Problem: "Index does not exist" error

**Error message:**
```
There is no procedure with the name `db.index.fulltext.queryNodes`
or index 'document-fulltext' does not exist
```

**Solution:**
1. Check if index exists:
   ```cypher
   SHOW INDEXES
   ```
2. If missing, re-run ingestion:
   ```bash
   poetry run python -m scripts.ingest_data --no-chunking --skip-scrape --skip-graph
   ```

### Problem: No results returned

**Possible causes:**
1. **Index not populated**: Wait a few seconds after creation
2. **Query too specific**: Try broader terms
3. **Typos**: Fulltext is exact-match, check spelling

**Debug query:**
```cypher
// Check if documents exist
MATCH (d:Document)
RETURN count(d) as doc_count

// Check if text property exists
MATCH (d:Document)
WHERE d.text IS NOT NULL
RETURN count(d) as docs_with_text
```

### Problem: Wrong results (relevance issues)

**Lucene scoring quirks:**
- Common words get low scores (TF-IDF)
- Short documents may score higher than long ones
- Word proximity affects score

**Solutions:**
1. **Use hybrid retrieval**: Combine with vector search
   ```bash
   poetry run python -m scripts.test_neo4j_retrieval \
       --retriever fulltext_vector \
       "your query"
   ```
2. **Tune query**: Add more specific keywords
3. **Try BM25**: May have better ranking for your use case

### Problem: "Procedure not found" during index creation

**Error:**
```
Neo.ClientError.Procedure.ProcedureNotFound:
There is no procedure with the name `db.index.fulltext.createNodeIndex`
```

**Cause:** Using Neo4j 4.x syntax on Neo4j 5.x

**Solution:** Already fixed in `create_fulltext_index()` - uses Neo4j 5.x syntax:
```cypher
CREATE FULLTEXT INDEX `document-fulltext` IF NOT EXISTS
FOR (n:Document) ON EACH [n.text]
```

---

## Comparison with Other Retrievers

### Full-Text vs Vector Search

| Aspect | Full-Text (Lucene) | Vector (Embeddings) |
|--------|-------------------|---------------------|
| **Method** | Keyword matching | Semantic similarity |
| **Query** | "Frost Guardian" | "Frost Guardian" |
| **Finds** | Documents with exact words | Semantically similar docs |
| **Example Match** | "Frost Guardian protects..." ✓ | "Ice protector shields..." ✓ |
| **Non-match** | "Ice protector shields..." ✗ | Still matches! |
| **Best For** | Exact terms, proper nouns | Concepts, synonyms |

### Full-Text vs BM25

| Aspect | Full-Text (Neo4j Lucene) | BM25 (rank-bm25) |
|--------|--------------------------|------------------|
| **Algorithm** | TF-IDF variants | BM25 (Okapi) |
| **Where runs** | Inside Neo4j | Python (in-memory) |
| **Performance** | Faster | Slower |
| **Ranking quality** | Good | Often better |
| **State-of-art** | Older | Newer standard |

**Note:** In research, you're comparing both to see which performs better for Wildfrost domain!

### Hybrid Retrievers

Combine strengths of multiple methods using **Reciprocal Rank Fusion (RRF)**:

**Available hybrid retrievers:**
1. `fulltext_vector` - Fulltext + Vector
2. `bm25_vector` - BM25 + Vector
3. `bm25_fulltext_vector` - All three!

**When to use hybrid:**
- Want both exact keyword matching AND semantic understanding
- Maximize recall (find all relevant docs)
- Research shows hybrids often outperform individual methods

**Example:**
```bash
# Query: "healing cards"
# - Fulltext finds: documents with "healing" keyword
# - Vector finds: documents about restoration, health recovery
# - RRF combines and ranks all results
poetry run python -m scripts.test_neo4j_retrieval \
    --retriever fulltext_vector \
    "healing cards"
```

---

## Advanced Topics

### Lucene Query Syntax

Neo4j fulltext supports Lucene's query syntax:

**Boolean operators:**
```cypher
// AND (both required)
CALL db.index.fulltext.queryNodes('document-fulltext', 'Frost AND Guardian')

// OR (either word)
CALL db.index.fulltext.queryNodes('document-fulltext', 'Frost OR Ice')

// NOT (exclude)
CALL db.index.fulltext.queryNodes('document-fulltext', 'Frost NOT Guardian')
```

**Wildcards:**
```cypher
// Prefix search
CALL db.index.fulltext.queryNodes('document-fulltext', 'Frost*')

// Matches: Frost, Frostbite, Frostguard, etc.
```

**Phrase search:**
```cypher
// Exact phrase (words in order)
CALL db.index.fulltext.queryNodes('document-fulltext', '"Frost Guardian"')
```

### Index Configuration (Future)

Neo4j allows configuring analyzers (tokenization rules):
- **Standard analyzer**: Default, splits on whitespace and punctuation
- **Custom analyzers**: Can add stemming, stop words, etc.

For now, we use the default. Future enhancement: experiment with analyzers for better retrieval.

---

## References

### Neo4j Documentation
- [Fulltext indexes (Neo4j 5.x)](https://neo4j.com/docs/cypher-manual/current/indexes-for-full-text-search/)
- [Querying fulltext indexes](https://neo4j.com/docs/cypher-manual/current/indexes-for-full-text-search/#query-fulltext-index)

### Lucene
- [Apache Lucene](https://lucene.apache.org/)
- [TF-IDF scoring](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

### Related Files
- Implementation: `src/rag/retrievers/neo4j_fulltext_search.py`
- Index creation: `src/neo4j_kg/vector_store.py`
- Configuration: `src/utils/config.py`
- Test script: `scripts/test_neo4j_retrieval.py`

---

## Next Steps

1. **Try it yourself**: Run test queries and see the results
2. **Compare retrievers**: Test same query with different methods
3. **Read hybrid docs**: Learn how RRF combines multiple retrievers (coming soon!)
4. **Explore evaluation**: See how fulltext performs on your query dataset

---

*Last Updated: 2026-01-18*
*WildFrostRAG v0.1.0*
