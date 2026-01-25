# Graph-Aware Retrievers

## Overview

Graph-aware retrievers combine traditional retrieval methods with Neo4j's graph traversal capabilities. Instead of just returning Document nodes, they enrich results with related entities (Cards, Tribes, Stats) from the knowledge graph.

**Available Graph-Aware Retrievers:**
- **VectorThenCypherRetriever**: Vector search followed by predefined Cypher traversal
- **Text2CypherVectorHybridRetriever**: LLM-generated Cypher + Vector fallback (RRF fusion)

---

## Table of Contents

1. [VectorThenCypherRetriever](#vectorthencypherretriever)
2. [Text2CypherVectorHybridRetriever](#text2cyphervectorhybridretriever)
3. [When to Use Which](#when-to-use-which)
4. [Usage Guide](#usage-guide)
5. [Traversal Patterns](#traversal-patterns)

---

## VectorThenCypherRetriever

### What It Does

Combines vector similarity search with **predefined** graph traversal patterns. The name makes the order explicit:

1. **Vector search FIRST** - Find relevant Document nodes by semantic similarity
2. **Cypher traversal SECOND** - Enrich with Card/Tribe/CardType data

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  User Query: "What tribe does Snoffel belong to?"          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Vector Search                                      │
│  - Embed query                                              │
│  - Find similar Document nodes                              │
│  - Returns: Documents mentioning Snoffel                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Graph Traversal (Cypher)                           │
│  MATCH (doc)<-[:HAS_DOCUMENT]-(card:Card)                   │
│  OPTIONAL MATCH (card)-[:BELONGS_TO_TRIBE]->(tribe:Tribe)   │
│  OPTIONAL MATCH (card)-[:HAS_CARD_TYPE]->(cardtype:CardType)│
│  RETURN doc, card, tribe, cardtype, score                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Enriched Result:                                           │
│  {                                                          │
│    "doc_text": "Snoffel is a companion...",                 │
│    "card_card_name": "Snoffel",                             │
│    "tribe_name": "Snowdwellers",                            │
│    "cardtype_name": "Companion",                            │
│    "score": 0.87                                            │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **Single Cypher query** - Vector search + traversal in one round-trip
- **Predefined patterns** - No LLM calls, deterministic behavior
- **Flexible output** - Base class auto-formats any returned properties

### Code Example

```python
from src.rag.retrievers import VectorThenCypherRetriever
from neo4j import GraphDatabase
from src.utils.config import settings

driver = GraphDatabase.driver(
    settings.neo4j_uri.get_secret_value(),
    auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value())
)

# Use default "full_card_context" pattern
retriever = VectorThenCypherRetriever(driver)

# Or specify a pattern
retriever = VectorThenCypherRetriever(
    driver,
    traversal_pattern="with_stats"  # Include stats
)

results = retriever.search("What tribe does Snoffel belong to?", k=5)
```

---

## Text2CypherVectorHybridRetriever

### What It Does

Combines **LLM-generated Cypher** (dynamic, precise) with **vector search** (semantic fallback) using Reciprocal Rank Fusion (RRF).

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  User Query: "What is Bombom's attack?"                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Text2Cypher (async) │  │  Vector Search       │
│                      │  │                      │
│  LLM generates:      │  │  Semantic similarity │
│  MATCH (c:Card)      │  │  on Document nodes   │
│  WHERE c.name =      │  │                      │
│    "Bombom"          │  │  Returns: docs about │
│  RETURN c.attack     │  │  Bombom, attacks     │
│                      │  │                      │
│  Returns: exact      │  │                      │
│  structured data     │  │                      │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           └───────────┬─────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Reciprocal Rank Fusion (RRF)                               │
│                                                             │
│  If Text2Cypher succeeded:                                  │
│    - Combine both result sets                               │
│    - RRF score = sum(1 / (k + rank))                        │
│    - Duplicates get boosted scores                          │
│                                                             │
│  If Text2Cypher failed:                                     │
│    - Fallback to vector-only results                        │
│    - search_type = "text2cypher_vector_fallback"            │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **Graceful fallback** - If Cypher generation/execution fails, vector results returned
- **Async Text2Cypher** - Non-blocking LLM calls
- **RRF fusion** - Combines precision (Cypher) with recall (Vector)
- **Tracks success** - `self.text2cypher_success` flag for analysis

### Code Example

```python
from src.rag.retrievers import Text2CypherVectorHybridRetriever
from prompts.text2cypher_prompts import TEXT2CYPHER_PROMPT_V1
from neo4j import GraphDatabase
from src.utils.config import settings
import asyncio

driver = GraphDatabase.driver(
    settings.neo4j_uri.get_secret_value(),
    auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value())
)

retriever = Text2CypherVectorHybridRetriever(
    driver,
    text2cypher_prompt=TEXT2CYPHER_PROMPT_V1
)

# Must use async
async def search():
    results = await retriever.search("What is Bombom's attack?", k=5)
    print(f"Text2Cypher succeeded: {retriever.text2cypher_success}")
    return results

results = asyncio.run(search())
```

---

## When to Use Which

| Scenario | Best Retriever | Why |
|----------|---------------|-----|
| **Structured queries** ("What is X's attack?") | Text2CypherVectorHybrid | Cypher can return exact values |
| **Relationship queries** ("What tribe is X in?") | VectorThenCypher | Traversal enriches with Tribe data |
| **Open-ended queries** ("Tell me about ice cards") | Text2CypherVectorHybrid | Vector provides semantic fallback |
| **Deterministic results needed** | VectorThenCypher | No LLM variability |
| **Maximum precision** | Text2CypherVectorHybrid | LLM generates targeted queries |
| **Simple lookups** | VectorThenCypher | Faster, no LLM latency |

---

## Usage Guide

### CLI Usage

```bash
# VectorThenCypher
python -m scripts.evaluate_retrievers \
    --run-num 1 \
    --retriever vector_then_cypher \
    --description "Vector + graph traversal"

# Text2Cypher + Vector Hybrid
python -m scripts.evaluate_retrievers \
    --run-num 1 \
    --retriever text2cypher_vector \
    --text2cypher-prompt TEXT2CYPHER_PROMPT_V1 \
    --description "Text2Cypher with vector fallback"
```

### Experiment Shortcut

```bash
python -m scripts.experiment retrieval \
    --retriever vector_then_cypher \
    --description "Testing graph-aware retrieval"
```

---

## Traversal Patterns

VectorThenCypherRetriever supports three predefined patterns:

### 1. `full_card_context` (Default)

Returns Document + Card + Tribe + CardType.

```cypher
MATCH (doc)<-[:HAS_DOCUMENT]-(card:Card)
OPTIONAL MATCH (card)-[:BELONGS_TO_TRIBE]->(tribe:Tribe)
OPTIONAL MATCH (card)-[:HAS_CARD_TYPE]->(cardtype:CardType)
RETURN doc, card, tribe, cardtype, score
ORDER BY score DESC
```

**Output fields:**
- `doc_text`, `doc_source_file` (Document properties)
- `card_card_name`, `card_attack`, `card_health` (Card properties)
- `tribe_name` (Tribe name)
- `cardtype_name` (CardType name)
- `score` (vector similarity)

**Use when:** You need comprehensive card context for RAG.

### 2. `card_only`

Lighter weight - just Document + Card.

```cypher
MATCH (doc)<-[:HAS_DOCUMENT]-(card:Card)
RETURN doc, card, score
ORDER BY score DESC
```

**Use when:** You only need card data, not relationships.

### 3. `with_stats`

Most comprehensive - includes Card stats.

```cypher
MATCH (doc)<-[:HAS_DOCUMENT]-(card:Card)
OPTIONAL MATCH (card)-[:BELONGS_TO_TRIBE]->(tribe:Tribe)
OPTIONAL MATCH (card)-[:HAS_CARD_TYPE]->(cardtype:CardType)
OPTIONAL MATCH (card)-[has_stat:HAS_STAT]->(stat:Stat)
RETURN doc, card, tribe, cardtype, collect({stat_name: stat.name, value: has_stat.value}) as stats, score
ORDER BY score DESC
```

**Output includes:**
- All fields from `full_card_context`
- `stats`: List of `{stat_name, value}` objects

**Use when:** Queries involve stats ("What's Snoffel's attack?").

### Custom Patterns

You can pass custom Cypher:

```python
custom_pattern = """
    MATCH (doc)<-[:HAS_DOCUMENT]-(card:Card)-[:HAS_KEYWORD]->(kw:Keyword)
    RETURN doc, card, collect(kw.name) as keywords, score
    ORDER BY score DESC
"""

retriever = VectorThenCypherRetriever(
    driver,
    traversal_pattern=custom_pattern
)
```

---

## How Results Are Formatted

The base class `_record_to_dict()` handles any Cypher result:

- **`node` variable** - Properties extracted WITHOUT prefix (`text`, `source_file`)
- **Other variables** - Properties prefixed with variable name (`doc_text`, `card_card_name`)
- **Scalars** - Kept as-is (`score`, `stats`)

This ensures no property collisions when multiple nodes are returned.

### Example Result

```json
{
  "doc_text": "Snoffel is a friendly companion card...",
  "doc_source_file": "cards/snoffel.html",
  "card_card_name": "Snoffel",
  "card_attack": 2,
  "card_health": 4,
  "tribe_name": "Snowdwellers",
  "cardtype_name": "Companion",
  "score": 0.87,
  "search_type": "vector_then_cypher_full_card_context",
  "rag_context": "Doc Text: Snoffel is a friendly companion card...\n..."
}
```

---

## Prerequisites

### Graph Must Be Ingested

These retrievers require the knowledge graph to exist:

```bash
python -m scripts.ingest_data --no-chunking
```

**Required relationships:**
- `(Card)-[:HAS_DOCUMENT]->(Document)`
- `(Card)-[:BELONGS_TO_TRIBE]->(Tribe)`
- `(Card)-[:HAS_CARD_TYPE]->(CardType)`
- `(Card)-[:HAS_STAT]->(Stat)` (for `with_stats` pattern)

### Verify Graph Exists

```cypher
// Check relationships exist
MATCH (c:Card)-[:HAS_DOCUMENT]->(d:Document)
RETURN count(*) as card_doc_relationships

MATCH (c:Card)-[:BELONGS_TO_TRIBE]->(t:Tribe)
RETURN count(*) as card_tribe_relationships
```

---

## Comparison with Other Retrievers

| Retriever | Graph Traversal | LLM Required | Deterministic | Fallback |
|-----------|----------------|--------------|---------------|----------|
| **Vector** | No | No | Yes | N/A |
| **Fulltext** | No | No | Yes | N/A |
| **Text2Cypher** | Yes (dynamic) | Yes | No | No |
| **VectorThenCypher** | Yes (predefined) | No | Yes | No |
| **Text2CypherVectorHybrid** | Yes (dynamic) | Yes | No | Yes (vector) |

---

## Troubleshooting

### Problem: Empty results from VectorThenCypher

**Possible causes:**
1. Graph relationships don't exist yet
2. Vector index not created

**Debug:**
```cypher
// Check if HAS_DOCUMENT relationships exist
MATCH (c:Card)-[:HAS_DOCUMENT]->(d:Document)
RETURN c.card_name, d.text LIMIT 5
```

### Problem: Text2CypherVectorHybrid always falls back

**Check:**
- `retriever.text2cypher_success` after search
- `retriever.last_individual_results['text2cypher']` for errors

**Common causes:**
- Invalid Cypher generated by LLM
- Schema mismatch in prompt
- Neo4j connection issues

---

## Related Files

- **VectorThenCypher**: `src/rag/retrievers/vector_then_cypher_retriever.py`
- **Text2CypherVectorHybrid**: `src/rag/retrievers/hybrid_retrievers.py`
- **Base class**: `src/rag/retrievers/base_neo4j_retriever.py`
- **Text2Cypher prompts**: `prompts/text2cypher_prompts.py`

---

*Last Updated: 2026-01-25*
