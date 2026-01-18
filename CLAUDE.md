# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WildFrostRAG is a research-oriented Retrieval-Augmented Generation (RAG) system for the game Wildfrost. The project benchmarks different retrieval strategies (BM25, vector search, hybrid methods, Text2Cypher, and Graph RAG) using game data scraped from the Wildfrost Wiki and stored in Neo4j.

This is an ablation study comparing retrieval and generation metrics across different RAG implementations, using an evaluation framework inspired by Hamel Husain's methodology.

## Development Commands

### Environment Setup
```bash
# Install dependencies (Python 3.12 required)
poetry install

# Activate virtual environment
poetry shell
```

### Neo4j Setup
- Requires Neo4j instance running at `bolt://localhost:7687` (default)
- Configure credentials in `.env` file (see `.env.example`)
- **Neo4j Version**: Neo4j 5.x server (confirmed compatible)
  - Neo4j Python driver: 6.0.3 (see `pyproject.toml` or `.venv/Lib/site-packages/neo4j/`)
  - Driver 6.x is compatible with Neo4j 5.x databases
  - **Important**: Uses Neo4j 5.x Cypher syntax for fulltext indexes:
    - Correct: `CREATE FULLTEXT INDEX ... FOR (n:Label) ON EACH [n.property]`
    - Old (4.x): `CALL db.index.fulltext.createNodeIndex(...)` ❌

### Data Ingestion Pipeline
```bash
# Full ETL pipeline: scrape → process → embed → ingest
# Default: uses --no-chunking (chunking still needs testing)
python -m scripts.ingest_data --no-chunking

# Skip web scraping (use cached data)
python -m scripts.ingest_data --no-chunking --skip-scrape

# Skip graph creation
python -m scripts.ingest_data --no-chunking --skip-graph

# Skip vector ingestion
python -m scripts.ingest_data --no-chunking --skip-vectors

# Clear database before running
python -m scripts.ingest_data --no-chunking --clear-db

# To test chunking (experimental):
python -m scripts.ingest_data
```

### Retrieval Evaluation
```bash
# Default: use --chunking no (chunking still needs testing)
# Run vector search
python -m scripts.evaluate_retrievers --run-num 1 --retriever vector --chunking no

# Run BM25
python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25 --chunking no

# Run fulltext search
python -m scripts.evaluate_retrievers --run-num 1 --retriever fulltext --chunking no

# Run hybrid retrievers
python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25_vector --chunking no
python -m scripts.evaluate_retrievers --run-num 1 --retriever fulltext_vector --chunking no
python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25_fulltext_vector --chunking no

# Run Text2Cypher (WIP)
python -m scripts.evaluate_retrievers --run-num 1 --retriever text2cypher --chunking no

# To test with chunking (experimental):
python -m scripts.evaluate_retrievers --run-num 1 --retriever vector --chunking yes
```

### LLM Generation
```bash
# Run LLM generation pipeline with retrieved context
python -m scripts.run_llm_generation --run-num 1
```

### Retrieval Metrics Calculation
```bash
# Calculate retrieval metrics (NDCG, Hit@k, MRR)
python -m scripts.calculate_retrieval_metrics --run-num 1
```

### Interactive Notebooks
- `annotation_gui.ipynb`: Manual GUI for query annotation and validation
- `test_scraping_and_ingestion.ipynb`: Testing scraping and Neo4j ingestion workflows

### Documentation
- `docs/retriever_docs/` - Detailed guides for each retrieval method
  - `fulltext_search.md` - How Neo4j Lucene full-text search works
  - More coming: vector, BM25, hybrid retrievers
- `docs/documentation_guide.md` - Best practices for documentation (meta-guide)

## Architecture Overview

### Core Data Flow

1. **Web Scraping** (`src/web_scraper/`) → Scrapes Wildfrost Wiki for card data
2. **Data Processing** (`src/data_processing/`) → Parses HTML, extracts structured card data, enriches with tribe information
3. **Knowledge Graph** (`src/neo4j_kg/`) → Populates Neo4j with nodes (Card, Tribe, CardType, Stat, Document) and relationships
4. **Embeddings** (`src/embeddings/`) → Generates vectors using sentence-transformers (default: all-MiniLM-L6-v2)
5. **Vector Store** (`src/neo4j_kg/vector_store.py`) → Stores embeddings in Neo4j vector index
6. **Retrieval** (`src/rag/retrievers/`) → Multiple strategies query the knowledge graph
7. **Generation** (`src/rag/augmented_generation/`) → LLM generates responses using retrieved context
8. **Evaluation** (`src/rag/evaluation/`) → Measures retrieval quality (NDCG, Hit@k, MRR) and generation quality

### Configuration Management

All configuration is centralized in `src/utils/config.py` using Pydantic Settings:
- Neo4j connection (URI, username, password)
- Embedding model configuration (model name, dimensions, index names)
- Retrieval parameters (RRF k1, index names)
- Web scraping settings (URLs, concurrency limits)
- Directory paths (data, outputs, schemas)
- OpenAI API settings (model, temperature, seed)

**Import pattern**: `from src.utils.config import settings`

The settings object loads from `.env` file and provides type-safe access to configuration.

### Neo4j Schema

**Nodes**:
- `Card`: Game cards with properties (card_name, attack, health, counter, etc.)
- `Tribe`: Three exclusive tribes (Snowdwellers, Shademancers, Clunkmasters)
- `CardType`: Card categories (Friendly, Item, Enemy, Boss, Miniboss, etc.)
- `Stat`: Card statistics
- `Document`: Text chunks for retrieval (with `text`, `source_file`, `embedding` properties)

**Relationships**:
- `(Card)-[:HAS_CARD_TYPE]->(CardType)`
- `(Card)-[:BELONGS_TO_TRIBE]->(Tribe)`
- `(Card)-[:HAS_STAT]->(Stat)`
- `(Document)-[:REFERENCES_CARD]->(Card)` - links chunks to their source cards

### Retrieval Strategies

All retrievers follow dependency injection pattern - Neo4j driver is passed in, not created internally. Base class: `BaseNeo4jRetriever`.

**Available retrievers** (in `src/rag/retrievers/`):
1. **Neo4jVectorSearch**: Cosine similarity on embeddings
2. **Neo4jFullTextSearch**: Neo4j's Lucene-based full-text search
3. **BM25Retriever**: Keyword-based retrieval using rank-bm25 library (with class-level caching)
4. **HybridRetriever**: Reciprocal Rank Fusion (RRF) combining multiple retrievers
   - `BM25VectorHybridRetriever`: Combines BM25 + Vector
   - `FulltextVectorHybridRetriever`: Combines Fulltext + Vector
   - `BM25FulltextVectorHybridRetriever`: Combines all three
5. **Text2CypherRetriever**: (WIP) Uses Neo4j's text2cypher extension to generate Cypher queries
6. **GraphRAGRetriever**: (WIP) Advanced graph traversal/community-based retrieval

**Reciprocal Rank Fusion (RRF)** formula:
- For each document, sum across all retrievers: `weight / (k1 + rank)`
- Default k1=60 (configurable via `settings.rrf_k1`)
- Results sorted by combined RRF score

### Dependency Injection Pattern

Neo4j driver management follows dependency injection:
- Driver is created at application level (in scripts)
- Passed to retrievers via constructor
- Retrievers never create their own driver connection
- This enables proper connection pooling and lifecycle management

Example from `scripts/evaluate_retrievers.py`:
```python
from neo4j import GraphDatabase
driver = GraphDatabase.driver(uri, auth=(username, password))
retriever = Neo4jVectorSearch(driver=driver)
```

### LLM Generation

`src/rag/augmented_generation/call_llm_generation.py` provides:
- **Zero-shot generation**: LLM with no retrieval context
- **RAG generation**: LLM with retrieved context injected into prompt
- Async batch processing for efficiency
- Uses OpenAI API (configurable model via `settings.openai_model_name`)
- Consistent system prompt across all methods for fair comparison

### Evaluation Framework

**Retrieval Metrics** (`src/rag/evaluation/retrieval_metrics.py`):
- **Hit@k**: Precision/recall at different k values (1, 2, 5, 10)
- **NDCG**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

**Generation Metrics**:
- User-defined criteria for answer quality
- Qualitative coding: "Open Coding" and "Axial Coding" to categorize failure modes
- Taxonomy stored in `queries/simple_reference_based_failure_mode_taxonomy.md`

**Query Dataset**: `queries/simple_reference_based_queries.csv`

### Directory Structure

```
data/
├── structured_outputs/    # Processed card data by CardType
├── schemas/              # JSON definitions of card types
└── raw_htmls/           # Scraped HTML files

outputs/
├── retrievers/          # Retrieval results by run number
└── generation/         # LLM generation outputs

src/
├── data_processing/    # HTML parsing, card extraction, enrichment
├── embeddings/        # Embedding generation
├── neo4j_kg/         # Neo4j utilities, vector store
├── rag/
│   ├── retrievers/             # All retrieval strategies
│   ├── augmented_generation/   # LLM generation
│   └── evaluation/            # Metrics, sampling, taxonomy
├── utils/            # Config, logging, GUI utilities
└── web_scraper/     # Wiki scraping logic

scripts/              # CLI entry points for pipeline stages
```

## Important Implementation Notes

### Code Style

**Imports**: Always place imports at the top of the file. Never import inside functions.

```python
# Good
from src.utils.config import settings
from src.utils.logger import logger

def my_function():
    logger.info("Using imported logger")

# Bad
def my_function():
    from src.utils.logger import logger  # Never do this
    logger.info("Don't import inside functions")
```

**Default settings**: Use `--no-chunking` by default since chunking still needs testing and can produce irrelevant chunks.

### Prompt Versioning Methodology

Follow chemistry notebook principles for experimental rigor:

1. **Version control**: Append `_V{num}` suffix to prompt filenames (e.g., `system_prompt_V1.py`, `system_prompt_V2.py`)
2. **Ceteris paribus**: Change only ONE variable at a time between versions
3. **Start simple**: Begin with the simplest possible prompt, then incrementally build complexity
4. **Document changes**: User maintains separate notebook tracking what changed between versions and why
5. **Reproducibility**: Each version should be runnable independently for comparison

Example progression:
- `system_prompt_V1.py`: Minimal baseline
- `system_prompt_V2.py`: Add one constraint (e.g., "Be concise")
- `system_prompt_V3.py`: Add another constraint (e.g., "Cite sources")

This allows isolating which prompt changes improve/degrade performance.

### When Adding New Retrievers

1. Inherit from `BaseNeo4jRetriever`
2. Accept `driver: Driver` in `__init__` (dependency injection)
3. Implement `search(query: str, k: int)` method returning `List[Dict[str, Any]]`
4. Use `self._execute_query()` for Neo4j queries
5. Add metadata with `self._add_metadata(results, search_type)`
6. Register in `scripts/evaluate_retrievers.py` retriever factory

### Logging

Logger configured in `src/utils/logger.py`:
- Import: `from src.utils.logger import logger`
- Currently saves to single file (needs refactoring to separate log files)
- Use `logger.info()`, `logger.debug()`, `logger.error()` instead of print statements

### Chunking

HTML splitter (`src/data_processing/html_splitter.py`) has known issues:
- Splits on headers, but some headers are just card names
- Results in irrelevant chunks during retrieval
- Can run pipeline with `--no-chunking` flag to ingest full documents

### Environment Variables

Required in `.env`:
- `NEO4J_URI`: Neo4j connection string (e.g., `bolt://localhost:7687`)
- `NEO4J_USERNAME`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `OPENAI_API_KEY`: OpenAI API key for generation

## Project Goals & Context

### Primary Goal (Current)
Answer the research question: "Is Graph RAG better than traditional retrieval methods for game domain question answering?"

Evaluation compares:
1. Base LLM (zero-shot, no retrieval)
2. Traditional RAG (BM25, cosine similarity, hybrid)
3. Text2Cypher (structured query generation)
4. Graph RAG (learning objective)

The evaluation follows rigorous methodology with manual query annotation and failure mode taxonomy to understand not just which method performs better, but why it succeeds or fails.

### Secondary Goal
Learn software engineering principles while conducting research. This is an educational project with the goal of becoming an AI engineer, which means:
- Following proper SWE practices (dependency injection, configuration management, type safety)
- Building production-ready systems, not just research code
- Understanding the full stack: backend, API, frontend

### Future Roadmap (Tentative)
1. **Current**: Complete Graph RAG evaluation and ablation study (showcase AI eval expertise)
2. **Next**: Build REST API and frontend chatbot interface
3. **Potential**: Claude Computer playing Wildfrost OR YouTube video ingestion mapped to Neo4j knowledge graph

The order prioritizes demonstrating core competencies before expanding scope.
