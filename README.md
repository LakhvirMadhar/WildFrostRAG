# WildFrostRAG Project Context

## Project Overview
**WildFrostRAG** is a research-oriented Retrieval-Augmented Generation (RAG) system for the game *Wildfrost*. The primary objective is to evaluate and compare different retrieval strategies—ranging from traditional vector search to advanced Graph RAG—using game data (Cards, Tribes, Stats) scraped from the Wildfrost Wiki and stored in a Neo4j Knowledge Graph.
This project can be viewed as an abalation study of retrieval and generation metrics.

## Core Research Goals
The project aims to benchmark the following RAG implementations:
1.  **Base LLM:** Zero-shot performance using innate model knowledge (no retrieval).
2.  **Traditional RAG:** 
    *   **BM25:** Keyword-based retrieval.
    *   **Cosine Similarity:** Standard vector-based retrieval.
    *   **Hybrid:** Reciprocal Rank Fusion (RRF) of BM25 and Cosine Similarity.
3.  **Text2Cypher:** Using Neo4j's Text2Cypher extension to generate Cypher queries against the established Wildfrost ontology in Neo4j.
4.  **Graph RAG:** Implementing advanced graph traversal or community-based retrieval (the primary learning objective).

## Evaluation Framework
The project follows a rigorous evaluation style inspired by **Hamel Husain’s framework**:
*   **Query Generation:** User-generated queries from random samples of documents based on game schemas (`query_generation.ipynb`).
*   **Retrieval Metrics:** Measuring how accurately the system finds relevant chunks or nodes using metrics like NDCG and Hit@k.
*   **Generation Metrics:** Evaluating the quality of the final answer based on user-defined criteria.
*   **Qualitative Coding:** Using "Open Coding" and "Axial Coding" to categorize failure modes in `queries/simple_reference_based_queries.csv`.

## Architecture & Components

### 1. Data Layer (`data/`)
*   `structured_outputs/`: Processed card data organized by `CardType`.
*   `schemas/`: JSON definitions of card types and game mechanics.

### 2. Knowledge Graph (`src/neo4j_kg/`)
*   Manages the Neo4j instance.
*   Maps relationships like `BELONGS_TO_TRIBE`, `HAS_CARD_TYPE`, and `HAS_STAT`.
*   Nodes: `Card`, `Tribe`, `Stat`, `CardType`, `Document`.

### 3. Model Integrations
*   **OpenAI:** Used for baseline zero-shot and RAG comparisons.

## Planned Further Steps & Roadmap

### 1. Notebook Cleanup
*   **`rag_eval_demo.ipynb`:** Currently acts as a scratchpad. Needs to be stripped of scraped logic, hardcoded paths, and unused imports. It should solely focus on *demonstrating* the pipeline, calling functions from the `src` directory.
*   **`query_generation.ipynb`:** Contains a massive `QueryAnnotationGUI` class and mixed logic for generation and evaluation. The GUI code should be moved to a module, and the notebook should focus on the interactive analysis loop.

### 2. Codebase Refactoring
*   **Modularization:** Move the `QueryAnnotationGUI` class to `src/utils/gui.py` or `src/evaluation/gui.py`.
*   **Consolidation:** Ensure `scrape_sitemap` and `process_sitemap_urls` are tightly integrated into a single ingestion pipeline rather than disjointed function calls.
*   **Type Safety:** Add Pydantic models for configuration to replace loose environment variable fetching scattered across files.

### 3. Orchestration Scripts
*   **`scripts/ingest_data.py`:** Create a dedicated CLI script to handle the full ETL pipeline (Scrape -> Process -> Vectorize -> Ingest to Neo4j). This removes the dependency on Jupyter for data setup.
    *   *Usage:* `python -m scripts.ingest_data --force-scrape`

### 4. Agent Suggestions (Recommended)
*   **Evaluation Pipeline Module:** Create `src/evaluation/metrics.py` to house the logic for `NDCG`, `Hit@k`, and `Recall`. This allows these metrics to be computed programmatically during CI/CD or batch runs, not just in notebooks.
*   **Configuration Management:** Implement a central `config.py` (using `pydantic-settings`) to manage model names, embedding dimensions, and database URIs. This prevents "magic strings" like `'all-MiniLM-L6-v2'` from appearing in multiple files.
*   **Dependency Separation:** The project currently mixes "application" dependencies (FastAPI, Streamlit) with "analysis" dependencies (Jupyter, Pandas). Consider grouping these in `pyproject.toml` (e.g., `tool.poetry.group.dev.dependencies`).

##  Current Plan of Action

The immediate goal is to decompose the monolithic `rag_eval_demo.ipynb` into a robust, repeatable pipeline.

### Phase 1: Separation of Concerns
The "Setup" phase in `rag_eval_demo.ipynb` will be extracted into a standalone Python script (`scripts/ingest_data.py`) with two distinct stages:

1.  **Web Scraping & Processing:**
    *   **Orchestration:** Scrape sitemaps and download raw HTML.
    *   **Specific Logic:** Handle special cases like "Leaders", "Companions" (parsing tribe tables), and "Items" (parsing exclusivity tables).
    *   **Enrichment:** Fill missing data gaps identified in the HTML.

2.  **Neo4j Ingestion & Indexing:**
    *   **Graph Population:** Ingest `CardInfo` objects into Neo4j nodes (`Card`, `Tribe`, `Stat`).
    *   **Vector Indexing:** Chunk the raw HTML, generate embeddings (using `sentence-transformers`), and populate the Neo4j Vector Index.
    *   **Cleanup:** Remove the Ollama/Local LLM generation steps from the ingestion pipeline (these belong in the evaluation phase).


### Phase 2: Setting up other vector searches + adding in manual retrieval metrics
1.  **BM25**
    *   Figure out a way to setup BM25 with this Neo4j setup

2.  **Cosine Similarity**
    *   I did this, but my evals was based on chunked documents, need to redo the manual eval on not chunked documents

3.  **Hybrid Similarity**
    *   Given the lexical and semanitc search, figure out a way to 

4.  **Neo4j's text2Cypher library**
    *   Use Neo4j's text2Cypher library 

5.  **Knowledge Graph**
    *   Idk how a knowledge graph is supposed to work, but this is the goal. Is a knowledge graph "better" than the other methods?

Given the above retrieval techniques, we need to setup the retrieval metrics
1.  **hit@k**
    *   This establishes stuff like precision@k and recall@k, which k being the number of documents retrieved (eg. 1, 2, 5, 10).

2.  **NCDG**
    *   Does this make sense? I don't actually know the expected order

3.  **MRR**
    *   Mean Recipricocal Rank, I think this one is applicable?

4.  **RRF**
    *   Recipricol Rank Fusion, I don't remember what this means


### Phase 3: Data Processing/Chunking
1.  **Chunking**
    *   Currently have a chunking option and no chunking option. No chunking is fine as is, it caputres the entire document.
    *   The HTML splitter needs some work, as it splits headers but some headers are just the name of the card, therefore it becomes an irrelevant chunk during retrieval
    *   There's also some junk at the end of the HTML that I need to parse out as it's redunant text

2.  **Bugs**
    *   BUG: When processing the html, sometimes words are missing (FIXED)
        - Also should remove the completely irrelevant info that shows up at the end that has no informatoin related to the cards itself? (FIXED)
    *   BUG: For cards like Infernoko that have multiple phases, we are not capturing the data for phases (should add the relation, Card -> has other phase(or something) -> Card)

3.  **Missing Features/Pages to Still Scrape**
    *   To card nodes, need to add:
            a. Other Stats field  (Resist Snow, Frenzy, etc. This is the Stats page, which needs scraping and processing: https://wildfrostwiki.com/Stats)
            b. Card Description field (flavor text vs ability)
    *   For abilities, there's the listed ability itself, and the canonical ability that the in the excel sheet.
        -   "Increase attack by 2" & "Increase attack by 1" is actually "Increase attack by <n>"
        -   Probably need to do a keyword or ability node: https://wildfrostwiki.com/Keywords
    *   Need to add logic to scrape the Leaders page. For relations, it should link them to the tribe and to a leader node (there can only be one leader) (should also link the leader page to a leader node)
    *   Pets should also get a pet node (there can only be one pet)
    *   Map has it's own ontology too actually: https://wildfrostwiki.com/Map. The map contains Zones. Zones contains fights and map events (the inbetween after each fight, depending on the zone.) Map Events has an ontology I can refer to
    *   Each fight has their own data. Enemies show up in these fights. Fights are made up of waves (waves add these enemies to the field, granted if there is space on the field).
    *   I'm not saving the village unlocks, but its a low priority atm.
    *   I need to scrape the difficulty bells, as you need 10 bells to even do the final fight in Map Events.
    *   Hm, probably need something for how fights themselves play out (1 card is played OR player hits their sun bell, then enemy turn progresses). A lot I have to do here.
    *   I'm not saving the card image anywhere, unsure if needed atm.
    *   Need to update the to_dict method to capture more information I'm missing
    *   **Important**: I need to make sure the to_dict method is expanded to make a very cleaned format of the HTML.
    *   **Important**: Logger is saving all to one file, we need to make it several files. Also need to fix print statements or tqdm write statements to instead be logger.
    *   


### Overall Theme
*   The `rag_eval_demo.ipynb` will be for any scraping and neo4j ingestion testing the user deems necessary.
*   The `query_generation.ipynb` will remain the primary entry point for running RAG experiments and evaluations.

## Building and Running

### Prerequisites
*   Python 3.12+ (managed by Poetry).
*   Neo4j instance (Local Bolt: `bolt://localhost:7687`).
*   Environment variables in `configs/.env`: `OPENAI_API_KEY`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`.

### Key Workflows
*   **Data Prep:** Use `rag_eval_demo.ipynb` to test scraping and parsing of data, and ingest data into Neo4j.
*   **Eval Run:** Use `query_generation.ipynb` to run experiments across different RAG versions.
*   **Analysis:** Use the `QueryAnnotationGUI` (built with ipywidgets) to manually validate and code responses.
