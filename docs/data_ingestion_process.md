# Data Ingestion Process

The following Mermaid sequence diagram illustrates the data ingestion process in the WildFrostRAG project, showing how different components interact during data ingestion:

```mermaid
sequenceDiagram
    participant User
    participant IngestScript as scripts/ingest_data.py
    participant WikiScraper as src/web_scraper/sitemap_scraper.py
    participant Cards as src/data_processing/cards.py
    participant Enrichment as src/data_processing/enrichment.py
    participant HTMLSplitter as src/data_processing/html_splitter.py
    participant EmbedGen as src/embeddings/generator.py
    participant GraphBuilder as src/neo4j_kg/graph_builder.py
    participant VectorStore as src/neo4j_kg/vector_store.py
    participant Config as src/utils/config.py
    participant Logger as src/utils/logger.py

    User->>IngestScript: 1. Run ingestion script
    IngestScript->>Config: 2. Load configuration
    IngestScript->>WikiScraper: 3. Scrape Wildfrost Wiki
    WikiScraper-->>IngestScript: 4. Return card data
    IngestScript->>Cards: 5. Process card data
    Cards->>HTMLSplitter: 6. Split HTML content
    HTMLSplitter-->>Cards: 7. Return processed content
    Cards->>Enrichment: 8. Enrich with additional data
    Enrichment-->>Cards: 9. Return enriched data
    IngestScript->>EmbedGen: 10. Generate embeddings
    EmbedGen-->>IngestScript: 11. Return embeddings
    IngestScript->>GraphBuilder: 12. Create graph nodes
    GraphBuilder-->>IngestScript: 13. Nodes created
    IngestScript->>VectorStore: 14. Store embeddings in vector index
    VectorStore-->>IngestScript: 15. Embeddings stored
    IngestScript->>Logger: 16. Log process completion
```

## Process Overview

This diagram shows the complete data ingestion process:

1. User initiates the ingestion script
2. Configuration is loaded
3. Web scraping occurs to gather URLs
4. Card data is processed and enriched
5. HTML content is split and processed
6. Embeddings are generated
7. Graph nodes are created in Neo4j
8. Embeddings are stored in the vector index
9. Process completion is logged