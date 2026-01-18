# Query Answering Process

The following Mermaid sequence diagram illustrates the query answering process in the WildFrostRAG project, showing how different components interact when processing user queries:

```mermaid
sequenceDiagram
    participant User
    participant Retriever as src/rag/retrievers/*
    participant Config as src/utils/config.py
    participant VectorStore as src/neo4j_kg/vector_store.py
    participant LLMGen as src/rag/augmented_generation/call_llm_generation.py
    participant Logger as src/utils/logger.py

    User->>Retriever: 1. Submit query
    Retriever->>Config: 2. Load configuration
    Retriever->>VectorStore: 3. Search vector index
    VectorStore-->>Retriever: 4. Return relevant chunks
    Retriever->>LLMGen: 5. Generate response with context
    LLMGen-->>User: 6. Return generated response
    LLMGen->>Logger: 7. Log query and response
```

## Process Overview

This diagram shows the complete query answering process:

1. User submits a query
2. Configuration is loaded
3. The retriever searches the vector store for relevant chunks
4. Relevant chunks are returned
5. The LLM generates a response using the context
6. The response is returned to the user
7. The query and response are logged