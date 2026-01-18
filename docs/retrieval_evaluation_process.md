# Retrieval Evaluation Process

The following Mermaid sequence diagram illustrates the retrieval evaluation process in the WildFrostRAG project, showing how different components interact during evaluation:

```mermaid
sequenceDiagram
    participant User
    participant Eval as src/rag/evaluation/*
    participant Retriever as src/rag/retrievers/*
    participant LLMGen as src/rag/augmented_generation/call_llm_generation.py
    participant Logger as src/utils/logger.py

    User->>Eval: 1. Run evaluation
    Eval->>Retriever: 2. Test retrieval methods
    Retriever-->>Eval: 3. Return results
    Eval->>LLMGen: 4. Evaluate generation quality
    LLMGen-->>Eval: 5. Return quality metrics
    Eval->>Logger: 6. Log evaluation results
```

## Process Overview

This diagram shows the complete retrieval evaluation process:

1. User initiates an evaluation
2. The evaluation module tests retrieval methods
3. Results are returned from the retriever
4. Generation quality is evaluated
5. Quality metrics are returned
6. Evaluation results are logged