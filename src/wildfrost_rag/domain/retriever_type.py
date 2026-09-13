"""Single source of truth for the retriever type identifiers.

These values are persisted verbatim into experiment config.json/registry data
(see RetrievalConfig.retriever_type), so member values must never change once
shipped - only new members may be added.
"""

from enum import StrEnum


class RetrieverType(StrEnum):
    """Every retriever strategy the pipeline knows how to construct."""

    VECTOR = "vector"
    FULLTEXT = "fulltext"
    BM25 = "bm25"
    BM25_VECTOR = "bm25_vector"
    FULLTEXT_VECTOR = "fulltext_vector"
    BM25_FULLTEXT_VECTOR = "bm25_fulltext_vector"
    TEXT2CYPHER = "text2cypher"
    TEXT2CYPHER_VECTOR = "text2cypher_vector"
    VECTOR_THEN_CYPHER = "vector_then_cypher"
    FULLTEXT_THEN_CYPHER = "fulltext_then_cypher"
