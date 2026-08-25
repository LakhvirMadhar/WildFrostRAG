"""External BM25 retriever for WildFrostRAG.

This module implements lexical retrieval using the rank_bm25 library for true BM25 scoring.
It retrieves documents from Neo4j, processes them with BM25, and returns ranked results.
"""

import warnings
from typing import Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from neo4j import Driver
from rank_bm25 import BM25Okapi
from models.retrieval import RetrievedChunk, to_retrieved_chunks
from utils.config import get_settings
from utils.logger import logger
from rag.retrievers.base_neo4j_retriever import BaseNeo4jRetriever


class BM25Retriever(BaseNeo4jRetriever):
    """Implements lexical similarity retrieval using true BM25 scoring.

    This corresponds to the 'BM25' approach in the research goals.
    """

    _shared_cache: dict[str, Any] = {
        "documents": None,
        "node_data": None,
        "bm25_model": None,
        "cache_key": None,
    }

    def __init__(
        self,
        driver: Driver,
        neo4j_database: str | None = None,
        remove_stopwords: bool = True,
        remove_stopwords_query: bool | None = None,
        remove_stopwords_docs: bool | None = None,
    ) -> None:
        """Initialize the BM25 retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            neo4j_database: Optional database name (default: None uses default database)
            remove_stopwords: Master flag — sets both query and docs (default: True)
            remove_stopwords_query: Override for query preprocessing only
            remove_stopwords_docs: Override for document preprocessing only
        """
        super().__init__(driver, neo4j_database)
        warnings.warn(
            "BM25Retriever loads all documents into memory. "
            "For production use with large datasets, prefer Neo4jFullTextSearch "
            "which uses Lucene's on-disk BM25-style scoring.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.index_name = get_settings().embedding.bm25_index_name
        # If specific overrides provided, use them; otherwise fall back to master flag
        self.remove_stopwords_query = (
            remove_stopwords_query if remove_stopwords_query is not None else remove_stopwords
        )
        self.remove_stopwords_docs = (
            remove_stopwords_docs if remove_stopwords_docs is not None else remove_stopwords
        )
        self.bm25_model: BM25Okapi | None = None
        self.documents: list[list[str]] = []
        self.node_data: list[dict[str, Any]] = []
        self._initialize_nltk()

    def _initialize_nltk(self) -> None:
        """Initialize NLTK resources for text preprocessing."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

    def _tokenize(self, text: str, remove_sw: bool) -> list[str]:
        """Tokenize text, optionally removing stop words.

        Args:
            text: Input text to tokenize
            remove_sw: Whether to remove stop words

        Returns:
            List of lowercase word tokens
        """
        tokens: list[str] = word_tokenize(text.lower())
        if remove_sw:
            stop_words = set(stopwords.words("english"))
            tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        else:
            tokens = [token for token in tokens if token.isalpha()]
        return tokens

    def _load_documents_from_neo4j(self) -> None:
        """Load all documents from Neo4j to build the BM25 index.

        Uses class-level cache to avoid reloading for multiple instances.
        """
        cache_key = f"{self.neo4j_database}:{self.index_name}:sw_docs={self.remove_stopwords_docs}"

        if (
            BM25Retriever._shared_cache["cache_key"] == cache_key
            and BM25Retriever._shared_cache["bm25_model"] is not None
        ):
            logger.info("Using cached BM25 index from previous instance")
            self.documents = BM25Retriever._shared_cache["documents"]
            self.node_data = BM25Retriever._shared_cache["node_data"]
            self.bm25_model = BM25Retriever._shared_cache["bm25_model"]
            return

        logger.info("Loading all documents from Neo4j for BM25 indexing...")

        with self.driver.session(database=self.neo4j_database) as session:
            query = f"""
            MATCH (d:{self.index_name})
            WHERE d.text IS NOT NULL
            RETURN d.text AS text, d
            """

            results = session.run(query)

            self.documents = []
            self.node_data = []

            for record in results:
                text = record["text"]
                node = record["d"]

                tokens = self._tokenize(text, self.remove_stopwords_docs)
                self.documents.append(tokens)

                node_dict = {}
                for key, value in node.items():
                    if key != "embedding":
                        node_dict[key] = value
                self.node_data.append(node_dict)

            self.bm25_model = BM25Okapi(self.documents)

            BM25Retriever._shared_cache = {
                "documents": self.documents,
                "node_data": self.node_data,
                "bm25_model": self.bm25_model,
                "cache_key": cache_key,
            }

            logger.info(f"BM25 index built with {len(self.documents)} documents")

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Retrieve the top-k most relevant document chunks using BM25 scoring.

        Args:
            query: Input query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of typed RetrievedChunk objects
        """
        if self.bm25_model is None:
            self._load_documents_from_neo4j()

        query_tokens = self._tokenize(query, self.remove_stopwords_query)

        if self.bm25_model is None:
            raise RuntimeError("BM25 model not initialized — call _load_documents_from_neo4j first")
        scores = self.bm25_model.get_scores(query_tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            node_data = self.node_data[idx].copy()
            node_data["score"] = score
            results.append(node_data)

        return to_retrieved_chunks(self._add_metadata(results, "bm25"))
