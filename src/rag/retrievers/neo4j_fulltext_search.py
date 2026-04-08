"""Neo4j-based full-text search retriever for WildFrostRAG.

This module implements lexical retrieval using Neo4j's built-in full-text search
capabilities, which are based on Apache Lucene.
"""

from typing import Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from neo4j import Driver
from utils.config import settings
from utils.logger import logger
from .base_neo4j_retriever import BaseNeo4jRetriever


class Neo4jFullTextSearch(BaseNeo4jRetriever):
    """Implements lexical similarity retrieval using Neo4j's full-text search.

    This corresponds to the 'BM25' (or lexical search) approach in the research goals,
    using Neo4j's Lucene-based full-text search as a proxy.
    """

    def __init__(
        self,
        driver: Driver,
        neo4j_database: str | None = None,
        index_name: str | None = None,
        remove_stopwords: bool = False,
    ) -> None:
        """Initialize the Neo4j full-text search retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            neo4j_database: Optional database name (default: None uses default database)
            index_name: Optional fulltext index name override (default: from settings)
            remove_stopwords: Whether to remove stop words from queries before sending to Lucene
        """
        super().__init__(driver, neo4j_database)
        self.index_name = index_name or settings.fulltext_index_name
        self.remove_stopwords = remove_stopwords
        if self.remove_stopwords:
            self._initialize_nltk()

    def _initialize_nltk(self) -> None:
        """Initialize NLTK resources for stop word removal."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

    def _preprocess_query(self, query: str) -> str:
        """Remove stop words from query before sending to Lucene."""
        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words("english"))
        filtered = [
            token for token in tokens if token.isalpha() and token not in stop_words
        ]
        return " ".join(filtered)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant document chunks from Neo4j based on lexical similarity.

        This method performs full-text search using Neo4j's built-in full-text indexing
        capabilities, which implements Lucene-based search algorithms.

        Note: The fulltext index must already exist (created during pipeline setup).
              If index doesn't exist, Neo4j will raise an error.

        Args:
            query: Input query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of dictionaries containing retrieved chunks with their metadata and scores
        """
        search_query_text = query
        if self.remove_stopwords:
            search_query_text = self._preprocess_query(query)
            logger.debug(
                f"Fulltext query after stop word removal: '{search_query_text}'"
            )

        # Perform full-text search (index must already exist)
        search_query = """
        CALL db.index.fulltext.queryNodes($index_name, $query)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT $k
        """

        params = {"index_name": self.index_name, "query": search_query_text, "k": k}

        try:
            results = self._execute_query(search_query, params)
            return self._add_metadata(results, "fulltext")
        except Exception as e:
            logger.error(
                f"Fulltext search failed. Index '{self.index_name}' may not exist. "
                f"Run 'python -m scripts.ingest_data --no-chunking' to create it. "
                f"Error: {e}"
            )
            raise
