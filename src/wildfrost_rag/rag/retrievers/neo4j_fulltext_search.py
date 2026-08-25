"""Neo4j-based full-text search retriever for WildFrostRAG.

This module implements lexical retrieval using Neo4j's built-in full-text search
capabilities, which are based on Apache Lucene.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from neo4j import Driver
from wildfrost_rag.models.retrieval import RetrievedChunk, to_retrieved_chunks
from wildfrost_rag.neo4j_kg.document_repository import DocumentRepository
from wildfrost_rag.utils.config import get_settings
from wildfrost_rag.utils.logger import logger
from .base_neo4j_retriever import BaseNeo4jRetriever


class Neo4jFullTextSearch(BaseNeo4jRetriever):
    """Implements lexical similarity retrieval using Neo4j's full-text search.

    This corresponds to the 'BM25' (or lexical search) approach in the research goals,
    using Neo4j's Lucene-based full-text search as a proxy.
    """

    def __init__(
        self,
        driver: Driver,
        document_repository: DocumentRepository,
        neo4j_database: str | None = None,
        index_name: str | None = None,
        remove_stopwords: bool = False,
    ) -> None:
        """Initialize the Neo4j full-text search retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            document_repository: Repository owning the fulltext-search Cypher query
            neo4j_database: Optional database name (default: None uses default database)
            index_name: Optional fulltext index name override (default: from settings.embedding)
            remove_stopwords: Whether to remove stop words from queries before sending to Lucene
        """
        super().__init__(driver, neo4j_database)
        self._document_repository = document_repository
        self.index_name = index_name or get_settings().embedding.fulltext_index_name
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
        filtered = [token for token in tokens if token.isalpha() and token not in stop_words]
        return " ".join(filtered)

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Retrieve the top-k most relevant document chunks from Neo4j based on lexical similarity.

        This method performs full-text search using Neo4j's built-in full-text indexing
        capabilities, which implements Lucene-based search algorithms.

        Note: The fulltext index must already exist (created during pipeline setup).
              If index doesn't exist, Neo4j will raise an error.

        Args:
            query: Input query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of typed RetrievedChunk objects
        """
        search_query_text = query
        if self.remove_stopwords:
            search_query_text = self._preprocess_query(query)
            logger.debug(f"Fulltext query after stop word removal: '{search_query_text}'")

        # Perform full-text search (index must already exist)
        try:
            results = self._document_repository.fulltext_search(
                self.index_name, search_query_text, k
            )
            self.last_cypher_query = self._document_repository.last_cypher_query
            return to_retrieved_chunks(self._add_metadata(results, "fulltext"))
        except Exception as e:
            logger.error(
                f"Fulltext search failed. Index '{self.index_name}' may not exist. "
                f"Run 'python -m scripts.ingest_data --no-chunking' to create it. "
                f"Error: {e}"
            )
            raise
