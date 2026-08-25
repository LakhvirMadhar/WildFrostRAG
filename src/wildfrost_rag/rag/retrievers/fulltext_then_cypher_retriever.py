"""FulltextThenCypherRetriever for WildFrostRAG.

Combines Neo4j fulltext search with graph traversal to enrich results
with related Card, Tribe, CardType, Keyword, Stat, and other graph data.

The name "FulltextThenCypher" makes the order explicit:
1. Fulltext search FIRST (find relevant documents via Lucene)
2. Cypher traversal SECOND (enrich with graph data)
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from neo4j import Driver

from wildfrost_rag.models.retrieval import RetrievedChunk, to_retrieved_chunks
from wildfrost_rag.utils.config import get_settings
from wildfrost_rag.utils.logger import logger
from wildfrost_rag.rag.retrievers.base_neo4j_retriever import BaseNeo4jRetriever
from wildfrost_rag.rag.retrievers.traversal_patterns import GRAPH_TRAVERSAL_QUERY


class FulltextThenCypherRetriever(BaseNeo4jRetriever):
    """Retriever that combines fulltext search with graph traversal enrichment.

    Flow:
        1. Fulltext search finds relevant Document nodes via Lucene index
        2. Cypher traversal enriches with Card -> Tribe -> CardType -> Keywords -> Stats -> etc.
        3. Return combined results with rag_context
    """

    def __init__(
        self,
        driver: Driver,
        neo4j_database: str | None = None,
        index_name: str | None = None,
        remove_stopwords: bool = False,
    ) -> None:
        """Initialize the FulltextThenCypherRetriever.

        Args:
            driver: Neo4j driver instance
            neo4j_database: Optional database name
            index_name: Fulltext index name (default: from settings.embedding)
            remove_stopwords: Whether to remove stop words from queries before searching
        """
        super().__init__(driver, neo4j_database)
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
        """Search using fulltext similarity + graph traversal.

        Args:
            query: Natural language query
            k: Number of results to return

        Returns:
            List of typed RetrievedChunk objects enriched with graph data
        """
        search_query_text = query
        if self.remove_stopwords:
            search_query_text = self._preprocess_query(query)
            logger.debug(f"Fulltext query after stop word removal: '{search_query_text}'")

        combined_query = f"""
        CALL db.index.fulltext.queryNodes($index_name, $query)
        YIELD node as doc, score
        {GRAPH_TRAVERSAL_QUERY}
        LIMIT $k
        """

        params = {
            "index_name": self.index_name,
            "query": search_query_text,
            "k": k,
        }

        results = self._execute_query(combined_query, params)
        return to_retrieved_chunks(self._add_metadata(results, "fulltext_then_cypher"))
