"""
FulltextThenCypherRetriever for WildFrostRAG.

Combines Neo4j fulltext search with graph traversal to enrich results
with related Card, Tribe, CardType, Keyword, Stat, and other graph data.

The name "FulltextThenCypher" makes the order explicit:
1. Fulltext search FIRST (find relevant documents via Lucene)
2. Cypher traversal SECOND (enrich with graph data)
"""

from typing import List, Dict, Any, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from neo4j import Driver

from utils.config import settings
from utils.logger import logger
from rag.retrievers.base_neo4j_retriever import BaseNeo4jRetriever
from rag.retrievers.traversal_patterns import GRAPH_TRAVERSAL_QUERY


class FulltextThenCypherRetriever(BaseNeo4jRetriever):
    """
    Retriever that combines fulltext search with graph traversal enrichment.

    Flow:
        1. Fulltext search finds relevant Document nodes via Lucene index
        2. Cypher traversal enriches with Card -> Tribe -> CardType -> Keywords -> Stats -> etc.
        3. Return combined results with rag_context
    """

    def __init__(
        self,
        driver: Driver,
        neo4j_database: Optional[str] = None,
        index_name: Optional[str] = None,
        remove_stopwords: bool = False,
    ):
        """
        Initialize the FulltextThenCypherRetriever.

        Args:
            driver: Neo4j driver instance
            neo4j_database: Optional database name
            index_name: Fulltext index name (default: from settings)
            remove_stopwords: Whether to remove stop words from queries before searching
        """
        super().__init__(driver, neo4j_database)
        self.index_name = index_name or settings.fulltext_index_name
        self.remove_stopwords = remove_stopwords
        if self.remove_stopwords:
            self._initialize_nltk()

    def _initialize_nltk(self):
        """Initialize NLTK resources for stop word removal."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def _preprocess_query(self, query: str) -> str:
        """Remove stop words from query before sending to Lucene."""
        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        filtered = [token for token in tokens if token.isalpha() and token not in stop_words]
        return " ".join(filtered)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using fulltext similarity + graph traversal.

        Args:
            query: Natural language query
            k: Number of results to return

        Returns:
            List of enriched results with Card/Tribe/CardType/Keyword/Stat data
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
        return self._add_metadata(results, 'fulltext_then_cypher')
