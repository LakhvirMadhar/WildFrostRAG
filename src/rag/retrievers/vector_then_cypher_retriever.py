"""
VectorThenCypherRetriever for WildFrostRAG.

Combines vector similarity search with graph traversal to enrich results
with related Card, Tribe, and CardType data from the knowledge graph.

The name "VectorThenCypher" makes the order explicit:
1. Vector search FIRST (find relevant documents)
2. Cypher traversal SECOND (enrich with graph data)
"""

from typing import List, Dict, Any, Optional
from neo4j import Driver
from src.utils.config import settings
from src.rag.retrievers.base_neo4j_retriever import BaseNeo4jRetriever
from src.rag.retrievers.neo4j_vector_search import Neo4jVectorSearch


class VectorThenCypherRetriever(BaseNeo4jRetriever):
    """
    Retriever that combines vector search with predefined graph traversal patterns.

    This is the "Graph RAG" approach: use semantic similarity to find relevant
    Document nodes, then traverse the graph to enrich results with structured
    Card/Tribe/CardType data.

    Flow:
        1. Embed query
        2. Vector search finds relevant Document nodes
        3. Cypher traversal enriches with Card -> Tribe -> CardType
        4. Return combined results with rag_context
    """

    # Predefined traversal patterns for Wildfrost knowledge graph
    TRAVERSAL_PATTERNS = {
        "full_card_context": """
            MATCH (d)<-[:HAS_DOCUMENT]-(c:Card)
            OPTIONAL MATCH (c)-[:BELONGS_TO_TRIBE]->(t:Tribe)
            OPTIONAL MATCH (c)-[:HAS_CARD_TYPE]->(ct:CardType)
            RETURN d, c, t, ct, score
            ORDER BY score DESC
        """,
        "card_only": """
            MATCH (d)<-[:HAS_DOCUMENT]-(c:Card)
            RETURN d, c, score
            ORDER BY score DESC
        """,
        "with_stats": """
            MATCH (d)<-[:HAS_DOCUMENT]-(c:Card)
            OPTIONAL MATCH (c)-[:BELONGS_TO_TRIBE]->(t:Tribe)
            OPTIONAL MATCH (c)-[:HAS_CARD_TYPE]->(ct:CardType)
            OPTIONAL MATCH (c)-[hs:HAS_STAT]->(s:Stat)
            RETURN d, c, t, ct, collect({stat_name: s.name, value: hs.value}) as stats, score
            ORDER BY score DESC
        """,
    }

    def __init__(
        self,
        driver: Driver,
        neo4j_database: Optional[str] = None,
        index_name: Optional[str] = None,
        traversal_pattern: str = "full_card_context"
    ):
        """
        Initialize the VectorThenCypherRetriever.

        Args:
            driver: Neo4j driver instance
            neo4j_database: Optional database name
            index_name: Vector index name (default: from settings)
            traversal_pattern: Either a key from TRAVERSAL_PATTERNS or custom Cypher
        """
        super().__init__(driver, neo4j_database)
        self.index_name = index_name or settings.vector_index_name
        self.pattern_name = traversal_pattern
        self.traversal_pattern = self._resolve_pattern(traversal_pattern)

    def _resolve_pattern(self, pattern: str) -> str:
        """
        Resolve pattern name to Cypher query.

        Args:
            pattern: Either a key from TRAVERSAL_PATTERNS or custom Cypher

        Returns:
            Cypher query string for the traversal
        """
        if pattern in self.TRAVERSAL_PATTERNS:
            return self.TRAVERSAL_PATTERNS[pattern]
        # Assume it's custom Cypher if not a known pattern
        return pattern

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using vector similarity + graph traversal.

        Args:
            query: Natural language query
            k: Number of results to return

        Returns:
            List of enriched results with Card/Tribe/CardType data
        """
        # Step 1: Embed the query (reuse cached model from Neo4jVectorSearch)
        model = Neo4jVectorSearch.get_embedding_model()
        query_embedding = model.encode(query).tolist()

        # Step 2: Build combined vector search + traversal query
        combined_query = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
        YIELD node as d, score
        {self.traversal_pattern}
        """

        params = {
            "index_name": self.index_name,
            "query_embedding": query_embedding,
            "k": k
        }

        # Step 3: Execute using base class _execute_query (handles any Cypher result)
        # Base class _record_to_dict() will prefix node properties: d_text, c_card_name, t_name, etc.
        # Base class _format_result_as_text() will format all properties dynamically
        results = self._execute_query(combined_query, params)
        return self._add_metadata(results, f'vector_then_cypher_{self.pattern_name}')
