"""Text2Cypher retriever for WildFrostRAG using LLM to generate Cypher queries.

This module implements retrieval by using an LLM to convert natural language
queries into Cypher queries based on the Neo4j schema.
"""

from typing import Any

from neo4j import Driver, ManagedTransaction, Record, Session

from wildfrost_rag.core.exceptions import CypherExecutionError
from wildfrost_rag.models.retrieval import RetrievedChunk, to_retrieved_chunks
from wildfrost_rag.utils.config import get_settings
from wildfrost_rag.utils.logger import logger
from wildfrost_rag.prompts.prompt_utils import format_prompt_tuple, VersionedPrompt
from wildfrost_rag.rag.augmented_generation.openai_client import call_openai_api
from wildfrost_rag.rag.retrievers.base_neo4j_retriever import BaseNeo4jRetriever


class Text2CypherRetriever(BaseNeo4jRetriever):
    """Implements retrieval by using an LLM to generate Cypher queries from natural language.

    This simulates the Text2Cypher functionality by using an LLM to understand the schema
    and generate appropriate Cypher queries.
    """

    def __init__(
        self,
        driver: Driver,
        text2cypher_prompt: VersionedPrompt,
        neo4j_database: str | None = None,
    ) -> None:
        """Initialize the Text2Cypher retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            text2cypher_prompt: VersionedPrompt containing the prompt template and version name
            neo4j_database: Optional database name (default: None uses default database)
        """
        super().__init__(driver, neo4j_database)

        self.prompt_version = text2cypher_prompt.prompt_version_name
        self.prompt_template = text2cypher_prompt.prompt_tuple

    def _get_schema(self, session: Session) -> dict[str, Any]:
        """Get the schema of the Neo4j database with relationship directions.

        Args:
            session: Neo4j session

        Returns:
            Dictionary containing the database schema with node properties and relationship patterns
        """
        nodes = self._get_node_schema(session)
        relationship_patterns = self._get_relationship_patterns(session)

        return {"nodes": nodes, "relationship_patterns": relationship_patterns}

    def _get_node_schema(self, session: Session) -> dict[str, list[dict[str, str]]]:
        """Get node labels and their properties with types from the database."""
        query = """
        CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName, propertyTypes, mandatory
        RETURN nodeType, collect({propertyName: propertyName, propertyTypes: propertyTypes, mandatory: mandatory}) as properties
        """

        def _read_tx(tx: ManagedTransaction) -> dict[str, list[dict[str, str]]]:
            result = tx.run(query)
            nodes: dict[str, list[dict[str, str]]] = {}
            for record in result:
                node_type = record["nodeType"]
                properties = record["properties"]
                nodes[node_type] = [
                    {
                        "name": prop["propertyName"],
                        "type": prop["propertyTypes"][0] if prop["propertyTypes"] else "Unknown",
                    }
                    for prop in properties
                ]
            return nodes

        return session.execute_read(_read_tx)

    def _get_relationship_patterns(self, session: Session) -> list[str]:
        """Get relationship patterns with directions from the database."""

        def _read_tx(tx: ManagedTransaction) -> list[str]:
            query = "CALL db.schema.visualization()"
            viz_result = tx.run(query)
            viz_record = viz_result.single()

            if not viz_record:
                return []

            relationships = viz_record.get("relationships", [])

            patterns = []
            for rel in relationships:
                start_node, end_node = rel.nodes
                start_label = list(start_node.labels)[0] if start_node.labels else "Unknown"
                end_label = list(end_node.labels)[0] if end_node.labels else "Unknown"
                pattern = f"({start_label})-[:{rel.type}]->({end_label})"
                patterns.append(pattern)

            return patterns

        return session.execute_read(_read_tx)

    def _format_schema_for_prompt(self, schema: dict[str, Any]) -> str:
        """Format schema into a string for the LLM prompt."""
        nodes_str = ""
        for node_label, properties in schema["nodes"].items():
            clean_label = node_label.strip(":").strip("`")
            nodes_str += f"\n  Label: `{clean_label}`\n  Properties:\n"
            for prop in properties:
                if isinstance(prop, dict):
                    nodes_str += f"    - {prop['name']} ({prop['type']})\n"
                else:
                    nodes_str += f"    - {prop}\n"

        patterns_str = "\n".join(f"  {pattern}" for pattern in schema["relationship_patterns"])

        return f"""Node labels and their properties:{nodes_str}
        Relationship patterns (with directions):
        {patterns_str}"""

    def _clean_cypher_response(self, response: str) -> str:
        """Remove markdown formatting and trailing semicolons from LLM response."""
        if response.startswith("```"):
            lines = response.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            response = "\n".join(lines)

        return response.strip().rstrip(";").strip()

    async def _generate_cypher_query(self, natural_query: str, schema: dict[str, Any]) -> str:
        """Generate a Cypher query from a natural language query using an LLM.

        Args:
            natural_query: Natural language query
            schema: Database schema

        Returns:
            Generated Cypher query string
        """
        schema_str = self._format_schema_for_prompt(schema)

        prompt = format_prompt_tuple(self.prompt_template, schema=schema_str, query=natural_query)

        settings = get_settings()
        response = await call_openai_api(
            messages=[{"role": "user", "content": prompt}],
            model=settings.openai.text2cypher_model,
            temperature=settings.openai.text2cypher_temperature,
        )

        cypher_query = response.strip()
        cypher_query = self._clean_cypher_response(cypher_query)

        return cypher_query

    def _add_limit_clause(self, cypher_query: str, k: int) -> str:
        """Add LIMIT clause to query if not present."""
        if "LIMIT" in cypher_query.upper():
            return cypher_query
        if "RETURN" in cypher_query.upper():
            return f"{cypher_query} LIMIT {k}"
        return cypher_query

    def _record_to_dict_with_cypher(
        self, record: Record, cypher_query: str, index: int
    ) -> dict[str, Any]:
        """Convert a Neo4j record to a result dictionary with Text2Cypher-specific fields.

        Uses base class _record_to_dict() for generic record handling,
        then adds Text2Cypher-specific metadata.
        """
        # Use base class for generic record → dict conversion
        result_dict = super()._record_to_dict(record)

        # Add Text2Cypher-specific fields
        result_dict["score"] = result_dict.get("score", 1.0)  # Default score if not in query
        result_dict["generated_cypher"] = cypher_query
        result_dict["result_index"] = index

        return result_dict

    def _execute_cypher_query(
        self, session: Session, cypher_query: str, k: int
    ) -> list[dict[str, Any]]:
        """Execute Cypher query as a read-only transaction and return results."""
        try:

            def _read_tx(tx: ManagedTransaction) -> list[dict[str, Any]]:
                result = tx.run(cypher_query)
                return [
                    self._record_to_dict_with_cypher(record, cypher_query, i)
                    for i, record in enumerate(result)
                    if i < k
                ]

            results = session.execute_read(_read_tx)

            if not results:
                return self._add_metadata(
                    [{"generated_cypher": cypher_query, "no_results": True}],
                    "text2cypher_llm_no_results",
                )

            return self._add_metadata(results, "text2cypher_llm")

        except Exception as e:
            raise CypherExecutionError(cypher_query=cypher_query, reason=str(e)) from e

    async def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Retrieve results by generating and executing a Cypher query from the natural language query.

        Args:
            query: Natural language query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of typed RetrievedChunk objects
        """
        with self.driver.session(database=self.neo4j_database) as session:
            schema = self._get_schema(session)

            cypher_query = await self._generate_cypher_query(query, schema)
            logger.info(f"Generated Cypher query:\n{cypher_query}")

            cypher_query = self._add_limit_clause(cypher_query, k)
            self.last_cypher_query = cypher_query
            raw_results = self._execute_cypher_query(session, cypher_query, k)

        # A "no results" placeholder is metadata for the caller (via last_cypher_query),
        # not a retrieved chunk — drop it before converting.
        raw_results = [r for r in raw_results if not r.get("no_results")]
        return to_retrieved_chunks(raw_results)
