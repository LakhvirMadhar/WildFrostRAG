"""
Text2Cypher retriever for WildFrostRAG using LLM to generate Cypher queries.

This module implements retrieval by using an LLM to convert natural language
queries into Cypher queries based on the Neo4j schema.
"""

from typing import Any

from neo4j import Driver

from src.utils.config import settings
from src.utils.logger import logger
from src.utils.prompt_utils import format_prompt_tuple, VersionedPrompt
from src.rag.augmented_generation.openai_client import call_openai_api
from src.rag.retrievers.base_neo4j_retriever import BaseNeo4jRetriever


class Text2CypherRetriever(BaseNeo4jRetriever):
    """
    Implements retrieval by using an LLM to generate Cypher queries from natural language.
    This simulates the Text2Cypher functionality by using an LLM to understand the schema
    and generate appropriate Cypher queries.
    """

    def __init__(
        self,
        driver: Driver,
        text2cypher_prompt: VersionedPrompt,
        neo4j_database: str | None = None
    ):
        """
        Initialize the Text2Cypher retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            text2cypher_prompt: VersionedPrompt containing the prompt template and version name
            neo4j_database: Optional database name (default: None uses default database)
        """
        super().__init__(driver, neo4j_database)
        self.llm_response = None  # Track LLM response for experiment tracking

        self.prompt_version = text2cypher_prompt.prompt_version_name
        self.prompt_template = text2cypher_prompt.prompt_tuple

    def _get_schema(self, session) -> dict[str, Any]:
        """
        Get the schema of the Neo4j database with relationship directions.

        Args:
            session: Neo4j session

        Returns:
            Dictionary containing the database schema with node properties and relationship patterns
        """
        nodes = self._get_node_schema(session)
        relationship_patterns = self._get_relationship_patterns(session)

        return {
            "nodes": nodes,
            "relationship_patterns": relationship_patterns
        }

    def _get_node_schema(self, session) -> dict[str, list[str]]:
        """Get node labels and their properties from the database."""
        query = """
        CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName, propertyTypes, mandatory
        RETURN nodeType, collect({propertyName: propertyName, propertyTypes: propertyTypes, mandatory: mandatory}) as properties
        """

        result = session.run(query)
        nodes = {}
        for record in result:
            node_type = record["nodeType"]
            properties = record["properties"]
            nodes[node_type] = [prop["propertyName"] for prop in properties]

        return nodes

    def _get_relationship_patterns(self, session) -> list[str]:
        """Get relationship patterns with directions from the database."""
        query = "CALL db.schema.visualization()"
        viz_result = session.run(query)
        viz_record = viz_result.single()

        print(f"DEBUG viz_record: {viz_record}")
        print(f"DEBUG viz_record keys: {list(viz_record.keys()) if viz_record else 'None'}")

        if not viz_record:
            return []

        relationships = viz_record.get("relationships", [])
        print(f"DEBUG: Found {len(relationships)} relationships")

        patterns = []
        for rel in relationships:
            start_node, end_node = rel.nodes
            start_label = list(start_node.labels)[0] if start_node.labels else "Unknown"
            end_label = list(end_node.labels)[0] if end_node.labels else "Unknown"
            pattern = f"({start_label})-[:{rel.type}]->({end_label})"
            print(f"DEBUG: Added pattern: {pattern}")
            patterns.append(pattern)

        return patterns

    def _format_schema_for_prompt(self, schema: dict[str, Any]) -> str:
        """Format schema into a string for the LLM prompt."""
        nodes_str = ""
        for node_label, properties in schema['nodes'].items():
            clean_label = node_label.strip(":").strip("`")
            nodes_str += f"\n  Label: `{clean_label}`\n  Properties:\n"
            for prop in properties:
                nodes_str += f"    - {prop}\n"

        patterns_str = "\n".join(f"  {pattern}" for pattern in schema['relationship_patterns'])

        return f"""Node labels and their properties:{nodes_str}
        Relationship patterns (with directions):
        {patterns_str}"""

    def _clean_cypher_response(self, response: str) -> str:
        """Remove markdown formatting from LLM response."""
        if not response.startswith("```"):
            return response

        lines = response.split('\n')
        if lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]

        return '\n'.join(lines).strip()

    async def _generate_cypher_query(self, natural_query: str, schema: dict[str, Any]) -> str:
        """
        Generate a Cypher query from a natural language query using an LLM.

        Args:
            natural_query: Natural language query
            schema: Database schema

        Returns:
            Generated Cypher query string
        """
        schema_str = self._format_schema_for_prompt(schema)

        prompt = format_prompt_tuple(
            self.prompt_template,
            schema=schema_str,
            query=natural_query
        )

        response = await call_openai_api(
            messages=[{"role": "user", "content": prompt}],
            model=settings.text2cypher_model,
            temperature=settings.text2cypher_temperature,
        )

        cypher_query = response.strip()
        cypher_query = self._clean_cypher_response(cypher_query)

        self.llm_response = cypher_query
        return cypher_query

    def _print_schema_debug(self, schema: dict[str, Any]) -> None:
        """Print schema for debugging purposes."""
        print("\n=== SCHEMA SEEN BY LLM ===")
        print("\nNode Labels and Properties:")

        for node_label, properties in schema['nodes'].items():
            clean_label = node_label.strip(":").strip("`")
            print(f"  Label: `{clean_label}`")
            print("  Properties:")
            for prop in properties:
                print(f"    - {prop}")
            print()

        print("Relationship Patterns:")
        for pattern in schema['relationship_patterns']:
            print(f"  {pattern}")
        print("\n===========================\n")

    def _add_limit_clause(self, cypher_query: str, k: int) -> str:
        """Add LIMIT clause to query if not present."""
        if "LIMIT" in cypher_query.upper():
            return cypher_query
        if "RETURN" in cypher_query.upper():
            return f"{cypher_query} LIMIT {k}"
        return cypher_query

    def _record_to_dict(self, record, cypher_query: str, index: int) -> dict[str, Any]:
        """Convert a Neo4j record to a result dictionary."""
        result_dict = {
            "score": 1.0,
            "generated_cypher": cypher_query,
            "result_index": index
        }

        for key, value in record.items():
            if not isinstance(value, dict):
                result_dict[key] = value
                continue

            # Extract properties from nested dicts (nodes/relationships)
            if hasattr(value, 'items'):
                for prop_key, prop_value in value.items():
                    result_dict[f"{key}_{prop_key}"] = prop_value
            else:
                result_dict[key] = str(value)

        return result_dict

    def _execute_cypher_query(
        self, session, cypher_query: str, k: int
    ) -> list[dict[str, Any]]:
        """Execute Cypher query and return results."""
        try:
            result = session.run(cypher_query)
            results = []

            for i, record in enumerate(result):
                if i >= k:
                    break
                results.append(self._record_to_dict(record, cypher_query, i))

            return self._add_metadata(results, 'text2cypher_llm')

        except Exception as e:
            error_results = [{
                "error": f"Generated Cypher query failed: {str(e)}",
                "generated_cypher": cypher_query
            }]
            return self._add_metadata(error_results, 'text2cypher_llm_error')

    async def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve results by generating and executing a Cypher query from the natural language query.

        Args:
            query: Natural language query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of dictionaries containing retrieved chunks with their metadata and scores
        """
        with self.driver.session(database=self.neo4j_database) as session:
            schema = self._get_schema(session)
            self._print_schema_debug(schema)

            cypher_query = await self._generate_cypher_query(query, schema)
            logger.info(f"Generated Cypher query:\n{cypher_query}")

            cypher_query = self._add_limit_clause(cypher_query, k)
            return self._execute_cypher_query(session, cypher_query, k)
