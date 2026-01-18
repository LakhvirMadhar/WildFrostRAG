"""
Text2Cypher retriever for WildFrostRAG using LLM to generate Cypher queries.

This module implements retrieval by using an LLM to convert natural language
queries into Cypher queries based on the Neo4j schema.
"""

from typing import List, Dict, Any, Optional
from neo4j import Driver
from openai import OpenAI
from src.utils.config import settings
from src.utils.logger import logger
from src.utils.utils import format_prompt_tuple
from src.rag.retrievers.base_neo4j_retriever import BaseNeo4jRetriever
from prompts.text2cypher_prompts import TEXT2CYPHER_PROMPT_V1


class Text2CypherRetriever(BaseNeo4jRetriever):
    """
    Implements retrieval by using an LLM to generate Cypher queries from natural language.
    This simulates the Text2Cypher functionality by using an LLM to understand the schema
    and generate appropriate Cypher queries.
    """

    def __init__(self, driver: Driver, neo4j_database: Optional[str] = None):
        """
        Initialize the Text2Cypher retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            neo4j_database: Optional database name (default: None uses default database)
        """
        super().__init__(driver, neo4j_database)
        self.client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self.model = settings.openai_model_name

    def _get_schema(self, session) -> Dict[str, Any]:
        """
        Get the schema of the Neo4j database with relationship directions.

        Args:
            session: Neo4j session

        Returns:
            Dictionary containing the database schema with node properties and relationship patterns
        """
        # Get all node properties schema
        node_query = """
        CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName, propertyTypes, mandatory
        RETURN nodeType, collect({propertyName: propertyName, propertyTypes: propertyTypes, mandatory: mandatory}) as properties
        """

        nodes_result = session.run(node_query)
        nodes = {}
        for record in nodes_result:
            node_type = record["nodeType"]
            properties = record["properties"]
            nodes[node_type] = [prop["propertyName"] for prop in properties]

        # Get relationship patterns with directions using schema visualization
        schema_viz_query = """
        CALL db.schema.visualization()
        """

        viz_result = session.run(schema_viz_query)
        viz_record = viz_result.single()

        # Debug: print what we got
        print(f"DEBUG viz_record: {viz_record}")
        print(f"DEBUG viz_record keys: {list(viz_record.keys()) if viz_record else 'None'}")

        relationship_patterns = []
        if viz_record:
            # Neo4j Record objects don't support 'in' operator reliably, access directly
            relationships = viz_record.get("relationships", [])
            print(f"DEBUG: Found {len(relationships)} relationships")
            for rel in relationships:
                # Extract start and end node labels from the relationship
                start_node, end_node = rel.nodes
                start_label = list(start_node.labels)[0] if start_node.labels else "Unknown"
                end_label = list(end_node.labels)[0] if end_node.labels else "Unknown"
                rel_type = rel.type

                # Format as: (StartLabel)-[:REL_TYPE]->(EndLabel)
                pattern = f"({start_label})-[:{rel_type}]->({end_label})"
                print(f"DEBUG: Added pattern: {pattern}")
                relationship_patterns.append(pattern)

        return {
            "nodes": nodes,
            "relationship_patterns": relationship_patterns
        }

    def _generate_cypher_query(self, natural_query: str, schema: Dict[str, Any]) -> str:
        """
        Generate a Cypher query from a natural language query using an LLM.

        Args:
            natural_query: Natural language query
            schema: Database schema

        Returns:
            Generated Cypher query string
        """
        # Format schema information with clean formatting
        nodes_str = ""
        for node_label, properties in schema['nodes'].items():
            # Extract label name without Neo4j syntax (remove :` and `)
            clean_label = node_label.strip(":").strip("`")
            nodes_str += f"\n  Label: `{clean_label}`\n  Properties:\n"
            for prop in properties:
                nodes_str += f"    - {prop}\n"

        patterns_str = "\n".join(f"  {pattern}" for pattern in schema['relationship_patterns'])

        schema_str = f"""Node labels and their properties:{nodes_str}
        Relationship patterns (with directions):
        {patterns_str}"""

        # Use versioned prompt template with utility function
        prompt = format_prompt_tuple(
            TEXT2CYPHER_PROMPT_V1,
            schema=schema_str,
            query=natural_query
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # Deterministic output
        )

        cypher_query = response.choices[0].message.content.strip()

        # Clean up the response if it contains markdown formatting
        if cypher_query.startswith("```cypher") or cypher_query.startswith("```"):
            # Remove the first and last lines if they contain markdown
            lines = cypher_query.split('\n')
            if lines[0].strip().startswith('```'):
                lines = lines[1:]
            if lines[-1].strip().startswith('```'):
                lines = lines[:-1]
            cypher_query = '\n'.join(lines).strip()

        return cypher_query

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve results by generating and executing a Cypher query from the natural language query.

        Args:
            query: Natural language query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of dictionaries containing retrieved chunks with their metadata and scores
        """
        with self.driver.session(database=self.neo4j_database) as session:
            # Get the current database schema
            schema = self._get_schema(session)

            # Print schema for debugging (not logged, just printed)
            print("\n=== SCHEMA SEEN BY LLM ===")
            print("\nNode Labels and Properties:")
            for node_label, properties in schema['nodes'].items():
                # Extract label name without Neo4j syntax (remove :` and `)
                clean_label = node_label.strip(":").strip("`")
                print(f"  Label: `{clean_label}`")
                print(f"  Properties:")
                for prop in properties:
                    print(f"    - {prop}")
                print()  # Empty line between labels

            print("Relationship Patterns:")
            for pattern in schema['relationship_patterns']:
                print(f"  {pattern}")
            print("\n===========================\n")

            # Generate a Cypher query from the natural language query
            cypher_query = self._generate_cypher_query(query, schema)
            logger.info(f"Generated Cypher query:\n{cypher_query}")

            # Execute the generated Cypher query
            # We need to modify the query to limit results and return appropriate data
            # If the query doesn't already have a LIMIT clause, add one
            if "LIMIT" not in cypher_query.upper():
                # Try to find where we might add a LIMIT clause
                # This is a simple approach - in practice, you'd want more sophisticated query parsing
                if "RETURN" in cypher_query.upper():
                    cypher_query += f" LIMIT {k}"

            try:
                result = session.run(cypher_query)

                results = []
                for i, record in enumerate(result):
                    if i >= k:
                        break

                    # Convert the result record to a dictionary
                    result_dict = {
                        "score": 1.0,  # Generated queries don't have a natural score
                        "generated_cypher": cypher_query,
                        "result_index": i
                    }

                    # Add all properties from the result record
                    for key, value in record.items():
                        if isinstance(value, dict):
                            # If the value is a node or relationship, extract its properties
                            if hasattr(value, 'items'):
                                for prop_key, prop_value in value.items():
                                    result_dict[f"{key}_{prop_key}"] = prop_value
                            else:
                                result_dict[key] = dict(value) if hasattr(value, 'items') else str(value)
                        else:
                            result_dict[key] = value

                    results.append(result_dict)

                return self._add_metadata(results, 'text2cypher_llm')

            except Exception as e:
                # If the generated query fails, return an error result
                error_results = [{
                    "error": f"Generated Cypher query failed: {str(e)}",
                    "generated_cypher": cypher_query
                }]
                return self._add_metadata(error_results, 'text2cypher_llm_error')