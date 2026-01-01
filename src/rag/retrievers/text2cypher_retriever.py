"""
Text2Cypher retriever for WildFrostRAG using LLM to generate Cypher queries.

This module implements retrieval by using an LLM to convert natural language
queries into Cypher queries based on the Neo4j schema.
"""

from typing import List, Dict, Any
from neo4j import GraphDatabase
from openai import OpenAI
from src.utils.config import settings


class Text2CypherRetriever:
    """
    Implements retrieval by using an LLM to generate Cypher queries from natural language.
    This simulates the Text2Cypher functionality by using an LLM to understand the schema
    and generate appropriate Cypher queries.
    """

    def __init__(self):
        """
        Initialize the Text2Cypher retriever.
        """
        self.uri = settings.neo4j_uri.get_secret_value()
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password.get_secret_value()
        self.client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self.model = settings.openai_model_name

    def _get_schema(self, session) -> Dict[str, Any]:
        """
        Get the schema of the Neo4j database.

        Args:
            session: Neo4j session

        Returns:
            Dictionary containing the database schema
        """
        # Get node labels and their properties
        node_query = """
        CALL db.labels() YIELD label
        WITH label
        CALL db.schema.nodeTypeProperties(label) YIELD nodeType, propertyName, propertyTypes, mandatory
        RETURN label, collect({propertyName: propertyName, propertyTypes: propertyTypes, mandatory: mandatory}) as properties
        """

        nodes_result = session.run(node_query)
        nodes = {}
        for record in nodes_result:
            label = record["label"]
            properties = record["properties"]
            nodes[label] = [prop["propertyName"] for prop in properties]

        # Get relationship types
        rel_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN collect(relationshipType) as relationships
        """

        rels_result = session.run(rel_query)
        relationships = rels_result.single()["relationships"] if rels_result.single() else []

        return {
            "nodes": nodes,
            "relationships": relationships
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
        schema_str = f"""
        Node labels and properties: {schema['nodes']}
        Relationship types: {schema['relationships']}
        """

        prompt = f"""
        You are an expert Neo4j Cypher query writer. Given the following Neo4j database schema:

        {schema_str}

        Convert the following natural language query into a valid Cypher query:
        "{natural_query}"

        Only return the Cypher query, nothing else. Make sure it's syntactically correct and appropriate for the schema.
        If the query is asking for information about cards, tribes, stats, or other entities in the WildFrost game,
        write a query that would return relevant information from the graph.

        The Cypher query:
        """

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
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        try:
            with driver.session() as session:
                # Get the current database schema
                schema = self._get_schema(session)

                # Generate a Cypher query from the natural language query
                cypher_query = self._generate_cypher_query(query, schema)

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
                            "search_type": "text2cypher_llm",
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

                    return results

                except Exception as e:
                    # If the generated query fails, return an error result
                    return [{
                        "error": f"Generated Cypher query failed: {str(e)}",
                        "generated_cypher": cypher_query,
                        "search_type": "text2cypher_llm_error"
                    }]

        finally:
            driver.close()