"""
Text2Cypher prompt templates for generating Cypher queries from natural language.

Version history:
- V1: Minimal baseline prompt with schema and query only
"""

TEXT2CYPHER_PROMPT_V1 = (
"""You are an expert Neo4j Cypher query writer.

Database Schema:
{schema}

Convert the following natural language query into a valid Cypher query:
"{query}"

Return ONLY the Cypher query, no explanations or markdown formatting.

Cypher query:""",
    "schema",
    "query"
)
