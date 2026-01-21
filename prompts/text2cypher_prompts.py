"""
Text2Cypher prompt templates for generating Cypher queries from natural language.
"""

from src.utils.prompt_utils import VersionedPrompt


TEXT2CYPHER_PROMPT_V1 = VersionedPrompt(
    prompt_version_name="TEXT2CYPHER_PROMPT_V1",
    prompt_tuple=(
"""You are an expert Neo4j Cypher query writer.

Convert the following natural language query into a valid Cypher query:
"{query}"

Database Schema:
{schema}


Return ONLY the Cypher query, no explanations or markdown formatting.

Cypher query:""",
    "query",
    "schema"
    )
)
