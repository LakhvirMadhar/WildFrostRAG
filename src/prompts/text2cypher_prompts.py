"""
Text2Cypher prompt templates for generating Cypher queries from natural language.
"""

from prompts.prompt_utils import VersionedPrompt


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


TEXT2CYPHER_PROMPT_V2 = VersionedPrompt(
    prompt_version_name="TEXT2CYPHER_PROMPT_V2",
    prompt_tuple=(
"""You are an expert Neo4j Cypher query writer.

Convert the following natural language query into a valid Cypher query:
"{query}"

Database Schema:
{schema}

Rules:
1. Use the graph structure to identify the relevant entity nodes, then traverse to their Document nodes via the HAS_DOCUMENT relationship.
2. Always return the Document node as `RETURN d AS node`.

Example:
Question: "What tribe does Pyra belong to?"
MATCH (c:Card {{card_name: "Pyra"}})-[:HAS_DOCUMENT]->(d:Document)
RETURN d AS node

Return ONLY the Cypher query, no explanations or markdown formatting.

Cypher query:""",
    "query",
    "schema"
    )
)


TEXT2CYPHER_PROMPT_V3 = VersionedPrompt(
    prompt_version_name="TEXT2CYPHER_PROMPT_V3",
    prompt_tuple=(
"""You are an expert Neo4j Cypher query writer for a Wildfrost game knowledge base.

Convert this natural language query into a Cypher query:
"{query}"

Database Schema:
{schema}

Rules:
1. Every answer lives in a Document node. Always traverse to it via HAS_DOCUMENT and return `RETURN d AS node`.
2. When the question mentions a card by name, match the Card node and return its Document. This is the most common pattern.
3. When the question mentions a bell, charm, fight, or other entity by name, match that entity node and return its Document.
4. Prefer simple queries. If the question asks about a specific named entity, just find that entity's Document — do NOT add extra filters on d.text or traverse through intermediate nodes.
5. Use case-insensitive matching with toLower() for card_name and entity names, since users may not capitalize exactly.
6. Use CONTAINS for partial name matching when a name might be slightly different (e.g., "Tinkerson Jr." vs "Tinkerson Jr").

Examples:
Question: "What tribe does Snoof belong to?"
MATCH (c:Card) WHERE toLower(c.card_name) = toLower("Snoof")
MATCH (c)-[:HAS_DOCUMENT]->(d:Document)
RETURN d AS node

Question: "What are Gobbler's stats?"
MATCH (c:Card) WHERE toLower(c.card_name) = toLower("Gobbler")
MATCH (c)-[:HAS_DOCUMENT]->(d:Document)
RETURN d AS node

Question: "What does Frenzy Bell do?"
MATCH (b:Bell) WHERE toLower(b.name) = toLower("Frenzy Bell")
MATCH (b)-[:HAS_DOCUMENT]->(d:Document)
RETURN d AS node

Return ONLY the Cypher query, no explanations or markdown formatting.

Cypher query:""",
    "query",
    "schema"
    )
)
