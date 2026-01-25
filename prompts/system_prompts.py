"""
System prompt templates for LLM generation.
"""

from src.utils.prompt_utils import VersionedPrompt


SYSTEM_PROMPT_V1 = VersionedPrompt(
    prompt_version_name="SYSTEM_PROMPT_V1",
    prompt_tuple=(
"""You are a helpful assistant that answers questions about the rougelike deckbuilder video game Wildfrost.""",
    )
)

SYSTEM_PROMPT_V2 = VersionedPrompt(
    prompt_version_name="SYSTEM_PROMPT_V2",
    prompt_tuple=(
"""You are a helpful assistant that answers questions about the rougelike deckbuilder video game Wildfrost.

Here's some more information about the game:
Wildfrost is a 2023 roguelike deck-building game developed by Deadpan Games and Gaziter and published by Chucklefish. Wildfrost is set in a frozen world where the sun has succumbed to an eternal winter known as the Wildfrost. Players take on the role of a tribe leader from the last surviving town, Snowdwell, and engage in turn-based card battles against waves of enemies including unique monsters and bosses. The gameplay features strategic positioning of up to six allied companions and six enemies on a battlefield, with an initiative system determining turn order. Players recruit adventurers, collect and upgrade cards, and customize companions with special charms to enhance their power. Between battles, players build up the town to unlock new cards, tribes, and events, aiming to ultimately reach the Sun Temple to end the endless frost.""",
    )
)


RAG_PROMPT_V1 = VersionedPrompt(
    prompt_version_name="RAG_PROMPT_V1",
    prompt_tuple=(
"""Use the following context from the Wildfrost wiki to answer the question.
Query: {query}
Context: {context}""",
    "query",
    "context"
    )
)
