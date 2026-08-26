"""Generation use cases (zero-shot vs. RAG) for WildFrostRAG.

These are the business-logic functions that decide *what* to ask the LLM —
whether to answer with no retrieved context (zero-shot) or with retrieved
context injected into the prompt (RAG). They orchestrate prompt formatting
and delegate the actual API call to the low-level client in
`wildfrost_rag.clients.openai_client`.
"""

from wildfrost_rag.clients.openai_client import call_openai_api
from wildfrost_rag.prompts.prompt_utils import VersionedPrompt, format_prompt_tuple


async def generate_zero_shot(query: str, system_prompt: VersionedPrompt) -> str:
    """Generate a zero-shot response (no context).

    Args:
        query: The user query
        system_prompt: VersionedPrompt containing the system prompt

    Returns:
        The generated response text
    """
    return await call_openai_api(
        messages=[
            {"role": "system", "content": system_prompt.prompt_tuple[0]},
            {"role": "user", "content": query},
        ]
    )


async def generate_rag(
    query: str,
    context: str,
    system_prompt: VersionedPrompt,
    rag_prompt: VersionedPrompt,
) -> str:
    """Generate a RAG response using provided context.

    Args:
        query: The user query
        context: The retrieved context (concatenated chunks)
        system_prompt: VersionedPrompt containing the system prompt
        rag_prompt: VersionedPrompt for formatting the user message with context

    Returns:
        The generated response text
    """
    user_message = format_prompt_tuple(rag_prompt.prompt_tuple, query=query, context=context)
    return await call_openai_api(
        messages=[
            {"role": "system", "content": system_prompt.prompt_tuple[0]},
            {"role": "user", "content": user_message},
        ]
    )
