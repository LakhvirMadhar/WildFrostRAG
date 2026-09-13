"""Generation use cases (zero-shot vs. RAG) for WildFrostRAG.

These are the business-logic functions that decide *what* to ask the LLM —
whether to answer with no retrieved context (zero-shot) or with retrieved
context injected into the prompt (RAG). They orchestrate prompt formatting
and delegate the actual API call to the low-level client in
`wildfrost_rag.clients.openai_client`.
"""

from mlflow.entities.model_registry.prompt_version import PromptVersion

from wildfrost_rag.clients.openai_client import call_openai_api
from wildfrost_rag.domain.chat_role import ChatRole
from wildfrost_rag.prompts.prompt_utils import render_text_prompt


async def generate_zero_shot(query: str, system_prompt: PromptVersion) -> str:
    """Generate a zero-shot response (no context).

    Args:
        query: The user query
        system_prompt: The registered system prompt version

    Returns:
        The generated response text
    """
    return await call_openai_api(
        messages=[
            {"role": ChatRole.SYSTEM.value, "content": render_text_prompt(system_prompt)},
            {"role": ChatRole.USER.value, "content": query},
        ]
    )


async def generate_rag(
    query: str,
    context: str,
    system_prompt: PromptVersion,
    rag_prompt: PromptVersion,
) -> str:
    """Generate a RAG response using provided context.

    Args:
        query: The user query
        context: The retrieved context (concatenated chunks)
        system_prompt: The registered system prompt version
        rag_prompt: The registered prompt version for formatting the user message with context

    Returns:
        The generated response text
    """
    user_message = render_text_prompt(rag_prompt, query=query, context=context)
    return await call_openai_api(
        messages=[
            {"role": ChatRole.SYSTEM.value, "content": render_text_prompt(system_prompt)},
            {"role": ChatRole.USER.value, "content": user_message},
        ]
    )
