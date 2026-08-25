"""Taxonomy prompt templates for generating axial codes from open codes."""

from wildfrost_rag.prompts.prompt_utils import VersionedPrompt


TAXONOMY_SYSTEM_PROMPT_V1 = VersionedPrompt(
    prompt_version_name="TAXONOMY_SYSTEM_PROMPT_V1",
    prompt_tuple=(
        """You are an expert at qualitative coding analysis, specifically creating axial codes from open codes.

You will be given a numbered list of open codes that describe various failure modes from an LLM evaluation.

Your task is to perform axial coding:
1. Analyze all the open codes
2. Identify common themes and patterns across the codes
3. Create higher-level axial codes (categories) that group related open codes
4. Each axial code should represent a broader conceptual category
5. Provide clear definitions for each axial code
6. Reference which open codes (by number) fall under each axial code

Format your response as a well-structured markdown document with:
- A title: "Failure Mode Taxonomy - Axial Codes"
- A brief introduction explaining the coding approach
- Axial codes as H2 headers (##)
- Sub-categories as H3 headers (###) if needed
- Clear descriptions of each axial code category
- List the relevant open code numbers that fall under each axial code
- A summary section with key insights

Be comprehensive but concise. Make the taxonomy useful for understanding and addressing these failure modes.""",
    ),
)


TAXONOMY_USER_PROMPT_V1 = VersionedPrompt(
    prompt_version_name="TAXONOMY_USER_PROMPT_V1",
    prompt_tuple=(
        """Here are the open codes from the failure analysis:

{codes_text}

Please create axial codes that group these open codes into higher-level categories. Reference the open codes by their numbers.""",
        "codes_text",
    ),
)
