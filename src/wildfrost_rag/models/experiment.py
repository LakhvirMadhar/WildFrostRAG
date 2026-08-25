"""Typed schemas for experiment registry records.

These models define what ExperimentRegistry's list/search methods return -
what gets read back from outputs/experiments.yaml.
"""

from typing import Any, Literal, Self

from pydantic import BaseModel


class RetrievalRecord(BaseModel):
    """A single retrieval experiment as stored in the registry."""

    reference: str
    timestamp: str | None = None
    retriever_type: str | None = None
    chunking: bool | None = None
    description: str = ""
    total_queries: int | None = None
    successful_queries: int | None = None
    run_num: int | None = None
    type: Literal["retrieval"] = "retrieval"

    @classmethod
    def from_registry_data(
        cls, reference: str, data: dict[str, Any], run_num: int | None = None
    ) -> Self:
        """Build a RetrievalRecord from a raw registry.yaml entry."""
        return cls(
            reference=reference,
            timestamp=data.get("timestamp"),
            retriever_type=data.get("retriever_type"),
            chunking=data.get("chunking"),
            description=data.get("description", ""),
            total_queries=data.get("total_queries"),
            successful_queries=data.get("successful_queries"),
            run_num=run_num,
        )


class GenerationRecord(BaseModel):
    """A single generation experiment as stored in the registry."""

    reference: str
    timestamp: str | None = None
    retrieval_reference: str | None = None
    system_prompt_version: str | None = None
    description: str = ""
    total_queries: int | None = None
    successful_queries: int | None = None
    run_num: int | None = None
    type: Literal["generation"] = "generation"

    @classmethod
    def from_registry_data(
        cls, reference: str, data: dict[str, Any], run_num: int | None = None
    ) -> Self:
        """Build a GenerationRecord from a raw registry.yaml entry."""
        return cls(
            reference=reference,
            timestamp=data.get("timestamp"),
            retrieval_reference=data.get("retrieval_reference"),
            system_prompt_version=data.get("system_prompt_version"),
            description=data.get("description", ""),
            total_queries=data.get("total_queries"),
            successful_queries=data.get("successful_queries"),
            run_num=run_num,
        )
