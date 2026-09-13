"""Validated result models for repository read methods.

Neo4j Cypher can return almost any shape (single nodes, multiple nodes,
scalars, LLM-generated queries), which is why `record_utils.record_to_dict()`
stays a generic flattener. These models sit one layer above it: each
repository method that reads a *specific, known* query shape builds one of
these before returning, so `dict[str, Any]`/raw tuples never leak past the
repository (see the `pydantic-boundaries` skill).
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field


class DocumentProperties(BaseModel):
    """Document node properties, minus embedding vectors.

    `text` is required: both `DocumentRepository.load_all_documents()` (via
    its `WHERE d.text IS NOT NULL` clause) and the vector/fulltext indexes
    only ever surface Documents that have text. `source_file`/`title`/
    `source_url` are set on ingestion (see `ingest_documents_into_neo4j`),
    but modeled optional because `record_to_dict()` silently drops any
    property that's `None` or absent - an older or partially-ingested node
    could legitimately be missing one.
    """

    text: str
    source_file: str | None = None
    title: str | None = None
    source_url: str | None = None


class DocumentSearchResult(DocumentProperties):
    """A single hit from a Document vector/fulltext index search: node + score."""

    score: float


class GraphTraversalResult(BaseModel):
    """One row from a Document index search enriched with graph-traversal data.

    Produced by `GRAPH_TRAVERSAL_QUERY` (see `repositories/traversal_patterns.py`)
    piped from either a vector or fulltext index lookup. `record_to_dict()`
    prefixes every non-`node` Cypher variable, and the traversal query's Document
    variable is `doc`, so its properties arrive as `doc_text`, `doc_source_file`,
    etc.

    The owning entity (Card/Bell/Charm/Fight/Crown/Stat/Zone/MapEvent/Shop) and
    its single-node traversals (Tribe/CardType) are deliberately NOT enumerated
    field-by-field here: each entity type has its own, evolving property set
    (see `data/schemas/`), so hardcoding them would duplicate that schema and
    break on every card-type change. Their flattened `<prefix>_<property>` keys
    land in `entity_fields` instead - everything left over once the fixed
    doc/score/list fields below are accounted for.
    """

    doc_text: str
    doc_source_file: str | None = None
    doc_title: str | None = None
    doc_source_url: str | None = None
    score: float
    keywords: list[str] = Field(default_factory=list)
    stats: list[str] = Field(default_factory=list)
    summons: list[str] = Field(default_factory=list)
    transforms_into: list[str] = Field(default_factory=list)
    can_recruit_as: list[str] = Field(default_factory=list)
    fight_enemies: list[str] = Field(default_factory=list)
    bell_charms: list[str] = Field(default_factory=list)
    bell_adds_cards: list[str] = Field(default_factory=list)
    bell_grants_keywords: list[str] = Field(default_factory=list)
    entity_fields: dict[str, Any] = Field(default_factory=dict)

    _LIST_FIELDS: ClassVar[tuple[str, ...]] = (
        "keywords",
        "stats",
        "summons",
        "transforms_into",
        "can_recruit_as",
        "fight_enemies",
        "bell_charms",
        "bell_adds_cards",
        "bell_grants_keywords",
    )

    @classmethod
    def from_flat_dict(cls, flat: dict[str, Any]) -> "GraphTraversalResult":
        """Build from the flat dict `record_to_dict()` produces for this query shape."""
        known_keys = {"doc_text", "doc_source_file", "doc_title", "doc_source_url", "score"}
        known_keys.update(cls._LIST_FIELDS)
        entity_fields = {k: v for k, v in flat.items() if k not in known_keys}
        return cls(
            doc_text=flat["doc_text"],
            doc_source_file=flat.get("doc_source_file"),
            doc_title=flat.get("doc_title"),
            doc_source_url=flat.get("doc_source_url"),
            score=flat["score"],
            keywords=flat.get("keywords", []),
            stats=flat.get("stats", []),
            summons=flat.get("summons", []),
            transforms_into=flat.get("transforms_into", []),
            can_recruit_as=flat.get("can_recruit_as", []),
            fight_enemies=flat.get("fight_enemies", []),
            bell_charms=flat.get("bell_charms", []),
            bell_adds_cards=flat.get("bell_adds_cards", []),
            bell_grants_keywords=flat.get("bell_grants_keywords", []),
            entity_fields=entity_fields,
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """Reconstruct the original flat shape for the generic result-formatting pipeline.

        `BaseNeo4jRetriever._add_metadata()` / `_format_result_as_text()` and
        `RetrievedChunk.from_raw_retriever_dict()` are intentionally generic -
        they format and archive whatever a retriever's Cypher happened to
        return, including future entity-specific fields. This is the
        adaptation step back to that flat shape, called once per search, right
        after the repository (the boundary) has already validated the fixed
        fields above.
        """
        flat: dict[str, Any] = {
            "doc_text": self.doc_text,
            "score": self.score,
        }
        if self.doc_source_file is not None:
            flat["doc_source_file"] = self.doc_source_file
        if self.doc_title is not None:
            flat["doc_title"] = self.doc_title
        if self.doc_source_url is not None:
            flat["doc_source_url"] = self.doc_source_url
        for field_name in self._LIST_FIELDS:
            value = getattr(self, field_name)
            if value:
                flat[field_name] = value
        flat.update(self.entity_fields)
        return flat


class MissingEmbeddingDocument(BaseModel):
    """A Document node still missing a given embedding property.

    Replaces the anonymous `(text, element_id)` tuple `VectorRepository`
    used to return - both fields have clear domain meaning (the text to
    embed, and the Neo4j element id to write the resulting vector back to),
    so a bare tuple was primitive obsession.
    """

    text: str
    element_id: str
