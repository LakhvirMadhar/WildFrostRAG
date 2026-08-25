"""Hybrid retrievers for WildFrostRAG using Reciprocal Rank Fusion (RRF).

This module implements hybrid retrieval by combining multiple retrieval methods
using Reciprocal Rank Fusion to produce a single ranked result list.

Available hybrid retrievers:
- BM25VectorHybridRetriever: BM25 + Vector
- FulltextVectorHybridRetriever: Fulltext + Vector
- BM25FulltextVectorHybridRetriever: BM25 + Fulltext + Vector
- Text2CypherVectorHybridRetriever: Text2Cypher + Vector (async, with fallback)
"""

from typing import Any
from collections.abc import Callable
from neo4j import Driver
from pydantic import BaseModel, Field
from core.exceptions import WildFrostRAGError
from models.retrieval import RetrievedChunk
from rag.retrievers.neo4j_vector_search import Neo4jVectorSearch
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.neo4j_fulltext_search import Neo4jFullTextSearch
from rag.retrievers.text2cypher_retriever import Text2CypherRetriever
from utils.config import get_settings
from utils.logger import logger
from prompts.prompt_utils import VersionedPrompt


class RRFScore(BaseModel):
    """RRF bookkeeping for one document as it's fused across retrievers.

    Transient - built and consumed entirely within _apply_rrf(), never
    serialized to disk (unlike models/retrieval.py's RetrievedChunk).
    """

    rrf_score: float
    chunk: RetrievedChunk
    source_retriever: str
    retriever_scores: dict[str, float] = Field(default_factory=dict)


class HybridRetriever:
    """Implements hybrid retrieval using Reciprocal Rank Fusion (RRF).

    Combines results from multiple retrieval methods.
    """

    def __init__(
        self,
        retrievers: list[Any],
        retriever_names: list[str],
        weights: list[float] | None = None,
        k1: int = 60,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            retrievers: List of retriever instances to combine
            retriever_names: List of names for each retriever (for identification)
            weights: Optional list of weights for each retriever (default: equal weights)
            k1: Smoothing parameter for RRF (default: 60)
        """
        self.retrievers = retrievers
        self.retriever_names = retriever_names
        self.k1 = k1
        self.last_individual_results: dict[str, list[RetrievedChunk]] | None = None

        if weights is None:
            # Default to equal weights
            self.weights = [1.0] * len(retrievers)
        else:
            if len(weights) != len(retrievers):
                raise ValueError("Number of weights must match number of retrievers")
            self.weights = weights

        if len(retrievers) != len(retriever_names):
            raise ValueError("Number of retrievers must match number of retriever names")

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Retrieve results using multiple retrievers and combine them with RRF.

        Args:
            query: Input query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of typed RetrievedChunk objects, fused via RRF
        """
        # Get results from each retriever
        all_results: list[tuple[list[RetrievedChunk], float]] = []
        individual_results: dict[str, list[RetrievedChunk]] = {}

        for i, (retriever, name) in enumerate(
            zip(self.retrievers, self.retriever_names, strict=False)
        ):
            results = retriever.search(query, k=k * 2)  # Get more results to allow for fusion
            all_results.append((results, self.weights[i]))
            individual_results[name] = results

        # Apply RRF to combine results
        fused_results = self._apply_rrf(all_results, k)

        # Store individual results as instance variable for experiment tracking
        # This allows evaluate_retrievers.py to save them separately
        self.last_individual_results = individual_results

        return fused_results

    def _apply_rrf(
        self, all_results: list[tuple[list[RetrievedChunk], float]], k: int
    ) -> list[RetrievedChunk]:
        """Apply Reciprocal Rank Fusion to combine results from multiple retrievers.

        Args:
            all_results: List of (results, weight) tuples from each retriever
            k: Number of top results to return

        Returns:
            List of fused RetrievedChunk objects sorted by RRF score
        """
        # Map document identifier -> RRF bookkeeping for that document
        doc_scores: dict[str, RRFScore] = {}

        for retriever_idx, (results, weight) in enumerate(all_results):
            for rank, chunk in enumerate(results, 1):  # RRF uses 1-based ranking
                doc_identifier = self._identify_doc(chunk)

                # Calculate RRF score: weight * 1 / (k1 + rank)
                rrf_score = weight * 1.0 / (self.k1 + rank)
                retriever_name = self.retriever_names[retriever_idx]

                if doc_identifier not in doc_scores:
                    doc_scores[doc_identifier] = RRFScore(
                        rrf_score=rrf_score,
                        chunk=chunk,
                        source_retriever=retriever_name,
                        retriever_scores={retriever_name: chunk.score},
                    )
                else:
                    # Handle duplicates: accumulate RRF score, keep the
                    # chunk/source from the last-seen occurrence across methods
                    existing = doc_scores[doc_identifier]
                    existing.rrf_score += rrf_score
                    existing.retriever_scores[retriever_name] = chunk.score
                    existing.chunk = chunk
                    existing.source_retriever = retriever_name

        # Sort by RRF score in descending order, return the top k as RetrievedChunks
        sorted_docs = sorted(doc_scores.values(), key=lambda s: s.rrf_score, reverse=True)
        return [self._to_fused_chunk(score) for score in sorted_docs[:k]]

    def _identify_doc(self, chunk: RetrievedChunk) -> str:
        """Build a document identifier for RRF deduplication across retrievers."""
        text_content = chunk.cypher_result.get("text", "") or chunk.retrieved_text
        source_file = chunk.cypher_result.get("source_file", "")
        if not text_content:
            return str(hash(str(chunk)))
        return f"{text_content[:50]}_{source_file}"

    def _to_fused_chunk(self, score: RRFScore) -> RetrievedChunk:
        """Build the final fused RetrievedChunk for one document's RRFScore."""
        cypher_result = dict(score.chunk.cypher_result)
        cypher_result["rrf_score"] = score.rrf_score
        cypher_result["source_retriever"] = score.source_retriever
        cypher_result["retriever_scores"] = score.retriever_scores
        return RetrievedChunk(
            score=score.rrf_score,
            search_type="hybrid_rrf",
            retrieved_text=score.chunk.retrieved_text,
            source_url=score.chunk.source_url,
            cypher_result=cypher_result,
        )


class BM25VectorHybridRetriever(HybridRetriever):
    """A specific hybrid retriever that combines BM25 search and vector search."""

    def __init__(
        self,
        driver: Driver,
        embed_fn: Callable[[str], list[float]],
        neo4j_database: str | None = None,
        index_name: str | None = None,
    ) -> None:
        """Initialize the BM25 and vector hybrid retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            embed_fn: Query embedding function for vector search
            neo4j_database: Optional database name (default: None uses default database)
            index_name: Optional vector index name (default: uses settings.embedding.vector_index_name)
        """
        settings = get_settings()
        bm25_retriever = BM25Retriever(driver, neo4j_database)
        vector_retriever = Neo4jVectorSearch(
            driver,
            embed_fn,
            neo4j_database,
            index_name=index_name or settings.embedding.vector_index_name,
        )

        super().__init__(
            retrievers=[bm25_retriever, vector_retriever],
            retriever_names=["bm25", "vector"],
            weights=[1.0, 1.0],  # Equal weights for both methods
            k1=settings.embedding.rrf_k1,
        )


class FulltextVectorHybridRetriever(HybridRetriever):
    """A specific hybrid retriever that combines fulltext search and vector search."""

    def __init__(
        self,
        driver: Driver,
        embed_fn: Callable[[str], list[float]],
        neo4j_database: str | None = None,
        index_name: str | None = None,
        remove_stopwords: bool = False,
    ) -> None:
        """Initialize the fulltext and vector hybrid retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            embed_fn: Query embedding function for vector search
            neo4j_database: Optional database name (default: None uses default database)
            index_name: Optional vector index name (default: uses settings.embedding.vector_index_name)
            remove_stopwords: Whether to remove stop words from fulltext queries
        """
        settings = get_settings()
        fulltext_retriever = Neo4jFullTextSearch(
            driver, neo4j_database, remove_stopwords=remove_stopwords
        )
        vector_retriever = Neo4jVectorSearch(
            driver,
            embed_fn,
            neo4j_database,
            index_name=index_name or settings.embedding.vector_index_name,
        )

        super().__init__(
            retrievers=[fulltext_retriever, vector_retriever],
            retriever_names=["fulltext", "vector"],
            weights=[1.0, 1.0],  # Equal weights for both methods
            k1=settings.embedding.rrf_k1,
        )


class BM25FulltextVectorHybridRetriever(HybridRetriever):
    """A specific hybrid retriever that combines BM25, fulltext, and vector search."""

    def __init__(
        self,
        driver: Driver,
        embed_fn: Callable[[str], list[float]],
        neo4j_database: str | None = None,
        index_name: str | None = None,
    ) -> None:
        """Initialize the BM25, fulltext, and vector hybrid retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            embed_fn: Query embedding function for vector search
            neo4j_database: Optional database name (default: None uses default database)
            index_name: Optional vector index name (default: uses settings.embedding.vector_index_name)
        """
        settings = get_settings()
        bm25_retriever = BM25Retriever(driver, neo4j_database)
        fulltext_retriever = Neo4jFullTextSearch(driver, neo4j_database)
        vector_retriever = Neo4jVectorSearch(
            driver,
            embed_fn,
            neo4j_database,
            index_name=index_name or settings.embedding.vector_index_name,
        )

        super().__init__(
            retrievers=[bm25_retriever, fulltext_retriever, vector_retriever],
            retriever_names=["bm25", "fulltext", "vector"],
            weights=[1.0, 1.0, 1.0],  # Equal weights for all methods
            k1=settings.embedding.rrf_k1,
        )


class Text2CypherVectorHybridRetriever(HybridRetriever):
    """Hybrid retriever combining Text2Cypher with Vector search.

    Key features:
    - Text2Cypher provides precise structured queries when it works
    - Vector search provides semantic fallback
    - Falls back to vector-only if Text2Cypher fails
    - Uses RRF fusion when both succeed
    """

    def __init__(
        self,
        driver: Driver,
        embed_fn: Callable[[str], list[float]],
        text2cypher_prompt: VersionedPrompt,
        neo4j_database: str | None = None,
        index_name: str | None = None,
    ) -> None:
        """Initialize the Text2Cypher + Vector hybrid retriever.

        Args:
            driver: Neo4j driver instance
            embed_fn: Query embedding function for vector search
            text2cypher_prompt: VersionedPrompt for Text2Cypher LLM
            neo4j_database: Optional database name
            index_name: Vector index name (default: from settings)
        """
        settings = get_settings()

        # Create component retrievers
        self.text2cypher = Text2CypherRetriever(driver, text2cypher_prompt, neo4j_database)
        self.vector = Neo4jVectorSearch(
            driver,
            embed_fn,
            neo4j_database,
            index_name=index_name or settings.embedding.vector_index_name,
        )

        # Store for config tracking
        self.text2cypher_prompt_version = text2cypher_prompt.prompt_version_name

        # Initialize parent HybridRetriever with equal weights
        super().__init__(
            retrievers=[self.text2cypher, self.vector],
            retriever_names=["text2cypher", "vector"],
            weights=[1.0, 1.0],
            k1=settings.embedding.rrf_k1,
        )

    async def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:  # type: ignore[override]
        """Search using Text2Cypher + Vector with RRF fusion.

        Override parent to handle async Text2Cypher and provide fallback.

        Args:
            query: Natural language query
            k: Number of results to return

        Returns:
            List of fused RetrievedChunk objects (or vector-only if Text2Cypher fails)
        """
        all_results: list[tuple[list[RetrievedChunk], float]] = []
        individual_results: dict[str, list[RetrievedChunk]] = {}

        # Try Text2Cypher (async) with error handling
        text2cypher_success = False
        try:
            text2cypher_results = await self.text2cypher.search(query, k=k * 2)
            text2cypher_success = True
            all_results.append((text2cypher_results, self.weights[0]))
            individual_results["text2cypher"] = text2cypher_results
        except WildFrostRAGError as e:
            logger.warning(f"Text2Cypher failed, falling back to vector-only: {e}")
            individual_results["text2cypher"] = []

        # Vector search (sync) - always run
        vector_results = self.vector.search(query, k=k * 2)
        all_results.append((vector_results, self.weights[1]))
        individual_results["vector"] = vector_results

        # Apply RRF if we have multiple result sets, otherwise just return vector
        if len(all_results) > 1:
            fused_results = self._apply_rrf(all_results, k)
        else:
            # Vector-only fallback
            fused_results = [
                RetrievedChunk(
                    score=chunk.score,
                    search_type="text2cypher_vector_fallback",
                    retrieved_text=chunk.retrieved_text,
                    source_url=chunk.source_url,
                    cypher_result=chunk.cypher_result,
                )
                for chunk in vector_results[:k]
            ]

        # Store individual results for experiment tracking
        self.last_individual_results = individual_results
        self.text2cypher_success = text2cypher_success

        return fused_results
