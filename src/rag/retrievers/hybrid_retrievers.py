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
from rag.retrievers.neo4j_vector_search import Neo4jVectorSearch
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.neo4j_fulltext_search import Neo4jFullTextSearch
from rag.retrievers.text2cypher_retriever import Text2CypherRetriever
from utils.config import settings
from utils.logger import logger
from prompts.prompt_utils import VersionedPrompt


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
        self.last_individual_results: dict[str, Any] | None = None

        if weights is None:
            # Default to equal weights
            self.weights = [1.0] * len(retrievers)
        else:
            if len(weights) != len(retrievers):
                raise ValueError("Number of weights must match number of retrievers")
            self.weights = weights

        if len(retrievers) != len(retriever_names):
            raise ValueError("Number of retrievers must match number of retriever names")

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieve results using multiple retrievers and combine them with RRF.

        Args:
            query: Input query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of dictionaries containing retrieved chunks with their metadata and scores
        """
        # Get results from each retriever
        all_results = []
        individual_results = {}  # Store individual retriever results for analysis

        for i, (retriever, name) in enumerate(
            zip(self.retrievers, self.retriever_names, strict=False)
        ):
            results = retriever.search(query, k=k * 2)  # Get more results to allow for fusion
            # Add source information to help distinguish results during testing
            for result in results:
                result["source_retriever"] = name
            all_results.append((results, self.weights[i]))
            individual_results[name] = results

        # Apply RRF to combine results
        fused_results = self._apply_rrf(all_results, k)

        # Store individual results as instance variable for experiment tracking
        # This allows evaluate_retrievers.py to save them separately
        self.last_individual_results = individual_results

        return fused_results

    def _apply_rrf(
        self, all_results: list[tuple[list[dict[str, Any]], float]], k: int
    ) -> list[dict[str, Any]]:
        """Apply Reciprocal Rank Fusion to combine results from multiple retrievers.

        Args:
            all_results: List of (results, weight) tuples from each retriever
            k: Number of top results to return

        Returns:
            List of fused results sorted by RRF score
        """
        # Create a mapping from document text to RRF score and metadata
        doc_scores: dict[str, dict[str, Any]] = {}

        for retriever_idx, (results, weight) in enumerate(all_results):
            for rank, doc in enumerate(results, 1):  # RRF uses 1-based ranking
                # Use a combination of text and other identifiers to uniquely identify documents
                # Create a more robust identifier that includes source to avoid false duplicates
                text_content = doc.get("text", "")
                source_file = doc.get("source_file", "")
                doc_identifier = (
                    f"{text_content[:50]}_{source_file}" if text_content else str(hash(str(doc)))
                )

                # Calculate RRF score: weight * 1 / (k1 + rank)
                rrf_score = weight * 1.0 / (self.k1 + rank)

                if doc_identifier not in doc_scores:
                    # Store the RRF score and the original document metadata
                    doc_scores[doc_identifier] = {
                        "rrf_score": rrf_score,
                        "metadata": doc,
                        "retriever_scores": {
                            self.retriever_names[retriever_idx]: doc.get("score", 0)
                        },
                    }
                else:
                    # Add to existing RRF score (this handles duplicates)
                    doc_scores[doc_identifier]["rrf_score"] += rrf_score
                    # Add the score from this retriever
                    doc_scores[doc_identifier]["retriever_scores"][
                        self.retriever_names[retriever_idx]
                    ] = doc.get("score", 0)
                    # Keep the metadata from the highest-ranked occurrence across methods
                    # (Actually, we should keep the original metadata, so we'll just update the score)
                    doc_scores[doc_identifier]["metadata"] = doc

        # Sort by RRF score in descending order
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)

        # Return top k results with updated scores
        top_results = []
        for _doc_identifier, data in sorted_docs[:k]:
            result = data["metadata"].copy()
            result["rrf_score"] = data["rrf_score"]  # Keep the RRF score
            result["score"] = data["rrf_score"]  # Replace with RRF score
            result["search_type"] = "hybrid_rrf"
            result["retriever_scores"] = data[
                "retriever_scores"
            ]  # Add original scores from each retriever
            top_results.append(result)

        return top_results


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
            index_name: Optional vector index name (default: uses settings.vector_index_name)
        """
        bm25_retriever = BM25Retriever(driver, neo4j_database)
        vector_retriever = Neo4jVectorSearch(
            driver,
            embed_fn,
            neo4j_database,
            index_name=index_name or settings.vector_index_name,
        )

        super().__init__(
            retrievers=[bm25_retriever, vector_retriever],
            retriever_names=["bm25", "vector"],
            weights=[1.0, 1.0],  # Equal weights for both methods
            k1=settings.rrf_k1,
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
            index_name: Optional vector index name (default: uses settings.vector_index_name)
            remove_stopwords: Whether to remove stop words from fulltext queries
        """
        fulltext_retriever = Neo4jFullTextSearch(
            driver, neo4j_database, remove_stopwords=remove_stopwords
        )
        vector_retriever = Neo4jVectorSearch(
            driver,
            embed_fn,
            neo4j_database,
            index_name=index_name or settings.vector_index_name,
        )

        super().__init__(
            retrievers=[fulltext_retriever, vector_retriever],
            retriever_names=["fulltext", "vector"],
            weights=[1.0, 1.0],  # Equal weights for both methods
            k1=settings.rrf_k1,
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
            index_name: Optional vector index name (default: uses settings.vector_index_name)
        """
        bm25_retriever = BM25Retriever(driver, neo4j_database)
        fulltext_retriever = Neo4jFullTextSearch(driver, neo4j_database)
        vector_retriever = Neo4jVectorSearch(
            driver,
            embed_fn,
            neo4j_database,
            index_name=index_name or settings.vector_index_name,
        )

        super().__init__(
            retrievers=[bm25_retriever, fulltext_retriever, vector_retriever],
            retriever_names=["bm25", "fulltext", "vector"],
            weights=[1.0, 1.0, 1.0],  # Equal weights for all methods
            k1=settings.rrf_k1,
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
        # Create component retrievers
        self.text2cypher = Text2CypherRetriever(driver, text2cypher_prompt, neo4j_database)
        self.vector = Neo4jVectorSearch(
            driver,
            embed_fn,
            neo4j_database,
            index_name=index_name or settings.vector_index_name,
        )

        # Store for config tracking
        self.text2cypher_prompt_version = text2cypher_prompt.prompt_version_name

        # Initialize parent HybridRetriever with equal weights
        super().__init__(
            retrievers=[self.text2cypher, self.vector],
            retriever_names=["text2cypher", "vector"],
            weights=[1.0, 1.0],
            k1=settings.rrf_k1,
        )

    async def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:  # type: ignore[override]
        """Search using Text2Cypher + Vector with RRF fusion.

        Override parent to handle async Text2Cypher and provide fallback.

        Args:
            query: Natural language query
            k: Number of results to return

        Returns:
            List of fused results (or vector-only if Text2Cypher fails)
        """
        all_results = []
        individual_results = {}

        # Try Text2Cypher (async) with error handling
        text2cypher_results = []
        text2cypher_success = False
        try:
            text2cypher_results = await self.text2cypher.search(query, k=k * 2)
            # Check if Text2Cypher returned an error result
            if not self._has_error(text2cypher_results):
                text2cypher_success = True
                for result in text2cypher_results:
                    result["source_retriever"] = "text2cypher"
                all_results.append((text2cypher_results, self.weights[0]))
                individual_results["text2cypher"] = text2cypher_results
            else:
                logger.warning("Text2Cypher returned error, falling back to vector-only")
                individual_results["text2cypher"] = text2cypher_results  # Keep error for tracking
        except Exception as e:
            logger.warning(f"Text2Cypher failed with exception, falling back to vector-only: {e}")
            individual_results["text2cypher"] = [{"error": str(e)}]

        # Vector search (sync) - always run
        vector_results = self.vector.search(query, k=k * 2)
        for result in vector_results:
            result["source_retriever"] = "vector"
        all_results.append((vector_results, self.weights[1]))
        individual_results["vector"] = vector_results

        # Apply RRF if we have multiple result sets, otherwise just return vector
        if len(all_results) > 1:
            fused_results = self._apply_rrf(all_results, k)
        else:
            # Vector-only fallback
            fused_results = vector_results[:k]
            for result in fused_results:
                result["search_type"] = "text2cypher_vector_fallback"

        # Store individual results for experiment tracking
        self.last_individual_results = individual_results
        self.text2cypher_success = text2cypher_success

        return fused_results

    def _has_error(self, results: list[dict[str, Any]]) -> bool:
        """Check if Text2Cypher returned an error result."""
        if not results:
            return True
        return any("error" in r for r in results)
