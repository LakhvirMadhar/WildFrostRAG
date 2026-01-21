"""
Hybrid retriever for WildFrostRAG using Reciprocal Rank Fusion (RRF).

This module implements hybrid retrieval by combining multiple retrieval methods
using Reciprocal Rank Fusion to produce a single ranked result list.
"""

from typing import List, Dict, Any, Optional
from neo4j import Driver
from src.rag.retrievers.neo4j_vector_search import Neo4jVectorSearch
from src.rag.retrievers.bm25_retriever import BM25Retriever
from src.rag.retrievers.neo4j_fulltext_search import Neo4jFullTextSearch
from src.utils.config import settings


class HybridRetriever:
    """
    Implements hybrid retrieval using Reciprocal Rank Fusion (RRF) to combine
    results from multiple retrieval methods.
    """

    def __init__(self, retrievers: List, retriever_names: List[str], weights: List[float] = None, k1: int = 60):
        """
        Initialize the hybrid retriever.

        Args:
            retrievers: List of retriever instances to combine
            retriever_names: List of names for each retriever (for identification)
            weights: Optional list of weights for each retriever (default: equal weights)
            k1: Smoothing parameter for RRF (default: 60)
        """
        self.retrievers = retrievers
        self.retriever_names = retriever_names
        self.k1 = k1
        self.last_individual_results = None  # Track individual results for experiment tracking

        if weights is None:
            # Default to equal weights
            self.weights = [1.0] * len(retrievers)
        else:
            if len(weights) != len(retrievers):
                raise ValueError("Number of weights must match number of retrievers")
            self.weights = weights

        if len(retrievers) != len(retriever_names):
            raise ValueError("Number of retrievers must match number of retriever names")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve results using multiple retrievers and combine them with RRF.

        Args:
            query: Input query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of dictionaries containing retrieved chunks with their metadata and scores
        """
        # Get results from each retriever
        all_results = []
        individual_results = {}  # Store individual retriever results for analysis

        for i, (retriever, name) in enumerate(zip(self.retrievers, self.retriever_names)):
            results = retriever.search(query, k=k*2)  # Get more results to allow for fusion
            # Add source information to help distinguish results during testing
            for result in results:
                result['source_retriever'] = name
            all_results.append((results, self.weights[i]))
            individual_results[name] = results

        # Apply RRF to combine results
        fused_results = self._apply_rrf(all_results, k)

        # Store individual results as instance variable for experiment tracking
        # This allows evaluate_retrievers.py to save them separately
        self.last_individual_results = individual_results

        return fused_results

    def _apply_rrf(self, all_results: List[tuple], k: int) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to combine results from multiple retrievers.

        Args:
            all_results: List of (results, weight) tuples from each retriever
            k: Number of top results to return

        Returns:
            List of fused results sorted by RRF score
        """
        # Create a mapping from document text to RRF score and metadata
        doc_scores = {}

        for retriever_idx, (results, weight) in enumerate(all_results):
            for rank, doc in enumerate(results, 1):  # RRF uses 1-based ranking
                # Use a combination of text and other identifiers to uniquely identify documents
                # Create a more robust identifier that includes source to avoid false duplicates
                text_content = doc.get('text', '')
                source_file = doc.get('source_file', '')
                doc_identifier = f"{text_content[:50]}_{source_file}" if text_content else str(hash(str(doc)))

                # Calculate RRF score: weight * 1 / (k1 + rank)
                rrf_score = weight * 1.0 / (self.k1 + rank)

                if doc_identifier not in doc_scores:
                    # Store the RRF score and the original document metadata
                    doc_scores[doc_identifier] = {
                        'rrf_score': rrf_score,
                        'metadata': doc,
                        'retriever_scores': {self.retriever_names[retriever_idx]: doc.get('score', 0)}
                    }
                else:
                    # Add to existing RRF score (this handles duplicates)
                    doc_scores[doc_identifier]['rrf_score'] += rrf_score
                    # Add the score from this retriever
                    doc_scores[doc_identifier]['retriever_scores'][self.retriever_names[retriever_idx]] = doc.get('score', 0)
                    # Keep the metadata from the highest-ranked occurrence across methods
                    # (Actually, we should keep the original metadata, so we'll just update the score)
                    doc_scores[doc_identifier]['metadata'] = doc

        # Sort by RRF score in descending order
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]['rrf_score'],
            reverse=True
        )

        # Return top k results with updated scores
        top_results = []
        for doc_identifier, data in sorted_docs[:k]:
            result = data['metadata'].copy()
            result['rrf_score'] = data['rrf_score']  # Keep the RRF score
            result['score'] = data['rrf_score']  # Replace with RRF score
            result['search_type'] = 'hybrid_rrf'
            result['retriever_scores'] = data['retriever_scores']  # Add original scores from each retriever
            top_results.append(result)

        return top_results


class BM25VectorHybridRetriever(HybridRetriever):
    """
    A specific hybrid retriever that combines BM25 search and vector search.
    """

    def __init__(self, driver: Driver, neo4j_database: Optional[str] = None, index_name: Optional[str] = None):
        """
        Initialize the BM25 and vector hybrid retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            neo4j_database: Optional database name (default: None uses default database)
            index_name: Optional vector index name (default: uses settings.vector_index_name)
        """
        bm25_retriever = BM25Retriever(driver, neo4j_database)
        vector_retriever = Neo4jVectorSearch(driver, neo4j_database, index_name=index_name or settings.vector_index_name)

        super().__init__(
            retrievers=[bm25_retriever, vector_retriever],
            retriever_names=['bm25', 'vector'],
            weights=[1.0, 1.0],  # Equal weights for both methods
            k1=settings.rrf_k1
        )


class FulltextVectorHybridRetriever(HybridRetriever):
    """
    A specific hybrid retriever that combines fulltext search and vector search.
    """

    def __init__(self, driver: Driver, neo4j_database: Optional[str] = None, index_name: Optional[str] = None):
        """
        Initialize the fulltext and vector hybrid retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            neo4j_database: Optional database name (default: None uses default database)
            index_name: Optional vector index name (default: uses settings.vector_index_name)
        """
        fulltext_retriever = Neo4jFullTextSearch(driver, neo4j_database)
        vector_retriever = Neo4jVectorSearch(driver, neo4j_database, index_name=index_name or settings.vector_index_name)

        super().__init__(
            retrievers=[fulltext_retriever, vector_retriever],
            retriever_names=['fulltext', 'vector'],
            weights=[1.0, 1.0],  # Equal weights for both methods
            k1=settings.rrf_k1
        )


class BM25FulltextVectorHybridRetriever(HybridRetriever):
    """
    A specific hybrid retriever that combines BM25, fulltext, and vector search.
    """

    def __init__(self, driver: Driver, neo4j_database: Optional[str] = None, index_name: Optional[str] = None):
        """
        Initialize the BM25, fulltext, and vector hybrid retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            neo4j_database: Optional database name (default: None uses default database)
            index_name: Optional vector index name (default: uses settings.vector_index_name)
        """
        bm25_retriever = BM25Retriever(driver, neo4j_database)
        fulltext_retriever = Neo4jFullTextSearch(driver, neo4j_database)
        vector_retriever = Neo4jVectorSearch(driver, neo4j_database, index_name=index_name or settings.vector_index_name)

        super().__init__(
            retrievers=[bm25_retriever, fulltext_retriever, vector_retriever],
            retriever_names=['bm25', 'fulltext', 'vector'],
            weights=[1.0, 1.0, 1.0],  # Equal weights for all methods
            k1=settings.rrf_k1
        )