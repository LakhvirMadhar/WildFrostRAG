"""
Hybrid retriever for WildFrostRAG using Reciprocal Rank Fusion (RRF).

This module implements hybrid retrieval by combining multiple retrieval methods
using Reciprocal Rank Fusion to produce a single ranked result list.
"""

from typing import List, Dict, Any
from src.rag.retrievers.neo4j_vector_search import Neo4jVectorSearch
from src.rag.retrievers.bm25_retriever import BM25Retriever
from src.utils.config import settings


class HybridRetriever:
    """
    Implements hybrid retrieval using Reciprocal Rank Fusion (RRF) to combine
    results from multiple retrieval methods.
    """

    def __init__(self, retrievers: List, weights: List[float] = None, k1: int = 60):
        """
        Initialize the hybrid retriever.

        Args:
            retrievers: List of retriever instances to combine
            weights: Optional list of weights for each retriever (default: equal weights)
            k1: Smoothing parameter for RRF (default: 60)
        """
        self.retrievers = retrievers
        self.k1 = k1

        if weights is None:
            # Default to equal weights
            self.weights = [1.0] * len(retrievers)
        else:
            if len(weights) != len(retrievers):
                raise ValueError("Number of weights must match number of retrievers")
            self.weights = weights

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
        for i, retriever in enumerate(self.retrievers):
            results = retriever.search(query, k=k*2)  # Get more results to allow for fusion
            all_results.append((results, self.weights[i]))

        # Apply RRF to combine results
        fused_results = self._apply_rrf(all_results, k)

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

        for results, weight in all_results:
            for rank, doc in enumerate(results, 1):  # RRF uses 1-based ranking
                # Use a combination of text and other identifiers to uniquely identify documents
                doc_identifier = doc.get('text', str(hash(str(doc)))[:10])

                # Calculate RRF score: weight * 1 / (k1 + rank)
                rrf_score = weight * 1.0 / (self.k1 + rank)

                if doc_identifier not in doc_scores:
                    # Store the RRF score and the original document metadata
                    doc_scores[doc_identifier] = {
                        'rrf_score': rrf_score,
                        'metadata': doc
                    }
                else:
                    # Add to existing RRF score
                    doc_scores[doc_identifier]['rrf_score'] += rrf_score
                    # Keep the metadata from the highest-ranked occurrence across methods
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
            result['score'] = data['rrf_score']  # Replace with RRF score
            result['search_type'] = 'hybrid_rrf'
            top_results.append(result)

        return top_results


class BM25VectorHybridRetriever(HybridRetriever):
    """
    A specific hybrid retriever that combines BM25 search and vector search.
    """

    def __init__(self):
        """
        Initialize the BM25 and vector hybrid retriever.
        """
        bm25_retriever = BM25Retriever()
        vector_retriever = Neo4jVectorSearch()
        # Use the vector index name from config
        vector_retriever.index_name = settings.vector_index_name

        super().__init__(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[1.0, 1.0],  # Equal weights for both methods
            k1=settings.rrf_k1
        )