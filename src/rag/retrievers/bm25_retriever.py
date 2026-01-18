"""
External BM25 retriever for WildFrostRAG.

This module implements lexical retrieval using the rank_bm25 library for true BM25 scoring.
It retrieves documents from Neo4j, processes them with BM25, and returns ranked results.
"""

from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from neo4j import Driver
from rank_bm25 import BM25Okapi
from src.utils.config import settings
from src.utils.logger import logger
from src.rag.retrievers.base_neo4j_retriever import BaseNeo4jRetriever


class BM25Retriever(BaseNeo4jRetriever):
    """
    Implements lexical similarity retrieval using true BM25 scoring.
    This corresponds to the 'BM25' approach in the research goals.
    """

    # Class-level cache to share BM25 index across instances
    _shared_cache = {
        'documents': None,
        'node_data': None,
        'bm25_model': None,
        'cache_key': None
    }

    def __init__(self, driver: Driver, neo4j_database: Optional[str] = None):
        """
        Initialize the BM25 retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            neo4j_database: Optional database name (default: None uses default database)
        """
        super().__init__(driver, neo4j_database)
        self.index_name = settings.bm25_index_name
        self.bm25_model = None
        self.documents = []
        self.node_data = []
        self._initialize_nltk()

    def _initialize_nltk(self):
        """
        Initialize NLTK resources for text preprocessing.
        """
        # Download required NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 by tokenizing into words and removing stop words.

        Args:
            text: Input text to tokenize

        Returns:
            List of lowercase word tokens without stop words
        """
        # Use NLTK for better tokenization and stop word removal
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        # Filter out stop words and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        return tokens

    def _load_documents_from_neo4j(self) -> None:
        """
        Load all documents from Neo4j to build the BM25 index.
        Uses class-level cache to avoid reloading for multiple instances.
        """
        # Create cache key based on database and index name
        cache_key = f"{self.neo4j_database}:{self.index_name}"

        # Check if we can use cached data
        if (BM25Retriever._shared_cache['cache_key'] == cache_key and
            BM25Retriever._shared_cache['bm25_model'] is not None):
            logger.info("Using cached BM25 index from previous instance")
            self.documents = BM25Retriever._shared_cache['documents']
            self.node_data = BM25Retriever._shared_cache['node_data']
            self.bm25_model = BM25Retriever._shared_cache['bm25_model']
            return

        logger.info(f"Loading all documents from Neo4j for BM25 indexing...")

        with self.driver.session(database=self.neo4j_database) as session:
            # Query all documents
            query = f"""
            MATCH (d:{self.index_name})
            WHERE d.text IS NOT NULL
            RETURN d.text AS text, d
            """

            results = session.run(query)

            self.documents = []
            self.node_data = []

            for record in results:
                text = record["text"]
                node = record["d"]

                # Preprocess the text for BM25
                tokens = self._preprocess_text(text)
                self.documents.append(tokens)

                # Store the full node data for retrieval
                node_dict = {}
                for key, value in node.items():
                    if key != "embedding":  # Exclude the large vector
                        node_dict[key] = value
                self.node_data.append(node_dict)

            # Initialize the BM25 model with the documents
            self.bm25_model = BM25Okapi(self.documents)

            # Update class-level cache
            BM25Retriever._shared_cache = {
                'documents': self.documents,
                'node_data': self.node_data,
                'bm25_model': self.bm25_model,
                'cache_key': cache_key
            }

            logger.info(f"BM25 index built with {len(self.documents)} documents")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant document chunks using BM25 scoring.

        Args:
            query: Input query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of dictionaries containing retrieved chunks with their metadata and scores
        """
        if self.bm25_model is None:
            # Load documents from Neo4j on first search
            self._load_documents_from_neo4j()

        # Preprocess the query
        query_tokens = self._preprocess_text(query)

        # Get BM25 scores for the query against all documents
        scores = self.bm25_model.get_scores(query_tokens)

        # Get the top-k document indices and scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        # Build the results
        results = []
        for idx in top_indices:
            score = float(scores[idx])  # Convert np.float64 to Python float
            node_data = self.node_data[idx].copy()
            node_data["score"] = score
            results.append(node_data)

        return self._add_metadata(results, 'bm25')