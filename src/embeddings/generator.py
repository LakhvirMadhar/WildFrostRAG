"""
Embedding generation module for WildFrostRAG.

This module handles loading embedding models and generating vector
embeddings from text chunks for use in vector similarity search.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from src.utils.logger import logger


class EmbeddingGenerator:
    """
    Wrapper for sentence transformer models to generate embeddings.
    
    This class provides a clean interface for embedding generation
    and caches the model to avoid reloading.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator with a specific model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self._model = None
        logger.info(f"Initialized EmbeddingGenerator with model: {model_name}")
    
    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy-load the sentence transformer model.
        
        Returns:
            Loaded SentenceTransformer model
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully")
        return self._model
    
    def generate_embeddings(
        self,
        texts: Union[List[str], List[Document]]
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts or Document objects.
        
        Args:
            texts: List of strings or LangChain Document objects to embed
        
        Returns:
            NumPy array of embeddings with shape (n_texts, embedding_dim)
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> texts = ["Hello world", "Goodbye world"]
            >>> embeddings = generator.generate_embeddings(texts)
            >>> embeddings.shape
            (2, 384)
        """
        # Extract text content from Document objects if needed
        if texts and isinstance(texts[0], Document):
            text_list = [doc.page_content for doc in texts]
            logger.debug(f"Extracting text from {len(texts)} Document objects")
        else:
            text_list = texts
        
        if not text_list:
            logger.warning("Empty text list provided for embedding generation")
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(text_list)} texts")
        embeddings = self.model.encode(text_list)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings produced by this model.
        
        Returns:
            Integer dimension of embedding vectors
        """
        # Get dimension from model config
        return self.model.get_sentence_embedding_dimension()


# Convenience functions for backward compatibility
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load a sentence transformer model.
    
    Args:
        model_name: Name of the sentence-transformers model to use
    
    Returns:
        Loaded SentenceTransformer model
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded successfully")
    return model


def generate_embeddings(
    texts: Union[List[str], List[Document]],
    model: SentenceTransformer
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using a pre-loaded model.
    
    Args:
        texts: List of strings or LangChain Document objects to embed
        model: Pre-loaded SentenceTransformer model
    
    Returns:
        NumPy array of embeddings with shape (n_texts, embedding_dim)
    """
    # Extract text content from Document objects if needed
    if texts and isinstance(texts[0], Document):
        text_list = [doc.page_content for doc in texts]
    else:
        text_list = texts
    
    if not text_list:
        logger.warning("Empty text list provided for embedding generation")
        return np.array([])
    
    logger.info(f"Generating embeddings for {len(text_list)} texts")
    embeddings = model.encode(text_list)
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    
    return embeddings
