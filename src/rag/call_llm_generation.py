"""
LLM Generation Pipeline for WildFrostRAG.

This module provides functionality to generate responses using various approaches:
- Zero-shot
- RAG with different retrieval strategies (vector, lexical, hybrid, etc.)
It handles batch processing of queries and manages the complete generation workflow.
"""

import asyncio
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Protocol
from openai import AsyncOpenAI
from src.utils.config import settings
from src.utils.logger import logger
from prompts.system_prompt import SYSTEM_PROMPT, RAG_PROMPT


class Retriever(Protocol):
    """
    Protocol defining the interface for different retrieval strategies.
    """
    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The input query
            k: Number of results to return

        Returns:
            List of retrieved documents/chunks with metadata
        """
        ...


class LLMGenerator:
    """
    Handles LLM generation using OpenAI API.
    """

    def __init__(self):
        """
        Initialize the LLM Generator.
        """
        self.client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        self.default_model = settings.openai_model_name
        self.default_seed = settings.openai_seed
        self.default_temperature = settings.openai_temperature


    async def _make_openai_call(
        self,
        user_message: str
    ) -> str:
        """
        Make a standardized OpenAI API call.

        Args:
            user_message: The user message content

        Returns:
            The generated response text
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.default_model,
                # Using the same system prompt through all retrieval methods for consistency
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.default_temperature,
                seed=self.default_seed
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error making OpenAI API call: {e}")
            return f"ERROR: {str(e)}"


    async def generate_zero_shot_response(
        self,
        query: str
    ) -> str:
        """
        Generate a zero-shot response from OpenAI.

        Args:
            query: The input query to process

        Returns:
            The generated response text
        """
        return await self._make_openai_call(query)


    async def generate_rag_response(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Generate a RAG response using provided context.

        Args:
            query: The input query to process
            context: The retrieved context to use

        Returns:
            The generated response text
        """
        rag_prompt = RAG_PROMPT[0].format(query=query, context=context)
        return await self._make_openai_call(rag_prompt)


class GenerationPipeline:
    """
    A pipeline for generating responses using different approaches.
    Supports zero-shot and various RAG strategies through dependency injection.
    """

    def __init__(self, llm_generator: LLMGenerator):
        """
        Initialize the Generation Pipeline.

        Args:
            llm_generator: Instance of LLMGenerator to handle API calls
        """
        self.llm_generator = llm_generator


    async def generate_response(
        self,
        query: str,
        retriever: Retriever
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a response using the provided retriever.

        Args:
            query: The input query
            retriever: Retriever instance to use for retrieval

        Returns:
            A tuple containing (response text, list of retrieved chunks)
        """
        # All approaches now use retrieval with default k from settings
        retrieved_chunks = retriever.search(query, k=settings.default_k)

        if not retrieved_chunks:
            return ("ERROR: No relevant documents found in the database.", [])

        context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
        response = await self.llm_generator.generate_rag_response(
            query, context
        )
        return response, retrieved_chunks


    async def process_batch(
        self,
        df: pd.DataFrame,
        target_column: str,
        retriever: Optional[Retriever] = None,  # If None, uses zero-shot, otherwise, use retrieval method
        batch_size: int = settings.default_batch_size,
        overwrite: bool = False
    ) -> pd.DataFrame:
        """
        Process a batch of queries from a DataFrame.

        Args:
            df: DataFrame containing query data
            target_column: Column name to store the results in
            retriever: Retriever instance to use for retrieval (if None, uses zero-shot)
            batch_size: Number of queries to process in each batch (uses default if None)
            overwrite: Whether to overwrite existing responses and clear open & axial coding for new response

        Returns:
            Updated DataFrame with generated responses
        """
        # Filter rows that need processing
        # If overwrite is True, we process all valid queries regardless of existing response
        query_exists_mask = df['query'].notna() & (df['query'] != '')

        if overwrite:
            valid_mask = query_exists_mask
        else:
            valid_mask = query_exists_mask & (df[target_column].isna() | (df[target_column] == ''))

        valid_indices = df[valid_mask].index.tolist()
        queries = df.loc[valid_mask, 'query'].tolist()

        if not queries:
            logger.info(f"No queries to process for target column {target_column} (Overwrite={overwrite})")
            return df

        # Determine the processing approach based on whether retriever is provided
        if retriever is None:
            logger.info(f"Starting zero-shot generation for {len(queries)} queries (Overwrite={overwrite})...")
        else:
            logger.info(f"Starting generation for {len(queries)} queries (Overwrite={overwrite})...")

        results = []
        chunks_list = []  # For storing retrieved chunks (empty for zero-shot)

        # Batch processing
        total_batches = (len(queries) + batch_size - 1) // batch_size

        for i in range(0, len(queries), batch_size):
            batch_num = (i // batch_size) + 1
            batch = queries[i:i + batch_size]

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} queries)...")

            # Process each query in the batch
            batch_results = []
            for query in batch:
                if retriever is None:
                    # Zero-shot mode
                    response = await self.llm_generator.generate_zero_shot_response(
                        query
                    )
                    batch_results.append((response, None))  # (response, no chunks)
                else:
                    # RAG mode
                    result = await self.generate_response(
                        query,
                        retriever=retriever
                    )
                    batch_results.append(result)

            # Separate responses and chunks
            responses = [r[0] for r in batch_results]
            retrieved_chunks = [r[1] for r in batch_results]

            results.extend(responses)
            chunks_list.extend(retrieved_chunks)

            # Rate limit handling
            if i + batch_size < len(queries):
                await asyncio.sleep(1)

        # Assign back to DataFrame
        df.loc[valid_indices, target_column] = results

        # Store retrieved chunks if any were retrieved (only for RAG)
        if retriever is not None and any(chunk is not None for chunk in chunks_list):
            # serialization for CSV storage
            df.loc[valid_indices, 'retrieved_chunks'] = [
                json.dumps(c) if c is not None else '' for c in chunks_list
            ]

        # If overwriting, clear the associated annotation columns
        if overwrite:
            # Define columns to clear based on target column
            cols_to_clear = [
                f"{target_column}_validation",
                f"{target_column} Open Coding",
                f"{target_column} Axial Coding"
            ]

            for col in cols_to_clear:
                if col in df.columns:
                    df.loc[valid_indices, col] = ""  # Reset to empty string

        return df