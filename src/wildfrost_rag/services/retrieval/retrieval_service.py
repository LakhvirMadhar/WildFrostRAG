"""Runs a retrieval experiment end to end: build a retriever, run every query, save results."""

import inspect
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Callable

import mlflow
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas as mlflow_dataset_from_pandas
from mlflow.entities.model_registry.prompt_version import PromptVersion
from neo4j import Driver
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from wildfrost_rag.core.config import DEFAULT_QUERIES_FILE, get_settings
from wildfrost_rag.core.exceptions import CypherExecutionError
from wildfrost_rag.core.logger import logger
from wildfrost_rag.domain.prompt_name import PromptName
from wildfrost_rag.domain.retrieval import CypherExecution, QueryResult
from wildfrost_rag.domain.retriever_type import RetrieverType
from wildfrost_rag.experiment_tracker.experiment_utils import (
    create_retrieval_config,
    get_next_experiment_id,
    save_config,
    save_individual_results,
    save_results,
)
from wildfrost_rag.models.experiment_config import RetrievalConfig
from wildfrost_rag.prompts import load_prompt
from wildfrost_rag.repositories.card_repository import CardRepository
from wildfrost_rag.repositories.document_repository import DocumentRepository
from wildfrost_rag.services.embeddings.embedder_type import EmbedderType
from wildfrost_rag.services.evaluation.mlflow_tracking import get_or_create_run
from wildfrost_rag.services.evaluation.auto_annotator import run_auto_annotation
from wildfrost_rag.services.retrieval import (
    BM25FulltextVectorHybridRetriever,
    BM25Retriever,
    BM25VectorHybridRetriever,
    FulltextThenCypherRetriever,
    FulltextVectorHybridRetriever,
    Neo4jFullTextSearch,
    Neo4jVectorSearch,
    Text2CypherRetriever,
    Text2CypherVectorHybridRetriever,
    VectorThenCypherRetriever,
)
from wildfrost_rag.services.retrieval.get_query_embedder import get_query_embed_fn
from wildfrost_rag.services.retrieval.hybrid_retrievers import HybridRetriever

# Retriever types that support stop word removal
SW_QUERY_RETRIEVERS = {
    RetrieverType.BM25,
    RetrieverType.FULLTEXT,
    RetrieverType.BM25_VECTOR,
    RetrieverType.FULLTEXT_VECTOR,
    RetrieverType.BM25_FULLTEXT_VECTOR,
    RetrieverType.FULLTEXT_THEN_CYPHER,
}
SW_DOCS_RETRIEVERS = {
    RetrieverType.BM25,
    RetrieverType.BM25_VECTOR,
    RetrieverType.BM25_FULLTEXT_VECTOR,
}

# Retriever types that use vector embeddings
VECTOR_BASED_RETRIEVERS = {
    RetrieverType.VECTOR,
    RetrieverType.BM25_VECTOR,
    RetrieverType.FULLTEXT_VECTOR,
    RetrieverType.BM25_FULLTEXT_VECTOR,
    RetrieverType.VECTOR_THEN_CYPHER,
    RetrieverType.TEXT2CYPHER_VECTOR,
}


@dataclass
class RetrievalExperimentResult:
    """Outcome of one retrieval experiment run."""

    results: list[QueryResult]
    experiment_dir: Path


class RetrievalService:
    """Builds a retriever, runs it over a query set, and saves the experiment artifacts."""

    def _get_vector_index_name(
        self, retriever_type: RetrieverType, embedder: EmbedderType
    ) -> str | None:
        """Get vector index name for vector-based retrievers."""
        if retriever_type not in VECTOR_BASED_RETRIEVERS:
            return None

        embedder_config = get_settings().embedding.embedding_configs[embedder.value]
        index_name: str = embedder_config["index_name"]
        logger.info(f"Using embedder '{embedder.value}' with index '{index_name}'")
        return index_name

    def _get_embedder_config(
        self, retriever_type: RetrieverType, embedder: EmbedderType
    ) -> tuple[str | None, str | None, str | None]:
        """Get (provider, model, vector_index_name) for config file.

        Returns (None, None, None) for non-vector-based retrievers.
        """
        if retriever_type not in VECTOR_BASED_RETRIEVERS:
            return None, None, None

        embedder_cfg = get_settings().embedding.embedding_configs[embedder.value]
        return embedder.value, embedder_cfg["model"], embedder_cfg["index_name"]

    def get_retriever(
        self,
        retriever_type: RetrieverType,
        driver: Driver,
        embedder: EmbedderType = EmbedderType.HF,
        sw_query: bool = True,
        sw_docs: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Factory: build the retriever instance for the given type."""
        index_name = self._get_vector_index_name(retriever_type, embedder)
        document_repository = DocumentRepository(driver)
        card_repository = CardRepository(driver)

        embed_fn = (
            get_query_embed_fn(embedder.value)
            if retriever_type in VECTOR_BASED_RETRIEVERS
            else None
        )

        non_vector_factory: dict[RetrieverType, Callable[[], Any]] = {
            RetrieverType.FULLTEXT: lambda: Neo4jFullTextSearch(
                driver, document_repository, remove_stopwords=sw_query
            ),
            RetrieverType.BM25: lambda: BM25Retriever(
                driver,
                document_repository,
                remove_stopwords_query=sw_query,
                remove_stopwords_docs=sw_docs,
            ),
            RetrieverType.TEXT2CYPHER: lambda: Text2CypherRetriever(driver, **kwargs),
            RetrieverType.FULLTEXT_THEN_CYPHER: lambda: FulltextThenCypherRetriever(
                driver, card_repository, remove_stopwords=sw_query
            ),
        }

        if retriever_type in non_vector_factory:
            return non_vector_factory[retriever_type]()

        if embed_fn is None:
            raise ValueError(f"embed_fn is required for vector-based retriever: {retriever_type}")

        vector_factory: dict[RetrieverType, Callable[[], Any]] = {
            RetrieverType.VECTOR: lambda: Neo4jVectorSearch(
                driver, embed_fn, document_repository, index_name=index_name
            ),
            RetrieverType.BM25_VECTOR: lambda: BM25VectorHybridRetriever(
                driver, embed_fn, index_name=index_name
            ),
            RetrieverType.FULLTEXT_VECTOR: lambda: FulltextVectorHybridRetriever(
                driver, embed_fn, index_name=index_name, remove_stopwords=sw_query
            ),
            RetrieverType.BM25_FULLTEXT_VECTOR: lambda: BM25FulltextVectorHybridRetriever(
                driver, embed_fn, index_name=index_name
            ),
            RetrieverType.TEXT2CYPHER_VECTOR: lambda: Text2CypherVectorHybridRetriever(
                driver, embed_fn, index_name=index_name, **kwargs
            ),
            RetrieverType.VECTOR_THEN_CYPHER: lambda: VectorThenCypherRetriever(
                driver, embed_fn, card_repository, index_name=index_name, **kwargs
            ),
        }

        if retriever_type not in vector_factory:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

        return vector_factory[retriever_type]()

    async def _process_single_query(
        self,
        retriever: Any,  # noqa: ANN401
        query: str,
        query_id: int,
        k: int,
    ) -> tuple[QueryResult, dict[str, Any] | None]:
        """Run one query through the retriever and return typed results."""
        try:
            result = retriever.search(query, k=k)
            retrieved_chunks = await result if inspect.iscoroutine(result) else result
        except CypherExecutionError as e:
            failed_result = QueryResult(
                query_id=query_id,
                query=query,
                cypher_execution=CypherExecution(
                    cypher_query=e.cypher_query,
                    cypher_execution_status="failed",
                    cypher_error_message=e.reason,
                ),
                retrieved_chunks=[],
                relevance_annotations=[],
            )
            return failed_result, None

        cypher_query = getattr(retriever, "last_cypher_query", None)
        cypher_execution = CypherExecution(
            cypher_query=cypher_query,
            cypher_execution_status="success",
            cypher_error_message=None,
        )

        query_result = QueryResult(
            query_id=query_id,
            query=query,
            cypher_execution=cypher_execution,
            retrieved_chunks=retrieved_chunks,
            relevance_annotations=[],
        )

        individual_entry = None
        if isinstance(retriever, HybridRetriever) and retriever.last_individual_results is not None:
            individual_entry = {
                "query_id": query_id,
                "query": query,
                "individual_results": {
                    name: [chunk.to_dict() for chunk in chunks]
                    for name, chunks in retriever.last_individual_results.items()
                },
            }

        return query_result, individual_entry

    async def _process_all_queries(
        self,
        df: pd.DataFrame,
        retriever: Any,  # noqa: ANN401
        retriever_type: RetrieverType,
        k: int,
    ) -> tuple[list[QueryResult], list[dict[str, Any]]]:
        """Run every query in df through the retriever, concurrently if it's async."""
        query_rows = [
            (row.get("query_id", idx), row["query"])
            for idx, row in df.iterrows()
            if not pd.isna(row["query"]) and row["query"] != ""
        ]

        is_async = inspect.iscoroutinefunction(getattr(retriever, "search", None))

        if is_async:
            logger.info(f"Running {len(query_rows)} queries concurrently")
            tasks = [
                self._process_single_query(retriever, query, query_id, k)
                for query_id, query in query_rows
            ]
            all_results = await tqdm_asyncio.gather(
                *tasks, desc=f"{retriever_type.value} queries", unit="query"
            )
        else:
            logger.info(f"Running {len(query_rows)} queries sequentially")
            all_results = []
            for query_id, query in tqdm(
                query_rows, desc=f"{retriever_type.value} queries", unit="query"
            ):
                all_results.append(await self._process_single_query(retriever, query, query_id, k))

        results: list[QueryResult] = []
        individual_results_list: list[dict[str, Any]] = []
        for query_result, individual_entry in all_results:
            results.append(query_result)
            if individual_entry:
                individual_results_list.append(individual_entry)

        return results, individual_results_list

    def _save_experiment_artifacts(
        self,
        experiment_dir: Path,
        config: RetrievalConfig,
        results: list[QueryResult],
        individual_results: list[dict[str, Any]],
        retriever: Any,  # noqa: ANN401
        retriever_type: RetrieverType,
        experiment_id: str,
        df: pd.DataFrame,
        dataset_path: str,
    ) -> None:
        """Save config.json, results.json, and (for hybrid retrievers) individual_results.json.

        Also logs to MLflow (params + config.json/results.json as artifacts) - kept
        alongside the on-disk files for now, not a replacement for them yet.
        """
        save_config(config, experiment_dir)

        results_dicts = [r.to_dict() for r in results]
        save_results(results_dicts, experiment_dir / "results.json")

        run_name = f"{retriever_type.value}/{experiment_id}"
        with get_or_create_run(run_name):
            mlflow.set_tag("run_number", config.run_number)
            query_dataset = mlflow_dataset_from_pandas(
                df, source=dataset_path, name="query_dataset"
            )
            mlflow.log_input(query_dataset, context="retrieval")
            mlflow.log_params(
                {
                    "retriever_type": config.retriever_type,
                    "chunking": config.chunking,
                    "k": config.k,
                    "description": config.description,
                    "dataset": config.dataset,
                    "embedding_model": config.embedding.model,
                    "embedding_provider": config.embedding.provider,
                    "vector_index_name": config.embedding.vector_index_name,
                    **config.additional_metadata,
                }
            )
            mlflow.log_metrics(
                {
                    "total_queries": config.query_stats.total,
                    "successful_queries": config.query_stats.successful,
                    "failed_queries": config.query_stats.failed,
                }
            )
            mlflow.log_artifact(str(experiment_dir / "config.json"))
            mlflow.log_artifact(str(experiment_dir / "results.json"))

        if isinstance(retriever, HybridRetriever) and individual_results:
            metadata = {
                "retrieval_id": f"{retriever_type.value}/{experiment_id}",
                "retriever_names": retriever.retriever_names,
                "timestamp": datetime.now().isoformat(),
            }
            save_individual_results(
                individual_results,
                experiment_dir / "individual_results.json",
                metadata=metadata,
            )

    async def run_experiment(
        self,
        driver: Driver,
        df: pd.DataFrame,
        retriever_type: RetrieverType,
        run_num: int,
        chunking: bool,
        k: int = 10,
        description: str = "",
        embedder: EmbedderType = EmbedderType.HF,
        sw_query: bool = True,
        sw_docs: bool = True,
        text2cypher_prompt_name: str = PromptName.TEXT2CYPHER_PROMPT,
        queries_json_path: Path | None = None,
        dataset_path: str = DEFAULT_QUERIES_FILE,
    ) -> RetrievalExperimentResult:
        """Build the configured retriever and run a full experiment with it.

        The single entry point both the CLI and the Dagster asset call: resolves
        the text2cypher prompt when needed, builds the retriever, and works out
        which stop-word settings are relevant metadata for this retriever type.
        """
        retriever_kwargs: dict[str, Any] = {}
        config_kwargs: dict[str, Any] = {}

        if retriever_type in (RetrieverType.TEXT2CYPHER, RetrieverType.TEXT2CYPHER_VECTOR):
            prompt = self.load_text2cypher_prompt(text2cypher_prompt_name)
            retriever_kwargs["text2cypher_prompt"] = prompt
            config_kwargs["text2cypher_prompt_version"] = prompt.uri

        retriever = self.get_retriever(
            retriever_type,
            driver,
            embedder=embedder,
            sw_query=sw_query,
            sw_docs=sw_docs,
            **retriever_kwargs,
        )

        if retriever_type in SW_QUERY_RETRIEVERS:
            config_kwargs["sw_query"] = sw_query
        if retriever_type in SW_DOCS_RETRIEVERS:
            config_kwargs["sw_docs"] = sw_docs
        logger.info(
            f"Using {retriever_type.value} retriever (sw_query={sw_query}, sw_docs={sw_docs})"
        )

        return await self._run_with_retriever(
            df=df,
            retriever=retriever,
            retriever_type=retriever_type,
            run_num=run_num,
            chunking=chunking,
            k=k,
            description=description,
            embedder=embedder,
            queries_json_path=queries_json_path,
            dataset_path=dataset_path,
            **config_kwargs,
        )

    async def _run_with_retriever(
        self,
        df: pd.DataFrame,
        retriever: Any,  # noqa: ANN401
        retriever_type: RetrieverType,
        run_num: int,
        chunking: bool,
        k: int = 10,
        description: str = "",
        embedder: EmbedderType = EmbedderType.HF,
        queries_json_path: Path | None = None,
        dataset_path: str = DEFAULT_QUERIES_FILE,
        **kwargs: Any,  # noqa: ANN401
    ) -> RetrievalExperimentResult:
        """Run an already-built retriever on the provided dataset and save raw results.

        Args:
            df: DataFrame containing queries
            retriever: Retriever instance to run
            retriever_type: Type of retriever being used
            run_num: Experiment run number
            chunking: Whether chunking was used during ingestion
            k: Number of chunks to retrieve per query
            description: Human-readable description of this experiment
            embedder: Embedding provider (for vector-based retrievers)
            queries_json_path: Path to queries JSON with doc_references for auto-annotation
            dataset_path: Path the query CSV was actually loaded from, recorded as
                this experiment's dataset (both in config.json and as an MLflow
                dataset input, so two runs can be proven to have used the same
                or a different query set)
            **kwargs: Additional metadata (e.g., text2cypher_prompt_version)

        Returns:
            The results and the experiment directory they were saved to
        """
        retriever_dir_name = retriever_type.value
        if retriever_type in VECTOR_BASED_RETRIEVERS:
            retriever_dir_name = f"{retriever_type.value}_{embedder.value}"

        base_path = (
            get_settings().paths.outputs_dir / f"run_{run_num}" / "retrievals" / retriever_dir_name
        )
        experiment_id = get_next_experiment_id(base_path)
        experiment_dir = base_path / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running {retriever_type.value} retriever in {experiment_dir}")
        logger.info(f"Experiment ID: {retriever_type.value}/{experiment_id}")

        results, individual_results_list = await self._process_all_queries(
            df, retriever, retriever_type, k
        )

        total_queries = len(
            [
                r
                for _, r in df.iterrows()
                if not pd.isna(r.get("query", "")) and r.get("query", "") != ""
            ]
        )
        embedding_provider, embedding_model, vector_index_name = self._get_embedder_config(
            retriever_type, embedder
        )

        config = create_retrieval_config(
            run_num=run_num,
            retriever_type=retriever_type,
            experiment_id=experiment_id,
            chunking=chunking,
            total_queries=total_queries,
            successful_queries=len(results),
            failed_queries=total_queries - len(results),
            description=description,
            k=k,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            vector_index_name=vector_index_name,
            dataset=dataset_path,
            **kwargs,
        )

        self._save_experiment_artifacts(
            experiment_dir,
            config,
            results,
            individual_results_list,
            retriever,
            retriever_type,
            experiment_id,
            df,
            dataset_path,
        )

        annotation_summary = run_auto_annotation(experiment_dir, queries_json_path)
        logger.info(f"Auto-annotation: {annotation_summary['auto_annotated']} chunks annotated")

        logger.info("Experiment completed successfully!")
        logger.info(f"Retrieval ID: {retriever_type.value}/{experiment_id}")
        logger.info(f"Results saved to {experiment_dir}")

        return RetrievalExperimentResult(results=results, experiment_dir=experiment_dir)

    def load_text2cypher_prompt(self, prompt_reference: str) -> PromptVersion:
        """Load a text2cypher prompt version by "name" or "name:version" reference."""
        return load_prompt(prompt_reference)

    def load_and_filter_queries(
        self,
        file_path: str,
        query_ids: list[int] | None = None,
        exclude_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        """Load queries from CSV and apply optional ID include/exclude filters.

        Raises:
            FileNotFoundError: file_path doesn't exist.
            ValueError: the filtered result is empty.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Queries file not found: {file_path}")

        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows.")

        if query_ids:
            df = df[df["query_id"].isin(query_ids)]
            logger.info(f"Filtered to {len(df)} queries with IDs: {query_ids}")

        if exclude_ids:
            df = df[~df["query_id"].isin(exclude_ids)]
            logger.info(f"Excluded {len(exclude_ids)} queries. Remaining: {len(df)} queries")

        if len(df) == 0:
            raise ValueError("No queries to process after filtering")

        return df
