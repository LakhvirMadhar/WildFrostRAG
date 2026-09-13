"""Dagster asset for running a retrieval experiment."""

from pathlib import Path

from dagster import Config, asset

from wildfrost_rag.defs.embeddings.assets import vector_index
from wildfrost_rag.defs.ingestion.assets import neo4j_documents
from wildfrost_rag.defs.resources import Neo4jResource
from wildfrost_rag.domain.prompt_name import PromptName
from wildfrost_rag.domain.retriever_type import RetrieverType
from wildfrost_rag.services.embeddings.embedder_type import EmbedderType
from wildfrost_rag.services.retrieval.retrieval_service import RetrievalService


class RetrievalRunConfig(Config):
    """Parameters for one retrieval experiment run."""

    retriever_type: RetrieverType
    run_num: int
    embedder: EmbedderType = EmbedderType.HF
    k: int = 10
    chunking: bool = False
    description: str = ""
    sw_query: bool = True
    sw_docs: bool = True
    query_ids: list[int] | None = None
    exclude_query_ids: list[int] | None = None
    text2cypher_prompt_name: str = PromptName.TEXT2CYPHER_PROMPT
    queries_json: str | None = None
    file: str = "queries/simple_reference_based_queries.csv"


@asset(deps=[neo4j_documents, vector_index])
async def retrieval_results(config: RetrievalRunConfig, neo4j: Neo4jResource) -> str:
    """Run a configured retriever over the query set and save the experiment artifacts.

    Returns the experiment directory the results were saved to, so downstream
    evaluation assets (e.g. retrieval_metrics) know exactly where to read from.
    """
    service = RetrievalService()
    df = service.load_and_filter_queries(config.file, config.query_ids, config.exclude_query_ids)
    queries_json_path = Path(config.queries_json) if config.queries_json else None

    with neo4j.get_driver() as driver:
        experiment = await service.run_experiment(
            driver=driver,
            df=df,
            retriever_type=config.retriever_type,
            run_num=config.run_num,
            chunking=config.chunking,
            k=config.k,
            description=config.description,
            embedder=config.embedder,
            sw_query=config.sw_query,
            sw_docs=config.sw_docs,
            text2cypher_prompt_name=config.text2cypher_prompt_name,
            queries_json_path=queries_json_path,
        )

    return str(experiment.experiment_dir)
