"""Unit tests for EmbeddingService.

VectorRepository is faked at the module boundary (all Neo4j I/O lives there,
not in EmbeddingService), and each provider's model/client is faked too, so
these tests never touch a real database, download a real model, or make a
real network call.

pytest-asyncio isn't a project dependency, so async calls are driven with
asyncio.run() from ordinary (synchronous) test functions - same pattern as
test_scraping_service.py.
"""

import asyncio
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.domain.repository_results import MissingEmbeddingDocument
from wildfrost_rag.core.embedder_type import EmbedderType
from wildfrost_rag.services.embeddings.embedding_service import EmbeddingService

_MODULE = "wildfrost_rag.services.embeddings.embedding_service"


@pytest.fixture(autouse=True)
def _fake_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub required Settings fields so get_settings() validates without real credentials."""
    monkeypatch.setenv("NEO4J_URI", "bolt://fake-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "fake")
    monkeypatch.setenv("NEO4J_PASSWORD", "fake")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fake_repository(missing_docs: list[MissingEmbeddingDocument]) -> MagicMock:
    repository = MagicMock()
    repository.documents_missing_property.return_value = missing_docs
    repository.set_document_embeddings.side_effect = (
        lambda element_ids, embeddings, property_name: len(element_ids)
    )
    return repository


def test_add_embeddings_skips_entirely_when_nothing_is_missing() -> None:
    """No Documents missing the property -> zero work, zero writes."""
    repository = _fake_repository([])

    with patch(f"{_MODULE}.VectorRepository", return_value=repository):
        result = asyncio.run(EmbeddingService().add_embeddings(MagicMock(), EmbedderType.HF))

    assert result == 0
    repository.set_document_embeddings.assert_not_called()


class _FakeVector:
    """Stands in for a numpy row: only .tolist() is ever called on it."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeSentenceTransformer:
    """A real class (not a Mock) so isinstance(model, SentenceTransformer) still passes.

    Patching SentenceTransformer as a Mock instead would replace the name isinstance()
    checks against too, breaking the exact runtime check this test exists to exercise.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> list[_FakeVector]:
        return [_FakeVector([0.1, 0.2]) for _ in texts]


def test_add_embeddings_only_processes_documents_missing_the_property() -> None:
    """The resume fix: repository is asked for missing docs, not all docs."""
    missing = [
        MissingEmbeddingDocument(text="first text", element_id="id-1"),
        MissingEmbeddingDocument(text="second text", element_id="id-2"),
    ]
    repository = _fake_repository(missing)

    with (
        patch(f"{_MODULE}.VectorRepository", return_value=repository),
        patch(f"{_MODULE}.SentenceTransformer", _FakeSentenceTransformer),
    ):
        result = asyncio.run(EmbeddingService().add_embeddings(MagicMock(), EmbedderType.HF))

    assert result == 2
    repository.documents_missing_property.assert_called_once_with("hf_embedding")
    repository.set_document_embeddings.assert_called_once()
    element_ids, embeddings, property_name = repository.set_document_embeddings.call_args[0]
    assert element_ids == ["id-1", "id-2"]
    assert embeddings == [[0.1, 0.2], [0.1, 0.2]]
    assert property_name == "hf_embedding"


def test_add_embeddings_openai_delegates_to_call_openai_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The openai branch calls the shared ACL function, not a local client."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    get_settings.cache_clear()

    missing = [MissingEmbeddingDocument(text="some text", element_id="id-1")]
    repository = _fake_repository(missing)

    async def _fake_call_openai_embeddings(
        texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        return [[0.5, 0.6] for _ in texts]

    with (
        patch(f"{_MODULE}.VectorRepository", return_value=repository),
        patch(f"{_MODULE}.call_openai_embeddings", side_effect=_fake_call_openai_embeddings),
    ):
        result = asyncio.run(EmbeddingService().add_embeddings(MagicMock(), EmbedderType.OPENAI))

    assert result == 1
    repository.documents_missing_property.assert_called_once_with("openai_embedding")


def test_add_embeddings_openai_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No OPENAI_API_KEY configured -> a typed EmbeddingError, not a confusing 500 later.

    Forces settings.openai.api_key to None directly rather than only unsetting the env
    var, since this machine's real .env (a real dev credential file, not test fixture
    data) may itself set OPENAI_API_KEY - unsetting os.environ alone wouldn't simulate
    "missing key" if pydantic-settings then falls back to reading that file.
    """
    from wildfrost_rag.core.exceptions import EmbeddingError

    monkeypatch.setattr(get_settings().openai, "api_key", None)
    repository = _fake_repository([MissingEmbeddingDocument(text="some text", element_id="id-1")])

    with patch(f"{_MODULE}.VectorRepository", return_value=repository):
        with pytest.raises(EmbeddingError):
            asyncio.run(EmbeddingService().add_embeddings(MagicMock(), EmbedderType.OPENAI))


def test_add_embeddings_gemma_delegates_to_ollama_async_client() -> None:
    """The gemma branch calls ollama.AsyncClient().embed(), never a local model."""
    missing = [MissingEmbeddingDocument(text="some text", element_id="id-1")]
    repository = _fake_repository(missing)
    fake_async_client = MagicMock()

    async def _fake_embed(model: str, input: list[str]) -> dict[str, list[list[float]]]:
        return {"embeddings": [[0.7, 0.8] for _ in input]}

    fake_async_client.embed = _fake_embed

    with (
        patch(f"{_MODULE}.VectorRepository", return_value=repository),
        patch(f"{_MODULE}.ollama.AsyncClient", return_value=fake_async_client),
    ):
        result = asyncio.run(EmbeddingService().add_embeddings(MagicMock(), EmbedderType.GEMMA))

    assert result == 1
    repository.documents_missing_property.assert_called_once_with("gemma_embedding")


def test_create_vector_index_uses_the_embedder_s_own_config() -> None:
    """create_vector_index looks up property/index/dimension from the given embedder."""
    repository = MagicMock()

    with patch(f"{_MODULE}.VectorRepository", return_value=repository):
        EmbeddingService().create_vector_index(MagicMock(), EmbedderType.HF)

    repository.create_embedding_index.assert_called_once_with(
        property_name="hf_embedding", index_name="document-embeddings-hf", dimension=384
    )
