"""Unit tests for HybridRetriever's Reciprocal Rank Fusion math.

_apply_rrf() was decomposed into _score_documents() (accumulation loop) and
_select_top_k() (sort + slice), so the RRF math can be exercised directly with
hand-built RetrievedChunk objects - no Neo4j driver, no live retriever needed.
"""

from wildfrost_rag.domain.retrieval import RetrievedChunk
from wildfrost_rag.services.retrieval.hybrid_retrievers import HybridRetriever, RRFScore


def _make_chunk(text: str, source_file: str = "Bombom.html", score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        score=score,
        search_type="test",
        retrieved_text=text,
        source_url=None,
        cypher_result={"text": text, "source_file": source_file},
    )


def _make_retriever(names: list[str], k1: int = 60) -> HybridRetriever:
    # search() is never called in these tests, so component retrievers can be dummies.
    return HybridRetriever(retrievers=[object() for _ in names], retriever_names=names, k1=k1)


def test_score_documents_sums_rrf_score_for_doc_found_by_two_retrievers() -> None:
    """A doc ranked #1 by both retrievers merges into one entry with summed RRF score."""
    hybrid = _make_retriever(["bm25", "vector"])
    chunk_from_bm25 = _make_chunk("Bombom deals damage.", score=1.5)
    chunk_from_vector = _make_chunk("Bombom deals damage.", score=0.9)
    all_results = [([chunk_from_bm25], 1.0), ([chunk_from_vector], 1.0)]

    doc_scores = hybrid._score_documents(all_results)

    assert len(doc_scores) == 1
    merged = next(iter(doc_scores.values()))
    expected_score = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert merged.rrf_score == expected_score
    assert merged.retriever_scores == {"bm25": 1.5, "vector": 0.9}
    # Bookkeeping (chunk/source) reflects the last-seen retriever, per existing behavior.
    assert merged.source_retriever == "vector"


def test_score_documents_keeps_rank_based_score_for_doc_found_by_one_retriever() -> None:
    """A doc found by only one retriever keeps its own rank-based RRF score."""
    hybrid = _make_retriever(["bm25", "vector"])
    shared_chunk = _make_chunk("Bombom deals damage.", source_file="Bombom.html")
    unique_chunk = _make_chunk("Foxee has high attack.", source_file="Foxee.html")
    all_results = [
        ([shared_chunk, unique_chunk], 1.0),  # bm25: rank 1, rank 2
        ([shared_chunk], 1.0),  # vector: rank 1
    ]

    doc_scores = hybrid._score_documents(all_results)

    assert len(doc_scores) == 2
    unique_doc_id = hybrid._identify_doc(unique_chunk)
    unique_score = doc_scores[unique_doc_id]
    assert unique_score.rrf_score == 1.0 / (60 + 2)
    assert unique_score.retriever_scores == {"bm25": 1.0}


def test_score_documents_applies_per_retriever_weight() -> None:
    """A retriever's weight scales its contribution to the RRF score."""
    hybrid = _make_retriever(["bm25", "vector"])
    hybrid.weights = [2.0, 1.0]
    chunk = _make_chunk("Bombom deals damage.")
    all_results = [([chunk], hybrid.weights[0]), ([], hybrid.weights[1])]

    doc_scores = hybrid._score_documents(all_results)

    only_score = next(iter(doc_scores.values()))
    assert only_score.rrf_score == 2.0 * (1.0 / (60 + 1))


def test_select_top_k_sorts_descending_and_truncates() -> None:
    """_select_top_k orders by RRF score descending and returns only the top k."""
    hybrid = _make_retriever(["bm25"])
    low = RRFScore(
        rrf_score=0.1, chunk=_make_chunk("low"), source_retriever="bm25", retriever_scores={}
    )
    high = RRFScore(
        rrf_score=0.9, chunk=_make_chunk("high"), source_retriever="bm25", retriever_scores={}
    )
    mid = RRFScore(
        rrf_score=0.5, chunk=_make_chunk("mid"), source_retriever="bm25", retriever_scores={}
    )

    top = hybrid._select_top_k({"low": low, "high": high, "mid": mid}, k=2)

    assert [chunk.score for chunk in top] == [0.9, 0.5]


def test_apply_rrf_orchestrates_scoring_and_selection() -> None:
    """_apply_rrf still produces the same end-to-end result via the two new helpers."""
    hybrid = _make_retriever(["bm25", "vector"], k1=60)
    shared_chunk = _make_chunk("Bombom deals damage.", source_file="Bombom.html")
    unique_chunk = _make_chunk("Foxee has high attack.", source_file="Foxee.html")
    all_results = [([shared_chunk], 1.0), ([shared_chunk, unique_chunk], 1.0)]

    fused = hybrid._apply_rrf(all_results, k=5)

    assert len(fused) == 2
    # Shared doc: rank 1 in both retrievers -> higher combined score than the unique doc.
    assert fused[0].retrieved_text == "Bombom deals damage."
    assert fused[0].score == 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert fused[1].retrieved_text == "Foxee has high attack."
    assert fused[1].score == 1.0 / (60 + 2)
    assert fused[0].search_type == "hybrid_rrf"
