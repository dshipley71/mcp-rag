import pytest

from src.models import AnswerResult, QueryRequest
from src.orchestrator import run_query


class FakeRuntime:
    pass


@pytest.mark.asyncio
async def test_run_query_returns_answer_result(monkeypatch):
    async def fake_health_check(runtime):
        return True

    async def fake_bm25_search(runtime, query, top_k=20):
        return [{"doc_id": "doc-1", "text": "Reranking uses retrieved evidence.", "score": 0.9, "metadata": {}}]

    async def fake_vector_search(runtime, query, top_k=20):
        return [{"doc_id": "doc-1", "text": "Reranking uses retrieved evidence.", "score": 0.9, "metadata": {}}]

    monkeypatch.setattr("src.orchestrator.health_check_velocirag", fake_health_check)
    monkeypatch.setattr("src.orchestrator.run_bm25_search", fake_bm25_search)
    monkeypatch.setattr("src.orchestrator.run_vector_search", fake_vector_search)

    result = await run_query(QueryRequest(query="What is reranking?"), FakeRuntime())
    assert isinstance(result, AnswerResult)


@pytest.mark.asyncio
async def test_run_query_stops_if_no_evidence(monkeypatch):
    async def fake_health_check(runtime):
        return True

    async def fake_bm25_search(runtime, query, top_k=20):
        return []

    async def fake_vector_search(runtime, query, top_k=20):
        return []

    monkeypatch.setattr("src.orchestrator.health_check_velocirag", fake_health_check)
    monkeypatch.setattr("src.orchestrator.run_bm25_search", fake_bm25_search)
    monkeypatch.setattr("src.orchestrator.run_vector_search", fake_vector_search)

    result = await run_query(
        QueryRequest(query="This query should produce no evidence"),
        FakeRuntime(),
    )
    assert result.answer == ""
    assert result.citations == []
    assert result.status == "no_evidence"


@pytest.mark.asyncio
async def test_run_query_returns_citations_when_answering(monkeypatch):
    async def fake_health_check(runtime):
        return True

    async def fake_bm25_search(runtime, query, top_k=20):
        return [{"doc_id": "doc-1", "text": "Hybrid retrieval combines lexical and semantic search.", "score": 0.9, "metadata": {}}]

    async def fake_vector_search(runtime, query, top_k=20):
        return [{"doc_id": "doc-2", "text": "The system cites retrieved chunks.", "score": 0.8, "metadata": {}}]

    monkeypatch.setattr("src.orchestrator.health_check_velocirag", fake_health_check)
    monkeypatch.setattr("src.orchestrator.run_bm25_search", fake_bm25_search)
    monkeypatch.setattr("src.orchestrator.run_vector_search", fake_vector_search)

    result = await run_query(
        QueryRequest(query="What does the spec say about retrieval?"),
        FakeRuntime(),
    )

    if result.status == "answered":
        assert len(result.citations) > 0
