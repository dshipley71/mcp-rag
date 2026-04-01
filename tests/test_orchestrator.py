from src.models import QueryRequest, RetrievedChunk, AnswerResult
from src.orchestrator import run_query


def test_run_query_returns_answer_result():
    result = run_query(QueryRequest(query="What is reranking?"))
    assert isinstance(result, AnswerResult)


def test_run_query_stops_if_no_evidence():
    result = run_query(QueryRequest(query="This query should produce no evidence"))
    assert result.answer == ""
    assert result.citations == []
    assert result.status == "no_evidence"


def test_run_query_returns_citations_when_answering():
    result = run_query(QueryRequest(query="What does the spec say about retrieval?"))
    if result.status == "answered":
        assert len(result.citations) > 0
