from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval import _extract_structured_payload, _run_velocirag_search


class FakeVelociRag:
    def __init__(self, result):
        self._result = result

    async def call_tool(self, name, args):
        assert name == "search"
        assert "query" in args
        return self._result


class FakeRuntime:
    def __init__(self, result):
        self.velocirag = FakeVelociRag(result)


def test_extract_structured_payload_from_text_json_block():
    tool_result = SimpleNamespace(
        content=[SimpleNamespace(text='{"results": [{"doc_id": "d1", "content": "hello", "score": 0.5}]}')]
    )

    payload = _extract_structured_payload(tool_result)
    assert isinstance(payload, dict)
    assert "results" in payload


@pytest.mark.asyncio
async def test_run_velocirag_search_accepts_payload_without_total_results():
    tool_result = SimpleNamespace(
        content=[
            SimpleNamespace(
                text='{"results": [{"doc_id": "d1", "content": "VelociRAG indexes markdown docs.", "score": 0.82}]}'
            )
        ]
    )
    runtime = FakeRuntime(tool_result)

    hits = await _run_velocirag_search(runtime, query="What is VelociRAG?", top_k=3)

    assert len(hits) == 1
    assert hits[0]["doc_id"] == "d1"
    assert hits[0]["text"] == "VelociRAG indexes markdown docs."
