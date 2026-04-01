from __future__ import annotations

from src.answerer import generate_answer
from src.config_loader import load_routing_rules
from src.models import AnswerResult, QueryRequest
from src.retrieval import (
    fetch_documents,
    health_check_velocirag,
    rerank_candidates,
    run_bm25_search,
    run_vector_search,
)
from src.utils import try_parse_json_text


def _combine_results(
    bm25_results: list[dict],
    vector_results: list[dict],
    max_candidates: int,
) -> list[dict]:
    """
    Minimal deterministic fusion.
    Deduplicate by doc_id and keep the max score seen.
    """
    by_id: dict[str, dict] = {}

    for item in bm25_results + vector_results:
        doc_id = item["doc_id"]
        if doc_id not in by_id:
            by_id[doc_id] = item
            continue

        if item.get("score", 0.0) > by_id[doc_id].get("score", 0.0):
            by_id[doc_id] = item

    combined = list(by_id.values())
    combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return combined[:max_candidates]


def rewrite_query(query: str) -> str:
    return f"{query} refined"


async def run_query(request: QueryRequest, runtime) -> AnswerResult:
    routing = load_routing_rules()
    defaults = routing["defaults"]

    max_retries = int(defaults["max_retries"])
    max_candidates = int(defaults["max_candidates"])
    max_final_chunks = int(defaults["max_final_chunks"])
    bm25_top_k = int(defaults["bm25_top_k"])
    vector_top_k = int(defaults["vector_top_k"])

    query = request.query
    retries = 0

    healthy = await health_check_velocirag(runtime)
    if not healthy:
        return AnswerResult(answer="", citations=[], status="no_evidence")

    while True:
        bm25_results = await run_bm25_search(runtime, query, top_k=bm25_top_k)
        vector_results = await run_vector_search(runtime, query, top_k=vector_top_k)

        combined = _combine_results(
            bm25_results=bm25_results,
            vector_results=vector_results,
            max_candidates=max_candidates,
        )

        chunks = fetch_documents(combined)
        reranked = rerank_candidates(query, chunks)

        if not reranked:
            if retries < max_retries:
                retries += 1
                query = rewrite_query(query)
                continue

            return AnswerResult(
                answer="",
                citations=[],
                status="no_evidence",
            )

        return generate_answer(query, reranked[:max_final_chunks])
