from src.models import QueryRequest, AnswerResult
from src.retrieval import (
    run_bm25_search,
    run_vector_search,
    fetch_documents,
    rerank_candidates,
)
from src.answerer import generate_answer
from src.utils import simple_rrf_fusion
from src.config_loader import load_routing_rules


def run_query(request: QueryRequest) -> AnswerResult:
    routing = load_routing_rules()

    max_retries = routing["defaults"]["max_retries"]
    retries = 0

    query = request.query

    while True:
        # Step 1: Retrieval
        bm25_results = run_bm25_search(query)
        vector_results = run_vector_search(query)

        # Step 2: Fusion
        fused = simple_rrf_fusion(bm25_results, vector_results)

        if not fused:
            if retries < max_retries:
                retries += 1
                query = rewrite_query(query)
                continue
            return AnswerResult(answer="", citations=[], status="no_evidence")

        # Step 3: Fetch documents
        top_doc_ids = [item["doc_id"] for item in fused[: routing["defaults"]["max_candidates"]]]
        chunks = fetch_documents(top_doc_ids)

        # Step 4: Rerank
        reranked = rerank_candidates(query, chunks)

        # Step 5: Answer
        result = generate_answer(query, reranked[: routing["defaults"]["max_final_chunks"]])

        return result


def rewrite_query(query: str) -> str:
    """
    Minimal deterministic rewrite
    """
    return query + " refined"
