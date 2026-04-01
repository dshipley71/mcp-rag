from typing import List
from src.models import RetrievedChunk


def run_bm25_search(query: str, top_k: int = 20) -> List[dict]:
    """
    Stub BM25 search
    """
    if "no evidence" in query.lower():
        return []

    return [{"doc_id": f"bm25-{i}", "score": 1.0 / (i + 1)} for i in range(top_k)]


def run_vector_search(query: str, top_k: int = 20) -> List[dict]:
    """
    Stub vector search
    """
    if "no evidence" in query.lower():
        return []

    return [{"doc_id": f"vec-{i}", "score": 1.0 / (i + 1)} for i in range(top_k)]


def fetch_documents(doc_ids: List[str]) -> List[RetrievedChunk]:
    """
    Stub document fetch
    """
    return [
        RetrievedChunk(
            chunk_id=doc_id,
            text=f"Content for {doc_id}",
            score=1.0,
        )
        for doc_id in doc_ids
    ]


def rerank_candidates(query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Stub reranker (identity sort)
    """
    return sorted(chunks, key=lambda x: x.score, reverse=True)
