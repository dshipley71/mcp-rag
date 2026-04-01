from typing import List, Dict


def simple_rrf_fusion(bm25_results: List[Dict], vector_results: List[Dict], k: int = 60):
    """
    Minimal Reciprocal Rank Fusion (RRF)
    Used for deterministic merging of results
    """
    scores = {}

    for rank, item in enumerate(bm25_results):
        doc_id = item["doc_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    for rank, item in enumerate(vector_results):
        doc_id = item["doc_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    fused = [{"doc_id": doc_id, "score": score} for doc_id, score in scores.items()]
    fused.sort(key=lambda x: x["score"], reverse=True)

    return fused
