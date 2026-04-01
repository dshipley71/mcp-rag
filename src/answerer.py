from typing import List
from src.models import RetrievedChunk, AnswerResult


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> AnswerResult:
    """
    Minimal deterministic answer generator
    """

    if not chunks:
        return AnswerResult(
            answer="",
            citations=[],
            status="no_evidence",
        )

    # Use top chunks only
    top_chunks = chunks[:3]

    answer_text = " ".join([chunk.text for chunk in top_chunks])
    citations = [chunk.chunk_id for chunk in top_chunks]

    return AnswerResult(
        answer=answer_text,
        citations=citations,
        status="answered",
    )
