from __future__ import annotations

from src.models import AnswerResult, RetrievedChunk


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> AnswerResult:
    """
    Minimal deterministic answerer.
    Uses retrieved context only.
    """
    _ = query

    if not chunks:
        return AnswerResult(
            answer="",
            citations=[],
            status="no_evidence",
        )

    top_chunks = chunks[:3]
    answer_text = " ".join(chunk.text for chunk in top_chunks if chunk.text.strip())
    citations = [chunk.chunk_id for chunk in top_chunks]

    if not answer_text.strip():
        return AnswerResult(
            answer="",
            citations=[],
            status="no_evidence",
        )

    return AnswerResult(
        answer=answer_text,
        citations=citations,
        status="answered",
    )
