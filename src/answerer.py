from __future__ import annotations

import re

from src.models import AnswerResult, RetrievedChunk


def _clean_line(line: str) -> str:
    """
    Remove obvious markdown and document wrapper noise while preserving content.
    """
    line = line.strip()
    if not line:
        return ""

    # Drop wrapper/meta lines
    if line.startswith("[Document:"):
        return ""
    if line.startswith("[Source:"):
        return ""
    if line == "---":
        return ""

    # Drop markdown headings entirely for v1 to avoid answers like "# Title"
    if line.startswith("#"):
        return ""

    # Remove bullet markers but keep content
    line = re.sub(r"^[-*+]\s+", "", line)

    # Remove numbered list prefix but keep content
    line = re.sub(r"^\d+\.\s+", "", line)

    # Collapse excessive whitespace
    line = re.sub(r"\s+", " ", line).strip()

    return line


def _split_into_candidate_sentences(text: str) -> list[str]:
    """
    Convert chunk text into cleaned candidate answer units.
    We keep this simple and deterministic for v1.
    """
    candidates: list[str] = []

    for raw_line in text.splitlines():
        cleaned = _clean_line(raw_line)
        if not cleaned:
            continue

        # If the line is long prose, split into sentences.
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        for part in parts:
            part = part.strip()
            if part:
                candidates.append(part)

    return candidates


def _select_answer_text(chunks: list[RetrievedChunk], max_sentences: int = 3) -> tuple[str, list[str]]:
    """
    Choose the first meaningful sentences from the highest-ranked chunks.
    Returns answer text and citations used.
    """
    selected_sentences: list[str] = []
    selected_citations: list[str] = []

    for chunk in chunks:
        candidates = _split_into_candidate_sentences(chunk.text)

        for sentence in candidates:
            # Skip very short fragments that are unlikely to answer the question
            if len(sentence) < 20:
                continue

            selected_sentences.append(sentence)
            if chunk.chunk_id not in selected_citations:
                selected_citations.append(chunk.chunk_id)

            if len(selected_sentences) >= max_sentences:
                answer_text = " ".join(selected_sentences).strip()
                return answer_text, selected_citations

    return " ".join(selected_sentences).strip(), selected_citations


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> AnswerResult:
    """
    Minimal deterministic grounded answer generator.

    Rules:
    - Use retrieved context only
    - Do not hallucinate missing information
    - Prefer meaningful prose over markdown headings or wrappers
    - Return no_evidence if no usable explanatory text is available
    """
    _ = query

    if not chunks:
        return AnswerResult(
            answer="",
            citations=[],
            status="no_evidence",
        )

    answer_text, citations = _select_answer_text(chunks[:3], max_sentences=3)

    if not answer_text:
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
