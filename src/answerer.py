from __future__ import annotations

import re

from src.models import AnswerResult, RetrievedChunk


def _clean_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    if line.startswith("[Document:"):
        return ""
    if line.startswith("[Source:"):
        return ""
    if line == "---":
        return ""

    # Keep heading text, but remove markdown markers
    line = re.sub(r"^#+\s*", "", line)

    # Remove bullet markers
    line = re.sub(r"^[-*+]\s+", "", line)

    # Remove numbered list prefix
    line = re.sub(r"^\d+\.\s+", "", line)

    # Collapse whitespace
    line = re.sub(r"\s+", " ", line).strip()

    return line


def _extract_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    for raw_line in text.splitlines():
        cleaned = _clean_line(raw_line)
        if not cleaned:
            continue

        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        for part in parts:
            part = part.strip()
            if part:
                candidates.append(part)

    return candidates


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> AnswerResult:
    _ = query

    if not chunks:
        return AnswerResult(answer="", citations=[], status="no_evidence")

    top_chunks = chunks[:3]

    selected_sentences: list[str] = []
    citations: list[str] = []

    for chunk in top_chunks:
        candidates = _extract_candidates(chunk.text)

        for sentence in candidates:
            if len(sentence) < 10:
                continue

            selected_sentences.append(sentence)
            if chunk.chunk_id not in citations:
                citations.append(chunk.chunk_id)

            if len(selected_sentences) >= 3:
                return AnswerResult(
                    answer=" ".join(selected_sentences).strip(),
                    citations=citations,
                    status="answered",
                )

    # Fallback: use first cleaned non-empty line from the best chunk
    for chunk in top_chunks:
        candidates = _extract_candidates(chunk.text)
        if candidates:
            return AnswerResult(
                answer=candidates[0],
                citations=[chunk.chunk_id],
                status="answered",
            )

    return AnswerResult(answer="", citations=[], status="no_evidence")
