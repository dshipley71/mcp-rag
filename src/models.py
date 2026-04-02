from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class QueryRequest:
    query: str


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    answer: str
    citations: List[str] = field(default_factory=list)
    status: str = "answered"


@dataclass
class IngestionResult:
    path: str
    doc_id: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
