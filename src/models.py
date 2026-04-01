from dataclasses import dataclass, field
from typing import List


@dataclass
class QueryRequest:
    query: str


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


@dataclass
class AnswerResult:
    answer: str
    citations: List[str] = field(default_factory=list)
    status: str = "answered"
