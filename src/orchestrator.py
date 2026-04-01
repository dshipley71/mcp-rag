from src.models import AnswerResult, QueryRequest


def run_query(request: QueryRequest) -> AnswerResult:
    if "no evidence" in request.query.lower():
        return AnswerResult(
            answer="",
            citations=[],
            status="no_evidence",
        )

    return AnswerResult(
        answer="Stub grounded answer.",
        citations=["chunk-001"],
        status="answered",
    )
