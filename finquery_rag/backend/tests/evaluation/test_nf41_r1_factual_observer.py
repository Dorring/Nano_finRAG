from src.generation.deterministic_answers import DeterministicAnswerExtractor
from src.generation.deterministic_observer import RecordingDeterministicAnswerObserver


def test_factual_observer_records_the_existing_selected_context_evidence():
    extractor = DeterministicAnswerExtractor()
    observer = RecordingDeterministicAnswerObserver()
    sources = [{
        "filename": "report.pdf", "document_id": "report.pdf", "page": 1,
        "candidate_key": "candidate:v1:factual", "candidate_rank": 1,
    }]
    answer = extractor.answer_factual_query_from_context(
        "What is the title shown on the cover?",
        "[report.pdf, p1]\n2025 Annual Report",
        sources,
        observer=observer,
    )
    assert answer is not None
    assert observer.routes == ["deterministic_factual_context"]
    assert observer.facts
    assert observer.facts[0].candidate_key == "candidate:v1:factual"
    assert observer.selected_fact_ids == [observer.facts[0].fact_id]
