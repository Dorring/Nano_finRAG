from src.generation.deterministic_answers import DeterministicAnswerExtractor
from src.generation.deterministic_observer import RecordingDeterministicAnswerObserver


def _chunk():
    return {
        "content": "Cash and cash equivalents were $42.2 million as of December 31, 2025.",
        "metadata": {"candidate_key": "candidate:v1:test", "candidate_rank": 1, "document_id": "report.pdf", "page": 3},
    }


def test_observer_does_not_change_answer_and_records_real_candidates():
    extractor = DeterministicAnswerExtractor()
    query = "How much cash and cash equivalents did the company have in 2025?"
    baseline = extractor.answer_numeric_query_from_chunks(query, [_chunk()])
    observer = RecordingDeterministicAnswerObserver()
    observed = extractor.answer_numeric_query_from_chunks(query, [_chunk()], observer=observer)
    assert observed == baseline
    assert observer.routes == ["deterministic_numeric_raw_child"]
    assert observer.facts
    assert observer.selected_fact_ids


def test_observer_is_disabled_by_default():
    answer = DeterministicAnswerExtractor().answer_numeric_query_from_chunks(
        "How much cash and cash equivalents did the company have in 2025?", [_chunk()]
    )
    assert answer is not None
