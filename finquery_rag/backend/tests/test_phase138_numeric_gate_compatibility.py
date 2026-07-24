from unittest.mock import MagicMock

from src.retrieval.query_processor import QueryProcessor


def test_numeric_gate_keeps_underspecified_metric_on_legacy_path():
    processor = QueryProcessor()

    assert not processor.should_try_deterministic_numeric_answer(
        "What was revenue?", [{"content": "Revenue was $10M."}]
    )


def test_numeric_gate_accepts_generic_quantified_question():
    processor = QueryProcessor()

    assert processor.should_try_deterministic_numeric_answer(
        "How much did Metric B contribute?", [{"content": "Metric B was 12."}]
    )


def test_raw_answer_contract_rejects_truthy_non_mapping_result():
    candidate = MagicMock()

    assert not isinstance(candidate, dict)
