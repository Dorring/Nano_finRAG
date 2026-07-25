from src.generation.deterministic_answers import DeterministicAnswerExtractor
from src.retrieval.query_processor import QueryProcessor


def _chunk(content, score=0.5):
    return {
        "content": content,
        "score": score,
        "metadata": {"doc_name": "report.pdf", "page": 7, "type": "text"},
    }


def test_raw_child_selector_prefers_metric_row_over_nearby_growth_row():
    extractor = DeterministicAnswerExtractor()
    result = extractor.answer_numeric_query_from_chunks(
        "What percentage of total revenue came from System A?",
        [
            _chunk("Revenue from System A increased by 6.1 per cent compared with 2019."),
            _chunk("System A fees accounted for 76.6 per cent of total revenue."),
        ],
    )

    assert result is not None
    assert "76.6 per cent" in result["answer"]
    assert "6.1 per cent" not in result["answer"].split("Evidence:", 1)[0]


def test_raw_child_selector_keeps_value_and_growth_from_same_evidence_row():
    extractor = DeterministicAnswerExtractor()
    result = extractor.answer_numeric_query_from_chunks(
        "What was Metric A revenue and its year-over-year growth rate?",
        [_chunk("Metric A revenue was $38.0 million, an increase of 70% year-over-year.")],
    )

    assert result is not None
    assert "$38 million" in result["answer"]
    assert "70%" in result["answer"]
    assert "Evidence:" not in result["answer"]
    assert "[report.pdf, p7]" in result["answer"]


def test_raw_child_selector_rejects_truncated_decimal_fragment():
    extractor = DeterministicAnswerExtractor()
    result = extractor.answer_numeric_query_from_chunks(
        "What was Platform revenue in 2025?",
        [
            _chunk(".9 million, or 15%, compared to prior year. Platform revenue increased."),
            _chunk("Platform revenue was $181 million in 2025."),
        ],
    )

    assert result is not None
    assert "$181 million" in result["answer"]
    assert "9 million" not in result["answer"]


def test_metric_phrase_bonus_is_not_applied_when_phrase_is_absent():
    extractor = DeterministicAnswerExtractor()
    query = "What was Platform revenue in 2025?"
    terms = extractor._important_query_terms(query)

    generic_score = extractor._raw_numeric_evidence_score(
        "The company reported $219 million in total revenue.", query, terms
    )
    metric_score = extractor._raw_numeric_evidence_score(
        "Platform revenue was $181 million in 2025.", query, terms
    )

    assert metric_score > generic_score


def test_numeric_extraction_gate_is_not_metric_allow_listed():
    processor = QueryProcessor()

    assert processor.should_try_deterministic_numeric_answer(
        "How much did Metric B contribute?",
        [{"content": "Metric B contributed 12 units."}],
    ) is True


def test_raw_child_selector_prefers_metric_amount_over_nearby_date():
    extractor = DeterministicAnswerExtractor()
    result = extractor.answer_numeric_query_from_chunks(
        "How much cash and cash equivalents did the company have as of December 31, 2025?",
        [
            _chunk(
                "As of December 31, 2025, cash and cash equivalents were "
                "$42.2 million, compared to $114.9 million as of December 31, 2024. "
                "Cash and cash equivalents held by subsidiaries were $6.7 million "
                "and $13.3 million, respectively."
            ),
        ],
    )

    assert result is not None
    answer = result["answer"].split("[", 1)[0]
    assert "$42.2 million" in answer
    assert "31" not in answer
    assert "2024" not in answer
    assert "$13.3 million" not in answer


def test_raw_child_selector_skips_table_note_before_metric_amount():
    extractor = DeterministicAnswerExtractor()
    result = extractor.answer_numeric_query_from_chunks(
        "What were cash and cash equivalents at year end?",
        [_chunk("Cash and cash equivalents | 3 | 143,540 | 206,031")],
    )

    assert result is not None
    answer = result["answer"].split("[", 1)[0]
    assert "143,540" in answer
    assert "Answer: 3" not in answer
