from src.generation.deterministic_answers import DeterministicAnswerExtractor


def _chunk(content: str) -> dict:
    return {
        "content": content,
        "score": 0.8,
        "metadata": {"doc_name": "neutral.pdf", "page": 7, "type": "text"},
    }


def test_rate_question_pairs_metric_amount_with_same_row_percentage():
    extractor = DeterministicAnswerExtractor()

    answer = extractor.answer_numeric_query_from_chunks(
        "What was Metric A revenue and its year-over-year growth rate?",
        [_chunk(
            "Metric A revenue was $38.0 million, an increase of $15.7 million, "
            "or 70%, compared to prior year."
        )],
    )

    assert answer is not None
    assert "$38 million, 70%" in answer["answer"]
    assert "$15.7 million" not in answer["answer"]


def test_component_question_binds_labels_to_nearby_amounts():
    extractor = DeterministicAnswerExtractor()

    answer = extractor.answer_numeric_query_from_chunks(
        "What were the two components of the credit facilities?",
        [_chunk(
            "The agreement provides for (a) a revolving facility in an "
            "aggregate principal amount of $45.0 million and (b) a term loan "
            "facility in an aggregate principal amount of $25.0 million."
        )],
    )

    assert answer is not None
    rendered = answer["answer"].lower()
    assert "revolving facility, $45 million" in rendered
    assert "term loan facility, $25 million" in rendered


def test_numeric_normalization_removes_space_before_percent():
    extractor = DeterministicAnswerExtractor()

    answer = extractor.answer_numeric_query_from_chunks(
        "What was the gross margin?",
        [_chunk("Gross margin was 72 % for the reported period.")],
    )

    assert answer is not None
    assert "72%" in answer["answer"]
    assert "72 %" not in answer["answer"]
