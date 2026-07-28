from src.generation.deterministic_answers import DeterministicAnswerExtractor
from src.retrieval.query_processor import QueryProcessor
from src.services.mineru_parser import append_table_row_children


def test_table_cell_children_preserve_header_value_alignment():
    parent = {
        "content": (
            "STATEMENT III (in thousands of Swiss francs)\n"
            "Accumulated Surpluses | Special Projects Reserve Surplus | "
            "Revaluation Reserve | Actuarial gains/(losses) through Net | "
            "Working Capital Funds | Net Assets Total\n"
            "Net Assets at December 31, 2020 | 565,601 | 28,173 | "
            "20,368 | -233,421 | 6,342 | 387,063"
        ),
        "metadata": {
            "type": "table",
            "doc_id": "user_1_demo.pdf::page_1::table_1",
            "page": 1,
            "doc_name": "demo.pdf",
        },
    }

    children = append_table_row_children([parent])
    cells = [
        child for child in children
        if child["metadata"].get("type") == "table_cell"
    ]

    target = next(
        child for child in cells
        if child["metadata"]["table_column"] == "Net Assets Total"
    )
    assert "Value: 387,063" in target["content"]
    assert target["metadata"]["table_alignment"] == "exact"

    extractor = DeterministicAnswerExtractor(query_processor=QueryProcessor())
    answer = extractor.answer_numeric_query_from_chunks(
        "What were net assets at December 31, 2020?",
        cells,
    )

    assert answer is not None
    assert "387,063" in answer["answer"]
    assert "565,601" not in answer["answer"]


def test_table_cell_children_skip_ambiguous_header_value_alignment():
    parent = {
        "content": (
            "Original Budget | Actual Expense\n"
            "Program | Program Title\n"
            "5 | The PCT System | 110,231 | 109,097 | 98,755 | 10,342"
        ),
        "metadata": {
            "type": "table",
            "doc_id": "user_1_demo.pdf::page_2::table_1",
            "page": 2,
            "doc_name": "demo.pdf",
        },
    }

    children = append_table_row_children([parent])

    assert any(child["metadata"].get("type") == "table_row" for child in children)
    assert not any(child["metadata"].get("type") == "table_cell" for child in children)


def test_layout_row_column_context_selects_requested_reporting_period():
    """Sparse coordinate rows retain enough context for safe year selection."""
    extractor = DeterministicAnswerExtractor(query_processor=QueryProcessor())
    chunks = [{
        "content": (
            "Cash and cash equivalents | 3 | 143,540 | 206,031\n"
            "Table column context: December 31, 2020 | December 31, 2019"
        ),
        "metadata": {
            "type": "table_row",
            "subtype": "layout_coordinate_row",
            "doc_name": "statement.pdf",
            "page": 4,
        },
        "score": 0.1,
    }]

    answer = extractor.answer_numeric_query_from_chunks(
        "What were cash and cash equivalents at December 31, 2020?",
        chunks,
    )

    assert answer is not None
    assert "143,540" in answer["answer"]
    assert "206,031" not in answer["answer"]


def test_layout_row_column_context_selects_requested_actual_column():
    """Column qualifiers are treated as evidence, not document-specific rules."""
    extractor = DeterministicAnswerExtractor(query_processor=QueryProcessor())
    chunks = [{
        "content": (
            "System A | 110,231 | 98,755 | 10,342\n"
            "Table column context: Budget 2020 | Actual 2020 | Variance"
        ),
        "metadata": {"type": "table_row", "doc_name": "statement.pdf", "page": 8},
        "score": 0.1,
    }]

    answer = extractor.answer_numeric_query_from_chunks(
        "What was the actual 2020 amount for System A?",
        chunks,
    )

    assert answer is not None
    assert "98,755" in answer["answer"]
    assert "110,231" not in answer["answer"]


def test_text_extracted_date_headers_and_packed_values_are_aligned_safely():
    """Broken PDF column spacing remains usable without document-specific rules."""
    extractor = DeterministicAnswerExtractor(query_processor=QueryProcessor())
    chunks = [{
        "content": (
            "December 31,\n2020\nDecember 31,\n2019\n"
            "Total cash and cash equivalents143,540206,031"
        ),
        "metadata": {"type": "text", "doc_name": "statement.pdf", "page": 9},
        "score": 0.1,
    }]

    answer = extractor.answer_numeric_query_from_chunks(
        "What were total cash and cash equivalents at December 31, 2020?",
        chunks,
    )

    assert answer is not None
    assert "143,540" in answer["answer"]
    assert "206,031" not in answer["answer"]


def test_malformed_glued_year_amount_is_not_a_numeric_claim():
    extractor = DeterministicAnswerExtractor(query_processor=QueryProcessor())
    chunks = [{
        "content": (
            "Net assets at December 31, 2018328,732\n"
            "At December 31, 2020, net assets were 387.1 million Swiss francs."
        ),
        "metadata": {"type": "text", "doc_name": "statement.pdf", "page": 12},
        "score": 0.1,
    }]

    answer = extractor.answer_numeric_query_from_chunks(
        "What were net assets at December 31, 2020?",
        chunks,
    )

    assert answer is not None
    assert "387.1 million" in answer["answer"]
    assert "2018328,732" not in answer["answer"]


def test_explicit_opening_balance_year_does_not_answer_requested_period():
    extractor = DeterministicAnswerExtractor(query_processor=QueryProcessor())
    chunks = [{
        "content": (
            "Net assets at December 31, 2018 were 261,412.\n"
            "At December 31, 2020, net assets were 387.1 million Swiss francs."
        ),
        "metadata": {"type": "text", "doc_name": "statement.pdf", "page": 12},
        "score": 0.1,
    }]

    answer = extractor.answer_numeric_query_from_chunks(
        "What were net assets at December 31, 2020?",
        chunks,
    )

    assert answer is not None
    assert "387.1 million" in answer["answer"]
    assert "261,412" not in answer["answer"]


def test_flattened_multiline_headers_ignore_table_number_column():
    extractor = DeterministicAnswerExtractor(query_processor=QueryProcessor())
    chunks = [{
        "content": (
            "Reserve and surplus | 1 | 2,00,000 | 5,00,000\n"
            "Table column context: Balance Sheet as at March 31, 2017 | "
            "No. | 2017 (Rs.) | 2016 (Rs.)"
        ),
        "metadata": {"type": "table_row", "doc_name": "statement.pdf", "page": 13},
        "score": 0.1,
    }]

    answer = extractor.answer_numeric_query_from_chunks(
        "What reserve and surplus amount is shown for March 31, 2017?",
        chunks,
    )

    assert answer is not None
    assert "2,00,000" in answer["answer"]
    assert "5,00,000" not in answer["answer"]
