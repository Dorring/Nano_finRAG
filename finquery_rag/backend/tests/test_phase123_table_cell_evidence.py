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

    extractor = DeterministicAnswerExtractor(query_processor=QueryProcessor())
    answer = extractor.answer_numeric_query_from_chunks(
        "What were net assets at December 31, 2020?",
        cells,
    )

    assert answer is not None
    assert "387,063" in answer["answer"]
    assert "565,601" not in answer["answer"]
