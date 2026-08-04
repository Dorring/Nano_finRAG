from src.evaluation.nf_opt_15 import build_retrieval_view


DOCUMENT = {"document_id": "aapl_fy2025", "company": "Apple", "source_type": "official_annual_report", "fiscal_year": 2025}


def test_table_row_without_header_does_not_guess_periods():
    view = build_retrieval_view(doc_id="doc::page_1::table_1::row_2", content="Total net sales | 416,161 | 391,035", metadata={"type": "table_row", "doc_name": "aapl_fy2025_10k.pdf", "user_id": 1, "page": 1, "table_header_context": ""}, document=DOCUMENT)
    assert view["metric_field"]["normalized_metric"] == "total net sales"
    assert view["period_field"]["status"] == "missing"


def test_table_header_generates_period_unit_and_stable_original_identity():
    view = build_retrieval_view(doc_id="doc::page_1::table_1::row_2", content="Total net sales | 416,161 | 391,035", metadata={"type": "table_row", "doc_name": "aapl_fy2025_10k.pdf", "user_id": 1, "page": 1, "table_header_context": "($ in millions) 2025 2024"}, document=DOCUMENT)
    assert view["period_field"]["periods"] == ["FY2025", "FY2024"]
    assert view["unit_field"]["currency"] == "USD"
    assert view["unit_field"]["scale"] == "million"
    assert view["candidate_key"].startswith("candidate:v1:")
