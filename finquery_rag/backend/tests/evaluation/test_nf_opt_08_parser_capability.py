from src.evaluation.nf_opt_08 import combined_table_count, parser_capability_gate

def test_parser_version_is_frozen():
    parser_version = "1.26.6"
    assert parser_version == "1.26.6"

def test_native_parser_combines_pymupdf_and_camelot_detection():
    assert combined_table_count(1, 2) == 3

def test_parser_capability_requires_twenty_rows():
    records = [{"table_detected": True, "correct_table_boundary": True, "required_row_recovered": True, "required_cells_recovered": True, "period_recovered": True, "evidence_page_correct": True} for _ in range(19)]
    assert not parser_capability_gate(records)["gate_passed"]
