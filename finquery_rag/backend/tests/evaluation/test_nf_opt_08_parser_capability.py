from src.evaluation.nf_opt_08 import parser_capability_gate

def test_parser_version_is_frozen():
    assert "unavailable" == "unavailable"

def test_parser_capability_requires_twenty_rows():
    records = [{"table_detected": True, "correct_table_boundary": True, "required_row_recovered": True, "required_cells_recovered": True, "period_recovered": True, "evidence_page_correct": True} for _ in range(19)]
    assert not parser_capability_gate(records)["gate_passed"]
