from src.evaluation.nf_opt_08 import parser_capability_gate

def _good():
    return {"table_detected": True, "correct_table_boundary": True, "required_row_recovered": True, "required_cells_recovered": True, "period_recovered": True, "scale_recovered": True, "currency_recovered": True, "evidence_page_correct": True, "wrong_table_selected": False, "wrong_row_mapped": False, "wrong_column_mapped": False, "cross_table_join": False, "page_mismatch": False}

def test_gate_accepts_fully_verified_parser_output():
    assert parser_capability_gate([_good() for _ in range(22)])["gate_passed"]

def test_gate_fails_closed_for_page_mismatch():
    rows = [_good() for _ in range(22)]
    rows[0]["page_mismatch"] = True
    assert not parser_capability_gate(rows)["gate_passed"]
