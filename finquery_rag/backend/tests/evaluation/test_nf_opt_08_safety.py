from src.evaluation.nf_opt_08 import parser_capability_gate

def test_wrong_table_blocks():
    records = [{"wrong_table_selected": True} for _ in range(22)]
    assert not parser_capability_gate(records)["gate_passed"]

def test_cross_page_join_requires_verified_relation():
    records = [{"cross_table_join": True} for _ in range(22)]
    assert not parser_capability_gate(records)["gate_passed"]

def test_production_default_is_unchanged():
    from src.evaluation.nf_opt_08 import ENABLE_TABLE_FACT_EXTRACTION
    assert not ENABLE_TABLE_FACT_EXTRACTION
