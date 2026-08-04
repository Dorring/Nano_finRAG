from src.evaluation.nf_opt_07 import AuditInput, has_numeric_cells


def test_dash_is_not_a_numeric_cell():
    item = AuditInput(
        "key", "doc", 1, "hash", "| Revenue | - | N/A |", {"type": "table_row"}, None
    )
    assert not has_numeric_cells(item)


def test_parentheses_are_detected_as_numeric_cell():
    item = AuditInput(
        "key", "doc", 1, "hash", "| Revenue | (1,234) |", {"type": "table_row"}, None
    )
    assert has_numeric_cells(item)
