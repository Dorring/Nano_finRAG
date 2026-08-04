from src.evaluation.nf_opt_07 import AuditInput, Recoverability, classify_recoverability


def _item(content, metadata=None, parent=None):
    return AuditInput(
        "key",
        "doc",
        1,
        "hash",
        content,
        metadata or {"type": "table_row", "evidence_id": "child"},
        parent,
    )


def test_self_contained_table_is_recoverable():
    state, _ = classify_recoverability(
        _item("| Revenue | 2025 | 2024 |\n| Revenue | 100 | 90 |")
    )
    assert state is Recoverability.SELF_CONTAINED_TABLE


def test_parent_header_requires_verified_relation():
    parent = _item(
        "| Revenue | 2025 | 2024 |", {"type": "table", "evidence_id": "parent"}
    )
    child = _item(
        "| Revenue | 100 | 90 |",
        {"type": "table_row", "parent_id": "parent", "evidence_id": "child"},
        parent,
    )
    assert classify_recoverability(child)[0] is Recoverability.RECOVERABLE_FROM_PARENT
