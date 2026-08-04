from src.evaluation.nf_opt_07 import AuditInput, Recoverability, classify_recoverability


def test_same_page_does_not_imply_same_table():
    item = AuditInput(
        "key",
        "doc",
        1,
        "hash",
        "| Revenue | 100 | 90 |",
        {"type": "table_row", "evidence_id": "child"},
        None,
    )
    assert classify_recoverability(item)[0] is Recoverability.NOT_RECOVERABLE
