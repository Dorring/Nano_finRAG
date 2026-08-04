from src.evaluation.nf_opt_07 import (
    AuditInput,
    Recoverability,
    classify_recoverability,
    has_scale,
)


def test_currency_and_scale_are_not_guessed():
    item = AuditInput(
        "key", "doc", 1, "hash", "| Revenue | 100 |", {"type": "table_row"}, None
    )
    assert not has_scale(item)


def test_missing_period_blocks():
    item = AuditInput(
        "key", "doc", 1, "hash", "| Revenue | 100 | 90 |", {"type": "table_row"}, None
    )
    assert classify_recoverability(item)[0] is Recoverability.NOT_RECOVERABLE
