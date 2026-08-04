from src.evaluation.nf_opt_07 import AuditInput, is_table_candidate


def test_candidate_identity_is_part_of_audit_input():
    item = AuditInput(
        "candidate", "doc", 1, "hash", "| Revenue | 100 |", {"type": "table_row"}, None
    )
    assert item.candidate_key == "candidate"
    assert is_table_candidate(item)
