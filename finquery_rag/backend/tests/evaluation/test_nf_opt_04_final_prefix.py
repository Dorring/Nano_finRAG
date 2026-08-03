from src.evaluation.nf_opt_04 import prefix_report, select_prefix

def candidate(key):
    return {"candidate_key": key, "document_id": "d", "content_hash": key}

def test_f5_is_prefix_of_f8():
    items = [candidate(str(index)) for index in range(10)]
    assert prefix_report(select_prefix(items, max_evidence=5), select_prefix(items, max_evidence=8))["passed"] is True

def test_f8_is_prefix_of_f10():
    items = [candidate(str(index)) for index in range(10)]
    assert prefix_report(select_prefix(items, max_evidence=8), select_prefix(items, max_evidence=10))["passed"] is True

def test_expanding_final_budget_cannot_remove_candidate():
    items = [candidate(str(index)) for index in range(10)]
    assert [item["candidate_key"] for item in select_prefix(items, max_evidence=5)] == [str(index) for index in range(5)]
