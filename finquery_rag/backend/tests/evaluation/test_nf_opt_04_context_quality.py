from src.evaluation.nf_opt_04 import context_quality

def candidate(key, page, content):
    return {"candidate_key": key, "content_hash": key, "document_id": "d", "page": page, "content": content, "metadata": {"row_label": "Revenue"}}

def test_duplicate_evidence_is_observed():
    first = candidate("a", 1, "Revenue 2024 100")
    second = {**candidate("b", 1, "Revenue 2025 120"), "content_hash": "a"}
    report = context_quality([first, second])
    assert report["duplicate_evidence_count"] == 1
    assert report["same_page_duplicate_count"] == 1

def test_conflicting_period_and_value_are_observed():
    report = context_quality([candidate("a", 1, "Revenue 2024 100"), candidate("b", 2, "Revenue 2025 120")])
    assert report["conflicting_period_count"] == 1
    assert report["conflicting_value_count"] == 1
