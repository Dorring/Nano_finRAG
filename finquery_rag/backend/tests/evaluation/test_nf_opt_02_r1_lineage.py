from src.evaluation.nf_opt_02_r1 import lineage_report

def _candidate(key, document="doc-1"):
    return {"candidate_key": key, "document_id": document}

def test_reranker_receives_complete_rrf_list():
    report = lineage_report(rrf_input=[_candidate("a"), _candidate("b")], reranker_output=[_candidate("b"), _candidate("a")], final_output=[_candidate("b")], allowed_document_ids={"doc-1"})
    assert report["reranker_input_source"] == "rrf_all"
    assert report["reranker_input_count"] == 2

def test_reranker_does_not_receive_only_top40():
    report = lineage_report(rrf_input=[_candidate(str(i)) for i in range(41)], reranker_output=[_candidate("0")], final_output=[_candidate("0")], allowed_document_ids={"doc-1"})
    assert report["reranker_input_count"] == 41

def test_reranker_output_is_subset_of_input():
    report = lineage_report(rrf_input=[_candidate("a")], reranker_output=[_candidate("new")], final_output=[], allowed_document_ids={"doc-1"})
    assert report["lineage_passed"] is False

def test_final_output_is_subset_of_reranker():
    report = lineage_report(rrf_input=[_candidate("a")], reranker_output=[_candidate("a")], final_output=[_candidate("new")], allowed_document_ids={"doc-1"})
    assert report["lineage_passed"] is False
