from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf37_metrics import all_source_coverage_at_k, case_hit_rate_at_k, source_recall_at_k

def test_case_hit_rate_and_source_recall_are_distinct():
    case = EvaluationCase("a","q",expected_sources=(ExpectedSource(filename="a.pdf",page=1),ExpectedSource(filename="a.pdf",page=2)))
    rankings={"a":[{"filename":"a.pdf","page":1}]}
    assert case_hit_rate_at_k([case],rankings,1) == 1.0
    assert source_recall_at_k([case],rankings,1) == 0.5
    assert all_source_coverage_at_k([case],rankings,1) == 0.0

def test_no_answer_cases_excluded_from_retrieval_denominator():
    case = EvaluationCase("no","q",expected_no_answer=True)
    assert case_hit_rate_at_k([case],{},5) == 1.0

def test_document_id_candidate_matches_expected_filename():
    case = EvaluationCase(
        "a",
        "q",
        expected_sources=(ExpectedSource(filename="a.pdf", page=3),),
    )
    rankings = {"a": [{"document_id": "a.pdf", "page": 3}]}
    assert case_hit_rate_at_k([case], rankings, 1) == 1.0
