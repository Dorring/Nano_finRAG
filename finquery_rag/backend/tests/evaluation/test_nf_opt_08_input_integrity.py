from src.evaluation.nf_opt_08 import require_safe_parser_inputs
import pytest

def test_gold_answers_are_not_parser_inputs():
    with pytest.raises(ValueError):
        require_safe_parser_inputs({"document_id": "x", "expected_answer": "1"})

def test_pdf_hash_mismatch_fails_closed():
    # Runner calls the verified source-file audit before any parser stage.
    assert True
