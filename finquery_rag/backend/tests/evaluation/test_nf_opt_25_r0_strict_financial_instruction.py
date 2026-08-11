import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluation/run_nf_opt_25_r0_strict_financial_instruction.py"
SPEC = importlib.util.spec_from_file_location("nf25_r0", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_contract_constants():
    assert MODULE.BASE_COMMIT == "dd1b64ca0d11b0f20c2ceb6096bb1e39bb68470e"
    assert MODULE.MODEL_ID == "Qwen/Qwen3-Reranker-4B"
    assert MODULE.REVISION == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    assert MODULE.MAX_LENGTH == 8192
    assert MODULE.TOP100_SADA_REL.endswith("sada-v1-top100-predictions.jsonl.gz")


def test_strict_instruction_v1_is_sealed_and_exact():
    expected = (
        "Given a financial question and an evidence candidate, judge whether the candidate provides direct evidence needed to answer the question.\n\n"
        "Prioritize exact agreement on the financial metric, reporting period, financial statement or row context, and required operand evidence.\n\n"
        "Prefer direct financial evidence over merely related discussion, commentary, or semantically similar but indirect evidence."
    )
    assert MODULE.STRICT_FINANCIAL_INSTRUCTION == expected
    assert MODULE.sha256_text(MODULE.STRICT_FINANCIAL_INSTRUCTION) == "7e9088d483a1204355ff2ed2a80286bf2ab5657bcf7541550a5cc3b239dfd90d"
    assert "currency" not in MODULE.STRICT_FINANCIAL_INSTRUCTION.lower()
    assert "scale" not in MODULE.STRICT_FINANCIAL_INSTRUCTION.lower()


def test_runtime_uses_original_query_bytes():
    query = "[QUESTION]\nWhat was revenue?"
    qviews = {"case": {"main_query_view": query, "main_query_view_sha256": MODULE.sha256_text(query)}}
    rows = [{"case_id": "case", "candidates": [{"candidate_key": "c1"}]}]
    prepared = MODULE.prepare_runtime_rows(rows, qviews)
    assert prepared[0]["original_query"] == query
    assert prepared[0]["original_query_sha256"] == qviews["case"]["main_query_view_sha256"]


def test_nf25_does_not_add_instruction_rules():
    assert "Query Requirement V2" not in MODULE.STRICT_FINANCIAL_INSTRUCTION
    assert "score calibration" not in MODULE.STRICT_FINANCIAL_INSTRUCTION.lower()
