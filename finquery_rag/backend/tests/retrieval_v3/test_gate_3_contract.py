from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_gate_3_prediction_does_not_import_governance_or_labels() -> None:
    source = (ROOT / "scripts/evaluation/run_pdf_v3_gate_3_predict.py").read_text(encoding="utf-8")
    assert "labels.golden.jsonl" not in source
    assert "benchmark-governance.jsonl" not in source
    assert "score_pdf_v3_gate_3_pool" not in source


def test_gate_3_prediction_has_no_reranker_or_final_selector() -> None:
    source = (ROOT / "scripts/evaluation/run_pdf_v3_gate_3_predict.py").read_text(encoding="utf-8")
    assert "HeuristicReranker" not in source
    assert ".rerank(" not in source
    assert "FinalSelector" not in source
    assert ".select_final(" not in source


def test_gate_3_scoring_is_seal_guarded() -> None:
    source = (ROOT / "scripts/evaluation/score_pdf_v3_gate_3_pool.py").read_text(encoding="utf-8")
    assert "prediction seal hash verification failed" in source
    assert "labels_read_before_seal" in source
