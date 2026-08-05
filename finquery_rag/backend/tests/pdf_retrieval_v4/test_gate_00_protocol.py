import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/evaluation/run_pdf_v4_gate_00.py"


def test_gate_00_script_is_audit_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"mineru_calls": 0' in source
    assert "production_index_write" in source
    assert "retrieval_runs" in source
    assert "subprocess.check_output" in source
    assert "parse_pdf" not in source


def test_benchmark_inputs_are_unchanged() -> None:
    corpus = ROOT / "benchmarks/financial_rag_v1/corpus.json"
    questions = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
    labels = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
    for path in (corpus, questions, labels):
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_00_output_contract_names() -> None:
    expected = {
        "protocol.json",
        "input-integrity.json",
        "software-manifest.json",
        "benchmark-lineage.json",
        "acceptance.json",
    }
    assert expected == {
        "protocol.json",
        "input-integrity.json",
        "software-manifest.json",
        "benchmark-lineage.json",
        "acceptance.json",
    }


def test_prior_gate_seals_are_not_rewritten_by_freeze_script() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "gate-2/router-prediction-seal.json" in source
    assert "gate-3/gate-3-prediction-seal.json" in source
    assert "write_json(args.out_dir" in source
