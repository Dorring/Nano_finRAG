import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/evaluation/run_pdf_v4_gate_01.py"
SHADOW = ROOT / "artifacts/evaluation/nf-opt-08/shadow-page-set-manifest.json"
ORACLE = ROOT / "artifacts/evaluation/nf-opt-08-r2/manual-mapping-review-package.json"


def test_gate_01_uses_fixed_two_backend_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"hybrid_high"' in source
    assert '"pipeline_auto_ocr"' in source
    assert '"hybrid-engine"' in source
    assert '"pipeline"' in source
    assert '"method": "auto"' in source
    assert '"effort": "high"' in source
    assert "parameter_scan" in source


def test_gate_01_input_contract_is_present() -> None:
    shadow = json.loads(SHADOW.read_text(encoding="utf-8"))
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    assert shadow["page_count"] == 84
    assert len(shadow["pages"]) == 84
    assert len(oracle["records"]) == 22


def test_gate_01_does_not_build_indexes_or_run_retrieval() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "retrieval_runs" in source
    assert "index_builds" in source
    assert "production_index_writes" in source
    assert "sentence_transformers" not in source
    assert "Reranker" not in source


def test_gate_01_parser_output_is_kept_separate_from_posthoc_oracle() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "oracle_annotations_read_posthoc" in source
    assert "runtime_gold_reads" in source
    assert "source_identity_is_parser_input" in source
    assert "legacy_candidate_key" in source


def test_gate_01_raw_artifact_contract_names_are_stable() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for name in ("*_middle.json", "*_content_list.json", "*_model.json", "mineru.log"):
        assert name in source
    for name in ("mineru-probe-protocol.json", "probe-input-manifest.json", "backend-results.json", "capability-metrics.json", "acceptance.json", "next-gate.json"):
        assert name in source
