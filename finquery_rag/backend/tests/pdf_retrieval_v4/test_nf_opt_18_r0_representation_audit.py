import importlib.util
import json
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[2]
ARTIFACT = BACKEND / "artifacts/evaluation/nf-opt-18-r0-reranker-representation-audit"
SCRIPT = BACKEND / "scripts/evaluation/run_nf_opt_18_r0_representation_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("nf_opt_18_r0_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_offline_decision_and_frozen_counts():
    decision = json.loads((ARTIFACT / "decision.json").read_text(encoding="utf-8"))
    assert decision["model_execution"] is False
    assert decision["retrieval_rerun"] is False
    assert decision["production_switch_allowed"] is False
    assert decision["internal_strict_sources"] == 80
    assert decision["current_top100_hits"] == 68
    assert decision["current_qwen_top5_hits"] == 43
    assert decision["bounded_top100_prediction_sha256"] == "ced014c357a3c8862a5ae3723a4c618556156542fd63aa2a8fd4bf766b60f01a"
    assert decision["bounded_top100_identity_mismatches"] == 0


def test_cohort_union_and_disjoint_identity():
    cohorts = json.loads((ARTIFACT / "failure-cohorts.json").read_text(encoding="utf-8"))
    rows = cohorts["rows"]
    assert len(rows) == 80
    assert len({(r["case_id"], r["source_index"]) for r in rows}) == 80
    assert sum(cohorts["counts"].values()) == 80
    assert cohorts["counts"]["C0_top5_success"] == 43
    assert cohorts["counts"]["C1_top100_present_top5_miss"] == 25
    assert cohorts["counts"]["C2_top100_absent"] == 12


def test_physical_gold_not_replaced_by_semantic_identity():
    inventory = json.loads((ARTIFACT / "internal-candidate-representation-inventory.json").read_text(encoding="utf-8"))
    assert "strict_source_binding_occurrences" in inventory["gold_candidates"]
    assert inventory["gold_candidates"]["strict_source_binding_occurrences"] == 80


def test_graph_availability_and_consumption_are_separate():
    graph = json.loads((ARTIFACT / "semantic-graph-consumption.json").read_text(encoding="utf-8"))
    for item in graph["fields"].values():
        assert item["consumed_on_available_count"] <= item["available_count"]


def test_ambiguity_rule_is_deterministic():
    module = load_module()
    base = module.parse_document_view(
        "[DOCUMENT]\nDocument: d\n[STRUCTURE]\nMetric Path: revenue\nRow: revenue\n[EVIDENCE]\nPeriod: FY2024\n[CONTENT]\nPage: 1\nBlock Type: table_row\nSource:\nrevenue | 1"
    )
    competitor = module.parse_document_view(
        "[DOCUMENT]\nDocument: d\n[STRUCTURE]\nMetric Path: revenue\nRow: revenue\n[EVIDENCE]\nPeriod: FY2024\n[CONTENT]\nPage: 2\nBlock Type: table_row\nSource:\nrevenue | 2"
    )
    left = {"parsed": base}
    right = {"parsed": competitor}
    assert module.ambiguity_record(left, right) == module.ambiguity_record(left, right)
    assert module.ambiguity_record(left, right)["representation_ambiguity"] in {"high", "medium", "low"}


def test_contract_fields_have_machine_status():
    contract = json.loads((ARTIFACT / "internal-reranker-contract.json").read_text(encoding="utf-8"))
    for field in contract["candidate_serialization"]["fields"].values():
        assert set(field) >= {"status", "included_in_model_input"}


def test_packet_design_is_gold_independent_and_identity_preserving():
    packet = json.loads((ARTIFACT / "evidence-packet-v1-design.json").read_text(encoding="utf-8"))
    assert packet["execution"] == "design_only_not_run"
    assert packet["constraints"]["gold_independent"] is True
    assert packet["constraints"]["candidate_identity_preserving"] is True
    assert packet["constraints"]["no_expected_answer"] is True
