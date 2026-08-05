import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks/financial_rag_v1"
GOVERNANCE = BENCHMARK / "governance"
ARTIFACTS = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-1"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_governance_covers_every_case_and_gold_source() -> None:
    records = _jsonl(GOVERNANCE / "benchmark-governance.jsonl")
    assert len(records) == 72
    assert sum(len(record["strict_gold_source_bindings"]) for record in records) == 80
    assert all(
        record["required_evidence_count"] == len(record["operand_slots"])
        for record in records
    )
    assert all(record["review_status"] == "ai_assisted_pending_manual_review" for record in records)


def test_no_answer_has_no_gold_source_and_operand_contract_is_consistent() -> None:
    records = _jsonl(GOVERNANCE / "benchmark-governance.jsonl")
    assert all(
        not record["strict_gold_source_bindings"]
        for record in records
        if record["query_type"] == "no_answer"
    )
    assert all(
        len(record["operand_slots"]) >= record["minimum_evidence_count"]
        for record in records
    )
    assert all(
        record["operation"]
        for record in records
        if record["query_type"] == "calculation_multi_operand"
    )


def test_governance_preserves_the_frozen_benchmark_and_family_identity() -> None:
    integrity = json.loads((GOVERNANCE / "governance-integrity.json").read_text())
    families = json.loads((GOVERNANCE / "evidence-family-map.json").read_text())["families"]
    acceptance = json.loads((ARTIFACTS / "acceptance.json").read_text())
    assert integrity["questions_golden_sha256_before"] == integrity["questions_golden_sha256_after"]
    assert integrity["labels_golden_sha256_before"] == integrity["labels_golden_sha256_after"]
    assert integrity["gold_expansion_count"] == 0
    assert integrity["candidate_family_conflict_count"] == 0
    assert integrity["source_identity_hash_from_governance"]
    assert integrity["question_hash_manifest_status"] in {
        "matched",
        "legacy_manifest_mismatch_recorded_no_mutation",
    }
    assert len({family["evidence_family_id"] for family in families}) == len(families)
    assert acceptance["gate_passed"] is True
    assert acceptance["next_gate"] == "query_profile_router"


def test_artifact_integrity_conflicts_and_strict_baseline_are_frozen() -> None:
    frozen = json.loads((ARTIFACTS / "frozen-benchmark-integrity.json").read_text())
    conflicts = json.loads((ARTIFACTS / "governance-conflict-report.json").read_text())
    baseline = json.loads((ARTIFACTS / "baseline-multigranularity-metrics.json").read_text())
    operand_audit = json.loads((ARTIFACTS / "operand-slot-audit.json").read_text())
    assert frozen["questions_unchanged"] is True
    assert frozen["labels_unchanged"] is True
    assert frozen["strict_gold_source_identity_set_hash"]
    assert conflicts["conflict_count"] == 0
    assert baseline["strict_candidate_recall_at_5"] == "13/80"
    assert operand_audit["all_slots_complete"] is True
