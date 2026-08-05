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
