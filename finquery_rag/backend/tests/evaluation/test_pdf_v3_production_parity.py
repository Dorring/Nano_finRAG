import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-0"


def test_production_final_identity_parity_is_exact() -> None:
    parity = json.loads((OUT / "baseline-case-parity.json").read_text())
    assert len(parity["authoritative_final_hits"]) == 13
    assert parity["authoritative_final_hits"] == parity["replayed_final_hits"]
    assert parity["missing_hits"] == []
    assert parity["unexpected_hits"] == []


def test_gate_uses_current_production_snapshot_not_historical_shadows() -> None:
    acceptance = json.loads((OUT / "acceptance.json").read_text())
    replay = json.loads((OUT / "production-stage-replay.json").read_text())["metrics"]
    ledger = json.loads((OUT / "metric-provenance-ledger.json").read_text())["records"]
    nestedness = json.loads((OUT / "stage-nestedness-audit.json").read_text())
    assert replay["strict_final_source_recall_at_5"] == 13
    assert replay["rrf_source_recall_at_40"] == 20
    assert replay["bm25_source_recall_at_200"] == 37
    assert replay["dense_source_recall_at_200"] == 14
    assert acceptance["gate_passed"] is True
    assert acceptance["next_gate"] == "benchmark_governance"
    assert all(
        not record["production_promoted"]
        for record in ledger
        if record["evaluation_role"].startswith("historical_shadow")
    )
    assert nestedness["unexplained_transition_count"] == 0
