"""CPU-safe checks for the NF-OPT-26 read-only freeze package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[2]
SCRIPT = BACKEND / "scripts/evaluation/run_nf_opt_26_r0_internal_retrieval_freeze.py"
OUT = BACKEND / "artifacts/evaluation/nf-opt-26-r0-internal-retrieval-freeze"
SPEC = importlib.util.spec_from_file_location("nf26_r0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate_is_read_only_and_selected_method_is_frozen():
    assert MODULE.BASE_COMMIT == "2ae5b577eddb5fa507cbc1176599cb249b2d554e"
    assert MODULE.MODEL == "Qwen/Qwen3-Reranker-4B"
    assert MODULE.MODEL_REVISION == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    assert MODULE.OUTER_FLAGS == {
        "model_execution": False,
        "retrieval_rerun": False,
        "admission_rerun": False,
        "training": False,
        "parameter_tuning": False,
    }
    assert "torch" not in SCRIPT.read_text(encoding="utf-8")


def test_percentage_is_recomputed_from_counts():
    assert MODULE.pct(78, 80) == 0.975
    assert MODULE.pct(46, 80) == 0.575
    assert MODULE.pct(4, 4) == 1.0


def test_freeze_artifacts_have_required_invariants():
    if not OUT.exists():
        return
    decision = read_json(OUT / "decision.json")
    metrics = read_json(OUT / "final-internal-retrieval-metrics.json")
    method = read_json(OUT / "internal-retrieval-method-freeze.json")
    assert decision["internal_retrieval_method_frozen"] is True
    assert decision["selected_internal_shadow_method"] == "sada_statement_aware_v1"
    assert decision["production_switch_allowed"] is False
    assert metrics["deep_supply"] == {"hits": 78, "total": 80, "recall": 0.975}
    assert metrics["sada_top100"] == {"hits": 78, "total": 80, "recall": 0.975}
    assert metrics["selected_method"]["r5_hits"] == 46
    assert metrics["selected_method"]["r100_hits"] == 78
    assert method["further_internal_retrieval_tuning_allowed"] is False


def test_rejected_routes_cannot_reopen_without_new_evidence():
    if not OUT.exists():
        return
    routes = read_json(OUT / "closed-routes.json")
    assert routes
    assert all(route["reopen_allowed"] is False for route in routes)
    assert {route["route"] for route in routes} >= {
        "Query Requirement V1",
        "Instruction V1",
        "LRRF tuning",
        "confidence-gated LRRF",
        "1.5B listwise selector",
        "representation packet",
        "semantic sibling collapse",
    }


def test_manifest_digest_is_deterministic_and_excludes_self_referential_files():
    if not OUT.exists():
        return
    manifest_path = OUT / "final-evidence-manifest.json"
    digest_path = OUT / "final-evidence-manifest.sha256"
    payload = manifest_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == digest_path.read_text(encoding="utf-8").strip()
    manifest = json.loads(payload)
    assert manifest["deterministic_order"] == "lexicographic relative path"
    assert set(manifest["self_referential_exclusions"]) == {
        "decision.json",
        "final-evidence-manifest.json",
        "final-evidence-manifest.sha256",
    }
    listed = {entry["path"] for entry in manifest["files"]}
    assert "decision.json" not in listed
    assert "README.md" in listed


def test_claim_registry_preserves_shadow_disclosure_and_no_leaderboard_claim():
    if not OUT.exists():
        return
    claims = read_json(OUT / "retrieval-claim-registry.json")["claims"]
    assert claims
    for claim in claims:
        assert claim["fresh_blind"] is False
        assert claim["artifact"]
        assert claim["commit"]
        assert "SOTA" not in claim.get("allowed_wording", "")
