#!/usr/bin/env python3
"""Gate 08 R3-A: Freeze Retriever Config / Baseline Parity.

Reads Gate 08 R2 protocol and re-hashes all retriever configs to ensure
R3 prediction uses the exact same parameters.

Gate checks:
  - BUDGETS match (rrf_k=60, lane_k=50, final_pool_k=40, candidate_pool_k=40)
  - RRF config matches (k=60, all weights=1.0)
  - Slot pool config matches (slot_top_k=20, slot_min_budget=10, total_k=40)
  - Embedding model matches (all-MiniLM-L6-v2)
  - Query plan predictions hash matches
  - Source code hashes recorded for audit
  - No parameter scan
  - No per-query oracle

If any config changed:
  decision = coverage_replay_config_parity_blocked
  Do NOT run formal prediction.

Usage:
    python3 scripts/evaluation/freeze_pdf_v4_gate_08_r3_config.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
GATE08_R2_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2"
GATE08_R2_PROTOCOL = GATE08_R2_DIR / "gate-08-r2-protocol.json"
GATE08_R2_PRED_MANIFEST = GATE08_R2_DIR / "prediction-manifest.json"
GATE07_PLANS = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07/query-plan-predictions.json"
)
GATE08_RAW_PARITY = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08/raw-parity.json"
)

SRC_DIR = ROOT / "src/pdf_retrieval_v4"
RETRIEVER_SRC_FILES = [
    "candidate_direct_retriever.py",
    "candidate_query_builder.py",
    "candidate_rrf.py",
    "candidate_slot_pool.py",
    "candidate_view_index.py",
    "candidate_aligned_view.py",
    "query_plan_models.py",
]

# Expected config (must match Gate 08 R2 exactly)
EXPECTED_BUDGETS = {
    "rrf_k": 60,
    "lane_k": 50,
    "final_pool_k": 40,
    "candidate_pool_k": 40,
}
EXPECTED_RRF = {"k": 60, "lane_weights": "all_1.0"}
EXPECTED_SLOT_POOL = {"slot_top_k": 20, "slot_min_budget": 10, "total_k": 40}
EXPECTED_EMBEDDING = "all-MiniLM-L6-v2"
EXPECTED_PARAMETER_SCAN = False
EXPECTED_PER_QUERY_ORACLE = False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def main() -> int:
    print("=" * 70)
    print("Gate 08 R3-A: Retriever Config Parity")
    print("=" * 70)

    R3_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Gate 08 R2 protocol
    print("\n--- Loading Gate 08 R2 Protocol ---")
    r2_protocol = json.loads(GATE08_R2_PROTOCOL.read_text(encoding="utf-8"))
    print(f"  Protocol loaded: {GATE08_R2_PROTOCOL.name}")

    r2_budgets = r2_protocol.get("budgets", {})
    r2_rrf = r2_protocol.get("rrf", {})
    r2_embedding = r2_protocol.get("embedding_model", "")
    r2_param_scan = r2_protocol.get("parameter_scan", True)
    r2_per_query_oracle = r2_protocol.get("per_query_oracle", True)

    print(f"  R2 budgets: {r2_budgets}")
    print(f"  R2 rrf: {r2_rrf}")
    print(f"  R2 embedding: {r2_embedding}")
    print(f"  R2 parameter_scan: {r2_param_scan}")
    print(f"  R2 per_query_oracle: {r2_per_query_oracle}")

    # 2. Compare configs
    print("\n--- Config Parity Check ---")
    gates: dict[str, bool] = {}

    gates["budgets_match"] = r2_budgets == EXPECTED_BUDGETS
    print(f"  budgets_match: {gates['budgets_match']} (expected={EXPECTED_BUDGETS})")

    gates["rrf_match"] = (
        r2_rrf.get("k") == EXPECTED_RRF["k"]
        and r2_rrf.get("lane_weights") == EXPECTED_RRF["lane_weights"]
    )
    print(f"  rrf_match: {gates['rrf_match']} (expected={EXPECTED_RRF})")

    gates["embedding_match"] = r2_embedding == EXPECTED_EMBEDDING
    print(
        f"  embedding_match: {gates['embedding_match']} (expected={EXPECTED_EMBEDDING})"
    )

    gates["no_parameter_scan"] = r2_param_scan == EXPECTED_PARAMETER_SCAN
    print(f"  no_parameter_scan: {gates['no_parameter_scan']}")

    gates["no_per_query_oracle"] = r2_per_query_oracle == EXPECTED_PER_QUERY_ORACLE
    print(f"  no_per_query_oracle: {gates['no_per_query_oracle']}")

    # 3. Hash query plan predictions
    print("\n--- Query Plan Predictions Hash ---")
    plans_hash = _sha256_file(GATE07_PLANS)
    print(f"  plans_hash: {plans_hash[:16]}...")
    print(f"  plans_file: {GATE07_PLANS.name}")

    # Verify plans hash matches R2 protocol
    r2_plans_hash = r2_protocol.get("input_hashes", {}).get("plans", "")
    gates["plans_hash_matches_r2"] = plans_hash == r2_plans_hash
    print(f"  plans_hash_matches_r2: {gates['plans_hash_matches_r2']}")

    # 4. Hash raw parity file
    print("\n--- Raw Parity Hash ---")
    raw_parity_hash = _sha256_file(GATE08_RAW_PARITY)
    print(f"  raw_parity_hash: {raw_parity_hash[:16]}...")

    r2_raw_parity_hash = r2_protocol.get("input_hashes", {}).get("raw_parity", "")
    gates["raw_parity_hash_matches_r2"] = raw_parity_hash == r2_raw_parity_hash
    print(f"  raw_parity_hash_matches_r2: {gates['raw_parity_hash_matches_r2']}")

    # 5. Hash source code files
    print("\n--- Source Code Hashes ---")
    src_hashes: dict[str, str] = {}
    for fname in RETRIEVER_SRC_FILES:
        fpath = SRC_DIR / fname
        if fpath.exists():
            h = _sha256_file(fpath)
            src_hashes[fname] = h
            print(f"  {fname}: {h[:16]}...")
        else:
            src_hashes[fname] = "NOT_FOUND"
            print(f"  {fname}: NOT_FOUND")
            gates[f"src_{fname}_exists"] = False

    # 6. Hash R2 prediction manifest
    print("\n--- R2 Prediction Manifest ---")
    r2_manifest = json.loads(GATE08_R2_PRED_MANIFEST.read_text(encoding="utf-8"))
    r2_pred_hash = r2_manifest.get("prediction_sha256", "")
    r2_record_count = r2_manifest.get("record_count", 0)
    print(f"  R2 prediction_hash: {r2_pred_hash[:16]}...")
    print(f"  R2 record_count: {r2_record_count}")

    gates["r2_record_count_is_72"] = r2_record_count == 72
    print(f"  r2_record_count_is_72: {gates['r2_record_count_is_72']}")

    # 7. Slot pool config (from source code, not protocol)
    # Verify slot_top_k=20, slot_min_budget=10, total_k=40 in source
    slot_pool_src = (SRC_DIR / "candidate_slot_pool.py").read_text(encoding="utf-8")
    gates["slot_top_k_is_20"] = "slot_top_k: int = 20" in slot_pool_src
    gates["slot_min_budget_is_10"] = "slot_min_budget: int = 10" in slot_pool_src
    gates["total_k_is_40"] = "total_k: int = 40" in slot_pool_src
    print(f"\n  slot_top_k_is_20: {gates['slot_top_k_is_20']}")
    print(f"  slot_min_budget_is_10: {gates['slot_min_budget_is_10']}")
    print(f"  total_k_is_40: {gates['total_k_is_40']}")

    # 8. Security
    gates["reranker_calls_zero"] = r2_protocol.get("reranker_calls", 1) == 0
    gates["calculator_calls_zero"] = r2_protocol.get("calculator_calls", 1) == 0
    gates["answer_generation_zero"] = r2_protocol.get("answer_generation_calls", 1) == 0
    gates["production_index_writes_zero"] = (
        r2_protocol.get("production_index_writes", 1) == 0
    )
    gates["production_switch_false"] = (
        r2_protocol.get("production_switch_allowed", True) is False
    )
    print(f"\n  reranker_calls=0: {gates['reranker_calls_zero']}")
    print(f"  calculator_calls=0: {gates['calculator_calls_zero']}")
    print(f"  answer_generation_calls=0: {gates['answer_generation_zero']}")
    print(f"  production_index_writes=0: {gates['production_index_writes_zero']}")
    print(f"  production_switch_allowed=False: {gates['production_switch_false']}")

    all_passed = all(gates.values())

    # 9. Output config-parity.json
    print("\n--- Output ---")
    config_parity = {
        "gate": "gate-08-r3-a",
        "description": "Retriever config parity check before R3 prediction",
        "r2_protocol_path": str(GATE08_R2_PROTOCOL),
        "r2_protocol_hash": _sha256_file(GATE08_R2_PROTOCOL),
        "expected_config": {
            "budgets": EXPECTED_BUDGETS,
            "rrf": EXPECTED_RRF,
            "slot_pool": EXPECTED_SLOT_POOL,
            "embedding_model": EXPECTED_EMBEDDING,
            "parameter_scan": EXPECTED_PARAMETER_SCAN,
            "per_query_oracle": EXPECTED_PER_QUERY_ORACLE,
        },
        "r2_config": {
            "budgets": r2_budgets,
            "rrf": r2_rrf,
            "embedding_model": r2_embedding,
            "parameter_scan": r2_param_scan,
            "per_query_oracle": r2_per_query_oracle,
        },
        "input_hashes": {
            "gate07_plans": plans_hash,
            "gate07_plans_r2_match": gates["plans_hash_matches_r2"],
            "gate08_raw_parity": raw_parity_hash,
            "gate08_raw_parity_r2_match": gates["raw_parity_hash_matches_r2"],
        },
        "source_hashes": src_hashes,
        "r2_prediction": {
            "manifest_path": str(GATE08_R2_PRED_MANIFEST),
            "prediction_sha256": r2_pred_hash,
            "record_count": r2_record_count,
        },
        "gates": gates,
        "all_gates_passed": all_passed,
        "security": {
            "reranker_calls": 0,
            "calculator_calls": 0,
            "answer_generation_calls": 0,
            "production_index_writes": 0,
            "production_switch_allowed": False,
            "parameter_scan": False,
            "per_query_tuning": False,
        },
    }

    parity_path = R3_DIR / "config-parity.json"
    parity_path.write_text(
        json.dumps(config_parity, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Config parity: {parity_path}")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CONFIG PARITY GATES PASSED")
        print("decision = coverage_replay_config_parity_passed")
        print("next_step = gate_08_r3_b_prediction")
    else:
        print("CONFIG PARITY GATES FAILED")
        for gate_name, gate_ok in gates.items():
            if not gate_ok:
                print(f"  FAIL: {gate_name}")
        print("decision = coverage_replay_config_parity_blocked")
        print("DO NOT run formal prediction")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
