#!/usr/bin/env python3
"""NF-V2-03 R1A formal Attempt 4 after offline binding-contract recovery.

This runner reuses the already frozen Attempt-3 transport/semantic path.  It
only changes the immutable artifact namespace and attempt metadata; no model,
prompt, schema, request, or evaluator behavior is changed here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import run_nf_v2_03_formal_semantic_evidence_binder as formal  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r0e_transport_resilience_and_attempt_3 as attempt3  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


BASE_COMMIT = "ead160d"
MODEL = "qwen3.7-plus"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-formal-attempt-4"
AUDIT = ROOT / "artifacts/evaluation/nf-v2-03-r1a-binding-contract-recovery"
R0E = ROOT / "artifacts/evaluation/nf-v2-03-r0e-transport-resilience"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_attempt_metadata(*, completed: bool, failure: dict[str, Any] | None = None) -> None:
    config_path = OUT / "formal-run-config.json"
    if config_path.exists():
        config = read_json(config_path)
    else:
        config = {}
    config.update({
        "gate": "NF-V2-03-R1A",
        "attempt_number": 4,
        "base_commit": BASE_COMMIT,
        "binder_model": MODEL,
        "binding_contract_recovered": True,
        "previous_attempt_1_invalidated": True,
        "previous_attempt_2_invalidated": True,
        "gold_reads_before_prediction_seal": 0,
        "production_default": "V1",
        "production_switch_allowed": False,
    })
    if (R0E / "transport-resilience-contract.sha256").exists():
        config["transport_resilience_contract_sha256"] = (R0E / "transport-resilience-contract.sha256").read_text(encoding="utf-8").strip()
    write_json(config_path, config)

    history = {
        "attempt_1": {
            "status": "invalidated",
            "reason": "provider_read_timeout",
            "gold_reads": 0,
            "semantic_metrics": "none",
        },
        "attempt_2": {
            "status": "invalidated",
            "reason": "provider_read_timeout",
            "gold_reads": 0,
            "semantic_metrics": "none",
        },
        "attempt_3": {
            "status": "completed_provider_schema_layer",
            "prediction_sealed": True,
            "gold_reads_before_seal": 0,
            "semantic_metrics": "superseded_by_binding_contract_audit",
        },
        "attempt_4": {
            "status": "completed" if completed else "invalidated_runner_or_provider_failure",
            "prediction_sealed": bool(completed and (OUT / "binder-prediction-seal.json").exists()),
            "gold_reads_before_seal": 0,
            "semantic_scoring": bool(completed),
            "failure": failure,
        },
    }
    write_json(OUT / "formal-attempt-history.json", history)

    seal_path = OUT / "binder-prediction-seal.json"
    if seal_path.exists():
        seal = read_json(seal_path)
        seal.update({"gate": "NF-V2-03-R1A", "attempt": 4, "base_commit": BASE_COMMIT})
        write_json(seal_path, seal)

    decision_path = OUT / "decision.json"
    decision = read_json(decision_path) if decision_path.exists() else {}
    decision.update({
        "gate": "NF-V2-03-R1A",
        "attempt": 4,
        "base_commit": BASE_COMMIT,
        "binder_model": MODEL,
        "binding_contract_recovered": True,
        "formal_run_complete": completed,
        "gold_reads_before_prediction_seal": 0,
        "production_default": "V1",
        "production_switch_allowed": False,
    })
    if not completed:
        decision.update({
            "semantic_evidence_binder_effective": "not_evaluated",
            "semantic_binder_frozen": False,
            "dominant_failure": "formal_attempt_4_runner_or_provider_failure",
            "next_gate": "nf_v2_03_binding_contract_failure_review",
        })
    write_json(decision_path, decision)


def main() -> int:
    audit_decision = read_json(AUDIT / "decision.json")
    if not audit_decision.get("binding_contract_recovered"):
        raise SystemExit("binding contract audit did not authorize Attempt 4")
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() != MODEL:
        raise SystemExit("V2_SUPERVISOR_MODEL must remain qwen3.7-plus")
    if not (R0E / "transport-resilience-contract.sha256").exists():
        raise SystemExit("frozen transport resilience contract is missing")

    OUT.mkdir(parents=True, exist_ok=True)
    config = attempt3.provider_config()
    frozen = legacy.load_frozen_inputs()
    verification = formal.verify_frozen_inputs(frozen)

    # The imported function is the exact frozen Attempt-3 formal path.  Only
    # its artifact destination and base metadata are redirected to Attempt 4.
    attempt3.FORMAL_OUT = OUT
    attempt3.BASE_COMMIT = BASE_COMMIT
    attempt3.MODEL = MODEL
    return_code = attempt3.formal_attempt_3(config, frozen, verification)
    failure = read_json(OUT / "formal-failure.json") if (OUT / "formal-failure.json").exists() else None
    rewrite_attempt_metadata(completed=return_code == 0, failure=failure)
    if return_code == 0:
        decision = read_json(OUT / "decision.json")
        print(json.dumps({
            "gate": "NF-V2-03-R1A",
            "attempt": 4,
            "formal_run_complete": decision.get("formal_run_complete"),
            "binder_calls": decision.get("binder_calls"),
            "binding_validator_pass": decision.get("binding_validator_pass"),
            "direct_strict_bindable": decision.get("direct_fact_strict_bindable"),
            "direct_strict_complete": decision.get("direct_fact_strict_complete"),
            "calculation_all_operand_bound": decision.get("calculation_all_operand_bound"),
            "multi_evidence_complete_bound": decision.get("multi_evidence_complete_bound"),
            "semantic_evidence_binder_effective": decision.get("semantic_evidence_binder_effective"),
        }, ensure_ascii=False, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
