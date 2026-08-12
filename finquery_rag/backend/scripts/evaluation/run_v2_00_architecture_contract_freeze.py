#!/usr/bin/env python3
"""V2-00 typed-contract/state-machine freeze (no model or retrieval calls)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts import Action, Intent  # noqa: E402
from rag_v2.orchestration import RepairBudget, State, load_question_envelopes  # noqa: E402


QUESTIONS = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
OUT = ROOT / "artifacts/evaluation/nf-v2-00-architecture-contract-freeze"
BASE_COMMIT = "d5422bbaa4b74722be8644ac98573cb0e150a0e7"
GATE = "V2-00"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    questions = load_question_envelopes(QUESTIONS)
    if len(questions) != 72:
        raise RuntimeError(f"V2 question schema gate expected 72, got {len(questions)}")
    if len({item.question_id for item in questions}) != 72:
        raise RuntimeError("V2 question schema gate found duplicate IDs")

    budget = RepairBudget()
    contract = {
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "evaluation_role": "development_shadow_v2_architecture_contract_freeze",
        "fresh_blind_evaluation": False,
        "model_calls": 0,
        "retrieval_calls": 0,
        "reranker_calls": 0,
        "production_switch_allowed": False,
        "production_architecture": "v1",
        "shadow_architecture": "v2",
        "intents": [item.value for item in Intent],
        "no_answer_intent": False,
        "actions": [item.value for item in Action],
        "states": [item.value for item in State],
        "contracts": [
            "QuestionEnvelope",
            "SupervisorPlan",
            "RequiredSlot",
            "EvidenceBinding",
            "VerifiedEvidencePacket",
            "CalculationResultPacket",
            "CanonicalAnswer",
            "AnswerEnvelope",
            "ValidationResult",
        ],
        "hard_rules": {
            "bound_required_before_generate": True,
            "complete_operands_required_before_calculate": True,
            "validation_pass_required_before_release": True,
            "repair_budget_exhaustion_abstains": True,
            "llm_proposes_only": True,
            "state_machine_executes_only_validated_actions": True,
        },
        "repair_budget": {
            "retrieval_repair_max": budget.retrieval_repair_max,
            "generation_repair_max": budget.generation_repair_max,
            "total_tool_steps_max": budget.total_tool_steps_max,
        },
        "question_schema": {
            "loaded": len(questions),
            "unique_ids": len({item.question_id for item in questions}),
            "answer_fields_loaded": False,
            "gold_fields_loaded": False,
        },
    }
    write_json(OUT / "v2-contract-freeze.json", contract)

    transitions = {
        "gate": GATE,
        "allowed": [
            {"from": "RECEIVED", "event": "PLAN", "to": "PLANNED"},
            {"from": "PLANNED", "action": "RETRIEVE", "to": "RETRIEVED"},
            {"from": "RETRIEVED", "event": "MATERIALIZE", "to": "MATERIALIZED"},
            {"from": "RETRIEVED|MATERIALIZED", "action": "BIND", "to": "BOUND", "guard": "EvidenceBinding.status=BOUND"},
            {"from": "BOUND", "action": "CALCULATE", "to": "CALCULATED", "guard": "plan.intent=CALCULATION"},
            {"from": "BOUND|CALCULATED", "action": "GENERATE", "to": "GENERATED", "guard": "complete bound evidence"},
            {"from": "GENERATED", "event": "VALIDATE", "to": "VALIDATED"},
            {"from": "VALIDATED", "event": "RELEASE", "to": "RELEASED", "guard": "ValidationDecision.PASS"},
            {"from": "VALIDATED", "event": "BEGIN_REPAIR", "to": "REPAIRING", "guard": "repair budget available"},
            {"from": "REPAIRING", "action": "REPAIR_RETRIEVAL", "to": "RETRIEVED", "guard": "retrieval repair budget"},
            {"from": "REPAIRING", "action": "REPAIR_GENERATION", "to": "GENERATED", "guard": "generation repair budget"},
            {"from": "ANY_NONTERMINAL", "action": "ABSTAIN", "to": "ABSTAINED"},
            {"from": "ANY_NONTERMINAL", "action": "STOP", "to": "FAILED"},
        ],
        "terminal_states": ["RELEASED", "ABSTAINED", "FAILED"],
        "illegal_transition_behavior": "reject; no silent repair",
    }
    write_json(OUT / "state-transition-contract.json", transitions)

    question_audit = {
        "gate": GATE,
        "question_count": len(questions),
        "unique_question_ids": len({item.question_id for item in questions}),
        "rows": [item.to_dict() for item in questions],
        "model_calls": 0,
        "retrieval_calls": 0,
        "gold_reads": 0,
        "reference_answer_reads": 0,
    }
    write_json(OUT / "question-schema-audit.json", question_audit)

    readme = """# nano_finance V2-00 — Architecture Contract Freeze

This is a development-shadow contract freeze created from V1 Closure. It
defines typed supervisor/evidence/answer/validation contracts, deterministic
plan validation, repair budgets, and the state-machine transition contract.

No model, retrieval, reranker, PDF parsing, training, or production switch was
performed. The V1 path remains the production default. V2-01 is the next gate.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    decision = {
        "gate": GATE,
        "evaluation_role": "development_shadow_v2_architecture_contract_freeze",
        "fresh_blind_evaluation": False,
        "model_calls": 0,
        "retrieval_calls": 0,
        "production_switch_allowed": False,
        "production_default": "v1",
        "v2_00_contract_frozen": True,
        "question_schema_loaded": "72/72",
        "illegal_state_transitions_rejected": True,
        "repair_budget": budget.__dict__,
        "next_gate": "v2_01_general_llm_supervisor",
        "stop_rule": "pause before supervisor/model execution",
    }
    write_json(OUT / "decision.json", decision)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(OUT.iterdir())
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    write_json(OUT / "artifact-manifest.json", {"gate": GATE, "files": artifact_hashes})
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
