"""Gate 07 prediction: build question-only Query Plans, never run retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from src.pdf_retrieval_v4.planner import ConceptResolver, build_query_plan


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/financial_rag_v1/data"
REGISTRY = ROOT / "artifacts/evaluation/pdf-query-representation-v2/concept-registry.json"
GATE2_SEAL = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-2/router-prediction-seal.json"
R4_SCHEMA = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r4/table-schema-classification.json"
R2_MANIFEST = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06-r2/retrieval-view-manifest.json"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scope_hash(questions: list[dict[str, object]]) -> str:
    scopes = [tuple(str(item) for item in question.get("document_scope") or ()) for question in questions]
    payload = json.dumps(scopes, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DATA / "questions.golden.jsonl")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--gate2-seal", type=Path, default=GATE2_SEAL)
    parser.add_argument("--r4-schema", type=Path, default=R4_SCHEMA)
    parser.add_argument("--r2-manifest", type=Path, default=R2_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    resolver = ConceptResolver(args.registry)
    plans = []
    for question in questions:
        scope = tuple(str(item) for item in question.get("document_scope") or ())
        plan = build_query_plan(str(question["question"]), scope, resolver)
        plans.append({"case_id": str(question["case_id"]), "plan": asdict(plan)})

    protocol = {
        "gate": "pdf_retrieval_v4_gate_07",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "code_commit": args.code_commit,
        "plan_version": "pdf-v4-query-plan-v1",
        "questions_sha256": _sha(args.questions),
        "document_scope_sha256": _scope_hash(questions),
        "gate2_prediction_seal_sha256": _sha(args.gate2_seal),
        "gate2_router_inputs": ["question_text", "document_scope"],
        "concept_registry_sha256": _sha(args.registry),
        "temporal_schema_sha256": _sha(args.r4_schema),
        "index_type_manifest_sha256": _sha(args.r2_manifest),
        "prediction_inputs": ["question_text", "document_scope", "gate2_runtime_router", "concept_registry"],
        "forbidden_inputs": ["benchmark-governance.jsonl", "labels.golden.jsonl", "gold_source", "expected_value", "reference_answer", "evidence-family-map.json", "runtime_indexes"],
        "index_reads": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "parameter_scan": False,
        "per_query_oracle": False,
    }
    _write(args.out_dir / "gate-07-protocol.json", protocol)
    _write(args.out_dir / "gate-07-input-integrity.json", {
        "questions_sha256": protocol["questions_sha256"],
        "document_scope_sha256": protocol["document_scope_sha256"],
        "gate2_prediction_seal_sha256": protocol["gate2_prediction_seal_sha256"],
        "concept_registry_sha256": protocol["concept_registry_sha256"],
        "temporal_schema_sha256": protocol["temporal_schema_sha256"],
        "index_type_manifest_sha256": protocol["index_type_manifest_sha256"],
    })
    _write(args.out_dir / "query-plan-predictions.json", {"plans": plans})
    prediction_hash = _sha(args.out_dir / "query-plan-predictions.json")
    _write(args.out_dir / "query-plan-prediction-seal.json", {
        "prediction_count": len(plans),
        "protocol_hash": _sha(args.out_dir / "gate-07-protocol.json"),
        "prediction_hash": prediction_hash,
        "index_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "gold_reads_before_seal": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "answer_generation_calls": 0,
        "sealed": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

