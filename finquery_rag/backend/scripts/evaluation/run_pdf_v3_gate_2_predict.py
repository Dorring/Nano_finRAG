"""Gate 2 prediction: question-only Query Profile generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from src.retrieval_v3.query_router import route_question


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/financial_rag_v1/data"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DATA / "questions.golden.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line]
    predictions = []
    for question in questions:
        profile = route_question(str(question["question"]), document_scope=tuple(question.get("document_scope") or ()))
        predictions.append({"case_id": question["case_id"], "raw_question": question["question"], "profile": asdict(profile)})
    _write(args.out_dir / "router-protocol.json", {"gate": "pdf_retrieval_v3_gate_2", "evaluation_type": "post_benchmark_iterative_evaluation", "code_commit": args.code_commit, "questions_sha256": _sha(args.questions), "prediction_inputs": ["question_text", "document_scope"], "forbidden_inputs": ["benchmark-governance.jsonl", "labels.golden.jsonl", "gold_source", "expected_value", "reference_answer"], "retrieval_calls": 0, "parameter_scan": False})
    _write(args.out_dir / "router-predictions.json", {"predictions": predictions})
    prediction_hash = _sha(args.out_dir / "router-predictions.json")
    _write(args.out_dir / "router-prediction-seal.json", {"prediction_count": len(predictions), "protocol_hash": _sha(args.out_dir / "router-protocol.json"), "prediction_hash": prediction_hash, "governance_reads_before_seal": 0, "label_reads_before_seal": 0, "gold_reads_before_seal": 0, "predictions_sealed": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
