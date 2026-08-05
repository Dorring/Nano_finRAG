"""NF-OPT-17 Gate C: independently review generated dev annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.nf_opt_17 import validate_generated_annotation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-17-gate-c"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _review_row(row: dict[str, Any]) -> dict[str, Any]:
    """Check query wording and hard-negative relations without answer values."""
    validate_generated_annotation(row)
    question = str(row["question"])
    positive = row["positive_candidate"]
    negatives = {item["negative_type"]: item["candidate"] for item in row["hard_negatives"]}
    issuer = str(row["issuer"])
    if issuer not in question or str(positive["period_end"]) not in question:
        raise ValueError("query does not state the positive issuer and period")
    wrong_period = negatives["same_row_wrong_period"]
    wrong_metric = negatives["same_table_wrong_metric"]
    if (positive["table_index"], positive["row_index"]) != (
        wrong_period["table_index"],
        wrong_period["row_index"],
    ):
        raise ValueError("same-row negative has incompatible row lineage")
    if positive["period_end"] == wrong_period["period_end"]:
        raise ValueError("same-row negative has the same period")
    if positive["table_index"] != wrong_metric["table_index"]:
        raise ValueError("same-table negative has incompatible table lineage")
    if positive["xbrl_concept"] == wrong_metric["xbrl_concept"]:
        raise ValueError("same-table negative has the same concept")
    return {
        "annotation_id": row["annotation_id"],
        "query_sha256": _sha(question),
        "positive_candidate_key": positive["candidate_key"],
        "negative_candidate_keys": sorted(
            candidate["candidate_key"] for candidate in negatives.values()
        ),
        "ai_review_status": "structural_and_lexical_pass",
        "human_review_status": "not_reviewed",
        "review_mode": "authorized_ai_assisted_independent_review",
    }


def run(args: argparse.Namespace) -> int:
    rows = _read_rows(args.annotations)
    if len(rows) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} annotations, got {len(rows)}")
    reviews = [_review_row(row) for row in rows]
    annotation_ids = [str(row["annotation_id"]) for row in rows]
    questions = [str(row["question"]) for row in rows]
    positive_keys = [str(row["positive_candidate"]["candidate_key"]) for row in rows]
    if len(set(annotation_ids)) != len(rows):
        raise ValueError("annotation IDs are not unique")
    if len(set(questions)) != len(rows):
        raise ValueError("development questions are not unique")
    if len(set(positive_keys)) != len(rows):
        raise ValueError("positive candidates are not unique")
    issuer_counts = Counter(str(row["issuer"]) for row in rows)
    review = {
        "schema": "nf-opt-17/independent-annotation-review/v1",
        "reviewer_type": "ai_reviewer",
        "reviewer_id": "authorized_ai_assisted_annotation_reviewer",
        "authorized_by_user": True,
        "annotation_count": len(rows),
        "review_records": sorted(reviews, key=lambda record: str(record["annotation_id"])),
    }
    quality = {
        "annotation_count": len(rows),
        "issuer_counts": dict(sorted(issuer_counts.items())),
        "unique_annotation_count": len(set(annotation_ids)),
        "unique_query_count": len(set(questions)),
        "unique_positive_candidate_count": len(set(positive_keys)),
        "ai_review_pass_count": len(reviews),
        "human_review_count": 0,
        "human_review_claimed": False,
        "expected_answer_values_inspected": False,
    }
    acceptance = {
        "schema": "nf-opt-17/gate-c/acceptance/v1",
        "annotation_count": len(rows),
        "ai_review_pass_count": len(reviews),
        "human_review_count": 0,
        "expected_answer_values_inspected": False,
        "frozen_benchmark_question_or_label_reads": 0,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "decision": "independent_hard_negative_development_set_ai_reviewed",
        "next_gate": "nf-opt-17-gate-d-shadow_candidate_corpus",
    }
    _write(args.out_dir / "independent-annotation-review-report.json", review)
    _write(args.out_dir / "annotation-quality-report.json", quality)
    _write(args.out_dir / "next-gate.json", {
        "decision": acceptance["decision"],
        "next_gate": acceptance["next_gate"],
        "production_switch_allowed": False,
    })
    _write(args.out_dir / "nf-opt-17-gate-c-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-count", type=int, default=80)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
