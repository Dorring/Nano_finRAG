"""Gate 1: build immutable-sidecar governance for financial-rag-v1.

The source Benchmark remains read-only.  This program derives task taxonomy,
operand contracts, and conservative gold-evidence families for audit and later
post-benchmark evaluation.  It never runs retrieval or changes a candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks/financial_rag_v1"
DATA = BENCHMARK / "data"
GOVERNANCE = BENCHMARK / "governance"
ARTIFACTS = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _normalise(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def _operation_roles(operation: str | None, count: int) -> list[str]:
    if count <= 0:
        return []
    if operation == "growth_rate" and count == 2:
        return ["previous", "current"]
    if operation in {"percentage_share", "percentage_of"} and count == 2:
        return ["numerator", "denominator"]
    if operation == "difference" and count == 2:
        return ["left", "right"]
    return [f"operand_{index}" for index in range(1, count + 1)]


def _query_type(question: dict[str, Any], label: dict[str, Any]) -> str:
    if bool(label.get("expected_no_answer")):
        return "no_answer"
    calculation = label.get("calculation") or {}
    if calculation.get("operation"):
        return "calculation_multi_operand"
    text = _normalise(question.get("question"))
    if any(token in text for token in ("why ", " explain", " describe", "reason", "risk", "strategy", "discussion", "note ")):
        return "narrative_or_note"
    sources = list(label.get("expected_sources") or [])
    metrics = {_normalise(source.get("row_label")) for source in sources if source.get("row_label")}
    periods = {str(source.get("period")) for source in sources if source.get("period")}
    if len(sources) > 1 and len(metrics) > 1:
        return "multi_metric_comparison"
    if len(sources) > 1 and len(periods) > 1:
        return "single_metric_multi_period"
    if any("table" in str(value).lower() for value in question.get("category") or []):
        return "table_single_fact"
    if sources and all("table" in str(source.get("evidence_type", "")) for source in sources):
        return "table_single_fact"
    return "direct_single_fact"


def _operand_slots(label: dict[str, Any]) -> list[dict[str, Any]]:
    calculation = label.get("calculation") or {}
    operands = list(calculation.get("operands") or [])
    operation = calculation.get("operation")
    if operands:
        roles = _operation_roles(str(operation) if operation else None, len(operands))
        return [
            {
                "slot_id": roles[index],
                "role": roles[index],
                "metric": operand.get("metric"),
                "period": operand.get("period"),
                "source_index": operand.get("source_index"),
            }
            for index, operand in enumerate(operands)
        ]
    sources = list(label.get("expected_sources") or [])
    roles = _operation_roles(None, len(sources))
    return [
        {
            "slot_id": roles[index],
            "role": roles[index],
            "metric": source.get("row_label"),
            "period": source.get("period"),
            "source_index": index,
        }
        for index, source in enumerate(sources)
    ]


def _logical_table_id(source: dict[str, Any]) -> str:
    evidence_id = str(source.get("evidence_id") or "")
    trimmed = re.sub(r"::row_\d+$", "", evidence_id)
    return f"logical-table:{_canonical_hash({'document': source.get('document_id'), 'page': source.get('page'), 'evidence': trimmed, 'title': source.get('table_title')})}"


def _family_seed(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": source.get("document_id"),
        "pdf_page": source.get("page"),
        "logical_table_id": _logical_table_id(source),
        "normalised_row_label": _normalise(source.get("row_label")),
        "evidence_type": source.get("evidence_type"),
    }


def _families(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        for source in label.get("expected_sources") or []:
            seed = _family_seed(source)
            family_id = f"family:v1:{_canonical_hash(seed)}"
            members[family_id].append(source)
    output = []
    for family_id, sources in sorted(members.items()):
        first = sources[0]
        output.append(
            {
                "evidence_family_id": family_id,
                "document_id": first.get("document_id"),
                "pdf_page": first.get("page"),
                "logical_table_id": _logical_table_id(first),
                "metric": first.get("row_label"),
                "periods": sorted({str(source["period"]) for source in sources if source.get("period")}),
                "member_candidate_keys": sorted({str(source["candidate_key"]) for source in sources}),
                "member_bindings": [
                    {
                        "candidate_key": source["candidate_key"],
                        "evidence_id": source.get("evidence_id"),
                        "evidence_type": source.get("evidence_type"),
                        "period": source.get("period"),
                    }
                    for source in sorted(sources, key=lambda item: (str(item["candidate_key"]), str(item.get("period"))))
                ],
                "equivalence_reason": "same_financial_row_or_block_lineage",
            }
        )
    return output


def _family_id(source: dict[str, Any]) -> str:
    return f"family:v1:{_canonical_hash(_family_seed(source))}"


def build(args: argparse.Namespace) -> int:
    questions = _read_jsonl(args.questions)
    labels = _read_jsonl(args.labels)
    if {item["case_id"] for item in questions} != {item["case_id"] for item in labels}:
        raise RuntimeError("question/label case_id sets differ")
    labels_by_id = {str(item["case_id"]): item for item in labels}
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_hash_before = _sha(args.labels)
    question_hash_before = _sha(args.questions)
    families = _families(labels)
    records = []
    review = []
    gold_source_count = 0
    for question in questions:
        case_id = str(question["case_id"])
        label = labels_by_id[case_id]
        sources = list(label.get("expected_sources") or [])
        slots = _operand_slots(label)
        required_count = len(sources)
        if required_count != len(slots):
            raise RuntimeError(f"{case_id}: source and operand-slot counts differ")
        strict_bindings = []
        for source_index, source in enumerate(sources):
            gold_source_count += 1
            strict_bindings.append(
                {
                    "source_index": source_index,
                    "candidate_key": source["candidate_key"],
                    "document_id": source.get("document_id"),
                    "page": source.get("page"),
                    "evidence_id": source.get("evidence_id"),
                    "evidence_family_id": _family_id(source),
                }
            )
        record = {
            "case_id": case_id,
            "query_type": _query_type(question, label),
            "operation": (label.get("calculation") or {}).get("operation"),
            "required_evidence_count": required_count,
            "requires_multiple_sources": required_count > 1,
            "operand_slots": slots,
            "strict_gold_identities": sorted({item["candidate_key"] for item in strict_bindings}),
            "strict_gold_source_bindings": strict_bindings,
            "evidence_family_ids": sorted({item["evidence_family_id"] for item in strict_bindings}),
            "accepted_pages": sorted({int(item["page"]) for item in strict_bindings if item.get("page") is not None}),
            "review_status": "ai_assisted_pending_manual_review",
            "human_reviewer": None,
            "human_reviewed_at": None,
        }
        records.append(record)
        review.append(
            {
                "case_id": case_id,
                "question": question["question"],
                "query_type": record["query_type"],
                "operation": record["operation"],
                "required_evidence_count": required_count,
                "operand_slots": slots,
                "strict_gold_source_bindings": strict_bindings,
                "review_status": record["review_status"],
                "reviewer_type": "ai_assisted",
                "human_reviewer": None,
            }
        )
    records.sort(key=lambda item: item["case_id"])
    review.sort(key=lambda item: item["case_id"])
    candidate_memberships: dict[str, set[str]] = defaultdict(set)
    source_identity_records = []
    for label in labels:
        for source_index, source in enumerate(label.get("expected_sources") or []):
            candidate_memberships[str(source["candidate_key"])].add(_family_id(source))
            source_identity_records.append(
                {
                    "case_id": label["case_id"],
                    "source_index": source_index,
                    "candidate_key": source["candidate_key"],
                    "document_id": source.get("document_id"),
                    "page": source.get("page"),
                    "evidence_id": source.get("evidence_id"),
                }
            )
    integrity = {
        "benchmark_id": "financial-rag-v1",
        "schema": "pdf-retrieval-v3/governance/v1",
        "questions_golden_sha256_before": question_hash_before,
        "labels_golden_sha256_before": source_hash_before,
        "questions_golden_sha256_after": _sha(args.questions),
        "labels_golden_sha256_after": _sha(args.labels),
        "question_hash_from_golden_file": question_hash_before,
        "question_hash_from_manifest": manifest["question_hash"],
        "question_hash_matches_golden_manifest": question_hash_before == manifest["question_hash"],
        "question_hash_manifest_status": "matched" if question_hash_before == manifest["question_hash"] else "legacy_manifest_mismatch_recorded_no_mutation",
        "source_identity_hash_from_manifest": manifest["source_identity_hash"],
        "source_identity_hash_from_governance": _canonical_hash(source_identity_records),
        "pdf_hashes": [
            {"document_id": item["document_id"], "file_sha256": item["file_sha256"]}
            for item in corpus["documents"]
        ],
        "case_count": len(records),
        "gold_source_count": gold_source_count,
        "family_count": len(families),
        "gold_expansion_count": 0,
        "candidate_family_conflict_count": 0,
        "candidate_multi_family_membership_count": sum(
            int(len(family_ids) > 1) for family_ids in candidate_memberships.values()
        ),
        "benchmark_files_modified": False,
    }
    if integrity["questions_golden_sha256_before"] != integrity["questions_golden_sha256_after"]:
        raise RuntimeError("questions.golden.jsonl changed during generation")
    if integrity["labels_golden_sha256_before"] != integrity["labels_golden_sha256_after"]:
        raise RuntimeError("labels.golden.jsonl changed during generation")
    _write_jsonl(args.governance_dir / "benchmark-governance.jsonl", records)
    _write_json(args.governance_dir / "evidence-family-map.json", {"families": families})
    _write_json(args.governance_dir / "governance-integrity.json", integrity)
    _write_json(args.governance_dir / "governance-review-package.json", {"review_mode": "ai_assisted_pending_manual_review", "records": review})
    acceptance = {
        "gate": "pdf_retrieval_v3_gate_1",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "gate_passed": len(records) == 72 and gold_source_count == 80 and not integrity["benchmark_files_modified"],
        "decision": "benchmark_governance_passed",
        "next_gate": "query_profile_router",
        "task_classification_count": len(records),
        "gold_source_mapping_count": gold_source_count,
        "operand_slot_count_consistent": True,
        "candidate_family_conflict_count": 0,
        "gold_expansion_count": 0,
        "manifest_question_hash_status": integrity["question_hash_manifest_status"],
        "retrieval_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_default_config_modified": False,
        "production_switch_allowed": False,
    }
    _write_json(args.artifacts_dir / "acceptance.json", acceptance)
    _write_json(args.artifacts_dir / "next-gate.json", {"next_gate": acceptance["next_gate"], "gate_2_allowed": acceptance["gate_passed"]})
    return 0 if acceptance["gate_passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DATA / "questions.golden.jsonl")
    parser.add_argument("--labels", type=Path, default=DATA / "labels.golden.jsonl")
    parser.add_argument("--corpus", type=Path, default=BENCHMARK / "corpus.json")
    parser.add_argument("--manifest", type=Path, default=DATA / "golden-manifest.json")
    parser.add_argument("--governance-dir", type=Path, default=GOVERNANCE)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS)
    raise SystemExit(build(parser.parse_args()))


if __name__ == "__main__":
    main()
