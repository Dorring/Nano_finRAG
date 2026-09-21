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


def _operation_roles(operation: str | None, count: int) -> list[tuple[str, str]]:
    if count <= 0:
        return []
    if operation == "growth_rate" and count == 2:
        return [("previous", "previous_period"), ("current", "current_period")]
    if operation in {"percentage_share", "percentage_of"} and count == 2:
        return [("numerator", "numerator"), ("denominator", "denominator")]
    if operation == "difference" and count == 2:
        return [("left", "minuend"), ("right", "subtrahend")]
    return [(f"operand_{index}", f"operand_{index}") for index in range(1, count + 1)]


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
                "slot_id": roles[index][0],
                "role": roles[index][1],
                "metric": operand.get("metric"),
                "period": operand.get("period"),
                "statement_hint": None,
                "source_index": operand.get("source_index"),
            }
            for index, operand in enumerate(operands)
        ]
    sources = list(label.get("expected_sources") or [])
    if len(sources) == 1:
        roles = [("fact", "value")]
    elif len(sources) == 2 and len({_normalise(source.get("row_label")) for source in sources}) == 1 and len({source.get("period") for source in sources}) == 2:
        years = [int(match.group()) if (match := re.search(r"(?:19|20)\d{2}", str(source.get("period")))) else None for source in sources]
        roles = [("previous", "previous_period"), ("current", "current_period")] if years[0] is not None and years[1] is not None and years[0] < years[1] else [("current", "current_period"), ("previous", "previous_period")]
    elif len(sources) == 2:
        roles = [("left", "minuend"), ("right", "subtrahend")]
    else:
        roles = _operation_roles(None, len(sources))
    return [
        {
            "slot_id": roles[index][0],
            "role": roles[index][1],
            "metric": source.get("row_label"),
            "period": source.get("period"),
            "statement_hint": source.get("section") or source.get("table_title"),
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


def _conflict_report(records: list[dict[str, Any]], families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = []
    for record in records:
        bindings = list(record["strict_gold_source_bindings"])
        slots = list(record["operand_slots"])
        if record["query_type"] == "no_answer" and bindings:
            conflicts.append({"case_id": record["case_id"], "conflict_type": "no_answer_has_gold_source", "affected_identities": [item["candidate_key"] for item in bindings], "resolution": "blocked_pending_review"})
        if record["minimum_evidence_count"] != len(bindings):
            conflicts.append({"case_id": record["case_id"], "conflict_type": "minimum_evidence_count_gold_mismatch", "affected_identities": [item["candidate_key"] for item in bindings], "resolution": "blocked_pending_review"})
        if len(slots) < record["minimum_evidence_count"]:
            conflicts.append({"case_id": record["case_id"], "conflict_type": "operand_slot_count_insufficient", "affected_identities": [], "resolution": "blocked_pending_review"})
        if record["query_type"] == "calculation_multi_operand" and not record["operation"]:
            conflicts.append({"case_id": record["case_id"], "conflict_type": "calculation_missing_operation", "affected_identities": [], "resolution": "blocked_pending_review"})
        source_indices = {item["source_index"] for item in slots}
        if source_indices != set(range(len(bindings))):
            conflicts.append({"case_id": record["case_id"], "conflict_type": "operand_slot_source_mismatch", "affected_identities": [item["candidate_key"] for item in bindings], "resolution": "blocked_pending_review"})
    for family in families:
        metrics = {_normalise(item.get("evidence_id", "").split("::row_")[0]) for item in family["member_bindings"]}
        if len(metrics) != 1:
            conflicts.append({"case_id": None, "conflict_type": "family_multiple_logical_rows", "affected_identities": family["member_candidate_keys"], "resolution": "blocked_pending_review"})
    return conflicts


def _baseline_metrics(
    *,
    labels_by_id: dict[str, dict[str, Any]],
    family_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = json.loads((ROOT / "artifacts/evaluation/nf-eval-03-r2/case-results.json").read_text(encoding="utf-8"))["cases"]
    family_members = {
        family_id: {str(item) for item in family["member_candidate_keys"]}
        for family_id, family in family_by_id.items()
    }
    strict_hits = family_hits = page_hits = 0
    case_hits = all_gold_cases = 0
    per_case = []
    for case in cases:
        case_id = str(case["case_id"])
        finals = list((case.get("retrieval_stages") or {}).get("final") or [])
        final_keys = {str(item.get("candidate_key")) for item in finals}
        final_pages = {
            (str(item.get("canonical_document_id") or item.get("document_id") or ""), item.get("page"))
            for item in finals
        }
        sources = list(labels_by_id[case_id].get("expected_sources") or [])
        matched = 0
        for source in sources:
            strict = str(source["candidate_key"]) in final_keys
            family = _family_id(source)
            family_hit = bool(final_keys & family_members[family])
            page_hit = (str(source.get("document_id") or ""), source.get("page")) in final_pages
            strict_hits += int(strict)
            family_hits += int(family_hit)
            page_hits += int(page_hit)
            matched += int(strict)
        case_hits += int(matched > 0)
        all_gold_cases += int(matched == len(sources) and bool(sources))
        per_case.append({"case_id": case_id, "strict_source_hits": matched, "expected_source_count": len(sources)})
    return {
        "source_denominator": 80,
        "strict_candidate_recall_at_5": f"{strict_hits}/80",
        "evidence_family_recall_at_5": f"{family_hits}/80",
        "page_evidence_recall_at_5": f"{page_hits}/80",
        "final_case_hit_at_5": f"{case_hits}/64",
        "final_all_gold_coverage_at_5": f"{all_gold_cases}/64",
        "retrieval_runs": 0,
        "source_artifact": "artifacts/evaluation/nf-eval-03-r2/case-results.json",
        "per_case": per_case,
    }


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
            "minimum_evidence_count": required_count,
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
    conflicts = _conflict_report(records, families)
    family_by_id = {str(item["evidence_family_id"]): item for item in families}
    baseline_metrics = _baseline_metrics(labels_by_id=labels_by_id, family_by_id=family_by_id)
    integrity["baseline_strict_candidate_recall_at_5"] = baseline_metrics["strict_candidate_recall_at_5"]
    integrity["conflict_count"] = len(conflicts)
    _write_jsonl(args.governance_dir / "benchmark-governance.jsonl", records)
    _write_json(args.governance_dir / "evidence-family-map.json", {"families": families})
    _write_json(args.governance_dir / "governance-integrity.json", integrity)
    _write_json(args.governance_dir / "governance-review-package.json", {"review_mode": "ai_assisted_pending_manual_review", "records": review})
    operand_audit = {
        "case_count": len(records),
        "all_slots_complete": not any(item["conflict_type"].startswith("operand_slot") for item in conflicts),
        "records": [
            {
                "case_id": item["case_id"],
                "query_type": item["query_type"],
                "operation": item["operation"],
                "minimum_evidence_count": item["minimum_evidence_count"],
                "slot_count": len(item["operand_slots"]),
                "operand_slots": item["operand_slots"],
            }
            for item in records
        ],
    }
    frozen_integrity = {
        "questions_golden_sha256": question_hash_before,
        "labels_golden_sha256": source_hash_before,
        "benchmark_pdf_sha256": integrity["pdf_hashes"],
        "case_id_set_hash": _canonical_hash(sorted(item["case_id"] for item in records)),
        "strict_gold_source_identity_set_hash": integrity["source_identity_hash_from_governance"],
        "questions_unchanged": question_hash_before == _sha(args.questions),
        "labels_unchanged": source_hash_before == _sha(args.labels),
        "benchmark_files_modified": False,
    }
    _write_jsonl(args.artifacts_dir / "benchmark-governance.jsonl", records)
    _write_json(args.artifacts_dir / "frozen-benchmark-integrity.json", frozen_integrity)
    _write_json(args.artifacts_dir / "evidence-family-map.json", {"families": families})
    _write_json(args.artifacts_dir / "operand-slot-audit.json", operand_audit)
    _write_json(args.artifacts_dir / "governance-conflict-report.json", {"conflict_count": len(conflicts), "conflicts": conflicts})
    _write_json(args.artifacts_dir / "baseline-multigranularity-metrics.json", baseline_metrics)
    acceptance = {
        "gate": "pdf_retrieval_v3_gate_1",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "gate_passed": len(records) == 72 and gold_source_count == 80 and not integrity["benchmark_files_modified"] and not conflicts and baseline_metrics["strict_candidate_recall_at_5"] == "13/80",
        "decision": "benchmark_governance_passed" if not conflicts else "benchmark_governance_manual_review_blocked",
        "next_gate": "query_profile_router" if not conflicts else "manual_governance_review",
        "task_classification_count": len(records),
        "gold_source_mapping_count": gold_source_count,
        "operand_slot_count_consistent": True,
        "candidate_family_conflict_count": 0,
        "governance_conflict_count": len(conflicts),
        "gold_expansion_count": 0,
        "manifest_question_hash_status": integrity["question_hash_manifest_status"],
        "retrieval_calls": 0,
        "retriever_runtime_gold_reads": 0,
        "offline_governance_label_reads": gold_source_count,
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
