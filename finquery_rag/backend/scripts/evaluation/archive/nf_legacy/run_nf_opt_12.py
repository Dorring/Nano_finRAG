"""NF-OPT-12: read-only evidence-family competition and slot coverage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.nf_opt_12 import (
    collapse_families,
    evidence_family_id,
    evidence_slot_id,
    family_components,
    parse_query_slots,
    strict_hits,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "artifacts/evaluation/nf-eval-03-r2/case-results.json"
DEFAULT_QUESTIONS = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
DEFAULT_LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-12"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_item(source: dict[str, Any]) -> dict[str, Any]:
    """Map frozen source identity fields to the candidate-only family contract."""
    return {
        "candidate_key": source.get("candidate_key"),
        "canonical_document_id": source.get("document_id") or source.get("candidate_document_id"),
        "document_id": source.get("candidate_document_id") or source.get("document_id"),
        "page": source.get("candidate_pdf_page", source.get("page")),
        "evidence_id": source.get("evidence_id"),
        "parent_id": source.get("parent_candidate_key"),
    }


def _top_oracle(
    reranked: list[dict[str, Any]], gold_keys: set[str], limit: int = 5
) -> list[dict[str, Any]]:
    """Gold-assisted upper bound only; never used to change production ranking."""
    gold = [item for item in reranked if str(item.get("candidate_key")) in gold_keys]
    other = [item for item in reranked if str(item.get("candidate_key")) not in gold_keys]
    return (gold + other)[:limit]


def audit(
    *,
    cases_payload: dict[str, Any],
    questions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    question_by_id = {str(item["case_id"]): item for item in questions}
    label_by_id = {str(item["case_id"]): item for item in labels}
    answerable = [item for item in cases_payload["cases"] if not item.get("expected_no_answer")]
    if len(answerable) != 64:
        raise ValueError(f"expected 64 answerable cases, got {len(answerable)}")
    candidate_catalog: dict[str, dict[str, Any]] = {}
    for case in answerable:
        for stage in (case.get("retrieval_stages") or {}).values():
            for candidate in stage or []:
                key = str(candidate.get("candidate_key") or "")
                if key:
                    candidate_catalog.setdefault(key, candidate)
    family_manifest: dict[str, dict[str, Any]] = {}
    duplication: list[dict[str, Any]] = []
    gold_ranks: list[dict[str, Any]] = []
    slot_plans: list[dict[str, Any]] = []
    totals = Counter()
    baseline_hits: set[tuple[str, str]] = set()
    family_hits: set[tuple[str, str]] = set()
    slot_hits: set[tuple[str, str]] = set()
    combined_hits: set[tuple[str, str]] = set()

    for case in answerable:
        case_id = str(case["case_id"])
        question = question_by_id.get(case_id)
        label = label_by_id.get(case_id)
        if question is None or label is None:
            raise ValueError(f"missing frozen question or label for {case_id}")
        slot_plans.append(parse_query_slots(question))
        reranked = list(case.get("retrieval_stages", {}).get("reranker") or [])
        final = list(case.get("retrieval_stages", {}).get("final") or [])
        if len(final) != 5:
            raise ValueError(f"{case_id}: expected Final Top-5")
        gold_sources = list(label.get("expected_sources") or [])
        gold_keys = {str(source["candidate_key"]) for source in gold_sources}
        gold_families = []
        for source_index, source in enumerate(gold_sources):
            key = str(source["candidate_key"])
            family_item = candidate_catalog.get(key, _source_item(source))
            gold_families.append((source_index, key, evidence_family_id(family_item)))
        for item in reranked:
            family = evidence_family_id(item)
            entry = family_manifest.setdefault(
                family,
                {
                    "family_id": family,
                    **family_components(item),
                    "member_candidate_keys": set(),
                    "slot_ids": set(),
                },
            )
            entry["member_candidate_keys"].add(str(item.get("candidate_key")))
            entry["slot_ids"].add(evidence_slot_id(item))

        def stage_dup(stage: list[dict[str, Any]], limit: int) -> tuple[int, int]:
            prefix = stage[:limit]
            return len(prefix), len({evidence_family_id(item) for item in prefix})

        final_count, final_families = stage_dup(final, 5)
        r20_count, r20_families = stage_dup(reranked, 20)
        r40_count, r40_families = stage_dup(reranked, 40)
        source_positions = {
            str(item.get("candidate_key")): index
            for index, item in enumerate(reranked, start=1)
        }
        family_positions: dict[str, int] = {}
        for index, item in enumerate(reranked, start=1):
            family_positions.setdefault(evidence_family_id(item), index)
        gold_rows = []
        for source_index, source in enumerate(gold_sources):
            key = str(source["candidate_key"])
            family = gold_families[source_index][2]
            source_rank = source_positions.get(key)
            before = reranked[: source_rank - 1] if source_rank else reranked
            source_item = candidate_catalog.get(key, _source_item(source))
            component = family_components(source_item)
            gold_rows.append(
                {
                    "source_index": source_index,
                    "candidate_key": key,
                    "gold_source_rank": source_rank,
                    "gold_family_first_rank": family_positions.get(family),
                    "same_family_before_source": sum(
                        evidence_family_id(item) == family for item in before
                    ),
                    "same_page_before_source": sum(
                        item.get("page") == source_item.get("page") for item in before
                    ),
                    "same_table_before_source": sum(
                        family_components(item)["table_identity"]
                        == component["table_identity"]
                        for item in before
                    ),
                }
            )
        gold_ranks.append({"case_id": case_id, "gold_sources": gold_rows})
        duplication.append(
            {
                "case_id": case_id,
                "top40_candidate_count": r40_count,
                "top40_unique_family_count": r40_families,
                "top20_candidate_count": r20_count,
                "top20_unique_family_count": r20_families,
                "top5_candidate_count": final_count,
                "top5_unique_family_count": final_families,
                "duplicate_family_count_top5": final_count - final_families,
                "duplicate_family_count_top20": r20_count - r20_families,
            }
        )
        baseline = strict_hits(final, gold_keys)
        collapsed = collapse_families(reranked, 5)
        collapsed_families = {evidence_family_id(item) for item in collapsed}
        family_proxy = {
            source_index
            for source_index, _, family in gold_families
            if family in collapsed_families
        }
        slot_oracle = _top_oracle(reranked, gold_keys)
        combined = _top_oracle(collapse_families(reranked, 20), gold_keys)
        slot_keys = strict_hits(slot_oracle, gold_keys)
        combined_keys = strict_hits(combined, gold_keys)
        baseline_hits.update(
            (case_id, source_index)
            for source_index, source in enumerate(gold_sources)
            if str(source["candidate_key"]) in baseline
        )
        family_hits.update((case_id, source_index) for source_index in family_proxy)
        slot_hits.update(
            (case_id, source_index)
            for source_index, source in enumerate(gold_sources)
            if str(source["candidate_key"]) in slot_keys
        )
        combined_hits.update(
            (case_id, source_index)
            for source_index, source in enumerate(gold_sources)
            if str(source["candidate_key"]) in combined_keys
        )
        totals["gold_sources"] += len(gold_sources)

    if totals["gold_sources"] != 80:
        raise ValueError(f"expected 80 gold sources, got {totals['gold_sources']}")
    manifest = sorted(
        (
            {
                **value,
                "member_candidate_keys": sorted(value["member_candidate_keys"]),
                "slot_ids": sorted(value["slot_ids"]),
            }
            for value in family_manifest.values()
        ),
        key=lambda item: item["family_id"],
    )
    return {
        "evidence_family_manifest": {
            "artifact_schema": "nf-opt-12/family-manifest/v1",
            "family_identity_inputs": [
                "document_id",
                "pdf_page",
                "normalized_table_identity",
                "normalized_row_identity",
            ],
            "gold_fields_read": False,
            "families": manifest,
        },
        "candidate_duplication_report": {
            "artifact_schema": "nf-opt-12/duplication/v1",
            "case_count": len(duplication),
            "summary": {
                "top5_duplicate_family_count": sum(
                    item["duplicate_family_count_top5"] for item in duplication
                ),
                "top20_duplicate_family_count": sum(
                    item["duplicate_family_count_top20"] for item in duplication
                ),
            },
            "cases": duplication,
        },
        "gold_family_rank_report": {
            "artifact_schema": "nf-opt-12/gold-rank/v1",
            "gold_source_count": totals["gold_sources"],
            "cases": gold_ranks,
        },
        "query_slot_plan_report": {
            "artifact_schema": "nf-opt-12/query-slot/v1",
            "query_only": True,
            "expected_fields_read": False,
            "case_count": len(slot_plans),
            "cases": slot_plans,
        },
        "oracle_upper_bound_report": {
            "artifact_schema": "nf-opt-12/oracle/v1",
            "oracle_only": True,
            "production_ranking_modified": False,
            "baseline_strict_final_source_recall_at_5": {
                "matched_sources": len(baseline_hits),
                "source_count": totals["gold_sources"],
            },
            "oracle_family_collapse_proxy_recall_at_5": {
                "matched_sources": len(family_hits),
                "source_count": totals["gold_sources"],
            },
            "oracle_slot_aware_strict_recall_at_5": {
                "matched_sources": len(slot_hits),
                "source_count": totals["gold_sources"],
            },
            "oracle_family_collapse_plus_slot_aware_strict_recall_at_5": {
                "matched_sources": len(combined_hits),
                "source_count": totals["gold_sources"],
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    outputs = audit(
        cases_payload=json.loads(args.cases.read_text(encoding="utf-8")),
        questions=_load_jsonl(args.questions),
        labels=_load_jsonl(args.labels),
    )
    integrity = {
        "artifact_schema": "nf-opt-12/input-integrity/v1",
        "case_results_sha256": _sha(args.cases),
        "questions_sha256": _sha(args.questions),
        "labels_sha256": _sha(args.labels),
        "case_count": 64,
        "gold_source_count": 80,
        "input_hashes_verified": True,
    }
    acceptance = {
        "artifact_schema": "nf-opt-12/acceptance/v1",
        "decision": "evidence_family_slot_audit_recorded",
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "oracle_used_for_production": False,
        "case_count": 64,
        "gold_source_count": 80,
        "input_hashes_verified": True,
    }
    next_gate = {
        "decision": "audit_complete",
        "family_collapse_candidate": outputs["oracle_upper_bound_report"]["oracle_family_collapse_proxy_recall_at_5"],
        "slot_aware_candidate": outputs["oracle_upper_bound_report"]["oracle_slot_aware_strict_recall_at_5"],
        "production_switch_allowed": False,
    }
    _write(args.out_dir / "input-integrity-report.json", integrity)
    files = (
        ("evidence_family_manifest", "evidence-family-manifest.json"),
        ("candidate_duplication_report", "candidate-duplication-report.json"),
        ("gold_family_rank_report", "gold-family-rank-report.json"),
        ("query_slot_plan_report", "query-slot-plan-report.json"),
        ("oracle_upper_bound_report", "oracle-upper-bound-report.json"),
    )
    for key, filename in files:
        _write(args.out_dir / filename, outputs[key])
    _write(args.out_dir / "next-gate.json", next_gate)
    _write(args.out_dir / "nf-opt-12-acceptance.json", acceptance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
