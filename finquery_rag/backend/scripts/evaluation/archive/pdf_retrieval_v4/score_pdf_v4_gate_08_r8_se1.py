#!/usr/bin/env python3
"""Post-seal strict semantic-fact scoring for Gate 08 R8-SE1."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "artifacts/evaluation"
P0 = EVAL / "pdf-retrieval-v4-gate-08-r8-se1-p0"
OUT = EVAL / "pdf-retrieval-v4-gate-08-r8-se1"
R33 = EVAL / "pdf-retrieval-v4-gate-08-r8-r3-3"
STRICT = EVAL / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
GOVERNANCE = EVAL / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
FAMILY_MAP = EVAL / "pdf-retrieval-v3-gate-1/evidence-family-map.json"
QUERY_PLAN = EVAL / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
CUTOFFS = (1, 3, 5, 10, 20, 100)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def normalize_metric(value: Any) -> str:
    """Remove only explicit terminal footnote markers from a normalized label."""

    metric = normalize(value)
    metric = re.sub(r"\s*\([a-z0-9]{1,3}\)$", "", metric)
    metric = re.sub(r"(?<=[a-z])\d{1,2}$", "", metric)
    return metric.strip()


def metric_compatible(target: Any, actual: Any) -> bool:
    """Exact normalized metric or exact metric-path leaf; no substring match."""

    target_norm = normalize_metric(target)
    actual_norm = normalize_metric(actual)
    if not target_norm or not actual_norm:
        return False
    actual_leaf = normalize_metric(actual_norm.rsplit("/", 1)[-1])
    target_leaf = normalize_metric(target_norm.rsplit("/", 1)[-1])
    return target_norm == actual_norm or target_norm == actual_leaf or target_leaf == actual_norm


def exact_decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value)) if value not in (None, "") else None
    except InvalidOperation:
        return None


def fact_value_at_base_scale(fact: dict[str, Any]) -> Decimal | None:
    value = exact_decimal(fact.get("normalized_base_value"))
    scale = exact_decimal(fact.get("normalized_scale"))
    if value is None:
        return None
    return value * scale if scale is not None else value


def expected_operand_value(label: dict[str, Any], source_index: int) -> Decimal | None:
    for operand in (label.get("calculation") or {}).get("operands") or []:
        if int(operand.get("source_index", -1)) == source_index:
            return exact_decimal(operand.get("value"))
    sources = label.get("expected_sources") or []
    answer = label.get("expected_answer") or {}
    if len(sources) == 1 and answer.get("value_type") in {"currency", "number", "percentage"}:
        return exact_decimal(answer.get("canonical_value"))
    return None


def resolve_target(
    binding: dict[str, Any],
    label: dict[str, Any],
    plan: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_index = int(binding["source_index"])
    sources = label.get("expected_sources") or []
    source = sources[source_index] if source_index < len(sources) else {}
    slots = plan.get("operand_slots") or []
    slot = slots[source_index] if source_index < len(slots) else {}
    target = {
        "document_id": normalize(binding.get("document_id")),
        "metric": source.get("row_label") or slot.get("raw_metric_phrase"),
        "period": source.get("period") or slot.get("period"),
        "segment": slot.get("segment_label"),
        "bucket": slot.get("bucket_label"),
        "value_at_base_scale": expected_operand_value(label, source_index),
    }
    candidate = registry.get(str(binding["candidate_key"]))
    matches: list[dict[str, Any]] = []
    if candidate:
        for fact in candidate.get("semantic_facts") or []:
            if normalize(fact.get("document_id")) != target["document_id"]:
                continue
            if not metric_compatible(target["metric"], fact.get("normalized_metric")):
                continue
            if normalize(fact.get("normalized_period")) != normalize(target["period"]):
                continue
            if target["segment"] and normalize(fact.get("normalized_segment")) != normalize(target["segment"]):
                continue
            if target["bucket"] and normalize(fact.get("normalized_bucket")) != normalize(target["bucket"]):
                continue
            if target["value_at_base_scale"] is not None and fact_value_at_base_scale(fact) != target["value_at_base_scale"]:
                continue
            matches.append(fact)
    by_id = {str(fact["semantic_fact_id"]): fact for fact in matches}
    if len(by_id) == 1:
        status = "resolved"
        semantic_fact_id, fact = next(iter(by_id.items()))
    elif len(by_id) > 1:
        status, semantic_fact_id, fact = "ambiguous", None, None
    else:
        status, semantic_fact_id, fact = "unresolved", None, None
    return {
        **binding,
        "semantic_target_status": status,
        "gold_semantic_fact_id": semantic_fact_id,
        "target_contract": {
            **target,
            "value_at_base_scale": str(target["value_at_base_scale"]) if target["value_at_base_scale"] is not None else None,
        },
        "matching_semantic_fact_ids": sorted(by_id),
        "gold_semantic_fact": fact,
    }


def full_ranking(final_record: dict[str, Any], main_record: dict[str, Any]) -> list[str]:
    """Extend the frozen composed Top5 with the frozen main-ranking residual."""

    ordered = [str(item["candidate_key"]) for item in final_record["candidates"]]
    seen = set(ordered)
    ordered.extend(
        str(item["candidate_key"])
        for item in main_record["ranked_candidates"]
        if str(item["candidate_key"]) not in seen
    )
    if len(ordered) != 100 or len(set(ordered)) != 100:
        raise RuntimeError("r3_3_semantic_ranking_contract_blocked")
    return ordered


def candidate_families() -> dict[str, set[str]]:
    payload = json.loads(FAMILY_MAP.read_text(encoding="utf-8"))
    families = payload.get("families", payload) if isinstance(payload, dict) else payload
    result: dict[str, set[str]] = defaultdict(set)
    iterable = families.values() if isinstance(families, dict) else families
    for family in iterable:
        family_id = str(family["evidence_family_id"])
        for key in family.get("member_candidate_keys", family.get("member_candidate_identities", [])) or []:
            result[str(key)].add(family_id)
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    p0_seal = json.loads((P0 / "prediction-seal.json").read_text(encoding="utf-8"))
    registry_path = P0 / "candidate-semantic-fact-registry.jsonl.gz"
    if not p0_seal.get("sealed") or p0_seal["registry_sha256"] != sha256(registry_path):
        raise RuntimeError("se1_p0_registry_seal_invalid")
    if p0_seal["gold_reads_before_seal"] != 0 or p0_seal["candidate_mutation"] != 0:
        raise RuntimeError("se1_p0_safety_contract_blocked")
    r33_seal = json.loads((R33 / "prediction-seal.json").read_text(encoding="utf-8"))
    if not r33_seal.get("sealed") or r33_seal["candidate_added"] or r33_seal["candidate_removed"]:
        raise RuntimeError("r3_3_prediction_contract_blocked")

    registry = {str(row["candidate_key"]): row for row in read_jsonl(registry_path)}
    labels = {str(row["case_id"]): row for row in read_jsonl(LABELS)}
    governance = {str(row["case_id"]): row for row in read_jsonl(GOVERNANCE)}
    bindings = list(read_jsonl(STRICT))
    plans_payload = json.loads(QUERY_PLAN.read_text(encoding="utf-8"))
    plans = {str(row["case_id"]): row["plan"] for row in plans_payload["plans"]}
    final = {str(row["case_id"]): row for row in read_jsonl(R33 / "slot_aware_top5_predictions.jsonl.gz")}
    main_predictions = {str(row["case_id"]): row for row in read_jsonl(R33 / "main_rerank_predictions.jsonl.gz")}
    rankings = {case_id: full_ranking(final[case_id], main_predictions[case_id]) for case_id in final}
    if len(bindings) != 80 or len(rankings) != 72:
        raise RuntimeError("se1_scoring_denominator_contract_blocked")

    targets = [resolve_target(binding, labels[binding["case_id"]], plans[binding["case_id"]], registry) for binding in bindings]
    target_counts = Counter(target["semantic_target_status"] for target in targets)
    write_jsonl(OUT / "gold-semantic-targets.jsonl", targets)

    family_by_candidate = candidate_families()
    strict_counts = Counter()
    family_counts = Counter()
    semantic_counts = Counter()
    rescue_rows: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    false_conflicts: list[dict[str, Any]] = []
    semantic_hits_by_cutoff: dict[int, set[str]] = {cutoff: set() for cutoff in CUTOFFS}
    family_hits_by_cutoff: dict[int, set[str]] = {cutoff: set() for cutoff in CUTOFFS}

    for target in targets:
        case_id = str(target["case_id"])
        binding_id = str(target["binding_id"])
        candidate_key = str(target["candidate_key"])
        ranking = rankings[case_id]
        per_cutoff: dict[str, Any] = {}
        for cutoff in CUTOFFS:
            prefix = ranking[:cutoff]
            strict_hit = candidate_key in prefix
            family_id = str(target.get("evidence_family_id") or "")
            family_hit = strict_hit or any(family_id in family_by_candidate.get(key, set()) for key in prefix)
            rescued_key = None
            rescued_fact = None
            semantic_hit = strict_hit
            gold_fact_id = target.get("gold_semantic_fact_id")
            if not semantic_hit and target["semantic_target_status"] == "resolved":
                for key in prefix:
                    candidate = registry.get(key) or {}
                    for fact in candidate.get("semantic_facts") or []:
                        if fact.get("semantic_fact_id") == gold_fact_id:
                            semantic_hit, rescued_key, rescued_fact = True, key, fact
                            break
                    if semantic_hit:
                        break
            strict_counts[cutoff] += strict_hit
            family_counts[cutoff] += family_hit
            semantic_counts[cutoff] += semantic_hit
            if family_hit:
                family_hits_by_cutoff[cutoff].add(binding_id)
            if semantic_hit:
                semantic_hits_by_cutoff[cutoff].add(binding_id)
            per_cutoff[str(cutoff)] = {
                "strict_hit": strict_hit,
                "evidence_family_hit": family_hit,
                "semantic_hit": semantic_hit,
                "semantic_rescue": bool(semantic_hit and not strict_hit),
                "rescued_candidate_key": rescued_key,
            }
            if cutoff == 5 and semantic_hit and not strict_hit and rescued_key and rescued_fact:
                gold_fact = target["gold_semantic_fact"]
                conflict_fields = [
                    field
                    for field in (
                        "document_id", "normalized_metric", "normalized_period", "normalized_segment",
                        "normalized_bucket", "normalized_base_value", "normalized_scale", "normalized_currency",
                    )
                    if gold_fact.get(field) != rescued_fact.get(field)
                ]
                if conflict_fields:
                    false_conflicts.append({"binding_id": binding_id, "conflict_fields": conflict_fields})
                rescue_rows.append(
                    {
                        "case_id": case_id,
                        "source_index": target["source_index"],
                        "binding_id": binding_id,
                        "strict_gold_candidate_key": candidate_key,
                        "rescued_candidate_key": rescued_key,
                        "gold_semantic_fact_id": gold_fact_id,
                        "candidate_semantic_fact_id": rescued_fact["semantic_fact_id"],
                        "metric": rescued_fact["normalized_metric"],
                        "period": rescued_fact["normalized_period"],
                        "value": rescued_fact["normalized_base_value"],
                        "scale": rescued_fact["normalized_scale"],
                        "currency": rescued_fact["normalized_currency"],
                        "segment": rescued_fact["normalized_segment"],
                        "bucket": rescued_fact["normalized_bucket"],
                        "gold_physical_provenance": gold_fact["physical_provenance"],
                        "rescued_physical_provenance": rescued_fact["physical_provenance"],
                        "review_status": "pending",
                        "verified": False,
                    }
                )
        binding_rows.append({**target, "cutoffs": per_cutoff})

    if false_conflicts:
        raise RuntimeError(f"automatic_false_equivalence:{len(false_conflicts)}")

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        by_case[str(target["case_id"])].append(target)
    multi_metrics: dict[int, int] = {}
    calculation_metrics: dict[int, int] = {}
    case_completeness: list[dict[str, Any]] = []
    for cutoff in (5, 10):
        multi_complete = calc_complete = 0
        for case_id, case_targets in by_case.items():
            required = {str(target["binding_id"]) for target in case_targets}
            complete = required <= semantic_hits_by_cutoff[cutoff]
            is_multi = bool(governance[case_id]["requires_multiple_sources"])
            is_calculation = governance[case_id]["query_type"] == "calculation_multi_operand"
            multi_complete += bool(is_multi and complete)
            calc_complete += bool(is_calculation and complete)
            case_completeness.append(
                {
                    "case_id": case_id,
                    "cutoff": cutoff,
                    "is_multi_evidence": is_multi,
                    "is_calculation": is_calculation,
                    "required_semantic_fact_ids": sorted(
                        target["gold_semantic_fact_id"] for target in case_targets if target["gold_semantic_fact_id"]
                    ),
                    "semantic_complete": complete,
                }
            )
        multi_metrics[cutoff] = multi_complete
        calculation_metrics[cutoff] = calc_complete

    strict_recall = {f"recall_at_{cutoff}": f"{strict_counts[cutoff]}/80" for cutoff in CUTOFFS}
    family_recall = {f"recall_at_{cutoff}": f"{family_counts[cutoff]}/80" for cutoff in CUTOFFS}
    semantic_recall = {f"recall_at_{cutoff}": f"{semantic_counts[cutoff]}/80" for cutoff in CUTOFFS}
    if strict_recall["recall_at_5"] != "43/80" or strict_recall["recall_at_100"] != "68/80":
        raise RuntimeError("historical_strict_metric_parity_blocked")

    write_json(OUT / "strict-physical-recall.json", strict_recall)
    write_json(OUT / "evidence-family-recall.json", family_recall)
    write_json(OUT / "semantic-fact-recall.json", semantic_recall)
    write_json(OUT / "semantic-target-coverage.json", {"counts": dict(sorted(target_counts.items())), "coverage": f"{target_counts['resolved']}/80"})
    write_json(OUT / "semantic-multi-evidence.json", {"complete_at_5": f"{multi_metrics[5]}/16", "complete_at_10": f"{multi_metrics[10]}/16"})
    write_json(OUT / "semantic-calculation.json", {"complete_at_5": f"{calculation_metrics[5]}/11", "complete_at_10": f"{calculation_metrics[10]}/11"})
    write_json(OUT / "semantic-case-completeness.json", {"cases": case_completeness})
    write_json(OUT / "semantic-rescue-review-package.json", {"review_status": "pending", "verified": False, "rescue_count": len(rescue_rows), "records": rescue_rows})
    write_json(OUT / "false-equivalence-audit.json", {"automatic_false_equivalence": len(false_conflicts), "conflicts": false_conflicts, "decision": "pass"})
    write_json(OUT / "binding-semantic-score.json", {"bindings": binding_rows})
    comparison = {
        "metric_table_at_5": [
            {"metric": "Strict Physical Source Recall", "score": "43/80"},
            {"metric": "Conservative Evidence Family Recall", "score": family_recall["recall_at_5"]},
            {"metric": "Semantic Fact Recall", "score": semantic_recall["recall_at_5"]},
            {"metric": "Multi-evidence Physical Complete", "score": "4/16"},
            {"metric": "Multi-evidence Semantic Complete", "score": f"{multi_metrics[5]}/16"},
            {"metric": "Calculation Physical Complete", "score": "4/11"},
            {"metric": "Calculation Semantic Complete", "score": f"{calculation_metrics[5]}/11"},
        ]
    }
    write_json(OUT / "final-metric-comparison.json", comparison)

    coverage = target_counts["resolved"]
    semantic_at_5 = semantic_counts[5]
    if coverage >= 76 and semantic_at_5 >= 68:
        decision = "semantic_evidence_recall_exceptional"
        next_gate = "equivalence_aware_deterministic_calculator"
    elif coverage >= 76 and semantic_at_5 >= 64:
        decision = "semantic_evidence_recall_strong_pass"
        next_gate = "equivalence_aware_deterministic_calculator"
    elif coverage >= 76 and semantic_at_5 >= 60:
        decision = "semantic_evidence_recall_passed"
        next_gate = "equivalence_aware_deterministic_calculator"
    else:
        decision = "semantic_evidence_recall_below_top5_target"
        next_gate = "top10_evidence_set_contract" if semantic_counts[10] >= 60 else "semantic_target_contract_review"
    manifest = {
        "se1_p0_registry_sha256": sha256(registry_path),
        "se1_p0_seal_sha256": sha256(P0 / "prediction-seal.json"),
        "r3_3_final_prediction_sha256": sha256(R33 / "slot_aware_top5_predictions.jsonl.gz"),
        "r3_3_main_prediction_sha256": sha256(R33 / "main_rerank_predictions.jsonl.gz"),
        "strict_source_contract_sha256": sha256(STRICT),
        "labels_sha256": sha256(LABELS),
        "governance_sha256": sha256(GOVERNANCE),
        "family_map_sha256": sha256(FAMILY_MAP),
        "query_plan_sha256": sha256(QUERY_PLAN),
        "scorer_source_sha256": sha256(Path(__file__)),
        "post_benchmark_diagnostic": True,
    }
    write_json(OUT / "scoring-manifest.json", manifest)
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r8_se1",
        "decision": decision,
        "next_gate": next_gate,
        "historical_strict_physical_source_recall_at_5": "43/80",
        "semantic_target_coverage": f"{coverage}/80",
        "evidence_family_recall_at_5": family_recall["recall_at_5"],
        "semantic_fact_recall_at_5": semantic_recall["recall_at_5"],
        "semantic_fact_recall_at_10": semantic_recall["recall_at_10"],
        "semantic_multi_evidence_complete_at_5": f"{multi_metrics[5]}/16",
        "semantic_calculation_complete_at_5": f"{calculation_metrics[5]}/11",
        "semantic_rescue_count_at_5": len(rescue_rows),
        "automatic_false_equivalence": 0,
        "post_benchmark_diagnostic": True,
        "retrieval_optimization_stopped": True,
        "production_switch_allowed": False,
    }
    write_json(OUT / "acceptance.json", acceptance)
    write_json(OUT / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
