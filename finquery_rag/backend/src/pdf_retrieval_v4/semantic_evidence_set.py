"""Deterministic Top10 semantic evidence-set construction for Gate09 R5."""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from itertools import product
from typing import Any

MAX_EVIDENCE_ITEMS = 5
ACCESS_TOP_K = 10


def normalize(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def normalize_metric(value: Any) -> str:
    metric = normalize(value)
    metric = re.sub(r"\s*\([a-z0-9]{1,3}\)$", "", metric)
    metric = re.sub(r"(?<=[a-z])\d{1,2}$", "", metric)
    return metric.strip()


def metric_compatible(slot: dict[str, Any], metric: Any) -> tuple[bool, str | None]:
    """Use only exact phrase/path/leaf/concept equality; never substrings."""

    actual = normalize_metric(metric)
    leaf = normalize_metric(actual.rsplit("/", 1)[-1])
    phrase = normalize_metric(slot.get("raw_metric_phrase"))
    if phrase and phrase == actual:
        return True, "exact_path"
    if phrase and phrase == leaf:
        return True, "exact_leaf"
    concepts = {normalize_metric(item) for item in slot.get("concept_candidates") or [] if normalize_metric(item)}
    if actual in concepts or leaf in concepts:
        return True, "exact_frozen_concept"
    return False, None


def build_access_universe(
    plan: dict[str, Any],
    main_candidates: list[dict[str, Any]],
    slot_rankings: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Route single-slot to U0 and multi-slot to fixed U1 without rescoring."""

    slots = list(plan.get("operand_slots") or [])
    main_top10 = main_candidates[:ACCESS_TOP_K]
    by_key: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(main_top10, 1):
        key = str(item["candidate_key"])
        by_key[key] = {
            "candidate_key": key,
            "main_rank": rank,
            "slot_ranks": {},
            "access_sources": ["main_top10"],
        }
    if len(slots) > 1:
        for slot in slots:
            slot_id = str(slot["slot_id"])
            for rank, item in enumerate(slot_rankings.get(slot_id, [])[:ACCESS_TOP_K], 1):
                key = str(item["candidate_key"])
                record = by_key.setdefault(
                    key,
                    {"candidate_key": key, "main_rank": None, "slot_ranks": {}, "access_sources": []},
                )
                record["slot_ranks"][slot_id] = rank
                source = f"slot:{slot_id}:top10"
                if source not in record["access_sources"]:
                    record["access_sources"].append(source)
    return sorted(by_key.values(), key=lambda item: (best_access_rank(item), item["candidate_key"]))


def best_access_rank(candidate: dict[str, Any]) -> int:
    ranks = [rank for rank in [candidate.get("main_rank"), *(candidate.get("slot_ranks") or {}).values()] if rank is not None]
    return min(ranks) if ranks else 10**6


def build_semantic_classes(
    access: list[dict[str, Any]], registry: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    by_fact: dict[str, dict[str, Any]] = {}
    for candidate in access:
        candidate_key = str(candidate["candidate_key"])
        registry_record = registry.get(candidate_key) or {}
        for fact in registry_record.get("semantic_facts") or []:
            fact_id = str(fact["semantic_fact_id"])
            semantic_class = by_fact.setdefault(
                fact_id,
                {
                    "semantic_fact_id": fact_id,
                    "document_id": fact.get("document_id"),
                    "metric": fact.get("normalized_metric"),
                    "period": fact.get("normalized_period"),
                    "segment": fact.get("normalized_segment"),
                    "bucket": fact.get("normalized_bucket"),
                    "value": fact.get("normalized_base_value"),
                    "scale": fact.get("normalized_scale"),
                    "currency": fact.get("normalized_currency"),
                    "supporting_candidate_keys": [],
                    "supporting_evidence_ids": [],
                    "physical_provenance": [],
                    "candidate_access": {},
                },
            )
            if candidate_key not in semantic_class["supporting_candidate_keys"]:
                semantic_class["supporting_candidate_keys"].append(candidate_key)
            semantic_class["candidate_access"][candidate_key] = {
                "main_rank": candidate.get("main_rank"),
                "slot_ranks": candidate.get("slot_ranks") or {},
                "best_access_rank": best_access_rank(candidate),
            }
            for provenance in fact.get("physical_provenance") or []:
                evidence_id = provenance.get("authoritative_evidence_id")
                if evidence_id and evidence_id not in semantic_class["supporting_evidence_ids"]:
                    semantic_class["supporting_evidence_ids"].append(evidence_id)
                if provenance not in semantic_class["physical_provenance"]:
                    semantic_class["physical_provenance"].append(provenance)
    for semantic_class in by_fact.values():
        semantic_class["supporting_candidate_keys"].sort()
        semantic_class["supporting_evidence_ids"].sort()
        semantic_class["physical_provenance"].sort(key=lambda item: str(sorted(item.items())))
    return [by_fact[fact_id] for fact_id in sorted(by_fact)]


def match_slot(
    plan: dict[str, Any], slot: dict[str, Any], semantic_class: dict[str, Any], calculation: bool
) -> dict[str, Any] | None:
    if str(slot.get("required_evidence_shape") or "") != "atomic_fact":
        return None
    document_scope = {normalize(value) for value in plan.get("document_scope") or []}
    if document_scope and normalize(semantic_class.get("document_id")) not in document_scope:
        return None
    metric_ok, metric_grade = metric_compatible(slot, semantic_class.get("metric"))
    if not metric_ok:
        return None
    if normalize(slot.get("period")) != normalize(semantic_class.get("period")):
        return None
    if slot.get("segment_label") and normalize(slot.get("segment_label")) != normalize(semantic_class.get("segment")):
        return None
    if slot.get("bucket_label") and normalize(slot.get("bucket_label")) != normalize(semantic_class.get("bucket")):
        return None
    value = str(semantic_class.get("value") or "")
    scale = str(semantic_class.get("scale") or "")
    currency = str(semantic_class.get("currency") or "")
    return {
        "slot_id": str(slot["slot_id"]),
        "role": slot.get("role"),
        "semantic_fact_id": semantic_class["semantic_fact_id"],
        "metric_grade": metric_grade,
        "dimension_exact": True,
        "numeric_value_present": bool(value),
        "scale_resolved": bool(scale),
        "currency_resolved": bool(currency),
        "calculation_runtime_compatible": (not calculation) or bool(value and scale and currency),
        "supporting_candidate_keys": semantic_class["supporting_candidate_keys"],
    }


def match_slots(
    plan: dict[str, Any], semantic_classes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    calculation = str(plan.get("task_type") or "") == "calculation_multi_operand"
    results: list[dict[str, Any]] = []
    for slot in plan.get("operand_slots") or []:
        matches = [
            match
            for semantic_class in semantic_classes
            if (match := match_slot(plan, slot, semantic_class, calculation)) is not None
        ]
        matches.sort(key=lambda item: item["semantic_fact_id"])
        if not matches:
            status = "undercovered"
        elif len(matches) == 1:
            status = "deterministic"
        else:
            status = "runtime_operand_ambiguity"
        results.append(
            {
                "slot_id": str(slot["slot_id"]),
                "role": slot.get("role"),
                "slot_status": status,
                "compatible_semantic_fact_ids": [item["semantic_fact_id"] for item in matches],
                "matches": matches,
            }
        )
    return results


def _candidate_cost(candidate_key: str, access_by_key: dict[str, dict[str, Any]]) -> tuple[int, str]:
    return best_access_rank(access_by_key[candidate_key]), candidate_key


def minimum_candidate_cover(
    slot_matches: list[dict[str, Any]],
    semantic_classes: list[dict[str, Any]],
    access: list[dict[str, Any]],
    max_items: int = MAX_EVIDENCE_ITEMS,
) -> dict[str, Any]:
    deterministic = [item for item in slot_matches if item["slot_status"] == "deterministic"]
    required = [item["compatible_semantic_fact_ids"][0] for item in deterministic]
    class_by_id = {item["semantic_fact_id"]: item for item in semantic_classes}
    access_by_key = {item["candidate_key"]: item for item in access}
    if not required:
        return {"selected_candidate_keys": [], "covered_semantic_fact_ids": [], "complete": False, "evidence_item_count": 0}
    options = [class_by_id[fact_id]["supporting_candidate_keys"] for fact_id in required]
    candidate_sets = {tuple(sorted(set(values))) for values in product(*options) if len(set(values)) <= max_items}
    if not candidate_sets:
        return {"selected_candidate_keys": [], "covered_semantic_fact_ids": [], "complete": False, "evidence_item_count": 0}

    def rank(candidate_set: tuple[str, ...]) -> tuple[Any, ...]:
        coverage = {
            fact_id
            for fact_id in required
            if set(candidate_set).intersection(class_by_id[fact_id]["supporting_candidate_keys"])
        }
        access_ranks = [_candidate_cost(key, access_by_key)[0] for key in candidate_set]
        return (-len(coverage), len(candidate_set), max(access_ranks), sum(access_ranks), candidate_set)

    selected = min(candidate_sets, key=rank)
    covered = sorted(
        fact_id
        for fact_id in required
        if set(selected).intersection(class_by_id[fact_id]["supporting_candidate_keys"])
    )
    all_slots_deterministic = len(deterministic) == len(slot_matches)
    return {
        "selected_candidate_keys": list(selected),
        "covered_semantic_fact_ids": covered,
        "complete": all_slots_deterministic and len(covered) == len(required),
        "evidence_item_count": len(selected),
    }


def operand_projection(
    plan: dict[str, Any], slot_matches: list[dict[str, Any]], semantic_classes: list[dict[str, Any]]
) -> dict[str, Any]:
    class_by_id = {item["semantic_fact_id"]: item for item in semantic_classes}
    operands: dict[str, Any] = {}
    ready = str(plan.get("task_type") or "") == "calculation_multi_operand"
    for slot_match in slot_matches:
        slot_id = slot_match["slot_id"]
        if slot_match["slot_status"] != "deterministic":
            ready = False
            operands[slot_id] = {"status": slot_match["slot_status"], "ready": False}
            continue
        match = slot_match["matches"][0]
        semantic_class = class_by_id[match["semantic_fact_id"]]
        value = str(semantic_class.get("value") or "")
        scale = str(semantic_class.get("scale") or "")
        currency = str(semantic_class.get("currency") or "")
        try:
            normalized_value = str((Decimal(value) * Decimal(scale)).normalize()) if value and scale else None
        except InvalidOperation:
            normalized_value = None
        operand_ready = bool(normalized_value and currency)
        ready = ready and operand_ready
        operands[slot_id] = {
            "status": "deterministic",
            "role": match.get("role"),
            "semantic_fact_id": match["semantic_fact_id"],
            "normalized_value": normalized_value,
            "scale": "1" if normalized_value else scale or None,
            "currency": currency or None,
            "supporting_candidate_keys": match["supporting_candidate_keys"],
            "deterministic": True,
            "ready": operand_ready,
        }
    ready = ready and len(operands) == len(plan.get("operand_slots") or [])
    return {
        "operation": plan.get("operation"),
        "operands": operands,
        "calculation_runtime_ready": ready,
        "blocked_reason": None if ready else "operand_contract_incomplete_or_ambiguous",
    }
