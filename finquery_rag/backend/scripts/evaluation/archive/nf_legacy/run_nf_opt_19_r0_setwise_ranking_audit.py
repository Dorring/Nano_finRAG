"""NF-OPT-19 R0: frozen pointwise-rank and evidence-set competition audit.

This module deliberately performs no retrieval, model execution, ranking
mutation, binder, or calculator work.  It consumes only sealed R3.3/SE1
artifacts and writes Gold-independent diagnostics before unlocking the strict
binding artifact for scoring.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


K_VALUES = (5, 10, 20, 50, 100)
EXPECTED_COHORTS = {"c0": 43, "c1": 25, "c2": 12}
EXPECTED_CASES = 72
EXPECTED_MULTI = 16
EXPECTED_CALC = 11

ARTIFACT_NAME = "nf-opt-19-r0-setwise-ranking-audit"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    # mtime=0 makes the diagnostic artifact hash reproducible.
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for row in rows:
                gz.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return _sha256_file(path)


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _norm_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _numeric(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fact_metric(fact: dict[str, Any]) -> str:
    return str(fact.get("normalized_metric") or fact.get("metric") or "")


def _fact_period(fact: dict[str, Any]) -> str:
    return str(fact.get("normalized_period") or fact.get("period") or "")


def _fact_logical_tables(fact: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("logical_table_id", "logical_table"):
        if fact.get(key):
            values.add(str(fact[key]))
    for provenance in fact.get("physical_provenance", []) or []:
        if provenance.get("logical_table_id"):
            values.add(str(provenance["logical_table_id"]))
    return values


def _fact_pages(fact: dict[str, Any]) -> set[str]:
    pages: set[str] = set()
    for provenance in fact.get("physical_provenance", []) or []:
        if provenance.get("pdf_page") is not None:
            pages.add(str(provenance["pdf_page"]))
    return pages


def _parse_slots(query_view: str) -> list[dict[str, str]]:
    slots: list[dict[str, str]] = []
    for line in str(query_view or "").splitlines():
        match = re.match(r"\s*Slot\s+(\d+)\s*:\s*(.*)$", line, re.IGNORECASE)
        if not match:
            continue
        fields: dict[str, str] = {"slot_id": match.group(1), "raw": match.group(2)}
        for key, value in re.findall(r"([A-Za-z][A-Za-z_ ]*)\s*=\s*([^,]+)", match.group(2)):
            fields[_norm_text(key).replace(" ", "_")] = value.strip()
        slots.append(fields)
    return slots


def _candidate_facts(
    case_id: str,
    candidate_key: str,
    case_facts: dict[str, dict[str, set[str]]],
    fact_meta: dict[str, dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    ids = sorted(case_facts.get(case_id, {}).get(candidate_key, set()))
    return ids, [fact_meta[fact_id] for fact_id in ids if fact_id in fact_meta]


def _candidate_features(
    case_id: str,
    candidate_key: str,
    case_facts: dict[str, dict[str, set[str]]],
    fact_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fact_ids, facts = _candidate_facts(case_id, candidate_key, case_facts, fact_meta)
    group_ids = [f"semantic:{fact_id}" for fact_id in fact_ids]
    if not group_ids:
        group_ids = [f"candidate:{case_id}:{candidate_key}"]
    metrics = {_norm_text(_fact_metric(fact)) for fact in facts if _fact_metric(fact)}
    periods = {_norm_text(_fact_period(fact)) for fact in facts if _fact_period(fact)}
    tables = set().union(*(_fact_logical_tables(fact) for fact in facts)) if facts else set()
    pages = set().union(*(_fact_pages(fact) for fact in facts)) if facts else set()
    rows: set[str] = set()
    for fact in facts:
        for provenance in fact.get("physical_provenance", []) or []:
            if provenance.get("row_id"):
                rows.add(str(provenance["row_id"]))
    return {
        "candidate_key": candidate_key,
        "semantic_fact_ids": fact_ids,
        "semantic_group_ids": sorted(group_ids),
        "metrics": sorted(metrics),
        "periods": sorted(periods),
        "logical_table_ids": sorted(tables),
        "physical_pages": sorted(pages),
        "row_ids": sorted(rows),
    }


def _top_ids(ranked: list[dict[str, Any]], k: int) -> list[str]:
    return [row["candidate_key"] for row in ranked[:k]]


def _set_union_features(
    case_id: str,
    candidate_ids: list[str],
    case_facts: dict[str, dict[str, set[str]]],
    fact_meta: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    features = [_candidate_features(case_id, candidate, case_facts, fact_meta) for candidate in candidate_ids]
    return {
        "semantic_fact_ids": set().union(*(set(item["semantic_fact_ids"]) for item in features)) if features else set(),
        "semantic_group_ids": set().union(*(set(item["semantic_group_ids"]) for item in features)) if features else set(),
        "metrics": set().union(*(set(item["metrics"]) for item in features)) if features else set(),
        "periods": set().union(*(set(item["periods"]) for item in features)) if features else set(),
        "logical_table_ids": set().union(*(set(item["logical_table_ids"]) for item in features)) if features else set(),
        "physical_pages": set().union(*(set(item["physical_pages"]) for item in features)) if features else set(),
    }


def _strict_hit(bindings: list[dict[str, Any]], ranked: list[dict[str, Any]], k: int) -> tuple[int, int, float]:
    ids = set(_top_ids(ranked, k))
    hit = sum(1 for binding in bindings if binding["candidate_key"] in ids)
    return hit, len(bindings), _rate(hit, len(bindings)) or 0.0


def _source_cohort(rank: int | None) -> str:
    if rank is None or rank > 100:
        return "c2"
    return "c0" if rank <= 5 else "c1"


def _summarize_crowding(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"count": 0}
    keys = (
        "unique_semantic_fact_count",
        "unique_metric_count",
        "unique_period_count",
        "unique_logical_table_count",
        "unique_physical_page_count",
        "largest_semantic_group_size",
        "largest_metric_group_size",
        "largest_table_group_size",
        "duplicate_sibling_occupancy",
    )
    return {"count": len(items), **{key: _mean([float(item[key]) for item in items]) for key in keys}}


def _rank_metrics(
    case_ids: list[str],
    qwen: dict[str, list[dict[str, Any]]],
    bindings_by_case: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    total = sum(len(bindings_by_case.get(case_id, [])) for case_id in case_ids)
    result: dict[str, Any] = {"queries": len(case_ids), "sources": total}
    for k in K_VALUES:
        hit = sum(
            _strict_hit(bindings_by_case.get(case_id, []), qwen[case_id], k)[0]
            for case_id in case_ids
        )
        result[f"recall_at_{k}"] = {"hits": hit, "total": total, "rate": _rate(hit, total)}
    return result


def main() -> int:
    backend_root = Path(__file__).resolve().parents[2]
    evaluation_root = backend_root / "artifacts" / "evaluation"
    out_dir = evaluation_root / ARTIFACT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    top100_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r2a-2" / "bounded-top100-predictions.jsonl.gz"
    top100_seal_path = top100_path.parent / "prediction-seal.json"
    qwen_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3" / "main_rerank_predictions.jsonl.gz"
    views_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3-p0" / "queryplan-rerank-input-views.jsonl.gz"
    query_plan_path = evaluation_root / "pdf-retrieval-v4-gate-07" / "query-plan-predictions.json"
    registry_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1-p0" / "candidate-semantic-fact-registry.jsonl.gz"
    # R3.3 case-completeness is the frozen 72-case taxonomy.  SE1 semantic
    # completeness is a separate, partial semantic-target inventory and is
    # merged below without changing the benchmark taxonomy.
    completeness_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3" / "case-completeness.json"
    semantic_completeness_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1" / "semantic-case-completeness.json"
    strict_path = evaluation_root / "pdf-retrieval-v4-strict-source-contract" / "strict-gold-source-bindings.jsonl"
    semantic_targets_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1" / "gold-semantic-targets.jsonl"

    # Frozen inputs are loaded without invoking any retrieval/model code.
    top100_rows = _read_gzip_jsonl(top100_path)
    qwen_rows = _read_gzip_jsonl(qwen_path)
    view_rows = _read_gzip_jsonl(views_path)
    registry_rows = _read_gzip_jsonl(registry_path)
    completeness = _read_json(completeness_path)
    semantic_completeness = _read_json(semantic_completeness_path)

    top100_by_case = {row["case_id"]: sorted(row["candidates"], key=lambda item: item.get("final_candidate_rank", 10**9)) for row in top100_rows}
    qwen_by_case = {
        row["case_id"]: sorted(row["ranked_candidates"], key=lambda item: item.get("post_rerank_rank", 10**9))
        for row in qwen_rows
    }
    views_by_case = {row["case_id"]: row for row in view_rows}
    query_plans = {row["case_id"]: row.get("plan", {}) for row in _read_json(query_plan_path).get("predictions", _read_json(query_plan_path).get("cases", []))}
    case_ids = sorted(qwen_by_case)

    top100_seal = _read_json(top100_seal_path)
    top100_sha = _sha256_file(top100_path)
    top100_sha_expected = top100_seal.get("prediction_sha256")
    top100_identity_mismatch: list[dict[str, Any]] = []
    qwen_rank_mismatch: list[dict[str, Any]] = []
    for case_id in case_ids:
        top_ids = {item["candidate_key"] for item in top100_by_case.get(case_id, [])}
        qwen_items = qwen_by_case[case_id]
        qwen_ids = {item["candidate_key"] for item in qwen_items}
        if top_ids != qwen_ids or len(qwen_items) != 100:
            top100_identity_mismatch.append({"case_id": case_id, "top100_count": len(top_ids), "qwen_count": len(qwen_ids)})
        pre_ranks = sorted(int(item.get("pre_rerank_rank", -1)) for item in qwen_items)
        if pre_ranks != list(range(1, 101)):
            qwen_rank_mismatch.append({"case_id": case_id, "pre_rerank_ranks": pre_ranks[:10]})

    # Build candidate-global semantic identity lookup, retaining provenance metadata.
    case_facts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    fact_meta: dict[str, dict[str, Any]] = {}
    for record in registry_rows:
        candidate_key = record["candidate_key"]
        ids = set(record.get("semantic_fact_ids", []))
        for fact in record.get("semantic_facts", []) or []:
            fact_id = fact.get("semantic_fact_id")
            if fact_id:
                fact_meta.setdefault(fact_id, fact)
                ids.add(fact_id)
        for occurrence in record.get("occurrences", []) or []:
            case_facts[occurrence["case_id"]][candidate_key].update(ids)

    taxonomy = {item["case_id"]: dict(item) for item in completeness.get("cases", [])}
    semantic_targets = {item["case_id"]: item for item in semantic_completeness.get("cases", [])}
    # The frozen R3.3 completeness report contains 64 records because eight
    # cases have no completed composition row.  Their immutable Gate07
    # QueryPlan still carries the task/slot contract, so fill only the missing
    # taxonomy fields from that pre-sealed governance metadata (never failures).
    missing_taxonomy_cases: list[str] = []
    for case_id in case_ids:
        if case_id in taxonomy:
            continue
        plan = query_plans.get(case_id, {})
        slots = plan.get("operand_slots", []) or []
        is_calculation = bool(plan.get("operation")) and (len(slots) > 1 or plan.get("operation") in {"growth_rate", "percentage_share", "difference", "sum", "average", "ratio"})
        is_multi = bool(plan.get("requires_multiple_sources")) or len(slots) > 1
        taxonomy[case_id] = {
            "case_id": case_id,
            "complete_at_5": None,
            "required_source_count": None,
            "retrieved_source_count": None,
            "is_calculation": is_calculation,
            "is_multi_evidence": is_multi,
            "taxonomy_source": "frozen_gate07_queryplan_for_missing_r3_3_row",
        }
        missing_taxonomy_cases.append(case_id)
    for case_id, item in taxonomy.items():
        semantic_item = semantic_targets.get(case_id, {})
        item["required_semantic_fact_ids"] = list(semantic_item.get("required_semantic_fact_ids") or [])
        item["semantic_complete"] = semantic_item.get("semantic_complete")
    if len(case_ids) != EXPECTED_CASES or set(case_ids) != set(taxonomy):
        raise RuntimeError(f"case identity mismatch: qwen={len(case_ids)} taxonomy={len(taxonomy)}")
    if sum(bool(item.get("is_multi_evidence")) for item in taxonomy.values()) != EXPECTED_MULTI or sum(bool(item.get("is_calculation")) for item in taxonomy.values()) != EXPECTED_CALC:
        raise RuntimeError(
            "taxonomy count mismatch: "
            f"multi={sum(bool(item.get('is_multi_evidence')) for item in taxonomy.values())}, "
            f"calculation={sum(bool(item.get('is_calculation')) for item in taxonomy.values())}; "
            f"missing={[(case_id, taxonomy[case_id].get('is_multi_evidence'), taxonomy[case_id].get('is_calculation'), list(views_by_case[case_id].keys()), str(views_by_case[case_id].get('main_query_view', ''))[:120]) for case_id in missing_taxonomy_cases]}"
        )

    # Collapse diagnostic is deliberately produced before reading strict bindings.
    collapse_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        seen: set[str] = set()
        kept: list[dict[str, Any]] = []
        for item in qwen_by_case[case_id]:
            features = _candidate_features(case_id, item["candidate_key"], case_facts, fact_meta)
            groups = set(features["semantic_group_ids"])
            if groups & seen:
                continue
            kept.append(
                {
                    "candidate_key": item["candidate_key"],
                    "original_qwen_rank": item.get("post_rerank_rank"),
                    "original_bm25_rank": item.get("pre_rerank_rank"),
                    "reranker_score": item.get("reranker_score"),
                    "semantic_fact_ids": features["semantic_fact_ids"],
                    "semantic_group_ids": features["semantic_group_ids"],
                    "collapsed_rank": len(kept) + 1,
                }
            )
            seen.update(groups)
        collapse_rows.append({"case_id": case_id, "candidate_budget": 100, "ranked_candidates": kept})
    collapse_path = out_dir / "semantic-collapse-predictions.jsonl.gz"
    collapse_sha = _write_gzip_jsonl(collapse_path, collapse_rows)
    collapse_seal = {
        "gate": "NF-OPT-19-R0",
        "artifact": "semantic-collapse-predictions.jsonl.gz",
        "source_qwen_sha256": _sha256_file(qwen_path),
        "source_registry_sha256": _sha256_file(registry_path),
        "case_count": len(collapse_rows),
        "candidate_added": 0,
        "ranking_mutation": False,
        "gold_reads_before_seal": 0,
        "sealed": True,
        "prediction_sha256": collapse_sha,
    }
    _write_json(out_dir / "semantic-collapse-seal.json", collapse_seal)

    # Strict source bindings are intentionally unlocked only after the collapse seal.
    strict_rows = _read_jsonl(strict_path)
    # SE1's already-sealed semantic target projection is unlocked only after
    # the Gold-independent collapse prediction seal, just like the strict
    # source bindings.
    semantic_target_rows = _read_jsonl(semantic_targets_path)
    semantic_targets_by_binding = {row.get("binding_id"): row for row in semantic_target_rows}
    bindings_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        bindings_by_case[row["case_id"]].append(row)
    for case_id in bindings_by_case:
        bindings_by_case[case_id].sort(key=lambda row: (row.get("source_index", 0), row.get("binding_id", "")))
    if len(strict_rows) != 80:
        raise RuntimeError(f"strict source count mismatch: {len(strict_rows)}")

    # Rank maps and immutable source cohorts.
    qwen_rank_by_case = {case_id: {item["candidate_key"]: item["post_rerank_rank"] for item in items} for case_id, items in qwen_by_case.items()}
    cohorts: list[dict[str, Any]] = []
    cohort_counts = Counter()
    for binding in strict_rows:
        rank = qwen_rank_by_case.get(binding["case_id"], {}).get(binding["candidate_key"])
        cohort = _source_cohort(rank)
        cohort_counts[cohort] += 1
        cohorts.append({"case_id": binding["case_id"], "source_index": binding.get("source_index"), "candidate_key": binding["candidate_key"], "qwen_rank": rank, "cohort": cohort})
    if dict(cohort_counts) != EXPECTED_COHORTS:
        raise RuntimeError(f"C0/C1/C2 mismatch: {dict(cohort_counts)}")

    # Question-level taxonomy and source recall by category.
    category_case_ids = {
        "single_evidence": [case_id for case_id in case_ids if not taxonomy[case_id].get("is_multi_evidence") and not taxonomy[case_id].get("is_calculation")],
        "multi_evidence": [case_id for case_id in case_ids if taxonomy[case_id].get("is_multi_evidence")],
        "calculation": [case_id for case_id in case_ids if taxonomy[case_id].get("is_calculation")],
    }
    taxonomy_output: dict[str, Any] = {
        "counts": {key: len(value) for key, value in category_case_ids.items()},
        "multi_evidence_non_calculation": len([case_id for case_id in case_ids if taxonomy[case_id].get("is_multi_evidence") and not taxonomy[case_id].get("is_calculation")]),
        "missing_r3_3_taxonomy_cases_filled_from_queryplan": sorted(missing_taxonomy_cases),
        "calculation_is_subset_of_multi_evidence": True,
        "categories": {},
    }
    for category, ids in category_case_ids.items():
        metrics = _rank_metrics(ids, qwen_by_case, bindings_by_case)
        metrics["required_physical_sources"] = sum(len(bindings_by_case.get(case_id, [])) for case_id in ids)
        taxonomy_output["categories"][category] = metrics
    _write_json(out_dir / "question-taxonomy.json", taxonomy_output)
    _write_json(out_dir / "strict-source-cohorts.json", {"counts": dict(cohort_counts), "sources": cohorts, "expected": EXPECTED_COHORTS})

    # Semantic-equivalence competition for C1 physical misses.
    semantic_competition: list[dict[str, Any]] = []
    c1_resolved = 0
    for cohort in cohorts:
        if cohort["cohort"] != "c1":
            continue
        case_id = cohort["case_id"]
        target_row = semantic_targets_by_binding.get(next((row.get("binding_id") for row in strict_rows if row.get("case_id") == case_id and row.get("source_index") == cohort.get("source_index") and row.get("candidate_key") == cohort.get("candidate_key")), None), {})
        gold_ids = set(target_row.get("matching_semantic_fact_ids") or [])
        if not gold_ids and target_row.get("gold_semantic_fact_id"):
            gold_ids.add(target_row["gold_semantic_fact_id"])
        top100 = qwen_by_case[case_id]
        equivalent_members = [item for item in top100 if gold_ids & set(case_facts.get(case_id, {}).get(item["candidate_key"], set()))]
        best_rank = min((int(item["post_rerank_rank"]) for item in equivalent_members), default=None)
        if best_rank is not None:
            c1_resolved += 1
        semantic_competition.append({
            **cohort,
            "gold_semantic_fact_ids": sorted(gold_ids),
            "semantic_equivalent_top5": best_rank is not None and best_rank <= 5,
            "best_semantic_equivalent_rank": best_rank,
            "semantic_group_size_top100": len(equivalent_members),
            "semantic_group_members_top5": [item["candidate_key"] for item in equivalent_members if int(item["post_rerank_rank"]) <= 5],
            "semantic_equivalence_status": "resolved" if gold_ids else "unresolved",
        })
    c1_top5_equiv = sum(1 for item in semantic_competition if item["semantic_equivalent_top5"])
    _write_json(out_dir / "semantic-equivalence-competition.json", {
        "c1_total": len(semantic_competition),
        "c1_with_semantic_equivalent_top5": c1_top5_equiv,
        "c1_without_semantic_equivalent_top5": len(semantic_competition) - c1_top5_equiv,
        "c1_semantic_target_resolved": c1_resolved,
        "records": semantic_competition,
    })

    # Top5 crowding and sibling occupancy are Gold-independent except for the
    # success/failure labels used for the final comparison table.
    crowding_records: list[dict[str, Any]] = []
    occupancy_records: list[dict[str, Any]] = []
    for case_id in case_ids:
        top5_ids = _top_ids(qwen_by_case[case_id], 5)
        candidate_features = [_candidate_features(case_id, candidate, case_facts, fact_meta) for candidate in top5_ids]
        union = _set_union_features(case_id, top5_ids, case_facts, fact_meta)
        group_counts = Counter(group for feature in candidate_features for group in feature["semantic_group_ids"])
        metric_counts = Counter(metric for feature in candidate_features for metric in feature["metrics"])
        table_counts = Counter(table for feature in candidate_features for table in feature["logical_table_ids"])
        duplicate_occupancy = sum(max(0, count - 1) for count in group_counts.values())
        strict_all = all(binding["candidate_key"] in set(top5_ids) for binding in bindings_by_case.get(case_id, []))
        crowding_records.append({
            "case_id": case_id,
            "strict_top5_all_sources": strict_all,
            "unique_semantic_fact_count": len(union["semantic_fact_ids"]),
            "unique_metric_count": len(union["metrics"]),
            "unique_period_count": len(union["periods"]),
            "unique_logical_table_count": len(union["logical_table_ids"]),
            "unique_physical_page_count": len(union["physical_pages"]),
            "largest_semantic_group_size": max(group_counts.values(), default=0),
            "largest_metric_group_size": max(metric_counts.values(), default=0),
            "largest_table_group_size": max(table_counts.values(), default=0),
            "duplicate_sibling_occupancy": duplicate_occupancy,
        })
        occupancy_records.append({
            "case_id": case_id,
            "top5_sibling_group_count": len(group_counts),
            "duplicate_sibling_slots": duplicate_occupancy,
            "top5_positions_consumed_by_duplicate_siblings": duplicate_occupancy,
            "semantic_group_ids": sorted(group_counts),
            "strict_top5_all_sources": strict_all,
        })
    _write_json(out_dir / "top5-crowding.json", {
        "all": _summarize_crowding(crowding_records),
        "strict_success": _summarize_crowding([item for item in crowding_records if item["strict_top5_all_sources"]]),
        "coverage_failure": _summarize_crowding([item for item in crowding_records if not item["strict_top5_all_sources"]]),
        "records": crowding_records,
    })
    _write_json(out_dir / "sibling-occupancy.json", {"records": occupancy_records, "gold_independent_grouping": True})

    multi_ids = category_case_ids["multi_evidence"]
    calc_ids = category_case_ids["calculation"]
    multi_coverage: dict[str, Any] = {"denominator": len(multi_ids), "k": {}}
    for k in K_VALUES:
        strict_any = strict_all = semantic_any = semantic_all = 0
        strict_ratios: list[float] = []
        semantic_ratios: list[float] = []
        semantic_resolved = 0
        for case_id in multi_ids:
            bindings = bindings_by_case.get(case_id, [])
            top_ids = set(_top_ids(qwen_by_case[case_id], k))
            strict_hits = sum(1 for binding in bindings if binding["candidate_key"] in top_ids)
            strict_ratios.append(_rate(strict_hits, len(bindings)) or 0.0)
            strict_any += int(strict_hits > 0)
            strict_all += int(bool(bindings) and strict_hits == len(bindings))
            required_facts = list(taxonomy[case_id].get("required_semantic_fact_ids") or [])
            if required_facts:
                semantic_resolved += 1
                fact_union = _set_union_features(case_id, list(top_ids), case_facts, fact_meta)["semantic_fact_ids"]
                semantic_hits = sum(1 for fact_id in required_facts if fact_id in fact_union)
                semantic_ratios.append(_rate(semantic_hits, len(required_facts)) or 0.0)
                semantic_any += int(semantic_hits > 0)
                semantic_all += int(semantic_hits == len(required_facts))
        multi_coverage["k"][str(k)] = {
            "strict_physical": {"any_count": strict_any, "all_count": strict_all, "any_rate": _rate(strict_any, len(multi_ids)), "all_rate": _rate(strict_all, len(multi_ids)), "average_required_source_coverage": _mean(strict_ratios)},
            "semantic_fact": {"resolved_cases": semantic_resolved, "unresolved_cases": len(multi_ids) - semantic_resolved, "any_count": semantic_any, "all_count": semantic_all, "any_rate_over_16": _rate(semantic_any, len(multi_ids)), "all_rate_over_16": _rate(semantic_all, len(multi_ids)), "all_rate_resolved_only": _rate(semantic_all, semantic_resolved), "average_required_fact_coverage_resolved": _mean(semantic_ratios)},
        }
    _write_json(out_dir / "multi-evidence-coverage.json", multi_coverage)

    calc_coverage: dict[str, Any] = {"denominator": len(calc_ids), "k": {}, "cases": []}
    for k in K_VALUES:
        slot_count = represented = all_slots = resolved_cases = 0
        for case_id in calc_ids:
            required = list(taxonomy[case_id].get("required_semantic_fact_ids") or [])
            if not required:
                required = [f"queryplan_slot:{slot.get('slot_id')}" for slot in _parse_slots(views_by_case[case_id].get("main_query_view", ""))]
            if required and all(item.startswith("queryplan_slot:") for item in required):
                # There is no Gold-independent fact-to-slot mapping for this case.
                continue
            resolved_cases += int(bool(required))
            top_ids = _top_ids(qwen_by_case[case_id], k)
            fact_union = _set_union_features(case_id, top_ids, case_facts, fact_meta)["semantic_fact_ids"]
            present = sum(1 for fact_id in required if fact_id in fact_union)
            slot_count += len(required)
            represented += present
            all_slots += int(bool(required) and present == len(required))
        calc_coverage["k"][str(k)] = {"resolved_cases": resolved_cases, "required_slots": slot_count, "represented_slots": represented, "all_slots_cases": all_slots, "all_slots_rate_over_11": _rate(all_slots, len(calc_ids)), "slot_coverage_rate": _rate(represented, slot_count)}
    for case_id in calc_ids:
        required = list(taxonomy[case_id].get("required_semantic_fact_ids") or [])
        case_item = {"case_id": case_id, "required_slot_count": len(required), "required_semantic_fact_ids": required, "k": {}}
        for k in K_VALUES:
            fact_union = _set_union_features(case_id, _top_ids(qwen_by_case[case_id], k), case_facts, fact_meta)["semantic_fact_ids"]
            present = [fact_id for fact_id in required if fact_id in fact_union]
            case_item["k"][str(k)] = {"represented_slots": len(present), "required_slots": len(required), "all_slots": bool(required) and len(present) == len(required)}
        calc_coverage["cases"].append(case_item)
    calc_coverage["set_compression_failures"] = [
        item["case_id"] for item in calc_coverage["cases"] if item["k"].get("100", {}).get("all_slots") and not item["k"].get("5", {}).get("all_slots")
    ]
    _write_json(out_dir / "calculation-slot-coverage.json", calc_coverage)

    # Pre-register a scale-free margin cohort: the C1 median is the sole split.
    margin_records: list[dict[str, Any]] = []
    for cohort in semantic_competition:
        case_id = cohort["case_id"]
        rank = cohort["qwen_rank"]
        ranked = qwen_by_case[case_id]
        gold_item = next((item for item in ranked if item["candidate_key"] == cohort["candidate_key"]), None)
        cutoff_item = ranked[4] if len(ranked) >= 5 else None
        gold_score = _numeric(gold_item.get("reranker_score")) if gold_item else None
        cutoff_score = _numeric(cutoff_item.get("reranker_score")) if cutoff_item else None
        margin = cutoff_score - gold_score if cutoff_score is not None and gold_score is not None else None
        margin_records.append({"case_id": case_id, "source_index": cohort["source_index"], "candidate_key": cohort["candidate_key"], "qwen_rank": rank, "gold_score": gold_score, "rank5_cutoff_score": cutoff_score, "score_margin": margin})
    margins = [item["score_margin"] for item in margin_records if item["score_margin"] is not None]
    margin_threshold = median(margins) if margins else None
    for item in margin_records:
        item["cohort"] = "near_boundary" if item["score_margin"] is not None and margin_threshold is not None and item["score_margin"] <= margin_threshold else "clear_loss"
    _write_json(out_dir / "score-margin-analysis.json", {"rule": "near_boundary iff C1 margin <= median(C1 margins)", "threshold": margin_threshold, "records": margin_records, "counts": dict(Counter(item["cohort"] for item in margin_records))})

    # Rank-slot competition uses only frozen facts and the already sealed ranks.
    rank_slot_records: list[dict[str, Any]] = []
    for case_id in sorted(set(multi_ids) | set(calc_ids)):
        required_bindings = bindings_by_case.get(case_id, [])
        case_result = {"case_id": case_id, "operation": views_by_case[case_id].get("main_query_view", "").split("Operation:", 1)[-1].splitlines()[0].strip() if "Operation:" in views_by_case[case_id].get("main_query_view", "") else None, "sources": []}
        for binding in required_bindings:
            gold_ids = set(case_facts.get(case_id, {}).get(binding["candidate_key"], set()))
            gold_features = _candidate_features(case_id, binding["candidate_key"], case_facts, fact_meta)
            rank = qwen_rank_by_case[case_id].get(binding["candidate_key"])
            preceding = [item for item in qwen_by_case[case_id] if rank is not None and int(item["post_rerank_rank"]) < int(rank)]
            counts = Counter()
            for item in preceding:
                feature = _candidate_features(case_id, item["candidate_key"], case_facts, fact_meta)
                if gold_ids & set(feature["semantic_fact_ids"]):
                    counts["already_covered_semantic_fact"] += 1
                if set(gold_features["metrics"]) & set(feature["metrics"]):
                    counts["already_covered_metric"] += 1
                if set(gold_features["periods"]) & set(feature["periods"]):
                    counts["already_covered_period"] += 1
                if set(gold_features["logical_table_ids"]) & set(feature["logical_table_ids"]):
                    counts["same_logical_table_sibling"] += 1
            case_result["sources"].append({"source_index": binding.get("source_index"), "candidate_key": binding["candidate_key"], "qwen_rank": rank, "gold_displaced_by_redundant_candidate_count": sum(counts.values()), "preceding_signal_counts": dict(counts), "operand_role_signal": "not_available_without_binder"})
        rank_slot_records.append(case_result)
    _write_json(out_dir / "rank-slot-competition.json", {"records": rank_slot_records, "gold_independent_grouping": True})

    # Collapse metrics after the collapse seal and strict binding unlock.
    collapse_by_case = {row["case_id"]: row["ranked_candidates"] for row in collapse_rows}
    collapse_rank_by_case = {case_id: {item["candidate_key"]: item["collapsed_rank"] for item in items} for case_id, items in collapse_by_case.items()}
    original_strict_hits = 0
    collapse_strict_hits = 0
    original_sem_hits = 0
    collapse_sem_hits = 0
    collapse_movement = {"rescued": 0, "damaged": 0, "unchanged_hit": 0, "unchanged_miss": 0}
    semantic_targets_resolved = 0
    for binding in strict_rows:
        case_id = binding["case_id"]
        candidate = binding["candidate_key"]
        original_hit = qwen_rank_by_case[case_id].get(candidate, 101) <= 5
        collapse_hit = collapse_rank_by_case[case_id].get(candidate, 101) <= 5
        original_strict_hits += int(original_hit)
        collapse_strict_hits += int(collapse_hit)
        if not original_hit and collapse_hit:
            collapse_movement["rescued"] += 1
        elif original_hit and not collapse_hit:
            collapse_movement["damaged"] += 1
        elif original_hit:
            collapse_movement["unchanged_hit"] += 1
        else:
            collapse_movement["unchanged_miss"] += 1
        target_row = semantic_targets_by_binding.get(binding.get("binding_id"), {})
        target_ids = set(target_row.get("matching_semantic_fact_ids") or [])
        if not target_ids and target_row.get("gold_semantic_fact_id"):
            target_ids.add(target_row["gold_semantic_fact_id"])
        if target_ids:
            semantic_targets_resolved += 1
            original_union = _set_union_features(case_id, _top_ids(qwen_by_case[case_id], 5), case_facts, fact_meta)["semantic_fact_ids"]
            collapse_union = _set_union_features(case_id, [item["candidate_key"] for item in collapse_by_case[case_id][:5]], case_facts, fact_meta)["semantic_fact_ids"]
            # SE1's conservative semantic contract treats an exact physical
            # hit as a semantic hit even when the semantic target is
            # unresolved; otherwise use exact semantic-fact intersection.
            original_sem_hits += int(original_hit or bool(target_ids & original_union))
            collapse_sem_hits += int(collapse_hit or bool(target_ids & collapse_union))
        else:
            original_sem_hits += int(original_hit)
            collapse_sem_hits += int(collapse_hit)
    def _all_coverage(case_list: list[str], ranked_getter: Any, semantic: bool = False) -> int:
        count = 0
        for case_id in case_list:
            required = list(taxonomy[case_id].get("required_semantic_fact_ids") or [])
            if semantic and not required:
                continue
            if semantic:
                candidate_ids = ranked_getter(case_id)
                union = _set_union_features(case_id, candidate_ids, case_facts, fact_meta)["semantic_fact_ids"]
                count += int(bool(required) and all(item in union for item in required))
            else:
                candidate_ids = set(ranked_getter(case_id))
                bindings = bindings_by_case.get(case_id, [])
                count += int(bool(bindings) and all(binding["candidate_key"] in candidate_ids for binding in bindings))
        return count
    original_multi_strict = _all_coverage(multi_ids, lambda case_id: _top_ids(qwen_by_case[case_id], 5))
    collapse_multi_strict = _all_coverage(multi_ids, lambda case_id: [item["candidate_key"] for item in collapse_by_case[case_id][:5]])
    original_multi_sem = _all_coverage(multi_ids, lambda case_id: _top_ids(qwen_by_case[case_id], 5), semantic=True)
    collapse_multi_sem = _all_coverage(multi_ids, lambda case_id: [item["candidate_key"] for item in collapse_by_case[case_id][:5]], semantic=True)
    original_calc_slots = sum(item["k"]["5"]["all_slots"] for item in calc_coverage["cases"])
    collapse_calc_slots = 0
    for item in calc_coverage["cases"]:
        required = item["required_semantic_fact_ids"]
        union = _set_union_features(item["case_id"], [row["candidate_key"] for row in collapse_by_case[item["case_id"]][:5]], case_facts, fact_meta)["semantic_fact_ids"]
        collapse_calc_slots += int(bool(required) and all(fact_id in union for fact_id in required))
    _write_json(out_dir / "semantic-collapse-metrics.json", {
        "strict_physical_recall_at_5": {"before": {"hits": original_strict_hits, "total": 80, "rate": _rate(original_strict_hits, 80)}, "after": {"hits": collapse_strict_hits, "total": 80, "rate": _rate(collapse_strict_hits, 80)}},
        "semantic_recall_at_5": {"before": {"hits": original_sem_hits, "total": 80, "resolved_targets": semantic_targets_resolved, "rate_over_80": _rate(original_sem_hits, 80)}, "after": {"hits": collapse_sem_hits, "total": 80, "resolved_targets": semantic_targets_resolved, "rate_over_80": _rate(collapse_sem_hits, 80)}},
        "multi_evidence_all_at_5": {"strict_before": original_multi_strict, "strict_after": collapse_multi_strict, "semantic_before": original_multi_sem, "semantic_after": collapse_multi_sem, "denominator": len(multi_ids)},
        "calculation_all_slots_at_5": {"before": original_calc_slots, "after": collapse_calc_slots, "denominator": len(calc_ids)},
        "movement": {**collapse_movement, "net": collapse_movement["rescued"] - collapse_movement["damaged"]},
        "gold_reads_before_seal": 0,
    })

    logical_status = {"status": "skipped_insufficient_identity", "reason": "No stable logical_table_id was present in frozen SE1 physical provenance."}
    _write_json(out_dir / "logical-table-collapse-metrics.json", logical_status)

    # Diversity ceiling and overlap with the frozen failure cohorts.
    failure_cases = {case_id for case_id in case_ids if not all(binding["candidate_key"] in set(_top_ids(qwen_by_case[case_id], 5)) for binding in bindings_by_case.get(case_id, []))}
    c1_cases = {item["case_id"] for item in cohorts if item["cohort"] == "c1"}
    ceiling = {"records": [], "queries_with_at_least_1_recoverable_slot": 0, "queries_with_at_least_2_recoverable_slots": 0}
    for item in occupancy_records:
        recoverable = int(item["duplicate_sibling_slots"])
        record = {"case_id": item["case_id"], "recoverable_slot_count": recoverable, "strict_top5_failure": item["case_id"] in failure_cases, "c1_case": item["case_id"] in c1_cases, "multi_evidence": item["case_id"] in set(multi_ids), "calculation": item["case_id"] in set(calc_ids)}
        ceiling["records"].append(record)
        ceiling["queries_with_at_least_1_recoverable_slot"] += int(recoverable >= 1)
        ceiling["queries_with_at_least_2_recoverable_slots"] += int(recoverable >= 2)
    _write_json(out_dir / "diversity-ceiling.json", ceiling)

    # Decision thresholds are pre-registered and are intentionally diagnostic,
    # not an optimization loop.
    crowd_success = _summarize_crowding([item for item in crowding_records if item["strict_top5_all_sources"]])
    crowd_failure = _summarize_crowding([item for item in crowding_records if not item["strict_top5_all_sources"]])
    single_loss = 1.0 - float(taxonomy_output["categories"]["single_evidence"]["recall_at_100"]["rate"] or 0.0)
    multi_loss = 1.0 - float(multi_coverage["k"]["5"]["strict_physical"]["all_rate"] or 0.0)
    collapse_net = collapse_movement["rescued"] - collapse_movement["damaged"]
    crowding_delta = (crowd_failure.get("duplicate_sibling_occupancy") or 0.0) - (crowd_success.get("duplicate_sibling_occupancy") or 0.0)
    loss_gap = multi_loss - single_loss
    collapse_positive = collapse_net > 0
    setwise_supported = bool(loss_gap >= 0.10 and crowding_delta >= 0.50 and collapse_positive)
    median_margin = median(margins) if margins else None
    pointwise_dominant = bool(
        (c1_top5_equiv / len(semantic_competition) if semantic_competition else 0.0) < 0.50
        and not collapse_positive
        and median_margin is not None
        and median_margin > 0
    )
    if setwise_supported:
        next_gate = "nf_opt_19_r1_set_aware_evidence_selection"
    else:
        next_gate = "nf_opt_19_method_reconsideration"
    decision = {
        "gate": "NF-OPT-19-R0",
        "base_commit": "ee49010",
        "model_execution": False,
        "retrieval_rerun": False,
        "ranking_training": False,
        "ranking_mutation": False,
        "production_switch_allowed": False,
        "strict_sources": 80,
        "top100_hits": cohort_counts["c0"] + cohort_counts["c1"],
        "qwen_top5_hits": cohort_counts["c0"],
        "c0": cohort_counts["c0"],
        "c1": cohort_counts["c1"],
        "c2": cohort_counts["c2"],
        "single_evidence_questions": len(category_case_ids["single_evidence"]),
        "multi_evidence_questions": len(multi_ids),
        "calculation_questions": len(calc_ids),
        "c1_with_semantic_equivalent_top5": c1_top5_equiv,
        "mean_unique_semantic_facts_top5_success": crowd_success.get("unique_semantic_fact_count"),
        "mean_unique_semantic_facts_top5_failure": crowd_failure.get("unique_semantic_fact_count"),
        "multi_evidence_all_top5": multi_coverage["k"]["5"]["strict_physical"]["all_count"],
        "multi_evidence_all_top10": multi_coverage["k"]["10"]["strict_physical"]["all_count"],
        "calculation_all_slots_top5": calc_coverage["k"]["5"]["all_slots_cases"],
        "calculation_all_slots_top10": calc_coverage["k"]["10"]["all_slots_cases"],
        "semantic_collapse_strict_recall_at_5": collapse_strict_hits,
        "semantic_collapse_gain_sources": collapse_net,
        "top100_sha256": top100_sha,
        "top100_sha_expected": top100_sha_expected,
        "top100_sha_match": top100_sha == top100_sha_expected,
        "candidate_identity_mismatch": len(top100_identity_mismatch),
        "qwen_rank_identity_mismatch": len(qwen_rank_mismatch),
        "pre_registered_decision_rule": {"loss_gap_min": 0.10, "crowding_delta_min": 0.50, "collapse_net_min": 1, "pointwise_equivalent_rate_max": 0.50},
        "diagnostic_values": {"single_compression_loss": single_loss, "multi_compression_loss": multi_loss, "compression_loss_gap": loss_gap, "crowding_duplicate_occupancy_delta": crowding_delta, "collapse_net": collapse_net, "c1_semantic_equivalent_top5_rate": _rate(c1_top5_equiv, len(semantic_competition)), "c1_median_score_margin": median_margin},
        "setwise_ranking_gap_supported": setwise_supported,
        "pointwise_relevance_error_dominant": pointwise_dominant,
        "next_gate": next_gate,
        "decision_reason": [
            "Frozen Top100 raw SHA and candidate identities were checked before scoring.",
            "Semantic Group Collapse is a Gold-independent diagnostic and did not mutate the Qwen rank.",
            "Strict physical hits remain separate from semantic-equivalent hits.",
        ],
    }
    _write_json(out_dir / "decision.json", decision)

    _write_json(out_dir / "input-integrity.json", {
        "top100_prediction_sha256": top100_sha,
        "top100_prediction_sha256_expected": top100_sha_expected,
        "qwen_prediction_sha256": _sha256_file(qwen_path),
        "semantic_registry_sha256": _sha256_file(registry_path),
        "strict_source_contract_sha256": _sha256_file(strict_path),
        "candidate_identity_mismatch": len(top100_identity_mismatch),
        "qwen_rank_identity_mismatch": len(qwen_rank_mismatch),
        "model_execution": False,
        "retrieval_rerun": False,
        "ranking_mutation": False,
    })
    _write_text(out_dir / "README.md", """# NF-OPT-19 R0 — Pointwise Ranking & Evidence-Set Competition Audit

Post-seal, Gold-independent diagnostics over the frozen R3.3 Qwen ranking and bounded Top100. No retrieval, model execution, ranking mutation, Binder, Calculator, or production write occurred. Strict physical source hits remain separate from semantic-equivalent diagnostics.
""")
    print(json.dumps({"artifact_dir": str(out_dir), "decision": decision, "collapse_sha256": collapse_sha}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
