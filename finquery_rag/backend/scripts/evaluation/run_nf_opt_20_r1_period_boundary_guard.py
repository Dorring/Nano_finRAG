"""NF-OPT-20 R1: frozen period-consistency secondary discriminator.

Prediction generation is deliberately Gold-blind.  The only runtime change is
an at-most-one rank-5/rank-6..10 swap using frozen query/candidate period
metadata and the executable NF-OPT-19 near-boundary threshold.  Gold is read
only after the prediction artifact is sealed for post-seal diagnostics.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


K_VALUES = (1, 3, 5, 10, 20, 50, 100)
OUT_NAME = "nf-opt-20-r1-period-boundary-guard"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return sha256_file(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def normalize_period(value: Any) -> str | None:
    text = norm(value)
    if not text:
        return None
    years = re.findall(r"(?:fiscal year|fy|year)?\s*(19\d{2}|20\d{2})\b", text)
    quarters = re.findall(r"\bq([1-4])\b", text)
    if years:
        prefix = "q" + quarters[0] + " " if quarters else ""
        return prefix + "fy" + years[-1]
    if quarters:
        return "q" + quarters[0]
    return None


def normalize_period_values(values: Iterable[Any]) -> tuple[set[str], bool]:
    normalized: set[str] = set()
    had_value = False
    unresolved = False
    for value in values:
        if value is None or not str(value).strip():
            continue
        had_value = True
        pieces = re.split(r"[|,;]+", str(value))
        for piece in pieces:
            period = normalize_period(piece)
            if period:
                normalized.add(period)
            elif piece.strip():
                unresolved = True
    return normalized, had_value and unresolved


def parse_plan_rows(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        rows = payload
    else:
        rows = payload.get("plans") or payload.get("predictions") or payload.get("cases") or []
    return {row["case_id"]: row.get("plan", {}) for row in rows or []}


def parse_document_view(view: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for line in str(view or "").splitlines():
        stripped = line.strip()
        match = re.match(r"^([A-Za-z][A-Za-z ]*):\s*(.*)$", stripped)
        if match:
            fields.setdefault(norm(match.group(1)).replace(" ", "_"), match.group(2).strip())
    return {
        "document_id": fields.get("document", ""),
        "page": fields.get("page"),
        "period": fields.get("period", ""),
        "raw": view,
    }


def build_fact_registry(registry_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict[str, Any]]]:
    case_facts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    fact_meta: dict[str, dict[str, Any]] = {}
    for record in registry_rows:
        ids = set(record.get("semantic_fact_ids", []))
        for fact in record.get("semantic_facts", []) or []:
            fact_id = fact.get("semantic_fact_id")
            if fact_id:
                ids.add(fact_id)
                fact_meta[fact_id] = fact
        for occurrence in record.get("occurrences", []) or []:
            case_facts[occurrence["case_id"]][record["candidate_key"]].update(ids)
    return case_facts, fact_meta


def candidate_periods(
    case_id: str,
    candidate_key: str,
    candidate_views: dict[str, dict[str, Any]],
    case_facts: dict[str, dict[str, set[str]]],
    fact_meta: dict[str, dict[str, Any]],
) -> tuple[set[str], list[str], bool]:
    view = parse_document_view(candidate_views.get(case_id, {}).get(candidate_key, {}).get("document_view", ""))
    raw_values: list[Any] = [view.get("period", "")]
    for fact_id in case_facts.get(case_id, {}).get(candidate_key, set()):
        fact = fact_meta.get(fact_id, {})
        raw_values.append(fact.get("normalized_period") or fact.get("period"))
    periods, unresolved = normalize_period_values(raw_values)
    return periods, [str(value) for value in raw_values if value], unresolved


def query_period_contract(plan: dict[str, Any]) -> dict[str, Any]:
    raw_periods = list(plan.get("periods") or [])
    periods, unresolved = normalize_period_values(raw_periods)
    if not raw_periods:
        status = "no_explicit_period"
    elif unresolved or not periods:
        status = "unresolved"
    elif len(periods) == 1:
        status = "explicit_single_period"
    else:
        status = "explicit_multi_period"
    return {"status": status, "raw_periods": raw_periods, "required_periods": sorted(periods)}


def period_status(query: dict[str, Any], periods: set[str], unresolved: bool) -> str:
    if query["status"] not in {"explicit_single_period", "explicit_multi_period"}:
        return "NEUTRAL"
    required = set(query["required_periods"])
    if not periods or unresolved:
        return "NEUTRAL"
    overlap = required & periods
    if required <= periods:
        return "FULL_MATCH"
    if overlap:
        return "PARTIAL_MATCH"
    return "EXPLICIT_CONFLICT"


def top_ids(items: list[dict[str, Any]], k: int) -> list[str]:
    return [item["candidate_key"] for item in items[:k]]


def rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    return {item["candidate_key"]: index + 1 for index, item in enumerate(items)}


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_strict_metrics(bindings_by_case: dict[str, list[dict[str, Any]]], ranked: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    total = sum(len(rows) for rows in bindings_by_case.values())
    output: dict[str, Any] = {}
    for k in K_VALUES:
        hits = sum(binding["candidate_key"] in set(top_ids(ranked[binding["case_id"]], k)) for rows in bindings_by_case.values() for binding in rows)
        output[f"@{k}"] = {"hits": hits, "total": total, "rate": rate(hits, total)}
    return output


def semantic_union(case_id: str, items: list[dict[str, Any]], k: int, case_facts: dict[str, dict[str, set[str]]]) -> set[str]:
    return set().union(*(case_facts.get(case_id, {}).get(item["candidate_key"], set()) for item in items[:k]))


def load_semantic_targets(path: Path) -> dict[str, dict[str, Any]]:
    return {row["binding_id"]: row for row in read_jsonl(path)}


def compute_semantic_metrics(
    strict_rows: list[dict[str, Any]],
    target_by_binding: dict[str, dict[str, Any]],
    ranked: dict[str, list[dict[str, Any]]],
    case_facts: dict[str, dict[str, set[str]]],
) -> tuple[dict[str, Any], dict[int, int]]:
    resolved = 0
    hits_by_k = {5: 0, 10: 0}
    for binding in strict_rows:
        target = target_by_binding.get(binding.get("binding_id"), {})
        target_ids = set(target.get("matching_semantic_fact_ids") or [])
        if not target_ids and target.get("gold_semantic_fact_id"):
            target_ids.add(target["gold_semantic_fact_id"])
        if target_ids:
            resolved += 1
        ranks = rank_map(ranked[binding["case_id"]])
        for k in hits_by_k:
            physical = ranks.get(binding["candidate_key"], 101) <= k
            union = semantic_union(binding["case_id"], ranked[binding["case_id"]], k, case_facts)
            hits_by_k[k] += int(physical or bool(target_ids & union))
    return {
        f"@{k}": {"hits": hits, "total": 80, "resolved_targets": resolved, "rate_over_80": rate(hits, 80)}
        for k, hits in hits_by_k.items()
    }, hits_by_k


def classify_case(case_id: str, plan: dict[str, Any], binding_count: int) -> str:
    is_calculation = bool(plan.get("operation")) and len(plan.get("operand_slots", []) or []) > 1
    if is_calculation:
        return "calculation"
    if binding_count > 1 or bool(plan.get("requires_multiple_sources")):
        return "multi_evidence"
    return "single_evidence"


def coverage_all(case_ids: list[str], bindings_by_case: dict[str, list[dict[str, Any]]], ranked: dict[str, list[dict[str, Any]]], k: int) -> int:
    return sum(
        int(bool(bindings_by_case.get(case_id)) and all(binding["candidate_key"] in set(top_ids(ranked[case_id], k)) for binding in bindings_by_case[case_id]))
        for case_id in case_ids
    )


def calculation_coverage(calc_cases: list[dict[str, Any]], ranked: dict[str, list[dict[str, Any]]], case_facts: dict[str, dict[str, set[str]]], k: int) -> tuple[int, list[dict[str, Any]]]:
    count = 0
    records: list[dict[str, Any]] = []
    for item in calc_cases:
        required = list(item.get("required_semantic_fact_ids") or [])
        union = semantic_union(item["case_id"], ranked[item["case_id"]], k, case_facts)
        all_slots = bool(required) and all(fact_id in union for fact_id in required)
        count += int(all_slots)
        records.append({"case_id": item["case_id"], "required_slot_count": len(required), "represented_slot_count": sum(fact_id in union for fact_id in required), "all_slots": all_slots})
    return count, records


def main() -> int:
    backend_root = Path(__file__).resolve().parents[2]
    evaluation_root = backend_root / "artifacts" / "evaluation"
    out_dir = evaluation_root / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    qwen_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3" / "main_rerank_predictions.jsonl.gz"
    qwen_seal_path = qwen_path.parent / "prediction-seal.json"
    bounded_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r2a-2" / "bounded-top100-predictions.jsonl.gz"
    bounded_seal_path = bounded_path.parent / "prediction-seal.json"
    views_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3-p0" / "queryplan-rerank-input-views.jsonl.gz"
    registry_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1-p0" / "candidate-semantic-fact-registry.jsonl.gz"
    query_plan_path = evaluation_root / "pdf-retrieval-v4-gate-07" / "query-plan-predictions.json"
    strict_path = evaluation_root / "pdf-retrieval-v4-strict-source-contract" / "strict-gold-source-bindings.jsonl"
    targets_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1" / "gold-semantic-targets.jsonl"
    nf19_root = evaluation_root / "nf-opt-19-r0-setwise-ranking-audit"
    nf20_root = evaluation_root / "nf-opt-20-r0-pointwise-discrimination-audit"

    qwen_rows = read_gzip_jsonl(qwen_path)
    bounded_rows = read_gzip_jsonl(bounded_path)
    view_rows = read_gzip_jsonl(views_path)
    query_plans = parse_plan_rows(read_json(query_plan_path))
    candidate_views: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in view_rows:
        for candidate in row.get("candidates", []) or []:
            candidate_views[row["case_id"]][candidate["candidate_key"]] = candidate
    registry_rows = read_gzip_jsonl(registry_path)
    case_facts, fact_meta = build_fact_registry(registry_rows)

    qwen_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in qwen_rows:
        ranked = sorted(row["ranked_candidates"], key=lambda item: int(item["post_rerank_rank"]))
        if len(ranked) != 100 or [int(item["post_rerank_rank"]) for item in ranked] != list(range(1, 101)):
            raise RuntimeError(f"invalid frozen Qwen Top100: {row['case_id']}")
        qwen_by_case[row["case_id"]] = ranked

    bounded_by_case = {row["case_id"]: row["candidates"] for row in bounded_rows}
    bounded_identity_mismatch = sum(
        set(item["candidate_key"] for item in bounded_by_case.get(case_id, []))
        != set(item["candidate_key"] for item in qwen_by_case.get(case_id, []))
        for case_id in qwen_by_case
    )
    if bounded_identity_mismatch:
        raise RuntimeError("frozen bounded Top100 and Qwen Top100 candidate identities differ")

    qwen_sha = sha256_file(qwen_path)
    qwen_seal = read_json(qwen_seal_path)
    bounded_sha = sha256_file(bounded_path)
    bounded_seal = read_json(bounded_seal_path)
    expected_qwen_sha = qwen_seal.get("main_prediction_sha256")
    if qwen_sha != expected_qwen_sha:
        raise RuntimeError("frozen Qwen prediction SHA mismatch")
    if bounded_sha != bounded_seal.get("prediction_sha256"):
        raise RuntimeError("frozen bounded Top100 prediction SHA mismatch")
    near_definition = read_json(nf19_root / "score-margin-analysis.json")
    threshold = float(near_definition["threshold"])
    if near_definition.get("rule") != "near_boundary iff C1 margin <= median(C1 margins)":
        raise RuntimeError("NF-OPT-19 near-boundary definition is not executable/frozen")

    # Gold-blind BPG-V1 prediction generation.
    resolution_records: list[dict[str, Any]] = []
    eligibility_records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    mutated_cases: set[str] = set()
    for case_id in sorted(qwen_by_case):
        original = qwen_by_case[case_id]
        query = query_period_contract(query_plans.get(case_id, {}))
        status_by_key: dict[str, str] = {}
        raw_periods_by_key: dict[str, list[str]] = {}
        unresolved_by_key: dict[str, bool] = {}
        for item in original:
            periods, raw_periods, unresolved = candidate_periods(case_id, item["candidate_key"], candidate_views, case_facts, fact_meta)
            status_by_key[item["candidate_key"]] = period_status(query, periods, unresolved)
            raw_periods_by_key[item["candidate_key"]] = raw_periods
            unresolved_by_key[item["candidate_key"]] = unresolved
        rank5_score = float(original[4]["reranker_score"])
        border: dict[str, bool] = {}
        for item in original[5:10]:
            margin = rank5_score - float(item["reranker_score"])
            border[item["candidate_key"]] = margin >= 0.0 and margin <= threshold
        incumbent = original[4]
        eligible = [
            item
            for item in original[5:10]
            if border.get(item["candidate_key"], False) and status_by_key[item["candidate_key"]] == "FULL_MATCH"
        ]
        eligible.sort(key=lambda item: (-float(item["reranker_score"]), int(item["pre_rerank_rank"]), item["candidate_key"]))
        challenger = eligible[0] if query["status"] in {"explicit_single_period", "explicit_multi_period"} and status_by_key[incumbent["candidate_key"]] == "EXPLICIT_CONFLICT" and eligible else None
        mutated = challenger is not None
        if mutated:
            mutated_cases.add(case_id)
        output = [dict(item) for item in original]
        if challenger is not None:
            challenger_index = next(index for index, item in enumerate(output) if item["candidate_key"] == challenger["candidate_key"])
            output[4], output[challenger_index] = output[challenger_index], output[4]
        for index, item in enumerate(output, start=1):
            item["bpg_rank"] = index
            item["original_qwen_rank"] = int(item["post_rerank_rank"])
            item["period_status"] = status_by_key[item["candidate_key"]]
            item["near_boundary"] = bool(border.get(item["candidate_key"], False))
            item["bpg_role"] = "promoted_challenger" if challenger is not None and item["candidate_key"] == challenger["candidate_key"] else "demoted_incumbent" if challenger is not None and item["candidate_key"] == incumbent["candidate_key"] else "unchanged"
        prediction_rows.append({"case_id": case_id, "candidate_budget": 100, "ranked_candidates": output})
        resolution_records.append({"case_id": case_id, "query_period": query, "candidates": [{"candidate_key": item["candidate_key"], "original_qwen_rank": int(item["post_rerank_rank"]), "period_status": status_by_key[item["candidate_key"]], "raw_periods": raw_periods_by_key[item["candidate_key"]], "metadata_unresolved": unresolved_by_key[item["candidate_key"]]} for item in original]})
        eligibility_records.append({"case_id": case_id, "query_period_status": query["status"], "rank5_candidate": incumbent["candidate_key"], "rank5_period_status": status_by_key[incumbent["candidate_key"]], "eligible_challengers": [{"candidate_key": item["candidate_key"], "original_qwen_rank": int(item["post_rerank_rank"]), "score": item["reranker_score"], "near_boundary": border[item["candidate_key"]], "period_status": status_by_key[item["candidate_key"]]} for item in eligible], "mutated": mutated, "selected_challenger": challenger["candidate_key"] if challenger else None, "max_promotions_per_query": 1})

    prediction_path = out_dir / "predictions.jsonl.gz"
    prediction_sha = write_gzip_jsonl(prediction_path, prediction_rows)
    prediction_manifest = {"gate": "NF-OPT-20-R1", "rows": len(prediction_rows), "candidate_count_per_query": 100, "prediction_sha256": prediction_sha, "gold_reads_during_prediction": 0, "model_execution": False, "retrieval_rerun": False}
    write_json(out_dir / "prediction-manifest.json", prediction_manifest)
    write_json(out_dir / "prediction-seal.json", {**prediction_manifest, "sealed": True})
    write_json(out_dir / "period-resolution-audit.json", {"gold_reads_before_seal": 0, "near_boundary_rule": near_definition["rule"], "near_boundary_threshold": threshold, "records": resolution_records})
    write_json(out_dir / "eligibility-audit.json", {"gold_reads_before_seal": 0, "queries": len(prediction_rows), "queries_mutated": len(mutated_cases), "records": eligibility_records})
    write_json(out_dir / "bpg-v1-contract.json", {"method": "Qwen Top5 Boundary Period Guard V1", "short_name": "BPG-V1", "candidate_source": "frozen_qwen_top100", "scope": "original_qwen_ranks_1_10", "ranking_change": "swap_original_rank5_with_best_rank6_10_eligible_challenger", "query_period_status": ["explicit_single_period", "explicit_multi_period", "no_explicit_period", "unresolved"], "full_match": "all_required_periods_present", "partial_match": "some_but_not_all_required_periods_present", "neutral": "no_resolvable_candidate_period_metadata", "explicit_conflict": "candidate_period_metadata_present_without_required_period_overlap", "near_boundary_rule": near_definition["rule"], "near_boundary_threshold": threshold, "tie_break": ["reranker_score_desc", "original_bm25_rank_asc", "candidate_key_lexical_asc"], "max_promotions_per_query": 1, "rank1_4_never_displaced": True, "feature_seal": "frozen_financial_semantic_graph_and_candidate_metadata", "gold_reads_during_prediction": 0})

    # Post-seal scoring starts here; prediction generation above has no Gold
    # bindings, C1/P1 IDs, or expected-source metadata.
    strict_rows = read_jsonl(strict_path)
    bindings_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in strict_rows:
        bindings_by_case[binding["case_id"]].append(binding)
    bpg_by_case = {row["case_id"]: sorted(row["ranked_candidates"], key=lambda item: item["bpg_rank"]) for row in prediction_rows}
    qwen_metrics = compute_strict_metrics(bindings_by_case, qwen_by_case)
    bpg_metrics = compute_strict_metrics(bindings_by_case, bpg_by_case)
    write_json(out_dir / "strict-metrics.json", {"before_qwen": qwen_metrics, "after_bpg": bpg_metrics, "strict_sources": 80})

    targets = load_semantic_targets(targets_path)
    semantic_before, _ = compute_semantic_metrics(strict_rows, targets, qwen_by_case, case_facts)
    semantic_after, _ = compute_semantic_metrics(strict_rows, targets, bpg_by_case, case_facts)
    write_json(out_dir / "semantic-metrics.json", {"before_qwen": semantic_before, "after_bpg": semantic_after, "semantic_fact_contract": "SE1 exact target intersection; physical hit remains semantic hit"})

    movement = Counter()
    source_rank_diffs: list[dict[str, Any]] = []
    for binding in strict_rows:
        case_id = binding["case_id"]
        key = binding["candidate_key"]
        before_rank = rank_map(qwen_by_case[case_id]).get(key, 101)
        after_rank = rank_map(bpg_by_case[case_id]).get(key, 101)
        before_hit = before_rank <= 5
        after_hit = after_rank <= 5
        if not before_hit and after_hit:
            movement["rescued"] += 1
        elif before_hit and not after_hit:
            movement["damaged"] += 1
        elif before_hit:
            movement["unchanged_hit"] += 1
        else:
            movement["unchanged_miss"] += 1
        source_rank_diffs.append({"case_id": case_id, "source_index": binding.get("source_index", 0), "candidate_key": key, "before_rank": before_rank, "after_rank": after_rank, "rank_delta": before_rank - after_rank})
    movement["net"] = movement["rescued"] - movement["damaged"]
    for key in ("rescued", "damaged", "unchanged_hit", "unchanged_miss"):
        movement.setdefault(key, 0)
    mutation_details = [record for record in eligibility_records if record["mutated"]]
    opportunity_cases = {record["case_id"] for record in mutation_details if bindings_by_case.get(record["case_id"]) and any(rank_map(qwen_by_case[record["case_id"]]).get(binding["candidate_key"], 101) <= 10 for binding in bindings_by_case[record["case_id"]])}
    rescued_cases = {row["case_id"] for row in source_rank_diffs if row["before_rank"] > 5 and row["after_rank"] <= 5}
    write_json(out_dir / "rank-movement.json", {"source_level": {**dict(movement), "denominator": 80}, "queries_mutated": len(mutated_cases), "queries_unchanged": len(qwen_by_case) - len(mutated_cases), "mutation_records": mutation_details, "source_rank_diffs": source_rank_diffs, "mutation_precision": rate(len(rescued_cases & opportunity_cases), len(opportunity_cases)), "mutation_precision_denominator_queries": len(opportunity_cases), "mutated_gold_relevant_opportunities": sorted(opportunity_cases)})

    # Post-seal P1/P2/P4 and wrong-period attribution.
    recoverability = read_json(nf20_root / "recoverability-classes.json")["records"]
    recoverability_by_key = {(row["case_id"], int(row.get("source_index", 0))): row["recoverability_class"] for row in recoverability}
    taxonomy = read_json(nf20_root / "failure-taxonomy.json")["records"]
    wrong_period_keys = {(row["case_id"], int(row.get("source_index", 0))) for row in taxonomy if "wrong_period" in row.get("tags", [])}
    p_records: list[dict[str, Any]] = []
    for binding in strict_rows:
        key = (binding["case_id"], int(binding.get("source_index", 0)))
        if key not in recoverability_by_key:
            continue
        before_hit = rank_map(qwen_by_case[binding["case_id"]]).get(binding["candidate_key"], 101) <= 5
        after_hit = rank_map(bpg_by_case[binding["case_id"]]).get(binding["candidate_key"], 101) <= 5
        p_records.append({"case_id": key[0], "source_index": key[1], "class": recoverability_by_key[key], "wrong_period": key in wrong_period_keys, "before_hit": before_hit, "after_hit": after_hit, "mutated_query": key[0] in mutated_cases, "outcome": "rescued" if not before_hit and after_hit else "damaged" if before_hit and not after_hit else "unchanged_hit" if before_hit else "unchanged_miss"})
    def summarize_class(name: str) -> dict[str, Any]:
        rows = [row for row in p_records if row["class"] == name]
        return {"total": len(rows), "eligible": sum(row["mutated_query"] for row in rows), "rescued": sum(row["outcome"] == "rescued" for row in rows), "damaged": sum(row["outcome"] == "damaged" for row in rows), "unchanged": sum(row["outcome"].startswith("unchanged") for row in rows), "records": rows}
    write_json(out_dir / "p1-p2-p4-analysis.json", {name: summarize_class(name) for name in ("P1", "P2", "P4")})
    wrong_rows = [row for row in p_records if row["wrong_period"]]
    write_json(out_dir / "wrong-period-analysis.json", {"total": len(wrong_rows), "eligible": sum(row["mutated_query"] for row in wrong_rows), "rescued": sum(row["outcome"] == "rescued" for row in wrong_rows), "damaged": sum(row["outcome"] == "damaged" for row in wrong_rows), "unchanged": sum(row["outcome"].startswith("unchanged") for row in wrong_rows), "records": wrong_rows})

    case_ids = sorted(qwen_by_case)
    multi_cases = sorted(case_id for case_id in case_ids if len(bindings_by_case.get(case_id, [])) > 1)
    calc_cases = sorted(case_id for case_id in case_ids if classify_case(case_id, query_plans.get(case_id, {}), len(bindings_by_case.get(case_id, []))) == "calculation")
    category_cases = {"multi_evidence": multi_cases, "calculation": calc_cases, "single_evidence": sorted(set(case_ids) - set(multi_cases))}
    type_output: dict[str, Any] = {}
    for category, ids in category_cases.items():
        category_bindings = {case_id: bindings_by_case.get(case_id, []) for case_id in ids}
        before = compute_strict_metrics(category_bindings, qwen_by_case)
        after = compute_strict_metrics(category_bindings, bpg_by_case)
        type_output[category] = {"query_count": len(ids), "strict_before": before["@5"], "strict_after": after["@5"], "any_at_5_before": sum(bool(set(top_ids(qwen_by_case[case_id], 5)) & {row["candidate_key"] for row in bindings_by_case.get(case_id, [])}) for case_id in ids), "any_at_5_after": sum(bool(set(top_ids(bpg_by_case[case_id], 5)) & {row["candidate_key"] for row in bindings_by_case.get(case_id, [])}) for case_id in ids), "all_at_5_before": coverage_all(ids, bindings_by_case, qwen_by_case, 5), "all_at_5_after": coverage_all(ids, bindings_by_case, bpg_by_case, 5)}
    write_json(out_dir / "question-type-analysis.json", {"calculation_is_subset_of_multi_evidence": True, "categories": type_output, "case_counts": {key: len(value) for key, value in category_cases.items()}})

    calc_coverage_path = nf19_root / "calculation-slot-coverage.json"
    calc_coverage_rows = read_json(calc_coverage_path)["cases"]
    calc_before_5, calc_before_records = calculation_coverage(calc_coverage_rows, qwen_by_case, case_facts, 5)
    calc_after_5, calc_after_records = calculation_coverage(calc_coverage_rows, bpg_by_case, case_facts, 5)
    calc_before_10, _ = calculation_coverage(calc_coverage_rows, qwen_by_case, case_facts, 10)
    calc_after_10, _ = calculation_coverage(calc_coverage_rows, bpg_by_case, case_facts, 10)
    write_json(out_dir / "calculation-slot-analysis.json", {"denominator": 11, "before": {"@5": calc_before_5, "@10": calc_before_10}, "after": {"@5": calc_after_5, "@10": calc_after_10}, "records_before_5": calc_before_records, "records_after_5": calc_after_records})

    # Hard safety checks after scoring; these only inspect the sealed
    # predictions and never alter them.
    top10_invariant = all(set(top_ids(qwen_by_case[case_id], 10)) == set(top_ids(bpg_by_case[case_id], 10)) for case_id in case_ids)
    top100_invariant = all(set(top_ids(qwen_by_case[case_id], 100)) == set(top_ids(bpg_by_case[case_id], 100)) for case_id in case_ids)
    top5_order_mutation_only = all(
        [item["candidate_key"] for item in bpg_by_case[case_id]][:4] == [item["candidate_key"] for item in qwen_by_case[case_id]][:4]
        and set(top_ids(qwen_by_case[case_id], 5)) | {item["candidate_key"] for item in bpg_by_case[case_id][5:10] if item.get("bpg_role") == "promoted_challenger"} >= set(top_ids(bpg_by_case[case_id], 5))
        for case_id in case_ids
    )
    semantic_regression = semantic_after["@5"]["hits"] < semantic_before["@5"]["hits"]
    multi_regression = type_output["multi_evidence"]["all_at_5_after"] < 4
    calc_regression = calc_after_5 < 5
    gain = bpg_metrics["@5"]["hits"] - 43
    damage = movement["damaged"]
    rescue = movement["rescued"]
    explicit_period_queries = sum(record["query_period_status"] in {"explicit_single_period", "explicit_multi_period"} for record in eligibility_records)
    rank5_conflict_queries = sum(
        record["query_period_status"] in {"explicit_single_period", "explicit_multi_period"}
        and record["rank5_period_status"] == "EXPLICIT_CONFLICT"
        for record in eligibility_records
    )
    eligible_challenger_queries = sum(
        record["query_period_status"] in {"explicit_single_period", "explicit_multi_period"}
        and record["rank5_period_status"] == "EXPLICIT_CONFLICT"
        and bool(record["eligible_challengers"])
        for record in eligibility_records
    )
    if gain >= 5 and damage <= 2 and not semantic_regression and not multi_regression and not calc_regression:
        effectiveness = True
        next_gate = "nf_opt_20_r2_independent_holdout_confirmation_design"
    elif 2 <= gain <= 4 and damage <= rescue and not semantic_regression and not multi_regression and not calc_regression:
        effectiveness = "marginal"
        next_gate = "nf_opt_20_r2_independent_holdout_confirmation_design"
    else:
        effectiveness = False
        next_gate = "nf_opt_20_method_reconsideration"
    input_integrity = {"qwen_prediction_sha256": qwen_sha, "qwen_prediction_sha_expected": expected_qwen_sha, "qwen_scores_unchanged": qwen_sha == expected_qwen_sha, "bounded_top100_sha256": bounded_sha, "bounded_top100_sha_expected": bounded_seal.get("prediction_sha256"), "bounded_qwen_identity_mismatch": bounded_identity_mismatch, "candidate_identity_mismatch": int(not top100_invariant) or bounded_identity_mismatch, "top10_candidate_set_invariant": top10_invariant, "top100_candidate_set_invariant": top100_invariant, "ranking_mutation": bool(mutated_cases)}
    write_json(out_dir / "input-integrity.json", input_integrity)
    decision = {"gate": "NF-OPT-20-R1", "base_commit": "f70e4dd2389d1da8a0fa41831f85f6917b55ad0c", "evaluation_role": "development_shadow_calibration", "fresh_blind_evaluation": False, "model_execution": False, "retrieval_rerun": False, "strict_sources": 80, "baseline_strict_recall_at_5_hits": 43, "bpg_strict_recall_at_5_hits": bpg_metrics["@5"]["hits"], "rescued": rescue, "damaged": damage, "net_gain": movement["net"], "queries_explicit_period": explicit_period_queries, "queries_rank5_explicit_conflict": rank5_conflict_queries, "queries_with_eligible_challenger": eligible_challenger_queries, "queries_mutated": len(mutated_cases), "queries_unchanged": len(qwen_by_case) - len(mutated_cases), "p1_cases": 10, "p1_rescued": summarize_class("P1")["rescued"], "p1_eligible": summarize_class("P1")["eligible"], "p1_damaged": summarize_class("P1")["damaged"], "p1_unchanged": summarize_class("P1")["unchanged"], "semantic_recall_at_5_before": semantic_before["@5"]["hits"], "semantic_recall_at_5_after": semantic_after["@5"]["hits"], "semantic_recall_at_10_before": semantic_before["@10"]["hits"], "semantic_recall_at_10_after": semantic_after["@10"]["hits"], "multi_evidence_all_at_5_before": 4, "multi_evidence_all_at_5_after": type_output["multi_evidence"]["all_at_5_after"], "multi_evidence_all_at_10_before": 9, "multi_evidence_all_at_10_after": coverage_all(multi_cases, bindings_by_case, bpg_by_case, 10), "calculation_all_slots_at_5_before": calc_before_5, "calculation_all_slots_at_5_after": calc_after_5, "calculation_all_slots_at_10_before": calc_before_10, "calculation_all_slots_at_10_after": calc_after_10, "wrong_period_total": len(wrong_rows), "wrong_period_eligible": sum(row["mutated_query"] for row in wrong_rows), "wrong_period_rescued": sum(row["outcome"] == "rescued" for row in wrong_rows), "wrong_period_damaged": sum(row["outcome"] == "damaged" for row in wrong_rows), "secondary_discriminator_materially_effective": effectiveness, "development_shadow_result": True, "production_switch_allowed": False, "hard_safety": {"top10_candidate_set_invariant": top10_invariant, "top100_candidate_set_invariant": top100_invariant, "rank1_4_unchanged": top5_order_mutation_only, "semantic_regression": semantic_regression, "multi_evidence_regression": multi_regression, "calculation_regression": calc_regression, "prediction_sealed_before_gold": True}, "next_gate": next_gate}
    write_json(out_dir / "decision.json", decision)
    write_text(out_dir / "README.md", """# NF-OPT-20 R1 — Frozen Period-Consistency Secondary Discriminator Shadow Test

Development-shadow, post-seal diagnostic only. BPG-V1 consumes frozen Qwen
Top100 ranking and Gold-independent period metadata, makes at most one rank-5
boundary swap, and never executes a model, retrieval, Binder, Calculator, or
production change. Gold is loaded only after predictions are sealed.
""")
    print(json.dumps({"artifact_dir": str(out_dir), "decision": decision, "prediction_sha256": prediction_sha}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
