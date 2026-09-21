"""NF-OPT-20 R0 — strict-source pointwise discrimination audit.

The audit consumes the sealed R3.3 Qwen ranking and frozen source/semantic
metadata.  It never executes a model, retrieval, reranker, Binder, or
Calculator and never mutates the frozen rank.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


OUT_NAME = "nf-opt-20-r0-pointwise-discrimination-audit"
EXPECTED_C1 = 25
EXPECTED_NEAR = 13
EXPECTED_CLEAR = 12


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for row in rows:
                gz.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return sha256_file(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def tokens(value: Any) -> set[str]:
    return set(norm(value).split())


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def parse_document_view(document_view: str) -> dict[str, Any]:
    structure: dict[str, str] = {}
    lines = str(document_view or "").splitlines()
    in_structure = False
    source_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "[STRUCTURE]":
            in_structure = True
            continue
        if stripped == "[CONTENT]":
            in_structure = False
            continue
        match = re.match(r"^([A-Za-z][A-Za-z ]*):\s*(.*)$", stripped)
        if match:
            key = norm(match.group(1)).replace(" ", "_")
            value = match.group(2).strip()
            if in_structure or key in {"document", "page", "block_type", "section", "source"}:
                structure.setdefault(key, value)
        if stripped and (stripped.startswith("Source:") or source_lines):
            source_lines.append(stripped)
    document = structure.get("document", "")
    page = structure.get("page")
    pages = {str(page)} if page else set()
    if not pages:
        pages.update(re.findall(r"\bPage:\s*(\d+)\b", document_view or "", re.IGNORECASE))
    return {
        "document_id": document,
        "page": page,
        "pages": sorted(pages),
        "statement": structure.get("statement", ""),
        "table": structure.get("table", ""),
        "section": structure.get("section", ""),
        "metric_path": structure.get("metric_path", ""),
        "row": structure.get("row", ""),
        "period": structure.get("period", ""),
        "column_header": structure.get("column_header", ""),
        "scale": structure.get("scale", ""),
        "currency": structure.get("currency", ""),
        "evidence_type": structure.get("type", ""),
        "block_type": structure.get("block_type", ""),
        "source_preview": " ".join(source_lines)[:2000],
        "raw_evidence": document_view,
    }


def fact_value_sets(facts: list[dict[str, Any]]) -> dict[str, set[str]]:
    metrics = {norm(f.get("normalized_metric") or f.get("metric")) for f in facts if f.get("normalized_metric") or f.get("metric")}
    periods = {norm(f.get("normalized_period") or f.get("period")) for f in facts if f.get("normalized_period") or f.get("period")}
    scales = {str(f.get("normalized_scale") or f.get("scale")) for f in facts if f.get("normalized_scale") or f.get("scale")}
    currencies = {norm(f.get("normalized_currency") or f.get("currency")) for f in facts if f.get("normalized_currency") or f.get("currency")}
    values = {str(f.get("normalized_base_value") or f.get("base_value")) for f in facts if f.get("normalized_base_value") or f.get("base_value")}
    tables: set[str] = set()
    rows: set[str] = set()
    pages: set[str] = set()
    evidence_ids: set[str] = set()
    for fact in facts:
        if fact.get("logical_table_id"):
            tables.add(str(fact["logical_table_id"]))
        for provenance in fact.get("physical_provenance", []) or []:
            if provenance.get("logical_table_id"):
                tables.add(str(provenance["logical_table_id"]))
            if provenance.get("row_id"):
                rows.add(str(provenance["row_id"]))
            if provenance.get("pdf_page") is not None:
                pages.add(str(provenance["pdf_page"]))
            if provenance.get("authoritative_evidence_id"):
                evidence_ids.add(str(provenance["authoritative_evidence_id"]))
    return {"metrics": metrics, "periods": periods, "scales": scales, "currencies": currencies, "values": values, "tables": tables, "rows": rows, "pages": pages, "evidence_ids": evidence_ids}


def set_relation(left: set[str], right: set[str]) -> dict[str, Any]:
    return {
        "overlap": sorted(left & right),
        "equal": bool(left or right) and left == right,
        "left_only": sorted(left - right),
        "right_only": sorted(right - left),
        "left_count": len(left),
        "right_count": len(right),
    }


def parse_query_plan_rows(payload: Any) -> dict[str, dict[str, Any]]:
    rows = payload.get("predictions") if isinstance(payload, dict) else payload
    if rows is None and isinstance(payload, dict):
        rows = payload.get("cases")
    if rows is None and isinstance(payload, dict):
        # Gate07 publishes the frozen plans under ``plans``.  Keep the
        # parser tolerant of the other historical envelope names without
        # inventing any taxonomy metadata.
        rows = payload.get("plans", [])
    return {row["case_id"]: row.get("plan", {}) for row in rows or []}


def candidate_snapshot(
    case_id: str,
    candidate_key: str,
    candidate_meta: dict[str, dict[str, dict[str, Any]]],
    case_facts: dict[str, dict[str, set[str]]],
    fact_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    item = candidate_meta.get(case_id, {}).get(candidate_key, {})
    parsed = parse_document_view(item.get("document_view", ""))
    fact_ids = sorted(case_facts.get(case_id, {}).get(candidate_key, set()))
    facts = [fact_meta[fact_id] for fact_id in fact_ids if fact_id in fact_meta]
    semantic = fact_value_sets(facts)
    for key in ("document_id", "page"):
        if parsed.get(key):
            continue
        if key == "document_id":
            parsed[key] = str(item.get("document_id") or "")
    if parsed.get("page"):
        semantic["pages"].add(str(parsed["page"]))
    return {
        "candidate_key": candidate_key,
        "context_status": item.get("context_status"),
        "candidate_granularity": parsed.get("evidence_type") or item.get("context_status"),
        "document_id": parsed.get("document_id", ""),
        "page": parsed.get("page"),
        "section": parsed.get("section", ""),
        "statement": parsed.get("statement", ""),
        "table": parsed.get("table", ""),
        "metric_path": parsed.get("metric_path", ""),
        "row": parsed.get("row", ""),
        "column_header": parsed.get("column_header", ""),
        "period_header": parsed.get("period", ""),
        "scale": parsed.get("scale", ""),
        "currency": parsed.get("currency", ""),
        "block_type": parsed.get("block_type", ""),
        "semantic_fact_ids": fact_ids,
        "semantic_metrics": sorted(semantic["metrics"]),
        "semantic_periods": sorted(semantic["periods"]),
        "semantic_values": sorted(semantic["values"]),
        "semantic_scales": sorted(semantic["scales"]),
        "semantic_currencies": sorted(semantic["currencies"]),
        "logical_table_ids": sorted(semantic["tables"]),
        "row_ids": sorted(semantic["rows"]),
        "physical_pages": sorted(semantic["pages"]),
        "evidence_ids": sorted(semantic["evidence_ids"]),
        "raw_evidence": parsed.get("raw_evidence", ""),
        "raw_source_preview": parsed.get("source_preview", ""),
    }


def question_features(plan: dict[str, Any], question: str) -> dict[str, Any]:
    periods = {norm(value) for value in plan.get("periods", []) if value}
    metrics = {norm(value) for value in plan.get("metric_phrases", []) if value}
    if not metrics:
        metrics = {norm(value) for value in plan.get("raw_metric_phrases", []) if value}
    return {
        "question": question,
        "query_periods": sorted(periods),
        "query_metrics": sorted(metrics),
        "query_document_scope": sorted(str(value) for value in plan.get("document_scope", []) if value),
        "operation": plan.get("operation"),
        "requires_multiple_sources": bool(plan.get("requires_multiple_sources")),
        "requires_multiple_periods": len(periods) > 1,
        "query_tokens": sorted(tokens(question)),
    }


def overlap_ratio(query_tokens: set[str], candidate_text: str) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & tokens(candidate_text)) / len(query_tokens)


def feature_diff(question: dict[str, Any], gold: dict[str, Any], competitor: dict[str, Any]) -> dict[str, Any]:
    q_periods = set(question["query_periods"])
    q_metrics = set(question["query_metrics"])
    gold_metrics = set(gold["semantic_metrics"]) | tokens(gold.get("metric_path")) | tokens(gold.get("row"))
    comp_metrics = set(competitor["semantic_metrics"]) | tokens(competitor.get("metric_path")) | tokens(competitor.get("row"))
    gold_periods = set(gold["semantic_periods"]) | tokens(gold.get("period_header")) | tokens(gold.get("period_header"))
    comp_periods = set(competitor["semantic_periods"]) | tokens(competitor.get("period_header"))
    return {
        "document_identity": {"equal": bool(gold["document_id"] and gold["document_id"] == competitor["document_id"]), "gold": gold["document_id"], "competitor": competitor["document_id"]},
        "page": {"equal": bool(gold["page"] and competitor["page"] and str(gold["page"]) == str(competitor["page"])), "gold": gold["page"], "competitor": competitor["page"]},
        "statement": set_relation({norm(gold.get("statement"))} - {""}, {norm(competitor.get("statement"))} - {""}),
        "table": set_relation({norm(gold.get("table"))} - {""}, {norm(competitor.get("table"))} - {""}),
        "metric_path": set_relation(tokens(gold.get("metric_path")), tokens(competitor.get("metric_path"))),
        "row_label": set_relation(tokens(gold.get("row")), tokens(competitor.get("row"))),
        "column_header": set_relation(tokens(gold.get("column_header")), tokens(competitor.get("column_header"))),
        "period": set_relation(gold_periods, comp_periods),
        "scale": set_relation(set(gold["semantic_scales"]), set(competitor["semantic_scales"])),
        "currency": set_relation(set(gold["semantic_currencies"]), set(competitor["semantic_currencies"])),
        "logical_table": set_relation(set(gold["logical_table_ids"]), set(competitor["logical_table_ids"])),
        "candidate_granularity": {"gold": gold.get("candidate_granularity"), "competitor": competitor.get("candidate_granularity"), "equal": gold.get("candidate_granularity") == competitor.get("candidate_granularity")},
        "raw_text_overlap": {"gold": overlap_ratio(set(question["query_tokens"]), gold.get("raw_evidence", "")), "competitor": overlap_ratio(set(question["query_tokens"]), competitor.get("raw_evidence", ""))},
        "normalized_metric_overlap": {"gold": sorted(q_metrics & gold_metrics), "competitor": sorted(q_metrics & comp_metrics)},
        "period_overlap": {"gold": sorted(q_periods & gold_periods), "competitor": sorted(q_periods & comp_periods)},
        "entity_overlap": {"gold": sorted(set(question["query_document_scope"]) & {gold["document_id"]}), "competitor": sorted(set(question["query_document_scope"]) & {competitor["document_id"]})},
    }


def derive_tags(question: dict[str, Any], gold: dict[str, Any], competitor: dict[str, Any], diff: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    q_periods = set(question["query_periods"])
    gold_periods = set(gold["semantic_periods"]) | tokens(gold.get("period_header"))
    comp_periods = set(competitor["semantic_periods"]) | tokens(competitor.get("period_header"))
    gold_metrics = set(gold["semantic_metrics"]) | tokens(gold.get("metric_path")) | tokens(gold.get("row"))
    comp_metrics = set(competitor["semantic_metrics"]) | tokens(competitor.get("metric_path")) | tokens(competitor.get("row"))
    same_metric = bool(gold_metrics & comp_metrics)
    if q_periods and (q_periods & gold_periods) and not (q_periods & comp_periods):
        tags.append("wrong_period")
    if same_metric and gold.get("statement") and competitor.get("statement") and norm(gold["statement"]) != norm(competitor["statement"]):
        tags.append("same_metric_wrong_statement")
    if same_metric and gold.get("row") and competitor.get("row") and norm(gold["row"]) != norm(competitor["row"]):
        tags.append("same_metric_wrong_row")
    if same_metric and gold.get("table") and competitor.get("table") and norm(gold["table"]) != norm(competitor["table"]):
        tags.append("same_metric_wrong_table")
    if q_periods and (q_periods & gold_periods) and (q_periods & comp_periods) and not same_metric:
        tags.append("same_period_wrong_metric")
    if question["requires_multiple_sources"]:
        tags.append("multi_slot_competition")
    if question.get("operation") and len(question.get("query_periods", [])) > 1:
        tags.append("calculation_operand_competition")
    specific_relation = {"wrong_period", "same_metric_wrong_statement", "same_metric_wrong_row", "same_metric_wrong_table", "same_period_wrong_metric", "strict_source_sibling_confusion", "numerically_similar_wrong_fact"}
    if gold["document_id"] and gold["document_id"] == competitor["document_id"] and (gold.get("page") != competitor.get("page") or gold.get("raw_evidence") != competitor.get("raw_evidence")) and not (specific_relation & set(tags)):
        tags.append("same_document_wrong_evidence")
    if set(gold["semantic_fact_ids"]) & set(competitor["semantic_fact_ids"]):
        tags.append("strict_source_sibling_confusion")
    if set(gold["semantic_values"]) & set(competitor["semantic_values"]) and not (set(gold["semantic_fact_ids"]) & set(competitor["semantic_fact_ids"])):
        tags.append("numerically_similar_wrong_fact")
    if not gold_metrics and not q_periods:
        tags.append("query_underspecified")
    if not tags:
        tags.append("generic_financial_similarity" if diff["raw_text_overlap"]["competitor"] > 0 else "other")
    return sorted(set(tags))


def classify_recoverability(cohort: str, tags: list[str], gold: dict[str, Any], competitor: dict[str, Any], diff: dict[str, Any]) -> str:
    discriminator_tags = {"wrong_period", "same_metric_wrong_statement", "same_metric_wrong_row", "same_metric_wrong_table", "same_period_wrong_metric", "cross_statement_confusion"}
    hard_difference = any(
        not diff[key].get("equal", False)
        for key in ("document_identity", "statement", "table", "metric_path", "row_label", "period", "scale", "currency")
        if isinstance(diff.get(key), dict)
    )
    sibling = "strict_source_sibling_confusion" in tags
    if cohort == "near_boundary" and bool(discriminator_tags & set(tags)):
        return "P1"
    if sibling and not bool(discriminator_tags & set(tags)) and not hard_difference:
        return "P3"
    if cohort == "clear_loss" and hard_difference:
        return "P2"
    if sibling and not hard_difference:
        return "P3"
    return "P4"


def main() -> int:
    backend_root = Path(__file__).resolve().parents[2]
    evaluation_root = backend_root / "artifacts" / "evaluation"
    out_dir = evaluation_root / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    top100_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r2a-2" / "bounded-top100-predictions.jsonl.gz"
    top100_seal_path = top100_path.parent / "prediction-seal.json"
    qwen_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3" / "main_rerank_predictions.jsonl.gz"
    qwen_seal_path = qwen_path.parent / "prediction-seal.json"
    views_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3-p0" / "queryplan-rerank-input-views.jsonl.gz"
    registry_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1-p0" / "candidate-semantic-fact-registry.jsonl.gz"
    strict_path = evaluation_root / "pdf-retrieval-v4-strict-source-contract" / "strict-gold-source-bindings.jsonl"
    query_plan_path = evaluation_root / "pdf-retrieval-v4-gate-07" / "query-plan-predictions.json"
    nf19_root = evaluation_root / "nf-opt-19-r0-setwise-ranking-audit"
    t2_root = evaluation_root / "t2-ragbench-05-qwen3-strong-reranker"

    top100_rows = read_gzip_jsonl(top100_path)
    qwen_rows = read_gzip_jsonl(qwen_path)
    view_rows = read_gzip_jsonl(views_path)
    strict_rows = read_jsonl(strict_path)
    query_plans = parse_query_plan_rows(read_json(query_plan_path))
    top100_by_case = {row["case_id"]: row["candidates"] for row in top100_rows}
    qwen_by_case = {row["case_id"]: sorted(row["ranked_candidates"], key=lambda item: item["post_rerank_rank"]) for row in qwen_rows}
    qwen_rank = {case_id: {item["candidate_key"]: int(item["post_rerank_rank"]) for item in items} for case_id, items in qwen_by_case.items()}
    qwen_item = {case_id: {item["candidate_key"]: item for item in items} for case_id, items in qwen_by_case.items()}

    registry_rows = read_gzip_jsonl(registry_path)
    case_facts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    fact_meta: dict[str, dict[str, Any]] = {}
    for record in registry_rows:
        ids = set(record.get("semantic_fact_ids", []))
        for fact in record.get("semantic_facts", []) or []:
            fact_id = fact.get("semantic_fact_id")
            if fact_id:
                ids.add(fact_id)
                fact_meta.setdefault(fact_id, fact)
        for occurrence in record.get("occurrences", []) or []:
            case_facts[occurrence["case_id"]][record["candidate_key"]].update(ids)

    candidate_meta: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in view_rows:
        for candidate in row.get("candidates", []) or []:
            candidate_meta[row["case_id"]][candidate["candidate_key"]] = candidate

    # Reuse the sealed C1/near-boundary definition; do not re-search a margin.
    margin_rows = read_json(nf19_root / "score-margin-analysis.json")["records"]
    c1_records = [row for row in margin_rows if row["cohort"] in {"near_boundary", "clear_loss"}]
    if len(c1_records) != EXPECTED_C1 or sum(row["cohort"] == "near_boundary" for row in c1_records) != EXPECTED_NEAR or sum(row["cohort"] == "clear_loss" for row in c1_records) != EXPECTED_CLEAR:
        raise RuntimeError("frozen C1 cohort mismatch")

    strict_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in strict_rows:
        strict_by_key[(row["case_id"], int(row.get("source_index", 0)), row["candidate_key"])] = row
    # Independently reconstruct the frozen strict source cohorts from the
    # physical bindings and the sealed Qwen Top100 rank.  The NF-OPT-19
    # margin cohort is reused for near/clear analysis, but it is not the sole
    # source of the C0/C1/C2 identity check.
    strict_cohorts: Counter[str] = Counter()
    recomputed_c1_keys: set[tuple[str, int, str]] = set()
    for row in strict_rows:
        key = (row["case_id"], int(row.get("source_index", 0)), row["candidate_key"])
        rank = qwen_rank.get(row["case_id"], {}).get(row["candidate_key"])
        if rank is None:
            strict_cohorts["C2"] += 1
        elif int(rank) <= 5:
            strict_cohorts["C0"] += 1
        else:
            strict_cohorts["C1"] += 1
            recomputed_c1_keys.add(key)
    if dict(strict_cohorts) != {"C0": 43, "C1": 25, "C2": 12}:
        raise RuntimeError(f"strict C0/C1/C2 mismatch: {dict(strict_cohorts)}")
    margin_c1_keys = {(row["case_id"], int(row.get("source_index", 0)), row["candidate_key"]) for row in c1_records}
    if margin_c1_keys != recomputed_c1_keys:
        raise RuntimeError("NF-OPT-19 C1 margin rows do not match recomputed strict C1 identity")

    pairs: list[dict[str, Any]] = []
    taxonomy_counts: dict[str, Counter[str]] = {"overall": Counter(), "near_boundary": Counter(), "clear_loss": Counter()}
    recoverability_counts = Counter()
    movement = Counter()
    review_package: list[dict[str, Any]] = []

    for margin in c1_records:
        case_id = margin["case_id"]
        gold_key = margin["candidate_key"]
        source_index = int(margin.get("source_index", 0))
        binding = strict_by_key.get((case_id, source_index, gold_key), {})
        plan = query_plans.get(case_id, {})
        question_text = str(plan.get("raw_question") or "")
        q = question_features(plan, question_text)
        ranked = qwen_by_case[case_id]
        cutoff_item = ranked[4]
        highest_wrong = next(item for item in ranked if item["candidate_key"] != gold_key)
        competitor_items = [cutoff_item] if cutoff_item["candidate_key"] == highest_wrong["candidate_key"] else [cutoff_item, highest_wrong]
        gold_snapshot = candidate_snapshot(case_id, gold_key, candidate_meta, case_facts, fact_meta)
        pair_records: list[dict[str, Any]] = []
        primary_tags: list[str] = []
        primary_recoverability = "P4"
        for competitor_item in competitor_items:
            competitor_key = competitor_item["candidate_key"]
            competitor_snapshot = candidate_snapshot(case_id, competitor_key, candidate_meta, case_facts, fact_meta)
            diff = feature_diff(q, gold_snapshot, competitor_snapshot)
            tags = derive_tags(q, gold_snapshot, competitor_snapshot, diff)
            recoverability = classify_recoverability(margin["cohort"], tags, gold_snapshot, competitor_snapshot, diff)
            primary_tags.extend(tags)
            if recoverability == "P1" or primary_recoverability == "P4":
                primary_recoverability = recoverability
            gold_bm25 = int(qwen_item[case_id][gold_key].get("pre_rerank_rank", 10**9))
            competitor_bm25 = int(competitor_item.get("pre_rerank_rank", 10**9))
            gold_qwen = int(qwen_item[case_id][gold_key]["post_rerank_rank"])
            competitor_qwen = int(competitor_item["post_rerank_rank"])
            bm25_correct = gold_bm25 < competitor_bm25
            qwen_correct = gold_qwen < competitor_qwen
            pair_records.append({
                "competitor_role": "rank5_cutoff" if competitor_key == cutoff_item["candidate_key"] else "highest_ranked_wrong",
                "case_id": case_id,
                "source_index": source_index,
                "question": question_text,
                "gold_candidate_id": gold_key,
                "gold_rank": gold_qwen,
                "gold_score": margin["gold_score"],
                "gold_bm25_rank": gold_bm25,
                "competitor_candidate_id": competitor_key,
                "competitor_rank": competitor_qwen,
                "competitor_score": finite_float(competitor_item.get("reranker_score")),
                "competitor_bm25_rank": competitor_bm25,
                "score_margin": float(competitor_item.get("reranker_score", 0.0)) - float(margin["gold_score"]),
                "bm25_ordering": "gold_above_competitor" if bm25_correct else "gold_below_competitor",
                "qwen_ordering": "gold_above_competitor" if qwen_correct else "gold_below_competitor",
                "gold_snapshot": gold_snapshot,
                "competitor_snapshot": competitor_snapshot,
                "feature_diff": diff,
                "diagnostic_tags": tags,
                "recoverability_class": recoverability,
            })
        # Movement is a case-level audit, not a count of the two diagnostic
        # competitor pairs.  Use the highest-ranked wrong candidate as the
        # deterministic pointwise comparator so the denominator remains the
        # 25 C1 cases requested by the contract.
        primary_competitor = highest_wrong
        gold_bm25 = int(qwen_item[case_id][gold_key].get("pre_rerank_rank", 10**9))
        competitor_bm25 = int(primary_competitor.get("pre_rerank_rank", 10**9))
        gold_qwen = int(qwen_item[case_id][gold_key]["post_rerank_rank"])
        competitor_qwen = int(primary_competitor["post_rerank_rank"])
        bm25_correct = gold_bm25 < competitor_bm25
        qwen_correct = gold_qwen < competitor_qwen
        movement["bm25_correct_qwen_wrong"] += int(bm25_correct and not qwen_correct)
        movement["bm25_wrong_qwen_wrong"] += int(not bm25_correct and not qwen_correct)
        movement["bm25_wrong_qwen_correct"] += int(not bm25_correct and qwen_correct)
        movement["bm25_correct_qwen_correct"] += int(bm25_correct and qwen_correct)
        primary_tags = sorted(set(primary_tags))
        taxonomy_counts["overall"].update(primary_tags)
        taxonomy_counts[margin["cohort"]].update(primary_tags)
        recoverability_counts[primary_recoverability] += 1
        gold_review = dict(gold_snapshot)
        competitor_review = dict(candidate_snapshot(case_id, highest_wrong["candidate_key"], candidate_meta, case_facts, fact_meta))
        review_package.append({
            "case_id": case_id,
            "source_index": source_index,
            "question": question_text,
            "cohort": margin["cohort"],
            "gold": {"candidate_key": gold_key, "qwen_rank": margin["qwen_rank"], "qwen_score": margin["gold_score"], "bm25_rank": qwen_item[case_id][gold_key].get("pre_rerank_rank"), "physical_binding": binding, "evidence": gold_review},
            "wrong_competitor": {"candidate_key": highest_wrong["candidate_key"], "qwen_rank": highest_wrong["post_rerank_rank"], "qwen_score": highest_wrong.get("reranker_score"), "bm25_rank": highest_wrong.get("pre_rerank_rank"), "evidence": competitor_review},
            "diff": pair_records[0]["feature_diff"],
            "diagnostic_tags": primary_tags,
            "review_status": "diagnostic_only",
        })
        pairs.append({"case_id": case_id, "source_index": source_index, "question": question_text, "cohort": margin["cohort"], "score_margin_to_rank5_cutoff": margin["score_margin"], "gold_candidate": gold_key, "gold_rank": margin["qwen_rank"], "gold_score": margin["gold_score"], "competitors": pair_records, "diagnostic_tags": primary_tags, "recoverability_class": primary_recoverability})

    pair_sha = write_gzip_jsonl(out_dir / "c1-pairs.jsonl.gz", pairs)
    write_json(out_dir / "human-review-package.json", {"review_status": "diagnostic_only", "c1_total": len(review_package), "records": review_package})

    def cohort_analysis(cohort: str) -> dict[str, Any]:
        rows = [row for row in pairs if row["cohort"] == cohort]
        return {"count": len(rows), "taxonomy_counts": dict(taxonomy_counts[cohort]), "recoverability_counts": dict(Counter(row["recoverability_class"] for row in rows)), "mean_score_margin_to_rank5_cutoff": sum(row["score_margin_to_rank5_cutoff"] for row in rows) / len(rows) if rows else None, "records": rows}

    write_json(out_dir / "failure-taxonomy.json", {"overall": dict(taxonomy_counts["overall"]), "near_boundary": dict(taxonomy_counts["near_boundary"]), "clear_loss": dict(taxonomy_counts["clear_loss"]), "records": [{"case_id": row["case_id"], "source_index": row["source_index"], "cohort": row["cohort"], "tags": row["diagnostic_tags"]} for row in pairs]})
    write_json(out_dir / "near-boundary-analysis.json", cohort_analysis("near_boundary"))
    write_json(out_dir / "clear-loss-analysis.json", cohort_analysis("clear_loss"))
    write_json(out_dir / "bm25-vs-qwen-movement.json", {"overall": dict(movement), "denominator": len(pairs), "bm25_correct_qwen_wrong_rate": rate(movement["bm25_correct_qwen_wrong"], len(pairs))})
    write_json(out_dir / "gold-vs-competitor-feature-diff.json", {"pair_count": len(pairs), "pairs": pairs})

    # Source-category comparison is deliberately based on frozen QueryPlan
    # metadata, not on the C1 outcome.
    question_type_records: list[dict[str, Any]] = []
    for row in pairs:
        plan = query_plans.get(row["case_id"], {})
        is_calc = bool(plan.get("operation")) and len(plan.get("operand_slots", []) or []) > 1
        is_multi = bool(plan.get("requires_multiple_sources")) or len(plan.get("operand_slots", []) or []) > 1
        category = "calculation" if is_calc else "multi_evidence" if is_multi else "single_evidence"
        question_type_records.append({"case_id": row["case_id"], "source_index": row["source_index"], "category": category, "cohort": row["cohort"], "recoverability_class": row["recoverability_class"], "tags": row["diagnostic_tags"]})
    type_summary: dict[str, Any] = {}
    for category in ("single_evidence", "multi_evidence", "calculation"):
        subset = [row for row in question_type_records if row["category"] == category]
        type_summary[category] = {"c1_count": len(subset), "near_boundary": sum(row["cohort"] == "near_boundary" for row in subset), "clear_loss": sum(row["cohort"] == "clear_loss" for row in subset), "near_boundary_rate": rate(sum(row["cohort"] == "near_boundary" for row in subset), len(subset)), "taxonomy_counts": dict(Counter(tag for row in subset for tag in row["tags"])), "recoverability_counts": dict(Counter(row["recoverability_class"] for row in subset))}
    write_json(out_dir / "question-type-analysis.json", {"categories": type_summary, "records": question_type_records})

    # Aggregate, already-sealed T2 contrast.  No T2 row is imported into the
    # internal source identity or used to alter any label.
    t2_period = read_json(t2_root / "period-cohort.json")
    t2_types = read_json(t2_root / "query-type-analysis.json")
    t2_subset = read_json(t2_root / "subset-analysis.json")
    contrast: dict[str, Any] = {"source": "sealed_t2_05_aggregate_artifacts", "period_cohort": {}, "query_types": {}, "subsets": {}}
    for key in ("period_required", "no_period_requirement"):
        item = t2_period[key]
        contrast["period_cohort"][key] = {"query_count": item["query_count"], "bm25_r_at_5_pct": item["bm25"]["recall_pct"]["@5"], "qwen_r_at_5_pct": item["qwen"]["recall_pct"]["@5"], "gain_pp": item["qwen"]["recall_pct"]["@5"] - item["bm25"]["recall_pct"]["@5"], "rescued_at_5": item["movement_vs_bm25"]["rescued_at_5"], "damaged_at_5": item["movement_vs_bm25"]["damaged_at_5"]}
    for key in ("percentage", "percentage_change", "difference", "ratio"):
        if key in t2_types:
            item = t2_types[key]
            contrast["query_types"][key] = {"query_count": item["query_count"], "bm25_r_at_5_pct": item["bm25_r_at_5_pct"], "qwen_r_at_5_pct": item["qwen_r_at_5_pct"], "gain_pp": item["qwen_r_at_5_pct"] - item["bm25_r_at_5_pct"]}
    for key in ("FinQA", "TAT-DQA"):
        item = t2_subset[key]
        contrast["subsets"][key] = {"query_count": item["query_count"], "bm25_r_at_5_pct": item["bm25"]["recall_pct"]["@5"], "qwen_r_at_5_pct": item["qwen"]["recall_pct"]["@5"], "gain_pp": item["qwen"]["recall_pct"]["@5"] - item["bm25"]["recall_pct"]["@5"]}
    write_json(out_dir / "t2-contrast-analysis.json", contrast)

    # Deterministic P1/P2/P3/P4 contract.  P1 requires the frozen near-boundary
    # cohort plus a hard candidate-independent structural discriminator; P3
    # requires sibling/indistinguishable evidence; clear-loss with a hard
    # distinction is P2; remaining cases are P4.
    p1 = recoverability_counts["P1"]
    p2 = recoverability_counts["P2"]
    p3 = recoverability_counts["P3"]
    p4 = recoverability_counts["P4"]
    if p1 >= 8:
        route_supported = True
        semantic_error_dominant = False
        granularity_gap = False
        next_gate = "nf_opt_20_r1_frozen_secondary_discriminator_shadow_test"
    elif p2 >= 13:
        route_supported = False
        semantic_error_dominant = True
        granularity_gap = False
        next_gate = "nf_opt_20_method_reconsideration"
    elif p3 >= 8:
        route_supported = False
        semantic_error_dominant = False
        granularity_gap = True
        next_gate = "nf_opt_20_evidence_unit_review"
    else:
        route_supported = False
        semantic_error_dominant = False
        granularity_gap = False
        next_gate = "nf_opt_20_method_reconsideration"
    qwen_seal = read_json(qwen_seal_path)
    top100_seal = read_json(top100_seal_path)
    qwen_sha = sha256_file(qwen_path)
    top100_sha = sha256_file(top100_path)
    identity_mismatch = sum(
        set(item["candidate_key"] for item in top100_by_case.get(case_id, [])) != set(qwen_item[case_id])
        for case_id in qwen_by_case
    )
    rank_mismatch = sum(sorted(qwen_rank[case_id].values()) != list(range(1, 101)) for case_id in qwen_by_case)
    decision = {
        "gate": "NF-OPT-20-R0",
        "base_commit": "4106488",
        "model_execution": False,
        "retrieval_rerun": False,
        "ranking_mutation": False,
        "strict_sources": 80,
        "c0": strict_cohorts["C0"],
        "c1": strict_cohorts["C1"],
        "c2": strict_cohorts["C2"],
        "c1_cases": len(pairs),
        "near_boundary": sum(row["cohort"] == "near_boundary" for row in pairs),
        "clear_loss": sum(row["cohort"] == "clear_loss" for row in pairs),
        "p1_potentially_calibratable": p1,
        "p2_model_semantic_error": p2,
        "p3_strict_source_indistinguishable": p3,
        "p4_other": p4,
        "dominant_failure_tags": [{"tag": tag, "count": count, "rate": rate(count, len(pairs))} for tag, count in taxonomy_counts["overall"].most_common()],
        "bm25_correct_qwen_wrong": movement["bm25_correct_qwen_wrong"],
        "bm25_correct_qwen_wrong_rate": rate(movement["bm25_correct_qwen_wrong"], len(pairs)),
        "top100_sha256": top100_sha,
        "top100_sha_expected": top100_seal.get("prediction_sha256"),
        "qwen_sha256": qwen_sha,
        "qwen_sha_expected": qwen_seal.get("main_prediction_sha256"),
        "candidate_identity_mismatch": identity_mismatch,
        "qwen_rank_mismatch": rank_mismatch,
        "qwen_scores_unchanged": qwen_sha == qwen_seal.get("main_prediction_sha256"),
        "qwen_ranks_unchanged": rank_mismatch == 0,
        "pointwise_calibration_route_supported": route_supported,
        "model_semantic_error_dominant": semantic_error_dominant,
        "strict_source_granularity_gap_supported": granularity_gap,
        "production_switch_allowed": False,
        "decision_rule": {"p1_min": 8, "p2_majority_min": 13, "p3_significant_min": 8, "near_boundary_definition": "sealed NF-OPT-19 median score-margin cohort"},
        "next_gate": next_gate,
    }
    write_json(out_dir / "recoverability-classes.json", {"counts": {"P1": p1, "P2": p2, "P3": p3, "P4": p4}, "records": [{"case_id": row["case_id"], "source_index": row["source_index"], "cohort": row["cohort"], "recoverability_class": row["recoverability_class"], "tags": row["diagnostic_tags"]} for row in pairs], "decision_rule": decision["decision_rule"]})
    write_json(out_dir / "decision.json", decision)
    write_json(out_dir / "input-integrity.json", {"top100_sha256": top100_sha, "top100_sha_expected": top100_seal.get("prediction_sha256"), "qwen_sha256": qwen_sha, "qwen_sha_expected": qwen_seal.get("main_prediction_sha256"), "candidate_identity_mismatch": identity_mismatch, "qwen_rank_mismatch": rank_mismatch, "model_execution": False, "retrieval_rerun": False, "ranking_mutation": False})
    write_text(out_dir / "README.md", """# NF-OPT-20 R0 — Strict Source Pointwise Discrimination Audit

Post-seal diagnostic only. The audit compares each C1 strict source with frozen Qwen competitors, preserves strict physical identity, and performs no model, retrieval, ranking, Binder, Calculator, or production mutation.
""")
    print(json.dumps({"artifact_dir": str(out_dir), "decision": decision, "c1_pair_sha256": pair_sha}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
