#!/usr/bin/env python3
"""NF-V2 architecture handoff R0: discriminative evidence-scoring audit.

The script freezes a query-independent RequiredSlot/BinderFactViewV2
serialization, scores every fact in each frozen packet with the cached
Qwen3-Reranker-4B checkpoint, seals those scores, and only then reads the
development-shadow review labels.  No generative Binder, retrieval, or repair
path is invoked.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.evidence import BindingStatus  # noqa: E402
from rag_v2.evidence.binder_fact_view import build_binder_fact_views_v2  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r3_fact_view_v2 as r3  # noqa: E402
from src.pdf_retrieval_v4.qwen3_reranker import build_input_ids, score_batch  # noqa: E402


BASE_COMMIT = "acc08e057153768e1d3a251062041975a201f077"
GATE = "NF-V2-ARCHITECTURE-HANDOFF-R0"
OUT = ROOT / "artifacts/evaluation/nf-v2-architecture-handoff-r0-discriminative-binder"
R2_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r2-distinguishability-review"
R3_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r3-binder-fact-view-v2"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
MODEL_ID = "Qwen/Qwen3-Reranker-4B"
MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
MODEL_PATH = Path(os.getenv("NF_RERANKER_MODEL_PATH", f"/home/mxf/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/{MODEL_REVISION}"))
INSTRUCTION = (
    "Score whether the evidence fact is compatible with the requested financial slot. "
    "Answer yes only when the fact satisfies the requested metric, period, scope, "
    "statement context, and operand role when explicit. Answer no when it conflicts "
    "or does not satisfy the requirement. Use only the serialized slot and fact."
)
MAX_LENGTH = 4096
BATCH_SIZE = 8


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def slot_payload(request: Any, slot: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metric": getattr(slot, "metric", None),
        "period": getattr(slot, "period", None),
        "scope": getattr(slot, "scope", None),
        "role": getattr(slot, "role", None),
        "value_type": enum_value(getattr(slot, "value_type", None)),
        "statement_requirement": getattr(slot, "statement", None) or getattr(slot, "statement_type", None),
    }
    if enum_value(request.plan.intent) == "CALCULATION":
        payload["operation"] = enum_value(request.plan.operation)
        payload["operand_role"] = getattr(slot, "role", None)
    return payload


def canonical_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def query_text(request: Any, slot: Any) -> str:
    return "RequiredSlotV1\n" + canonical_text(slot_payload(request, slot))


def document_text(view: Mapping[str, Any]) -> str:
    return "BinderFactViewV2\n" + canonical_text(dict(view))


def serialization_contract() -> dict[str, Any]:
    return {
        "contract": "SlotFactSerializationV1",
        "version": 1,
        "query_side": "RequiredSlotV1 only; original question text excluded",
        "query_fields": ["metric", "period", "scope", "role", "value_type", "statement_requirement", "operation", "operand_role"],
        "fact_side": "BinderFactViewV2 deterministic JSON, all exposed fields preserved",
        "fact_fields": [
            "fact_handle", "raw_metric", "normalized_metric", "raw_period", "normalized_period",
            "raw_value", "parsed_numeric_value", "raw_scale", "normalized_scale", "currency", "unit",
            "row_label", "row_path", "row_hierarchy", "column_label", "column_header_path",
            "multi_level_column_headers", "table_title", "statement_title", "statement_type",
            "section_title", "section_path", "page", "table_id", "row_id", "column_id", "cell_id",
            "physical_source_id", "document_id", "pdf_page", "period_value_bindings", "candidate_rank",
        ],
        "instruction": INSTRUCTION,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "max_length": MAX_LENGTH,
        "gold_independent": True,
        "question_specific_aliases": 0,
        "original_question_in_default_input": False,
    }


def structural_candidate_ok(fact: Mapping[str, Any], packet: list[Mapping[str, Any]], source_map: Mapping[str, Mapping[str, Any]]) -> bool:
    fact_id = str(fact.get("fact_id") or "")
    if not fact_id or fact_id not in {str(item.get("fact_id")) for item in packet}:
        return False
    if fact.get("provenance_complete") is not True:
        return False
    linked_ids = {str(fact.get("candidate_id") or "")} | {str(item) for item in fact.get("candidate_ids", []) if item}
    linked_sources = [source_map[cid] for cid in linked_ids if cid in source_map]
    if not linked_sources:
        return False
    physical = str(fact.get("physical_source_id") or "")
    return bool(physical and any(str(source.get("physical_source_id") or "") == physical for source in linked_sources))


def load_views() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    frozen, source_map = r3.load_frozen()
    views = {
        qid: build_binder_fact_views_v2(list(request.facts), source_map)
        for qid, request in frozen["requests"].items()
    }
    return frozen, source_map, views


def score_all_packets(frozen: Mapping[str, Any], views: Mapping[str, list[dict[str, Any]]], tokenizer: Any, model: Any, device: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total_pairs = 0
    total_input_tokens = 0
    query_latencies: list[float] = []
    max_query_tokens = 0
    max_query: str | None = None
    started_all = time.perf_counter()
    for qid in sorted(frozen["requests"]):
        request = frozen["requests"][qid]
        query_started = time.perf_counter()
        slot_records: list[dict[str, Any]] = []
        query_pair_count = 0
        query_tokens = 0
        for slot in request.plan.required_slots:
            qtext = query_text(request, slot)
            prepared: list[tuple[dict[str, Any], list[int], dict[str, Any]]] = []
            for view, fact in zip(views[qid], request.facts, strict=True):
                ids, audit = build_input_ids(tokenizer, INSTRUCTION, qtext, document_text(view), MAX_LENGTH)
                view = {**view, "_fact_id": str(fact.get("fact_id"))}
                prepared.append((view, ids, audit))
            scored: list[dict[str, Any]] = []
            for offset in range(0, len(prepared), BATCH_SIZE):
                batch = prepared[offset : offset + BATCH_SIZE]
                outputs = score_batch(model, tokenizer, [item[1] for item in batch])
                for (view, _, audit), output in zip(batch, outputs, strict=True):
                    scored.append({
                        "fact_handle": view["fact_handle"],
                        "fact_id": str(view["_fact_id"]),
                        "score": float(output["reranker_score"]),
                        "yes_logit": float(output["yes_logit"]),
                        "no_logit": float(output["no_logit"]),
                        "input_tokens": int(audit["final_token_count"]),
                        "truncated": bool(audit["truncated"]),
                    })
            scored.sort(key=lambda item: (-item["score"], item["fact_handle"]))
            for rank, item in enumerate(scored, 1):
                item["rank"] = rank
            slot_records.append({"slot_id": slot.slot_id, "scores": scored, "query_sha256": stable_sha(qtext), "pair_count": len(scored)})
            query_pair_count += len(scored)
            query_tokens += sum(int(item["input_tokens"]) for item in scored)
        elapsed = round((time.perf_counter() - query_started) * 1000.0, 3)
        query_latencies.append(elapsed)
        total_pairs += query_pair_count
        total_input_tokens += query_tokens
        if query_tokens > max_query_tokens:
            max_query_tokens = query_tokens
            max_query = qid
        records.append({
            "question_id": qid,
            "intent": enum_value(request.plan.intent),
            "fact_count": len(request.facts),
            "slot_count": len(request.plan.required_slots),
            "slots": slot_records,
            "query_input_tokens": query_tokens,
            "latency_ms": elapsed,
        })
    runtime = {
        "total_pairs": total_pairs,
        "total_input_tokens": total_input_tokens,
        "output_tokens": 0,
        "query_count": len(records),
        "query_latencies_ms": query_latencies,
        "average_query_latency_ms": statistics.mean(query_latencies) if query_latencies else 0.0,
        "p50_query_latency_ms": statistics.median(query_latencies) if query_latencies else 0.0,
        "p95_query_latency_ms": percentile(query_latencies),
        "max_query_latency_ms": max(query_latencies) if query_latencies else 0.0,
        "wall_time_ms": round((time.perf_counter() - started_all) * 1000.0, 3),
        "largest_input_query": max_query,
        "largest_input_tokens": max_query_tokens,
        "device": device,
    }
    return records, runtime


def percentile(values: list[float], quantile: float = 0.95) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))]


def prediction_slots(predictions: list[Mapping[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        for slot in row.get("slots", []):
            result[(str(row["question_id"]), str(slot["slot_id"]))] = list(slot["scores"])
    return result


def rank_for_handles(scores: list[Mapping[str, Any]], handles: set[str]) -> int | None:
    ranks = [int(item["rank"]) for item in scores if str(item["fact_handle"]) in handles]
    return min(ranks) if ranks else None


def margin_for_handles(scores: list[Mapping[str, Any]], handles: set[str]) -> tuple[float | None, float | None, float | None]:
    correct = [float(item["score"]) for item in scores if str(item["fact_handle"]) in handles]
    incorrect = [float(item["score"]) for item in scores if str(item["fact_handle"]) not in handles]
    if not correct:
        return None, None, None
    top_correct = max(correct)
    top_incorrect = max(incorrect) if incorrect else None
    return top_correct, top_incorrect, top_correct - top_incorrect if top_incorrect is not None else None


def load_review_cohorts() -> dict[str, Any]:
    direct = read_json(R3_OUT / "direct-v2-distinguishability.json")["rows"]
    calc = read_json(R3_OUT / "calculation-v2-distinguishability.json")["rows"]
    indist = read_json(R3_OUT / "remaining-indistinguishable.json")["direct"]
    unbind = read_json(R2_OUT / "unbindable-false-binding-review.json")["rows"]
    return {
        "direct_unique": {str(row["question_id"]): row for row in direct if row.get("v2_visible_unique_bindable")},
        "calc": {(str(row["question_id"]), str(slot["slot_id"])): slot for row in calc for slot in row.get("slots", []) if slot.get("v2_visible_unique_operand")},
        "indistinguishable": {str(row["question_id"]): row for row in indist},
        "unbindable": {str(row["question_id"]): row for row in unbind},
    }


def score_distribution(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    top1 = [float(row["top1_score"]) for row in rows if row.get("top1_score") is not None]
    top2 = [float(row["top2_score"]) for row in rows if row.get("top2_score") is not None]
    margins = [float(row["margin"]) for row in rows if row.get("margin") is not None]
    return {
        "count": len(rows),
        "top1_scores": top1,
        "top2_scores": top2,
        "margins": margins,
        "top1_mean": statistics.mean(top1) if top1 else None,
        "top1_median": statistics.median(top1) if top1 else None,
        "top1_p05": percentile(top1, 0.05) if top1 else None,
        "top1_p95": percentile(top1, 0.95) if top1 else None,
        "margin_mean": statistics.mean(margins) if margins else None,
        "margin_median": statistics.median(margins) if margins else None,
        "positive_margin": sum(int(value > 1e-6) for value in margins),
        "near_zero_margin": sum(int(abs(value) <= 1e-6) for value in margins),
        "negative_margin": sum(int(value < -1e-6) for value in margins),
    }


def build_stage_a_metrics(predictions: list[Mapping[str, Any]], cohorts: Mapping[str, Any]) -> dict[str, Any]:
    slots = prediction_slots(predictions)
    slot_id_by_qid = {
        str(row["question_id"]): str(row["slots"][0]["slot_id"])
        for row in predictions
        if row.get("slots")
    }
    direct_rows: list[dict[str, Any]] = []
    for qid, reviewed in sorted(cohorts["direct_unique"].items()):
        slot_scores = slots.get((qid, slot_id_by_qid.get(qid, "slot_1")), [])
        handles = {str(item) for item in reviewed.get("gold_compatible_fact_handles", [])}
        rank = rank_for_handles(slot_scores, handles)
        correct_score, incorrect_score, margin = margin_for_handles(slot_scores, handles)
        direct_rows.append({"question_id": qid, "correct_handles": sorted(handles), "rank": rank, "rank_at_1": rank == 1, "rank_at_2": bool(rank and rank <= 2), "rank_at_3": bool(rank and rank <= 3), "correct_score": correct_score, "highest_incorrect_score": incorrect_score, "margin": margin, "top1_score": slot_scores[0]["score"] if slot_scores else None, "top2_score": slot_scores[1]["score"] if len(slot_scores) > 1 else None})
    calc_rows: list[dict[str, Any]] = []
    for (qid, slot_id), reviewed in sorted(cohorts["calc"].items()):
        slot_scores = slots.get((qid, slot_id), [])
        handles = {str(item) for item in reviewed.get("gold_compatible_fact_handles", [])}
        rank = rank_for_handles(slot_scores, handles)
        correct_score, incorrect_score, margin = margin_for_handles(slot_scores, handles)
        calc_rows.append({"question_id": qid, "slot_id": slot_id, "role": reviewed.get("role"), "correct_handles": sorted(handles), "rank": rank, "rank_at_1": rank == 1, "rank_at_2": bool(rank and rank <= 2), "rank_at_3": bool(rank and rank <= 3), "correct_score": correct_score, "highest_incorrect_score": incorrect_score, "margin": margin, "top1_score": slot_scores[0]["score"] if slot_scores else None, "top2_score": slot_scores[1]["score"] if len(slot_scores) > 1 else None})
    indist_rows: list[dict[str, Any]] = []
    for qid in sorted(cohorts["indistinguishable"]):
        slot_scores = slots.get((qid, slot_id_by_qid.get(qid, "slot_1")), [])
        indist_rows.append({"question_id": qid, "top1_score": slot_scores[0]["score"] if slot_scores else None, "top2_score": slot_scores[1]["score"] if len(slot_scores) > 1 else None, "margin": float(slot_scores[0]["score"] - slot_scores[1]["score"]) if len(slot_scores) > 1 else None})
    unbind_rows: list[dict[str, Any]] = []
    for qid in sorted(cohorts["unbindable"]):
        slot_scores = slots.get((qid, slot_id_by_qid.get(qid, "slot_1")), [])
        unbind_rows.append({"question_id": qid, "top1_score": slot_scores[0]["score"] if slot_scores else None, "top2_score": slot_scores[1]["score"] if len(slot_scores) > 1 else None, "margin": float(slot_scores[0]["score"] - slot_scores[1]["score"]) if len(slot_scores) > 1 else None})
    direct_ranks = [row["rank"] for row in direct_rows if row.get("rank")]
    calc_ranks = [row["rank"] for row in calc_rows if row.get("rank")]
    direct_mrr = sum(1.0 / rank for rank in direct_ranks) / len(direct_rows) if direct_rows else 0.0
    calc_mrr = sum(1.0 / rank for rank in calc_ranks) / len(calc_rows) if calc_rows else 0.0
    visible_dist = score_distribution(direct_rows)
    visible_dist["rows"] = direct_rows
    calc_dist = score_distribution(calc_rows)
    calc_dist["rows"] = calc_rows
    indist_dist = score_distribution(indist_rows)
    unbind_dist = score_distribution(unbind_rows)
    separation = {
        "visible_unique_median_margin": visible_dist["margin_median"],
        "indistinguishable_median_margin": indist_dist["margin_median"],
        "unbindable_median_margin": unbind_dist["margin_median"],
        "visible_unique_median_top1": visible_dist["top1_median"],
        "indistinguishable_median_top1": indist_dist["top1_median"],
        "unbindable_median_top1": unbind_dist["top1_median"],
        "margin_separated_from_indistinguishable": bool(visible_dist["margin_median"] is not None and indist_dist["margin_median"] is not None and visible_dist["margin_median"] > indist_dist["margin_median"]),
        "score_separated_from_unbindable": bool(visible_dist["top1_median"] is not None and unbind_dist["top1_median"] is not None and visible_dist["top1_median"] > unbind_dist["top1_median"]),
    }
    separation["useful_separation"] = bool(separation["margin_separated_from_indistinguishable"] and separation["score_separated_from_unbindable"])
    return {
        "direct": {"denominator": 21, "rank_at_1": sum(int(row["rank_at_1"]) for row in direct_rows), "rank_at_2": sum(int(row["rank_at_2"]) for row in direct_rows), "rank_at_3": sum(int(row["rank_at_3"]) for row in direct_rows), "mrr": round(direct_mrr, 6), "rows": direct_rows, "score_margin": visible_dist},
        "calculation": {"denominator": 12, "rank_at_1": sum(int(row["rank_at_1"]) for row in calc_rows), "rank_at_2": sum(int(row["rank_at_2"]) for row in calc_rows), "rank_at_3": sum(int(row["rank_at_3"]) for row in calc_rows), "mrr": round(calc_mrr, 6), "all_operands_rank_at_1": sum(int(all(row["rank_at_1"] for row in calc_rows if row["question_id"] == qid)) for qid in sorted({row["question_id"] for row in calc_rows})), "rows": calc_rows, "score_margin": calc_dist},
        "indistinguishable": {"denominator": 6, "rows": indist_rows, "distribution": indist_dist},
        "unbindable": {"denominator": 7, "rows": unbind_rows, "distribution": unbind_dist},
        "separation": separation,
    }


def threshold_grid() -> dict[str, Any]:
    return {
        "threshold_calibration_role": "development_shadow",
        "fresh_blind": False,
        "compatibility_thresholds": [-2.0, -1.0, -0.75, -0.5, -0.25, -0.1],
        "margin_thresholds": [0.0, 0.02, 0.05, 0.1, 0.2, 0.4],
        "selection_rule": "one global threshold and one global margin; maximize safe coverage on a zero-false-binding stable plateau",
        "question_specific_thresholds": 0,
        "gold_source_rules": 0,
    }


def predict_admission(qid: str, request: Any, slot_scores: Mapping[tuple[str, str], list[dict[str, Any]]], threshold: float, margin_threshold: float, source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    packet = list(request.facts)
    slots: list[dict[str, Any]] = []
    for slot in request.plan.required_slots:
        scores = list(slot_scores.get((qid, slot.slot_id), []))
        if not scores or float(scores[0]["score"]) < threshold:
            slots.append({"slot_id": slot.slot_id, "status": BindingStatus.MISSING.value, "selected_fact_id": None, "top1_score": scores[0]["score"] if scores else None, "top2_score": scores[1]["score"] if len(scores) > 1 else None, "margin": scores[0]["score"] - scores[1]["score"] if len(scores) > 1 else None})
            continue
        margin = float(scores[0]["score"] - scores[1]["score"]) if len(scores) > 1 else float("inf")
        if len(scores) > 1 and margin < margin_threshold:
            slots.append({"slot_id": slot.slot_id, "status": BindingStatus.AMBIGUOUS.value, "selected_fact_id": None, "top1_score": scores[0]["score"], "top2_score": scores[1]["score"], "margin": margin})
            continue
        selected = next((fact for fact in packet if str(fact.get("fact_id")) == str(next(item["fact_id"] for item in scores if item["fact_handle"] == scores[0]["fact_handle"]))), None)
        ok = bool(selected and structural_candidate_ok(selected, packet, source_map))
        slots.append({"slot_id": slot.slot_id, "status": BindingStatus.BOUND.value if ok else BindingStatus.MISSING.value, "selected_fact_id": str(selected.get("fact_id")) if ok else None, "selected_handle": scores[0]["fact_handle"] if ok else None, "top1_score": scores[0]["score"], "top2_score": scores[1]["score"] if len(scores) > 1 else None, "margin": margin, "structural_valid": ok})
    if any(row["status"] == BindingStatus.AMBIGUOUS.value for row in slots):
        status = BindingStatus.AMBIGUOUS.value
    elif all(row["status"] == BindingStatus.BOUND.value for row in slots):
        status = BindingStatus.BOUND.value
    else:
        status = BindingStatus.MISSING.value
    return {"question_id": qid, "status": status, "released": status == BindingStatus.BOUND.value, "slots": slots}


def strict_result(qid: str, request: Any, result: Mapping[str, Any], labels: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], reviewed_ids: set[str], reviewed_fact_ids: Mapping[str, set[str]], generic_direct_ids: set[str]) -> tuple[bool, int]:
    facts = {str(fact.get("fact_id")): fact for fact in request.facts}
    correct_slots = 0
    for slot in request.plan.required_slots:
        row = next(item for item in result["slots"] if item["slot_id"] == slot.slot_id)
        fact = facts.get(str(row.get("selected_fact_id"))) if row.get("selected_fact_id") else None
        correct = bool(fact and r1d.slot_is_strict(qid, slot, fact, labels[qid], source_map, reviewed_ids, reviewed_fact_ids, generic_direct_ids))
        correct_slots += int(correct)
    return correct_slots == len(request.plan.required_slots), correct_slots


def evaluate_grid(frozen: Mapping[str, Any], predictions: list[Mapping[str, Any]], labels: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], grid: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[float, float], dict[str, Any]]]:
    slots = prediction_slots(predictions)
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()
    current_view = read_json(ROOT / "artifacts/evaluation/nf-v2-03-r1c-supply-and-protocol-recovery/current-vs-view-bindability.json") if (ROOT / "artifacts/evaluation/nf-v2-03-r1c-supply-and-protocol-recovery/current-vs-view-bindability.json").exists() else {}
    generic_direct_ids = {str(item) for item in current_view.get("generic_recovered_strict_questions", [])}
    grid_rows: list[dict[str, Any]] = []
    lookup: dict[tuple[float, float], dict[str, Any]] = {}
    direct_ids = [qid for qid, request in frozen["requests"].items() if str(enum_value(request.plan.intent)) == "DIRECT_FACT"]
    for threshold in grid["compatibility_thresholds"]:
        for margin_threshold in grid["margin_thresholds"]:
            results: dict[str, Any] = {}
            bound = correct = false = 0
            for qid in direct_ids:
                result = predict_admission(qid, frozen["requests"][qid], slots, float(threshold), float(margin_threshold), source_map)
                results[qid] = result
                if result["released"]:
                    bound += 1
                    strict, _ = strict_result(qid, frozen["requests"][qid], result, labels, source_map, reviewed_ids, reviewed_fact_ids, generic_direct_ids)
                    correct += int(strict)
                    false += int(not strict)
            row = {"compatibility_threshold": threshold, "margin_threshold": margin_threshold, "bound": bound, "strict_correct": correct, "false_binding": false, "precision": correct / bound if bound else None}
            grid_rows.append(row)
            lookup[(float(threshold), float(margin_threshold))] = {"summary": row, "results": results}
    return grid_rows, lookup


def select_plateau(rows: list[Mapping[str, Any]], grid: Mapping[str, Any]) -> dict[str, Any] | None:
    safe = {(float(row["compatibility_threshold"]), float(row["margin_threshold"])): row for row in rows if int(row["false_binding"]) == 0}
    plateau: list[dict[str, Any]] = []
    thresholds = grid["compatibility_thresholds"]
    margins = grid["margin_thresholds"]
    for key, row in safe.items():
        ti = thresholds.index(key[0])
        mi = margins.index(key[1])
        neighbors = []
        for ni, nm in ((ti - 1, mi), (ti + 1, mi), (ti, mi - 1), (ti, mi + 1)):
            if 0 <= ni < len(thresholds) and 0 <= nm < len(margins):
                neighbors.append((float(thresholds[ni]), float(margins[nm])))
        zero_neighbors = sum(int(item in safe) for item in neighbors)
        if zero_neighbors:
            plateau.append({**row, "zero_false_neighbors": zero_neighbors})
    if not plateau:
        return None
    plateau.sort(key=lambda row: (-int(row["bound"]), -int(row["zero_false_neighbors"]), -float(row["compatibility_threshold"]), -float(row["margin_threshold"])))
    return plateau[0]


def evaluate_final(frozen: Mapping[str, Any], predictions: list[Mapping[str, Any]], labels: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], threshold: float, margin_threshold: float) -> dict[str, Any]:
    slots = prediction_slots(predictions)
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()
    current_view = read_json(ROOT / "artifacts/evaluation/nf-v2-03-r1c-supply-and-protocol-recovery/current-vs-view-bindability.json") if (ROOT / "artifacts/evaluation/nf-v2-03-r1c-supply-and-protocol-recovery/current-vs-view-bindability.json").exists() else {}
    generic_direct_ids = {str(item) for item in current_view.get("generic_recovered_strict_questions", [])}
    outputs: dict[str, list[dict[str, Any]]] = {"DIRECT_FACT": [], "CALCULATION": [], "MULTI_EVIDENCE": []}
    direct_rows: list[dict[str, Any]] = []
    calc_operands: list[dict[str, Any]] = []
    calc_questions: list[dict[str, Any]] = []
    multi_rows: list[dict[str, Any]] = []
    for qid, request in sorted(frozen["requests"].items()):
        intent = str(enum_value(request.plan.intent))
        result = predict_admission(qid, request, slots, threshold, margin_threshold, source_map)
        strict, correct_slots = strict_result(qid, request, result, labels, source_map, reviewed_ids, reviewed_fact_ids, generic_direct_ids)
        outputs[intent].append(result)
        if intent == "DIRECT_FACT":
            direct_rows.append({**result, "strict_correct": strict, "false_binding": bool(result["released"] and not strict), "correct_slots": correct_slots})
        elif intent == "CALCULATION":
            operand_rows = []
            for slot_result in result["slots"]:
                fact = next((fact for fact in request.facts if str(fact.get("fact_id")) == str(slot_result.get("selected_fact_id"))), None)
                slot = next(slot for slot in request.plan.required_slots if slot.slot_id == slot_result["slot_id"])
                slot_correct = bool(fact and r1d.slot_is_strict(qid, slot, fact, labels[qid], source_map, reviewed_ids, reviewed_fact_ids, generic_direct_ids))
                operand = {"question_id": qid, "slot_id": slot.slot_id, "status": slot_result["status"], "selected_fact_id": slot_result.get("selected_fact_id"), "strict_correct": slot_correct}
                operand_rows.append(operand)
                calc_operands.append(operand)
            calc_questions.append({"question_id": qid, "status": result["status"], "all_operands_bound": result["released"], "all_operands_strict_correct": strict, "operands": operand_rows})
        else:
            multi_rows.append({**result, "strict_correct": strict, "false_binding": bool(result["released"] and not strict), "correct_slots": correct_slots})
    direct_bound = [row for row in direct_rows if row["released"]]
    calc_bound = [row for row in calc_operands if row["status"] == BindingStatus.BOUND.value]
    return {
        "threshold": threshold,
        "margin_threshold": margin_threshold,
        "direct": {"total": 56, "bound": len(direct_bound), "strict_correct": sum(int(row["strict_correct"]) for row in direct_bound), "false_binding": sum(int(row["false_binding"]) for row in direct_rows), "missing": sum(int(row["status"] == BindingStatus.MISSING.value) for row in direct_rows), "ambiguous": sum(int(row["status"] == BindingStatus.AMBIGUOUS.value) for row in direct_rows), "rows": direct_rows},
        "calculation": {"total": 11, "bound_operand_slots": len(calc_bound), "strict_correct_operands": sum(int(row["strict_correct"]) for row in calc_bound), "false_operand_binding": sum(int(row["status"] == BindingStatus.BOUND.value and not row["strict_correct"]) for row in calc_operands), "ready": sum(int(row["all_operands_bound"]) for row in calc_questions), "strict_ready": sum(int(row["all_operands_strict_correct"]) for row in calc_questions), "partial": sum(int(not row["all_operands_bound"] and any(item["status"] == BindingStatus.BOUND.value for item in row["operands"])) for row in calc_questions), "not_ready": sum(int(not row["all_operands_bound"] and not any(item["status"] == BindingStatus.BOUND.value for item in row["operands"])) for row in calc_questions), "questions": calc_questions, "operands": calc_operands},
        "multi": {"total": 5, "complete": sum(int(row["released"]) for row in multi_rows), "partial": sum(int(not row["released"] and row["status"] == BindingStatus.AMBIGUOUS.value) for row in multi_rows), "missing": sum(int(row["status"] == BindingStatus.MISSING.value) for row in multi_rows), "false_binding": sum(int(row["false_binding"]) for row in multi_rows), "rows": multi_rows},
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    contract = serialization_contract()
    contract_path = OUT / "slot-fact-serialization-contract.json"
    write_json(contract_path, contract)
    serialization_sha = sha256_file(contract_path)
    (OUT / "serialization.sha256").write_text(serialization_sha + "  slot-fact-serialization-contract.json\n", encoding="utf-8")
    prediction_path = OUT / "ranking-predictions.jsonl.gz"
    frozen, source_map, views = load_views()
    if "--replay-sealed" in sys.argv:
        if not prediction_path.exists():
            raise RuntimeError("sealed_ranking_predictions_missing")
        predictions = read_jsonl_gz(prediction_path)
        stored_runtime = read_json(OUT / "cost-latency.json")
        runtime = {
            "total_pairs": stored_runtime.get("reranker_pairs", 0),
            "total_input_tokens": stored_runtime.get("input_tokens", 0),
            "output_tokens": stored_runtime.get("output_tokens", 0),
            "wall_time_ms": stored_runtime.get("total_latency_ms", 0.0),
            "average_query_latency_ms": stored_runtime.get("average_per_query_ms", 0.0),
            "p50_query_latency_ms": stored_runtime.get("p50_per_query_ms", 0.0),
            "p95_query_latency_ms": stored_runtime.get("p95_per_query_ms", 0.0),
            "max_query_latency_ms": stored_runtime.get("max_per_query_ms", 0.0),
        }
        prediction_sha = sha256_file(prediction_path)
        seal = read_json(OUT / "ranking-seal.json")
        if prediction_sha != seal.get("prediction_sha256"):
            raise RuntimeError("sealed_ranking_prediction_sha_mismatch")
    else:
        device = os.getenv("NF_RERANKER_DEVICE", "cuda:0")
        local_only = True
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), padding_side="left", local_files_only=local_only)
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.encode("yes", add_special_tokens=False) != [tokenizer.convert_tokens_to_ids("yes")]:
            raise RuntimeError("yes_token_contract_failed")
        if tokenizer.encode("no", add_special_tokens=False) != [tokenizer.convert_tokens_to_ids("no")]:
            raise RuntimeError("no_token_contract_failed")
        model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), torch_dtype=torch.bfloat16, local_files_only=local_only).to(device).eval()
        predictions, runtime = score_all_packets(frozen, views, tokenizer, model, device)
        write_jsonl_gz(prediction_path, predictions)
        prediction_sha = sha256_file(prediction_path)
        write_json(OUT / "ranking-seal.json", {"gate": GATE, "sealed": True, "prediction_sha256": prediction_sha, "prediction_count": len(predictions), "generative_binder_calls": 0, "reranker_pairs": runtime["total_pairs"], "gold_reads_before_prediction_seal": 0, "sealed_before_review_labels": True, "model": MODEL_ID, "revision": MODEL_REVISION, "serialization_sha256": serialization_sha})
        if sha256_file(prediction_path) != prediction_sha:
            raise RuntimeError("ranking_prediction_sha_mismatch")
    # Development-shadow review labels and cohort definitions are opened only
    # after the complete score file has been sealed.
    cohorts = load_review_cohorts()
    stage_a = build_stage_a_metrics(predictions, cohorts)
    write_json(OUT / "direct-ranking-capability.json", stage_a["direct"])
    write_json(OUT / "calculation-ranking-capability.json", stage_a["calculation"])
    write_json(OUT / "indistinguishable-score-analysis.json", stage_a["indistinguishable"])
    write_json(OUT / "unbindable-score-analysis.json", stage_a["unbindable"])
    write_json(OUT / "cost-latency.json", {"reranker_pairs": runtime["total_pairs"], "total_latency_ms": runtime["wall_time_ms"], "average_per_query_ms": runtime["average_query_latency_ms"], "p50_per_query_ms": runtime["p50_query_latency_ms"], "p95_per_query_ms": runtime["p95_query_latency_ms"], "max_per_query_ms": runtime["max_query_latency_ms"], "input_tokens": runtime["total_input_tokens"], "output_tokens": runtime["output_tokens"], "generative_binder_calls_avoided_if_adopted": 62, "generative_binder_calls_actual": 0, "generative_binder_reference": "NF-V2-04 R0 observed 62 calls"})
    ranking_pass = bool(stage_a["direct"]["rank_at_1"] >= 15 and stage_a["calculation"]["rank_at_1"] >= 8 and stage_a["separation"]["useful_separation"])
    grid = threshold_grid()
    stage_a_decision = {"direct_rank_at_1": f"{stage_a['direct']['rank_at_1']}/21", "calculation_rank_at_1": f"{stage_a['calculation']['rank_at_1']}/12", "useful_score_separation": stage_a["separation"]["useful_separation"], "ranking_feasible": ranking_pass}
    stage_b_executed = False
    selected: dict[str, Any] | None = None
    grid_rows: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    if ranking_pass:
        stage_b_executed = True
        labels = {str(row["case_id"]): row for row in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if row}
        grid_rows, grid_lookup = evaluate_grid(frozen, predictions, labels, source_map, grid)
        selected = select_plateau(grid_rows, grid)
        if selected is not None:
            key = (float(selected["compatibility_threshold"]), float(selected["margin_threshold"]))
            final = evaluate_final(frozen, predictions, labels, source_map, *key)
    write_json(OUT / "threshold-grid.json", {**grid, "stage_a": stage_a_decision, "stage_b_executed": stage_b_executed, "rows": grid_rows, "selected_operating_point": selected})
    admission_contract = {"contract": "DiscriminativeBindingAdmissionV1", "version": 1, "stage_b_executed": stage_b_executed, "threshold_calibration_role": "development_shadow", "fresh_blind": False, "conditions": ["top1_score >= global compatibility threshold", "top1_minus_top2 >= global margin threshold", "selected fact exists in packet", "provenance complete", "source relation valid", "Binding Validator pass"], "unknown_policy": "MISSING or AMBIGUOUS; never guess", "question_specific_thresholds": 0, "gold_source_rules": 0, "selected_threshold": selected.get("compatibility_threshold") if selected else None, "selected_margin": selected.get("margin_threshold") if selected else None}
    write_json(OUT / "discriminative-admission-contract.json", admission_contract)
    if final is not None:
        write_json(OUT / "direct-admission-results.json", final["direct"])
        write_json(OUT / "calculation-admission-results.json", final["calculation"])
        write_json(OUT / "multi-admission-results.json", final["multi"])
    else:
        write_json(OUT / "direct-admission-results.json", {"executed": False, "reason": "stage_a_failed_or_no_zero_false_binding_plateau"})
        write_json(OUT / "calculation-admission-results.json", {"executed": False, "reason": "stage_a_failed_or_no_zero_false_binding_plateau"})
        write_json(OUT / "multi-admission-results.json", {"executed": False, "reason": "stage_a_failed_or_no_zero_false_binding_plateau"})
    ablation = {"global_qwen3_plus": {"direct_visible_unique": "8/21", "calculation_visible_unique_operands": "1/12"}, "slotwise_qwen3_plus": {"direct_visible_unique": "9/21", "calculation_visible_unique_operands": "5/12"}, "pairwise_qwen3_plus": {"direct_visible_unique": "7/21", "calculation_visible_unique_operands": "1/12"}, "discriminative_reranker": {"direct_rank_at_1": f"{stage_a['direct']['rank_at_1']}/21", "calculation_rank_at_1": f"{stage_a['calculation']['rank_at_1']}/12", "stage_b_executed": stage_b_executed, "direct_bound": final["direct"]["bound"] if final else None, "calculation_ready": final["calculation"]["ready"] if final else None}}
    write_json(OUT / "generative-vs-discriminative-ablation.json", ablation)
    decision = {"gate": GATE, "base_commit": BASE_COMMIT, "scorer": MODEL_ID, "revision": MODEL_REVISION, "generative_binder_calls": 0, "retrieval_calls": 0, "serialization_sha256": serialization_sha, "stage_a": stage_a_decision, "stage_b_executed": stage_b_executed, "selected_threshold": selected.get("compatibility_threshold") if selected else None, "selected_margin": selected.get("margin_threshold") if selected else None, "discriminative_binder_feasible": ranking_pass, "discriminative_binder_effective": bool(final and final["direct"]["bound"] >= 8 and final["direct"]["false_binding"] == 0 and final["calculation"]["ready"] >= 3 and final["calculation"]["false_operand_binding"] == 0), "generative_binder_retired": False, "evidence_binding_architecture": "discriminative_reranker_plus_deterministic_admission" if final else "not_adopted", "supervisor_role": "General LLM Supervisor remains control plane; reranker is a trusted tool-plane scorer", "next_gate": "v2_05_calculation_recovery" if final and final["direct"]["bound"] >= 8 and final["direct"]["false_binding"] == 0 and final["calculation"]["ready"] >= 3 and final["calculation"]["false_operand_binding"] == 0 else "v2_architecture_scope_freeze", "production_default": "V1", "production_switch_allowed": False}
    write_json(OUT / "architecture-decision.json", decision)
    write_json(OUT / "README.md", {"gate": GATE, "summary": "Qwen3-Reranker-4B was evaluated as a downstream RequiredSlot-to-BinderFactViewV2 compatibility scorer. Serialization and scores were sealed before review labels; no generative Binder or retrieval path was called.", "decision": decision, "runtime": runtime})
    print(json.dumps({"stage_a": stage_a_decision, "stage_b_executed": stage_b_executed, "selected": selected, "final": final, "decision": decision, "runtime": runtime}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
