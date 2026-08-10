#!/usr/bin/env python3
"""T2-02B offline Dense residual value review.

This is a post-seal accounting gate.  It consumes the sealed T2-01
predictions and T2-02A rank artifacts, never runs a retriever or model, and
never changes the BM25 order.  Dense candidates are only appended as a
deduplicated residual.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 23_088
EXPECTED_DATASET_COMMIT = "adf7fe1541ac37351ce1142544d8e3b43010ed92"
BASE_COMMIT = "eb49617"
SUBSETS = ("FinQA", "ConvFinQA", "TAT-DQA")
K_VALUES = (5, 10, 20, 50, 100)
RESIDUAL_BASELINES = (50, 100)
TOP_K = 100
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def metadata_paths(root: Path) -> list[tuple[str, str, Path]]:
    paths = [
        ("FinQA", split, root / "data" / "FinQA" / split / "metadata.jsonl")
        for split in ("train", "dev", "test")
    ]
    paths.append(("ConvFinQA", "turn_0", root / "data" / "ConvFinQA" / "turn_0.jsonl"))
    paths.extend(
        ("TAT-DQA", split, root / "data" / "TAT-DQA" / split / "metadata.jsonl")
        for split in ("train", "dev", "test")
    )
    return paths


def load_dataset_records(dataset_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[int], int]:
    records: dict[str, dict[str, Any]] = {}
    contexts: dict[str, str] = {}
    context_lengths: list[int] = []
    empty_questions = 0
    for subset, default_split, path in metadata_paths(dataset_root):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                query_id = str(source["id"])
                if query_id in records:
                    raise RuntimeError(f"duplicate_query_id:{query_id}")
                context_id = str(source["context_id"])
                context = str(source.get("context") or "")
                existing_context = contexts.get(context_id)
                if existing_context is not None and existing_context != context:
                    raise RuntimeError(f"context_id_content_conflict:{context_id}")
                if existing_context is None:
                    contexts[context_id] = context
                    context_lengths.append(len(context))
                question = source.get("question")
                if question == "":
                    empty_questions += 1
                records[query_id] = {
                    "query_id": query_id,
                    "subset": subset,
                    "split": source.get("split", default_split),
                    "question": question,
                    "company_name": source.get("company_name"),
                    "gold_context_id": context_id,
                }
    if len(records) != EXPECTED_ROWS:
        raise RuntimeError(f"row_count:{len(records)}")
    if empty_questions != 11:
        raise RuntimeError(f"empty_question_count:{empty_questions}")
    return records, contexts, context_lengths, empty_questions


def nearest_rank(values: list[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def tokens(text: Any) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(str(text or ""))}


def diagnostic_tags(record: dict[str, Any], context: str, long_threshold: int) -> list[str]:
    query_tokens = tokens(f"{record.get('company_name') or ''} {record.get('question') or ''}")
    context_tokens = tokens(context)
    company_tokens = tokens(record.get("company_name"))
    overlap = len(query_tokens & context_tokens) / max(len(query_tokens), 1)
    tags: list[str] = []
    if company_tokens and company_tokens & context_tokens:
        tags.append("entity_match")
    if overlap < 0.25:
        tags.append("lexical_mismatch")
    if overlap < 0.5 and company_tokens and company_tokens & context_tokens:
        tags.append("paraphrase")
    if len(context) >= long_threshold:
        tags.append("long_context")
    if context.count("|") >= 4:
        tags.append("table_heavy")
    if not tags:
        tags.append("other")
    return tags


def validate_input_artifacts(prediction_root: Path, audit_root: Path) -> dict[str, Any]:
    prediction_seal_path = prediction_root / "prediction-seal.json"
    prediction_seal = read_json(prediction_seal_path)
    if prediction_seal.get("sealed") is not True:
        raise RuntimeError("t2_01_prediction_not_sealed")
    if prediction_seal.get("prediction_count") != EXPECTED_ROWS:
        raise RuntimeError("t2_01_prediction_count")
    for baseline in ("bm25", "dense", "hybrid"):
        path = prediction_root / f"{baseline}-predictions.jsonl.gz"
        if sha256(path) != prediction_seal["output_sha256"][baseline]:
            raise RuntimeError(f"t2_01_prediction_hash:{baseline}")
    audit_seal = read_json(audit_root / "prediction-seal.json")
    if audit_seal.get("sealed") is not True:
        raise RuntimeError("t2_02a_not_sealed")
    expected_input_hash = sha256(prediction_seal_path)
    if audit_seal.get("input_prediction_seal_sha256") != expected_input_hash:
        raise RuntimeError("t2_02a_input_hash")
    decision = read_json(audit_root / "decision.json")
    if decision.get("prediction_sha256") != expected_input_hash:
        raise RuntimeError("t2_02a_decision_input_hash")
    return {
        "prediction_seal": prediction_seal,
        "audit_seal": audit_seal,
        "prediction_seal_sha256": expected_input_hash,
    }


def empty_supply_stats() -> dict[str, Any]:
    return {"query_count": 0, "bm25_hits": 0, "dense_hits": 0, "union_hits": 0, "dense_unique_hits": 0}


def finalize_supply(stats: dict[str, Any]) -> dict[str, Any]:
    denominator = stats["query_count"]
    stats["bm25_recall"] = stats["bm25_hits"] / denominator
    stats["dense_recall"] = stats["dense_hits"] / denominator
    stats["union_recall"] = stats["union_hits"] / denominator
    stats["absolute_gain"] = stats["union_recall"] - stats["bm25_recall"]
    stats["gain_pp"] = stats["absolute_gain"] * 100.0
    return stats


def decide_mainline(union_gain_pp: float, average_extra_candidates: float, candidate_cost_increase_pct: float) -> bool:
    """Registered T2-02B reject rule; exact boundary is intentionally tested."""
    return union_gain_pp <= 1.0 and average_extra_candidates >= 1.0 and candidate_cost_increase_pct >= 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    prediction_root = args.prediction_root.resolve()
    audit_root = args.audit_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    validated = validate_input_artifacts(prediction_root, audit_root)
    records, contexts, context_lengths, empty_questions = load_dataset_records(dataset_root)
    long_threshold = nearest_rank(context_lengths, 0.95)

    supply: dict[str, dict[str, Any]] = {str(k): empty_supply_stats() for k in K_VALUES}
    protected: dict[str, dict[str, Any]] = {
        str(k): {
            "query_count": 0,
            "bm25_candidate_count": k,
            "residual_counts": [],
            "union_candidate_counts": [],
            "new_gold_covered_queries": 0,
            "residual_total_candidates": 0,
        }
        for k in RESIDUAL_BASELINES
    }
    subset_residual: dict[str, dict[str, dict[str, Any]]] = {
        subset: {str(k): {"query_count": 0, "bm25_hits": 0, "union_hits": 0, "dense_unique_rescue": 0} for k in RESIDUAL_BASELINES}
        for subset in SUBSETS
    }
    cohort_candidates: list[dict[str, Any]] = []

    handles = [
        gzip.open(prediction_root / f"{name}-predictions.jsonl.gz", "rt", encoding="utf-8")
        for name in ("bm25", "dense")
    ]
    try:
        for line_number, (bm25_line, dense_line) in enumerate(zip(*handles), start=1):
            bm25_record = json.loads(bm25_line)
            dense_record = json.loads(dense_line)
            query_id = str(bm25_record["query_id"])
            if query_id != str(dense_record["query_id"]):
                raise RuntimeError(f"prediction_alignment:{line_number}")
            gold = records.get(query_id)
            if gold is None:
                raise RuntimeError(f"query_not_in_dataset:{query_id}")
            subset = gold["subset"]
            bm25_ids = [str(item["context_id"]) for item in bm25_record["ranked_contexts"]]
            dense_ids = [str(item["context_id"]) for item in dense_record["ranked_contexts"]]
            if len(bm25_ids) != TOP_K or len(dense_ids) != TOP_K:
                raise RuntimeError(f"top100_contract:{query_id}")
            if len(set(bm25_ids)) != TOP_K or len(set(dense_ids)) != TOP_K:
                raise RuntimeError(f"duplicate_prediction_context:{query_id}")
            bm25_rank = next((idx for idx, value in enumerate(bm25_ids, start=1) if value == gold["gold_context_id"]), None)
            dense_rank = next((idx for idx, value in enumerate(dense_ids, start=1) if value == gold["gold_context_id"]), None)
            for k in K_VALUES:
                stats = supply[str(k)]
                stats["query_count"] += 1
                bm25_hit = bm25_rank is not None and bm25_rank <= k
                dense_hit = dense_rank is not None and dense_rank <= k
                union_hit = bm25_hit or dense_hit
                stats["bm25_hits"] += int(bm25_hit)
                stats["dense_hits"] += int(dense_hit)
                stats["union_hits"] += int(union_hit)
                stats["dense_unique_hits"] += int(not bm25_hit and dense_hit)
            for k in RESIDUAL_BASELINES:
                bm25_set = set(bm25_ids[:k])
                dense_residual = [context_id for context_id in dense_ids if context_id not in bm25_set]
                residual = protected[str(k)]
                residual["query_count"] += 1
                residual["residual_counts"].append(len(dense_residual))
                residual["union_candidate_counts"].append(k + len(dense_residual))
                residual["residual_total_candidates"] += len(dense_residual)
                new_gold = gold["gold_context_id"] in dense_residual
                residual["new_gold_covered_queries"] += int(new_gold)
                subset_stats = subset_residual[subset][str(k)]
                subset_stats["query_count"] += 1
                bm25_hit = bm25_rank is not None and bm25_rank <= k
                union_hit = bm25_hit or new_gold
                subset_stats["bm25_hits"] += int(bm25_hit)
                subset_stats["union_hits"] += int(union_hit)
                subset_stats["dense_unique_rescue"] += int(new_gold)
            if bm25_rank is None and dense_rank is not None:
                cohort_candidates.append(
                    {
                        "query_id": query_id,
                        "subset": subset,
                        "question": gold["question"],
                        "bm25_gold_rank": bm25_rank,
                        "dense_gold_rank": dense_rank,
                        "gold_context_id": gold["gold_context_id"],
                    }
                )
    finally:
        for handle in handles:
            handle.close()

    for stats in supply.values():
        finalize_supply(stats)
    protected_output: dict[str, Any] = {}
    for key, stats in protected.items():
        residual_counts = stats.pop("residual_counts")
        union_counts = stats.pop("union_candidate_counts")
        base_pairs = stats["query_count"] * stats["bm25_candidate_count"]
        residual_total = stats["residual_total_candidates"]
        stats["average_residual_candidates_per_query"] = residual_total / stats["query_count"]
        stats["p50_residual_candidates_per_query"] = nearest_rank(residual_counts, 0.50)
        stats["p95_residual_candidates_per_query"] = nearest_rank(residual_counts, 0.95)
        stats["maximum_residual_candidates_per_query"] = max(residual_counts) if residual_counts else 0
        stats["average_union_candidates_per_query"] = sum(union_counts) / stats["query_count"]
        stats["candidate_cost_increase"] = residual_total / base_pairs if base_pairs else 0.0
        stats["candidate_cost_increase_pct"] = stats["candidate_cost_increase"] * 100.0
        stats["new_gold_recall"] = stats["new_gold_covered_queries"] / stats["query_count"]
        stats["gain_per_10000_additional_pairs"] = (
            stats["new_gold_covered_queries"] / residual_total * 10000.0 if residual_total else 0.0
        )
        stats["additional_gold_hits_per_candidate"] = (
            stats["new_gold_covered_queries"] / residual_total if residual_total else 0.0
        )
        stats["union_candidate_count_min"] = min(union_counts) if union_counts else 0
        stats["union_candidate_count_max"] = max(union_counts) if union_counts else 0
        protected_output[key] = stats

    subset_output: dict[str, dict[str, Any]] = {}
    for subset, values in subset_residual.items():
        subset_output[subset] = {}
        for key, stats in values.items():
            denominator = stats["query_count"]
            stats["bm25_recall"] = stats["bm25_hits"] / denominator
            stats["union_recall"] = stats["union_hits"] / denominator
            stats["absolute_gain"] = stats["union_recall"] - stats["bm25_recall"]
            stats["gain_pp"] = stats["absolute_gain"] * 100.0
            subset_output[subset][key] = stats

    cohort_candidates.sort(key=lambda row: (row["subset"], row["query_id"]))
    cohort_rows: list[dict[str, Any]] = []
    for row in cohort_candidates[:50]:
        context = contexts[row["gold_context_id"]]
        row["diagnostic_tags"] = diagnostic_tags(records[row["query_id"]], context, long_threshold)
        row["context_char_count"] = len(context)
        cohort_rows.append(row)
    cohort_path = output_root / "dense-rescue-cohort.jsonl"
    with cohort_path.open("w", encoding="utf-8") as handle:
        for row in cohort_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    union_top100_gain_pp = supply["100"]["gain_pp"]
    avg_extra = protected_output["100"]["average_residual_candidates_per_query"]
    cost_pct = protected_output["100"]["candidate_cost_increase_pct"]
    rejected = decide_mainline(union_top100_gain_pp, avg_extra, cost_pct)
    decision_reason = [
        f"BM25 union Dense @100 gain = {union_top100_gain_pp:.6f} percentage points",
        f"Dense residual @100 adds {avg_extra:.6f} candidates/query on average",
        f"Protected residual candidate cost increase = {cost_pct:.6f}%",
        f"Dense-only new Gold coverage @100 = {supply['100']['dense_unique_hits']}/{EXPECTED_ROWS}",
    ]
    if rejected:
        decision_reason.append("union gain is at or below the 1.0 pp gate while residual candidate cost is material")
    else:
        decision_reason.append("union gain/cost evidence does not satisfy the registered reject rule")

    write_json(
        output_root / "candidate-supply-analysis.json",
        {
            "gate": "T2-02B",
            "base_commit": BASE_COMMIT,
            "dataset_commit": EXPECTED_DATASET_COMMIT,
            "formal_query_denominator": EXPECTED_ROWS,
            "empty_question_rows_retained": empty_questions,
            "by_k": supply,
        },
    )
    write_json(
        output_root / "protected-residual-accounting.json",
        {
            "policy": "BM25 candidates are retained in original order; Dense appends only non-duplicate Top100 contexts",
            "bm25_order_preserved": True,
            "dense_append_only": True,
            "candidate_identity_dedup": True,
            "dense_source_top_k": TOP_K,
            "by_bm25_budget": protected_output,
            "cohort_total_bm25_miss_dense_hit_at_100": len(cohort_candidates),
            "cohort_exported": len(cohort_rows),
        },
    )
    write_json(
        output_root / "subset-residual-analysis.json",
        {
            "by_subset": subset_output,
            "overall": {
                str(k): {
                    "bm25_recall": supply[str(k)]["bm25_recall"],
                    "union_recall": supply[str(k)]["union_recall"],
                    "dense_unique_rescue": supply[str(k)]["dense_unique_hits"],
                    "gain_pp": supply[str(k)]["gain_pp"],
                }
                for k in RESIDUAL_BASELINES
            },
        },
    )
    decision = {
        "gate": "T2-02B",
        "base_commit": BASE_COMMIT,
        "dataset_commit": EXPECTED_DATASET_COMMIT,
        "formal_query_denominator": EXPECTED_ROWS,
        "retrieval_rerun": False,
        "model_execution": False,
        "parameter_tuning": False,
        "input_prediction_seal_sha256": validated["prediction_seal_sha256"],
        "bm25_top100_recall": supply["100"]["bm25_recall"],
        "union_top100_recall": supply["100"]["union_recall"],
        "union_top100_gain_pp": union_top100_gain_pp,
        "dense_unique_rescue_top100": supply["100"]["dense_unique_hits"],
        "protected_dense_residual_mainline_rejected": rejected,
        "current_dense_role": "diagnostic_baseline_only" if rejected else "protected_residual_candidate_supply",
        "first_stage_retriever": "bm25" if rejected else "bm25_with_protected_dense_residual",
        "next_gate": "t2_03_qwen3_cross_encoder" if rejected else "t2_03_candidate_pool_review",
        "decision_reason": decision_reason,
        "threshold_contract": {
            "union_top100_gain_pp_max_for_reject": 1.0,
            "minimum_material_average_extra_candidates_per_query": 1.0,
            "minimum_material_candidate_cost_increase_pct": 1.0,
        },
        "protected_residual_top50": protected_output["50"],
        "protected_residual_top100": protected_output["100"],
        "subset_residual": subset_output,
    }
    write_json(output_root / "decision.json", decision)
    write_json(
        output_root / "prediction-seal.json",
        {
            "sealed": True,
            "gate": "T2-02B",
            "base_prediction_seal_sha256": validated["prediction_seal_sha256"],
            "formal_query_denominator": EXPECTED_ROWS,
            "retrieval_rerun": False,
            "model_execution": False,
            "parameter_tuning": False,
            "gold_used_for_candidate_accounting_only": True,
            "candidate_identity_mutation": 0,
        },
    )
    (output_root / "README.md").write_text(
        "# T2-02B Dense Residual Value Review\n\n"
        "This is a post-seal accounting gate. BM25 order is retained exactly; Dense contributes only deduplicated residual contexts. "
        "No retrieval, embedding, model execution, or parameter tuning is performed. See `decision.json` for the registered mainline decision.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

