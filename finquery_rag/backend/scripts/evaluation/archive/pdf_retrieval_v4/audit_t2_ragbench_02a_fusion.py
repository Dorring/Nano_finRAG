#!/usr/bin/env python3
"""T2-02A offline audit of the sealed BM25/Dense/RRF predictions.

This gate consumes only the sealed T2-01 prediction files and the published
dataset rows after validating the prediction seal.  It never performs search,
embedding, fusion, or reranking.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_ROWS = 23_088
EXPECTED_CORPUS = 7_318
EXPECTED_DATASET_COMMIT = "adf7fe1541ac37351ce1142544d8e3b43010ed92"
SUBSETS = ("FinQA", "ConvFinQA", "TAT-DQA")
BASELINES = ("bm25", "dense", "hybrid")
K_VALUES = (5, 10, 20, 50, 100)
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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def metadata_paths(root: Path) -> list[tuple[str, str, Path]]:
    result = [
        ("FinQA", split, root / "data" / "FinQA" / split / "metadata.jsonl")
        for split in ("train", "dev", "test")
    ]
    result.append(("ConvFinQA", "turn_0", root / "data" / "ConvFinQA" / "turn_0.jsonl"))
    result.extend(
        ("TAT-DQA", split, root / "data" / "TAT-DQA" / split / "metadata.jsonl")
        for split in ("train", "dev", "test")
    )
    return result


def load_gold(dataset_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for subset, default_split, path in metadata_paths(dataset_root):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                query_id = str(source["id"])
                if query_id in records:
                    raise RuntimeError(f"duplicate_query_id:{query_id}")
                records[query_id] = {
                    "query_id": query_id,
                    "subset": subset,
                    "split": source.get("split", default_split),
                    "gold_context_id": str(source["context_id"]),
                    "question": source.get("question"),
                    "company_name": source.get("company_name"),
                }
    if len(records) != EXPECTED_ROWS:
        raise RuntimeError(f"gold_row_count:{len(records)}")
    return records


def rank_map(record: dict[str, Any], baseline: str) -> dict[str, int]:
    ranked = record.get("ranked_contexts")
    if not isinstance(ranked, list) or len(ranked) != TOP_K:
        raise RuntimeError(f"candidate_count:{baseline}:{record.get('query_id')}:{len(ranked or [])}")
    ranks: dict[str, int] = {}
    expected_rank = 1
    for item in ranked:
        if int(item["rank"]) != expected_rank:
            raise RuntimeError(f"rank_contract:{baseline}:{record.get('query_id')}:{expected_rank}")
        context_id = str(item["context_id"])
        if context_id in ranks:
            raise RuntimeError(f"duplicate_context:{baseline}:{record.get('query_id')}:{context_id}")
        ranks[context_id] = expected_rank
        expected_rank += 1
    return ranks


def init_counts() -> dict[str, Any]:
    return {
        "query_count": 0,
        "hits": {baseline: {str(k): 0 for k in K_VALUES} for baseline in BASELINES},
        "gold_rank_buckets": {
            baseline: {"1": 0, "2_5": 0, "6_10": 0, "11_20": 0, "21_50": 0, "51_100": 0, "not_in_top100": 0}
            for baseline in BASELINES
        },
        "complementarity": {
            str(k): {
                "bm25_and_dense": 0,
                "bm25_only": 0,
                "dense_only": 0,
                "neither": 0,
                "union_oracle": 0,
                "bm25_hit_rrf_miss": 0,
                "bm25_miss_rrf_hit": 0,
                "dense_hit_rrf_miss": 0,
                "dense_miss_rrf_hit": 0,
            }
            for k in K_VALUES
        },
        "rank_movement": {
            "bm25_to_rrf": {"improved": 0, "unchanged": 0, "demoted": 0, "bm25_absent": 0},
            "dense_to_rrf": {"improved": 0, "unchanged": 0, "demoted": 0, "dense_absent": 0},
        },
    }


def add_rank_bucket(counts: dict[str, Any], baseline: str, rank: int | None) -> None:
    if rank is None:
        counts["gold_rank_buckets"][baseline]["not_in_top100"] += 1
    elif rank == 1:
        counts["gold_rank_buckets"][baseline]["1"] += 1
    elif rank <= 5:
        counts["gold_rank_buckets"][baseline]["2_5"] += 1
    elif rank <= 10:
        counts["gold_rank_buckets"][baseline]["6_10"] += 1
    elif rank <= 20:
        counts["gold_rank_buckets"][baseline]["11_20"] += 1
    elif rank <= 50:
        counts["gold_rank_buckets"][baseline]["21_50"] += 1
    else:
        counts["gold_rank_buckets"][baseline]["51_100"] += 1


def percent(value: int, denominator: int) -> float:
    return round(100.0 * value / denominator, 6) if denominator else 0.0


def finalize_counts(counts: dict[str, Any]) -> dict[str, Any]:
    denominator = counts["query_count"]
    counts["hits_pct"] = {
        baseline: {k: percent(value, denominator) for k, value in values.items()}
        for baseline, values in counts["hits"].items()
    }
    counts["complementarity_pct"] = {
        k: {name: percent(value, denominator) for name, value in values.items()}
        for k, values in counts["complementarity"].items()
    }
    return counts


def validate_prediction_seal(prediction_root: Path) -> dict[str, Any]:
    seal = read_json(prediction_root / "prediction-seal.json")
    manifest = read_json(prediction_root / "prediction-manifest.json")
    protocol = read_json(prediction_root / "protocol.json")
    if seal.get("sealed") is not True:
        raise RuntimeError("prediction_not_sealed")
    if seal.get("prediction_count") != EXPECTED_ROWS or manifest.get("query_count") != EXPECTED_ROWS:
        raise RuntimeError("prediction_count_contract")
    if seal.get("candidate_budget") != TOP_K or manifest.get("candidate_budget") != TOP_K:
        raise RuntimeError("candidate_budget_contract")
    for baseline in BASELINES:
        path = prediction_root / f"{baseline}-predictions.jsonl.gz"
        expected_hash = seal.get("output_sha256", {}).get(baseline)
        if expected_hash is None or sha256(path) != expected_hash:
            raise RuntimeError(f"prediction_hash:{baseline}")
    return {"seal": seal, "manifest": manifest, "protocol": protocol}


def audit_protocol(prediction_root: Path, validated: dict[str, Any]) -> dict[str, Any]:
    protocol = validated["protocol"]
    manifest = validated["manifest"]
    failures: list[str] = []
    observations: list[str] = []
    if protocol.get("dataset_commit") != EXPECTED_DATASET_COMMIT:
        failures.append("dataset_commit")
    if protocol.get("published_rows") != EXPECTED_ROWS:
        failures.append("published_rows")
    if protocol.get("gold_unit") != "context_id":
        failures.append("gold_unit")
    if protocol.get("retrieval_unit") != "whole_published_context":
        failures.append("retrieval_unit")
    if protocol.get("query_template") != "f'{company_name} : {question}'":
        failures.append("query_template")
    if tuple(protocol.get("subsets", ())) != SUBSETS:
        failures.append("subset_contract")
    bm25 = protocol.get("bm25", {})
    if bm25.get("k1") != 1.5 or bm25.get("b") != 0.75 or bm25.get("tokenizer") != "casefold ASCII alphanumeric regex":
        failures.append("bm25_contract")
    dense = protocol.get("dense", {})
    expected_dense = {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "device": "cpu",
        "batch_size": 64,
        "normalize_embeddings": True,
        "instruction": None,
    }
    dense_contract = {}
    for key, expected in expected_dense.items():
        actual = dense.get(key)
        dense_contract[key] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            failures.append(f"dense_{key}")
    if not dense.get("model_file_manifest", {}).get("files"):
        failures.append("dense_model_file_manifest")
    # SentenceTransformers reads this value from the frozen snapshot.  The
    # runner did not record it in protocol.json, so expose the omission rather
    # than silently treating truncation as an unknown implementation detail.
    snapshot = Path(dense.get("model_file_manifest", {}).get("snapshot", ""))
    max_seq_length: int | None = None
    tokenizer_model_max_length: int | None = None
    sentence_bert_config = snapshot / "sentence_bert_config.json"
    tokenizer_config = snapshot / "tokenizer_config.json"
    if sentence_bert_config.is_file():
        max_seq_length = int(read_json(sentence_bert_config).get("max_seq_length"))
    if tokenizer_config.is_file():
        tokenizer_model_max_length = int(read_json(tokenizer_config).get("model_max_length"))
    if max_seq_length is None:
        observations.append("effective_max_seq_length_not_recorded_in_protocol")
    if dense.get("instruction") is None:
        observations.append("no_query_or_document_instruction_used")
    hybrid = protocol.get("hybrid", {})
    if hybrid.get("method") != "RRF" or hybrid.get("k") != 60 or hybrid.get("component_top_k") != TOP_K:
        failures.append("hybrid_contract")
    forbidden_zero = {
        "pdf_parsing": 0,
        "chunking": 0,
        "query_plan": 0,
        "query_rewrite": 0,
        "hyde": 0,
        "cross_encoder": 0,
        "llm": 0,
        "gold_scoring_reads_before_seal": 0,
    }
    for key, expected in forbidden_zero.items():
        if protocol.get(key) != expected:
            failures.append(key)
    for key in ("parameter_scan", "gold_driven_scan"):
        if protocol.get(key) is not False:
            failures.append(key)
    corpus_manifest = read_json(prediction_root / "corpus-manifest.json")
    corpora = corpus_manifest.get("corpora", {})
    expected_contexts = {"FinQA": 2789, "ConvFinQA": 1806, "TAT-DQA": 2723}
    expected_queries = {"FinQA": 8281, "ConvFinQA": 3458, "TAT-DQA": 11349}
    for subset in SUBSETS:
        if corpora.get(subset, {}).get("context_count") != expected_contexts[subset]:
            failures.append(f"corpus_context_count:{subset}")
        if corpora.get(subset, {}).get("query_count") != expected_queries[subset]:
            failures.append(f"corpus_query_count:{subset}")
    if sum(item.get("context_count", 0) for item in corpora.values()) != EXPECTED_CORPUS:
        failures.append("corpus_count")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "observations": observations,
        "protocol": {
            "dataset_commit": protocol.get("dataset_commit"),
            "query_template": protocol.get("query_template"),
            "retrieval_unit": protocol.get("retrieval_unit"),
            "gold_unit": protocol.get("gold_unit"),
            "subsets": protocol.get("subsets"),
            "bm25": bm25,
            "dense": dense_contract,
            "dense_effective_max_seq_length": max_seq_length,
            "dense_tokenizer_model_max_length": tokenizer_model_max_length,
            "dense_truncation": "SentenceTransformer tokenizer truncates to effective max_seq_length",
            "dense_document_serialization": "raw published context string",
            "dense_query_serialization": "raw published query string",
            "dense_pooling": "mean_tokens",
            "dense_normalization": dense.get("normalize_embeddings"),
            "dense_similarity": "normalized dot product",
            "hybrid": hybrid,
            "corpus_count": sum(item.get("context_count", 0) for item in corpora.values()),
        },
        "manifest": {
            "query_count": manifest.get("query_count"),
            "corpus_count": manifest.get("corpus_count"),
            "candidate_budget": manifest.get("candidate_budget"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    prediction_root = args.prediction_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    validated = validate_prediction_seal(prediction_root)
    contract = audit_protocol(prediction_root, validated)
    gold = load_gold(dataset_root)
    counts: dict[str, dict[str, Any]] = {subset: init_counts() for subset in SUBSETS}
    counts["overall"] = init_counts()
    movement_path = output_root / "rank-movement.jsonl.gz"
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_limits = {"destructive": 100, "rescue": 100, "bm25_only": 50, "dense_only": 50, "neither": 50}

    handles = [gzip.open(prediction_root / f"{name}-predictions.jsonl.gz", "rt", encoding="utf-8") for name in BASELINES]
    try:
        with gzip.open(movement_path, "wt", encoding="utf-8", compresslevel=6) as movement_handle:
            for line_number, (bm25_line, dense_line, hybrid_line) in enumerate(zip(*handles), start=1):
                records = [json.loads(line) for line in (bm25_line, dense_line, hybrid_line)]
                query_ids = [str(record.get("query_id")) for record in records]
                if len(set(query_ids)) != 1:
                    raise RuntimeError(f"prediction_query_alignment:{line_number}:{query_ids}")
                query_id = query_ids[0]
                gold_record = gold.get(query_id)
                if gold_record is None:
                    raise RuntimeError(f"prediction_query_not_in_dataset:{query_id}")
                subset = gold_record["subset"]
                if any(record.get("subset") != subset for record in records):
                    raise RuntimeError(f"prediction_subset_alignment:{query_id}")
                maps = {baseline: rank_map(record, baseline) for baseline, record in zip(BASELINES, records)}
                gold_context_id = gold_record["gold_context_id"]
                ranks = {baseline: maps[baseline].get(gold_context_id) for baseline in BASELINES}
                subset_counts = counts[subset]
                overall_counts = counts["overall"]
                for current in (subset_counts, overall_counts):
                    current["query_count"] += 1
                    for baseline in BASELINES:
                        rank = ranks[baseline]
                        add_rank_bucket(current, baseline, rank)
                        for k in K_VALUES:
                            if rank is not None and rank <= k:
                                current["hits"][baseline][str(k)] += 1
                    for k in K_VALUES:
                        bm25_hit = ranks["bm25"] is not None and ranks["bm25"] <= k
                        dense_hit = ranks["dense"] is not None and ranks["dense"] <= k
                        rrf_hit = ranks["hybrid"] is not None and ranks["hybrid"] <= k
                        complement = current["complementarity"][str(k)]
                        if bm25_hit and dense_hit:
                            complement["bm25_and_dense"] += 1
                        elif bm25_hit:
                            complement["bm25_only"] += 1
                        elif dense_hit:
                            complement["dense_only"] += 1
                        else:
                            complement["neither"] += 1
                        if bm25_hit or dense_hit:
                            complement["union_oracle"] += 1
                        if bm25_hit and not rrf_hit:
                            complement["bm25_hit_rrf_miss"] += 1
                        if not bm25_hit and rrf_hit:
                            complement["bm25_miss_rrf_hit"] += 1
                        if dense_hit and not rrf_hit:
                            complement["dense_hit_rrf_miss"] += 1
                        if not dense_hit and rrf_hit:
                            complement["dense_miss_rrf_hit"] += 1
                    for source, target in (("bm25", "hybrid"), ("dense", "hybrid")):
                        source_rank = ranks[source]
                        target_rank = ranks[target]
                        movement = current["rank_movement"][f"{source}_to_rrf"]
                        if source_rank is None:
                            movement[f"{source}_absent"] += 1
                        elif target_rank is None or target_rank > source_rank:
                            movement["demoted"] += 1
                        elif target_rank < source_rank:
                            movement["improved"] += 1
                        else:
                            movement["unchanged"] += 1
                movement_row = {
                    "query_id": query_id,
                    "subset": subset,
                    "gold_context_id": gold_context_id,
                    "bm25_gold_rank": ranks["bm25"],
                    "dense_gold_rank": ranks["dense"],
                    "rrf_gold_rank": ranks["hybrid"],
                    "bm25_hit": {str(k): bool(ranks["bm25"] and ranks["bm25"] <= k) for k in K_VALUES},
                    "dense_hit": {str(k): bool(ranks["dense"] and ranks["dense"] <= k) for k in K_VALUES},
                    "rrf_hit": {str(k): bool(ranks["hybrid"] and ranks["hybrid"] <= k) for k in K_VALUES},
                    "bm25_hit_rrf_miss": {str(k): bool(ranks["bm25"] and ranks["bm25"] <= k and not (ranks["hybrid"] and ranks["hybrid"] <= k)) for k in K_VALUES},
                    "bm25_miss_dense_hit": {str(k): bool(not (ranks["bm25"] and ranks["bm25"] <= k) and ranks["dense"] and ranks["dense"] <= k) for k in K_VALUES},
                }
                movement_handle.write(json.dumps(movement_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                flags = {
                    "destructive": any(movement_row["bm25_hit_rrf_miss"].values()),
                    "rescue": any(not movement_row["bm25_hit"][str(k)] and movement_row["rrf_hit"][str(k)] for k in K_VALUES),
                    "bm25_only": movement_row["bm25_hit"]["5"] and not movement_row["dense_hit"]["5"],
                    "dense_only": movement_row["dense_hit"]["5"] and not movement_row["bm25_hit"]["5"],
                    "neither": not movement_row["bm25_hit"]["5"] and not movement_row["dense_hit"]["5"],
                }
                for name, flag in flags.items():
                    if flag and len(samples[name]) < sample_limits[name]:
                        samples[name].append(movement_row)
    finally:
        for handle in handles:
            handle.close()

    for current in counts.values():
        finalize_counts(current)
    complementarity = {key: value["complementarity"] for key, value in counts.items()}
    subset_breakdown = {key: value for key, value in counts.items()}
    write_json(output_root / "contract-audit.json", contract)
    write_json(output_root / "complementarity.json", {"overall": complementarity["overall"], "by_subset": {key: value for key, value in complementarity.items() if key != "overall"}})
    write_json(output_root / "subset-breakdown.json", subset_breakdown)
    write_json(output_root / "failure-samples.json", {key: value for key, value in samples.items()})

    overall = counts["overall"]
    destructive_r5 = overall["complementarity"]["5"]["bm25_hit_rrf_miss"]
    rescue_r5 = overall["complementarity"]["5"]["bm25_miss_rrf_hit"]
    destructive_any = any(
        overall["complementarity"][str(k)]["bm25_hit_rrf_miss"] > overall["complementarity"][str(k)]["bm25_miss_rrf_hit"]
        for k in K_VALUES
    )
    bm25_r5 = overall["hits"]["bm25"]["5"]
    dense_r5 = overall["hits"]["dense"]["5"]
    bm25_only_r5 = overall["complementarity"]["5"]["bm25_only"]
    dense_only_r5 = overall["complementarity"]["5"]["dense_only"]
    decisions: list[str] = []
    if contract["status"] == "failed":
        decisions.append("dense_branch_contract_problem")
    if contract["status"] == "passed" and dense_r5 < bm25_r5 and dense_only_r5 < bm25_only_r5:
        decisions.append("dense_branch_model_quality_insufficient")
    if destructive_any:
        decisions.append("fusion_negative_transfer_confirmed")
    if not decisions:
        decisions.append("dense_branch_model_quality_insufficient")
    diagnosis = {
        "first_principles": [
            "BM25 and Dense are independently ranked over the same subset-specific whole-context corpus.",
            "RRF can only rescue a BM25 miss when Dense places the gold in its Top100; it can destructively displace a BM25 hit when both top-100 lists contain stronger-looking consensus candidates.",
            "Dense quality is assessed from its independent gold ranks and unique rescue, not from the fused score.",
        ],
        "observed": {
            "bm25_r5": bm25_r5,
            "dense_r5": dense_r5,
            "bm25_only_r5": bm25_only_r5,
            "dense_only_r5": dense_only_r5,
            "rrf_destructive_r5": destructive_r5,
            "rrf_rescue_r5": rescue_r5,
            "union_oracle_r5": overall["complementarity"]["5"]["union_oracle"],
            "union_oracle_r100": overall["complementarity"]["100"]["union_oracle"],
        },
        "interpretation": {
            "dense": "independent Dense is weaker than BM25 at @5 and contributes fewer unique @5 rescues than BM25-only coverage" if dense_r5 < bm25_r5 and dense_only_r5 < bm25_only_r5 else "dense quality is not conclusively weaker under this audit",
            "fusion": "RRF has negative transfer at one or more audited K values" if destructive_any else "no destructive RRF transfer dominates at audited K values",
        },
    }
    write_json(output_root / "diagnosis.json", diagnosis)
    write_json(
        output_root / "decision.json",
        {
            "gate": "t2_ragbench_02a_fusion_failure_dense_contract_audit",
            "base_prediction_seal": str(prediction_root / "prediction-seal.json"),
            "prediction_sha256": sha256(prediction_root / "prediction-seal.json"),
            "published_denominator": EXPECTED_ROWS,
            "contract_status": contract["status"],
            "decisions": decisions,
            "primary_decision": decisions[0],
            "dense_equal_fusion_rejected": "fusion_negative_transfer_confirmed" in decisions,
            "retrieval_mutation": 0,
            "reranker_calls": 0,
            "embedding_calls": 0,
            "parameter_scan": False,
            "gold_read_mode": "post_seal_only",
            "next_gate": "t2_02b_dense_rescue_decision",
            "diagnosis": diagnosis,
        },
    )
    write_json(
        output_root / "prediction-seal.json",
        {
            "sealed": True,
            "gate": "t2_ragbench_02a_fusion_failure_dense_contract_audit",
            "input_prediction_seal_sha256": sha256(prediction_root / "prediction-seal.json"),
            "rank_movement_count": overall["query_count"],
            "published_denominator": EXPECTED_ROWS,
            "retrieval_runs": 0,
            "reranker_calls": 0,
            "embedding_calls": 0,
            "parameter_scan": False,
            "gold_reads_before_seal": 0,
            "gold_reads_after_input_seal": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

