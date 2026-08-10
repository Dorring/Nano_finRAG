#!/usr/bin/env python3
"""T2-04B.1 PCR-V1 reproduction and method freeze."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SEAL_SHA = "98204451ca98046fb7bed2338ad346f511b10f90195b6e7d78f084c52131641d"
DATASET_COMMIT = "adf7fe1541ac37351ce1142544d8e3b43010ed92"
COUNTS = {"train": 15314, "dev": 2025, "test": 2291}
SUBSETS = ("FinQA", "TAT-DQA")
KS = (1, 3, 5, 10, 20, 50)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_obj(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["query_id"])].append(row)
    return dict(output)


def load_sealed_inputs(feature_root: Path) -> tuple[
    dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    seal_path = feature_root / "feature-seal.json"
    if sha256_file(seal_path) != SEAL_SHA:
        raise RuntimeError("a2_feature_seal_mismatch")
    seal = read_json(seal_path)
    if seal["candidate_depth"] != 50 or seal["candidate_mutation"] != 0:
        raise RuntimeError("a2_candidate_contract")
    if seal["retrieval_rerun"] or seal["model_execution"]:
        raise RuntimeError("a2_runtime_contract")
    for name, expected in seal["feature_files"].items():
        if sha256_file(feature_root / name) != expected:
            raise RuntimeError(f"a2_feature_hash_mismatch:{name}")
    query_map = {}
    for row in read_gz(feature_root / "query-structure.jsonl.gz"):
        qid = str(row["query_id"])
        if qid in query_map or row["split"] not in {"train", "dev"}:
            raise RuntimeError("a2_query_structure_contract")
        query_map[qid] = row
    rows = read_gz(feature_root / "dev-candidate-features.jsonl.gz")
    by_query = groups(rows)
    if len(by_query) != COUNTS["dev"]:
        raise RuntimeError("a2_dev_count_contract")
    for qid, candidates in by_query.items():
        if query_map[qid]["split"] != "dev":
            raise RuntimeError("a2_split_mismatch")
        ranks = sorted(int(row["bm25_rank"]) for row in candidates)
        ids = [str(row["candidate_context_id"]) for row in candidates]
        if ranks != list(range(1, 51)) or len(set(ids)) != 50:
            raise RuntimeError("a2_top50_contract")
    return query_map, rows, seal


def load_dev_gold(dataset_root: Path, query_ids: set[str]) -> dict[str, str]:
    targets: dict[str, str] = {}
    for subset in SUBSETS:
        path = dataset_root / "data" / subset / "dev" / "metadata.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    qid = str(row["id"])
                    if qid in query_ids:
                        targets[qid] = str(row["context_id"])
    if set(targets) != query_ids:
        raise RuntimeError("dev_gold_identity_mismatch")
    return targets


def bm25_order(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        qid: [
            str(row["candidate_context_id"])
            for row in sorted(
                candidates,
                key=lambda r: (int(r["bm25_rank"]), str(r["candidate_context_id"])),
            )
        ]
        for qid, candidates in groups(rows).items()
    }


def pcr_order(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        qid: [
            str(row["candidate_context_id"])
            for row in sorted(
                candidates,
                key=lambda r: (
                    -float(r["features"]["required_period_coverage"]),
                    int(r["bm25_rank"]),
                    str(r["candidate_context_id"]),
                ),
            )
        ]
        for qid, candidates in groups(rows).items()
    }


def score(order: dict[str, list[str]], qids: list[str], gold: dict[str, str]) -> dict[str, Any]:
    hits = {str(k): 0 for k in KS}
    mrr = ndcg = 0.0
    for qid in qids:
        try:
            rank = order[qid].index(gold[qid]) + 1
        except ValueError:
            rank = None
        for cutoff in KS:
            hits[str(cutoff)] += int(rank is not None and rank <= cutoff)
        if rank is not None and rank <= 5:
            mrr += 1.0 / rank
        if rank is not None and rank <= 10:
            ndcg += 1.0 / math.log2(rank + 1)
    n = len(qids)
    return {
        "count": n,
        "hits": hits,
        "recall": {f"@{k}": f"{hits[str(k)]}/{n}" for k in KS},
        "recall_pct": {f"@{k}": round(100 * hits[str(k)] / n, 6) for k in KS},
        "mrr_at_5_pct": round(100 * mrr / n, 6),
        "ndcg_at_10_pct": round(100 * ndcg / n, 6),
    }


def movement(
    baseline: dict[str, list[str]],
    pcr: dict[str, list[str]],
    gold: dict[str, str],
) -> dict[str, int]:
    rescued = damaged = unchanged = 0
    for qid, gid in gold.items():
        before = gid in baseline[qid][:5]
        after = gid in pcr[qid][:5]
        if after and not before:
            rescued += 1
        elif before and not after:
            damaged += 1
        else:
            unchanged += 1
    return {
        "rescued_at_5": rescued,
        "damaged_at_5": damaged,
        "net_top5_gain": rescued - damaged,
        "unchanged_at_5": unchanged,
    }


def scoped_metrics(
    query_map: dict[str, dict[str, Any]],
    gold: dict[str, str],
    baseline: dict[str, list[str]],
    pcr: dict[str, list[str]],
) -> dict[str, Any]:
    output = {}
    for subset in SUBSETS:
        qids = sorted(
            qid for qid, row in query_map.items()
            if row["split"] == "dev" and row["subset"] == subset
        )
        output[subset] = {
            "query_count": len(qids),
            "bm25": score({qid: baseline[qid] for qid in qids}, qids, gold),
            "pcr_v1": score({qid: pcr[qid] for qid in qids}, qids, gold),
        }
    return output


def query_type_metrics(
    query_map: dict[str, dict[str, Any]],
    gold: dict[str, str],
    baseline: dict[str, list[str]],
    pcr: dict[str, list[str]],
) -> dict[str, Any]:
    output = {}
    for operation in sorted({row["operation_intent"] for row in query_map.values()}):
        qids = sorted(
            qid for qid, row in query_map.items()
            if row["split"] == "dev" and row["operation_intent"] == operation
        )
        if not qids:
            continue
        base_order = {qid: baseline[qid] for qid in qids}
        pcr_ordered = {qid: pcr[qid] for qid in qids}
        output[operation] = {
            "query_count": len(qids),
            "bm25": score(base_order, qids, gold),
            "pcr_v1": score(pcr_ordered, qids, gold),
            "movement": movement(base_order, pcr_ordered, {qid: gold[qid] for qid in qids}),
        }
    return output


def main(dataset_root: Path, feature_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    query_map, rows, seal = load_sealed_inputs(feature_root)
    qids = sorted(
        qid for qid, row in query_map.items()
        if row["split"] == "dev"
    )
    if len(qids) != COUNTS["dev"]:
        raise RuntimeError("dev_query_count_contract")
    baseline = bm25_order(rows)
    pcr = pcr_order(rows)
    for qid in qids:
        if set(baseline[qid]) != set(pcr[qid]):
            raise RuntimeError("pcr_candidate_mutation")
    no_period_qids = [
        qid for qid in qids if not query_map[qid].get("periods")
    ]
    no_period_invariant = all(
        baseline[qid] == pcr[qid] for qid in no_period_qids
    )
    contract = {
        "gate": "T2-04B.1",
        "method": "Financial Period-Consistency Reranker V1",
        "short_name": "PCR-V1",
        "candidate_source": "frozen_bm25_top50",
        "candidate_depth": 50,
        "feature": "required_period_coverage",
        "ranking": [
            "required_period_coverage_desc",
            "original_bm25_rank_asc",
            "context_id_asc",
        ],
        "no_period_behavior": "preserve_bm25_order",
        "no_period_query_count": len(no_period_qids),
        "no_period_order_invariant": no_period_invariant,
        "feature_seal": SEAL_SHA,
        "new_feature_extraction": False,
        "feature_weight_search": False,
        "scope_search": False,
        "test_and_transfer_locked": True,
    }
    write_json(output_root / "pcr-v1-contract.json", contract)
    # Gold is read only after sealed input hashes and PCR candidate invariants.
    gold = load_dev_gold(dataset_root, set(qids))
    base_metrics = score(baseline, qids, gold)
    pcr_metrics = score(pcr, qids, gold)
    move = movement(baseline, pcr, gold)
    reproduction_passed = (
        pcr_metrics["hits"]["5"] == 1478
        and move == {
            "rescued_at_5": 64,
            "damaged_at_5": 33,
            "net_top5_gain": 31,
            "unchanged_at_5": 1928,
        }
        and pcr_metrics["hits"]["50"] == 1857
    )
    dev_reproduction = {
        "expected": {
            "bm25_r_at_5_pct": 71.45679,
            "pcr_r_at_5_pct": 72.987654,
            "rescued_at_5": 64,
            "damaged_at_5": 33,
            "net_top5_gain": 31,
            "r_at_50": "1857/2025",
        },
        "bm25": base_metrics,
        "pcr_v1": pcr_metrics,
        "movement": move,
        "candidate_set_invariant": True,
        "r_at_50_invariant": pcr_metrics["hits"]["50"] == base_metrics["hits"]["50"],
        "reproduction_passed": reproduction_passed,
    }
    write_json(output_root / "dev-reproduction.json", dev_reproduction)
    subset = scoped_metrics(query_map, gold, baseline, pcr)
    write_json(output_root / "subset-analysis.json", subset)
    write_json(
        output_root / "query-type-analysis.json",
        query_type_metrics(query_map, gold, baseline, pcr),
    )
    write_json(output_root / "rank-movement.json", {
        "baseline": "bm25",
        "method": "pcr_v1",
        "movement": move,
        "candidate_depth": 50,
    })
    selected = reproduction_passed and (
        pcr_metrics["recall_pct"]["@5"] > base_metrics["recall_pct"]["@5"]
        and move["net_top5_gain"] > 0
    )
    selected_method = {
        "method": "Financial Period-Consistency Reranker V1",
        "short_name": "PCR-V1",
        "candidate_source": "frozen_bm25_top50",
        "feature": "required_period_coverage",
        "ranking": [
            "required_period_coverage_desc",
            "original_bm25_rank_asc",
            "context_id_asc",
        ],
        "no_period_behavior": "preserve_bm25_order",
        "feature_seal": SEAL_SHA,
        "dev_recall_at_5": pcr_metrics["recall_pct"]["@5"],
        "test_gold_used": False,
        "convfinqa_gold_used": False,
        "selected": selected,
    }
    write_json(output_root / "selected-method.json", selected_method)
    manifest = {
        "method": "financial_period_consistency_reranker_v1",
        "selected": selected,
        "feature_seal": SEAL_SHA,
        "candidate_depth": 50,
        "ranking": selected_method["ranking"],
        "no_period_behavior": "preserve_bm25_order",
        "dev_reproduction_sha256": sha256_file(output_root / "dev-reproduction.json"),
        "primary_test_gold_reads": 0,
        "convfinqa_gold_reads": 0,
    }
    write_json(output_root / "selected-method-manifest.json", manifest)
    method_hash = sha256_obj(manifest)
    (output_root / "selected-method-sha256.txt").write_text(
        method_hash + "\n", encoding="utf-8"
    )
    decision = {
        "gate": "T2-04B.1",
        "base_commit": "8e7d4d4",
        "dataset_commit": DATASET_COMMIT,
        "feature_seal": SEAL_SHA,
        "primary_train_queries": COUNTS["train"],
        "primary_dev_queries": COUNTS["dev"],
        "primary_test_queries": COUNTS["test"],
        "candidate_depth": 50,
        "retrieval_rerun": False,
        "new_feature_extraction": False,
        "primary_test_gold_reads": 0,
        "convfinqa_gold_reads": 0,
        "pcr_v1_freeze_allowed": reproduction_passed,
        "pcr_v1_selected": selected,
        "historical_t2_04b_structure_reranker_selected": False,
        "structure_reranker_selected": selected,
        "selected_method": (
            "financial_period_consistency_reranker_v1" if selected else None
        ),
        "selected_method_hash": method_hash if selected else None,
        "next_gate": "t2_04c_frozen_test_evaluation" if selected else "t2_04_method_reconsideration",
        "decision_reason": [
            "PCR-V1 was reproduced solely from the sealed A.2 feature artifact.",
            "Multi-feature B1/B2 methods remain rejected as historical T2-04B results.",
            "Primary Test and ConvFinQA Gold were not read.",
        ],
    }
    write_json(output_root / "decision.json", decision)
    (output_root / "README.md").write_text(
        "# T2-04B.1 PCR-V1 Freeze\n\n"
        "PCR-V1 reuses the sealed required_period_coverage feature and preserves "
        "BM25 order within coverage buckets. Test and ConvFinQA Gold remain locked.\n",
        encoding="utf-8",
    )
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(main(args.dataset_root, args.feature_root, args.output_root), sort_keys=True))
