#!/usr/bin/env python3
"""T2-04B guarded financial structure-aware reranker."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_SEAL = (
    "98204451ca98046fb7bed2338ad346f511b10f90195b6e7d78f084c52131641d"
)
DATASET_COMMIT = "adf7fe1541ac37351ce1142544d8e3b43010ed92"
COUNTS = {"train": 15314, "dev": 2025, "test": 2291}
SUBSETS = ("FinQA", "TAT-DQA")
KS = (1, 3, 5, 10, 20, 50)
CALC_OPS = {
    "difference", "sum", "ratio", "percentage", "percentage_change",
    "average", "comparison", "multi_operand_other",
}
B2_FEATURES = (
    "bm25_rank", "metric_exact_match", "metric_normalized_match",
    "period_any_match", "required_period_coverage", "period_in_table_header",
    "metric_in_row_label", "row_header_coherence", "multi_period_coverage",
    "operation_evidence_compatibility",
)
NEGATIVE_RANKS = (11, 21, 31, 41, 50)


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


def load_inputs(root: Path) -> tuple[
    dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    seal_path = root / "feature-seal.json"
    if sha256_file(seal_path) != EXPECTED_SEAL:
        raise RuntimeError("feature_seal_mismatch")
    seal = read_json(seal_path)
    if seal.get("candidate_depth") != 50 or seal.get("candidate_mutation") != 0:
        raise RuntimeError("feature_contract_mismatch")
    if seal.get("retrieval_rerun") or seal.get("model_execution"):
        raise RuntimeError("feature_generation_changed")
    for name, expected in seal["feature_files"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"feature_hash_mismatch:{name}")
    queries = {}
    for row in read_gz(root / "query-structure.jsonl.gz"):
        qid = str(row["query_id"])
        if qid in queries or row.get("split") not in {"train", "dev"}:
            raise RuntimeError("query_structure_contract")
        queries[qid] = row
    train = read_gz(root / "train-candidate-features.jsonl.gz")
    dev = read_gz(root / "dev-candidate-features.jsonl.gz")
    if len(queries) != COUNTS["train"] + COUNTS["dev"]:
        raise RuntimeError("query_count_contract")
    for rows, split, expected in ((train, "train", COUNTS["train"]), (dev, "dev", COUNTS["dev"])):
        by_query = groups(rows)
        if len(by_query) != expected:
            raise RuntimeError(f"{split}_count_contract")
        for qid, candidates in by_query.items():
            if qid not in queries or queries[qid]["split"] != split:
                raise RuntimeError("candidate_query_contract")
            ranks = sorted(int(row["bm25_rank"]) for row in candidates)
            ids = [str(row["candidate_context_id"]) for row in candidates]
            if ranks != list(range(1, 51)) or len(set(ids)) != 50:
                raise RuntimeError("bm25_top50_contract")
    return queries, train, dev, seal


def load_gold(dataset_root: Path, split: str, query_ids: set[str]) -> dict[str, str]:
    if split not in {"train", "dev"}:
        raise ValueError("test_and_transfer_gold_forbidden")
    targets: dict[str, str] = {}
    for subset in SUBSETS:
        path = dataset_root / "data" / subset / split / "metadata.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    qid = str(row["id"])
                    if qid in query_ids:
                        targets[qid] = str(row["context_id"])
    if set(targets) != query_ids:
        raise RuntimeError(f"{split}_gold_identity_mismatch")
    return targets


def bm25_order(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        qid: [
            str(row["candidate_context_id"])
            for row in sorted(
                candidates,
                key=lambda row: (int(row["bm25_rank"]), str(row["candidate_context_id"])),
            )
        ]
        for qid, candidates in groups(rows).items()
    }


def feature_order(rows: list[dict[str, Any]], feature: str) -> dict[str, list[str]]:
    return {
        qid: [
            str(row["candidate_context_id"])
            for row in sorted(
                candidates,
                key=lambda row: (
                    -float(row["features"][feature]),
                    int(row["bm25_rank"]),
                    str(row["candidate_context_id"]),
                ),
            )
        ]
        for qid, candidates in groups(rows).items()
    }


def score(orders: dict[str, list[str]], qids: list[str], gold: dict[str, str]) -> dict[str, Any]:
    hits = {str(k): 0 for k in KS}
    mrr = ndcg = 0.0
    for qid in qids:
        try:
            rank = orders[qid].index(gold[qid]) + 1
        except ValueError:
            rank = None
        for k in KS:
            hits[str(k)] += int(rank is not None and rank <= k)
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
        "mrr_at_5_pct": round(100 * mrr / n, 6) if n else 0.0,
        "ndcg_at_10_pct": round(100 * ndcg / n, 6) if n else 0.0,
    }


def movement(base: dict[str, list[str]], order: dict[str, list[str]], gold: dict[str, str]) -> dict[str, int]:
    rescued = damaged = unchanged = 0
    for qid, gid in gold.items():
        before = gid in base[qid][:5]
        after = gid in order[qid][:5]
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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def displacement(base: dict[str, list[str]], order: dict[str, list[str]], qids: list[str]) -> dict[str, Any]:
    values: list[float] = []
    top5 = top10 = 0
    for qid in qids:
        old = {cid: i + 1 for i, cid in enumerate(base[qid])}
        new = {cid: i + 1 for i, cid in enumerate(order[qid])}
        values.extend(abs(old[cid] - new[cid]) for cid in old)
        top5 += len(set(base[qid][:5]) - set(order[qid][:5]))
        top10 += len(set(base[qid][:10]) - set(order[qid][:10]))
    return {
        "mean_absolute_rank_displacement": sum(values) / len(values),
        "median_rank_displacement": statistics.median(values),
        "p95_rank_displacement": percentile(values, 0.95),
        "top5_candidates_displaced_out": top5,
        "top10_candidates_displaced_out": top10,
    }


def strong_metric(row: dict[str, Any]) -> bool:
    f = row["features"]
    return f["metric_exact_match"] == 1 or f["metric_normalized_match"] == 1


def promotion_tier(row: dict[str, Any], query: dict[str, Any]) -> str | None:
    f = row["features"]
    has_period_req = bool(query.get("period_requirement_present"))
    if has_period_req and float(f["required_period_coverage"]) == 0.0:
        return None
    if query.get("normalized_metric_terms") and not strong_metric(row):
        return None
    metric = strong_metric(row)
    period = (not has_period_req) or float(f["required_period_coverage"]) == 1.0
    table = f["row_header_coherence"] == 1 or (
        f["metric_in_row_label"] == 1 and float(f["period_in_table_header"]) > 0
    )
    operation = f["operation_evidence_compatibility"] == 1
    op = query.get("operation_intent", "direct_fact")
    if metric and period and table:
        return "A"
    if op in CALC_OPS and metric and float(f["required_period_coverage"]) == 1.0 and operation:
        return "B"
    if op == "direct_fact" and metric and period:
        return "C"
    return None


def guarded_order(
    rows: list[dict[str, Any]],
    queries: dict[str, dict[str, Any]],
    max_rank: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    result = {}
    eligible_total = promoted_total = queries_total = 0
    tiers = {"A": 0, "B": 0, "C": 0}
    for qid, candidates in groups(rows).items():
        ordered = sorted(candidates, key=lambda r: (int(r["bm25_rank"]), str(r["candidate_context_id"])))
        eligible = [
            row for row in ordered
            if 6 <= int(row["bm25_rank"]) <= max_rank
            and promotion_tier(row, queries[qid]) is not None
        ]
        eligible_total += len(eligible)
        eligible = eligible[:1]
        for row in eligible:
            tiers[promotion_tier(row, queries[qid])] += 1
        promoted_total += len(eligible)
        queries_total += int(bool(eligible))
        eligible_ids = {id(row) for row in eligible}
        remaining = [
            row for row in ordered if id(row) not in eligible_ids
        ]
        promoted = [
            str(row["candidate_context_id"]) for row in eligible
        ]
        result[qid] = [
            str(row["candidate_context_id"]) for row in remaining[:4]
        ] + promoted + [
            str(row["candidate_context_id"]) for row in remaining[4:]
        ]
    return result, {
        "max_promotion_rank": max_rank,
        "eligible_candidate_count": eligible_total,
        "promoted_candidate_count": promoted_total,
        "promoted_query_count": queries_total,
        "max_promotions_per_query": 1,
        "promotion_position": "rank5_boundary",
        "tier_counts": tiers,
    }


def vector(row: dict[str, Any]) -> list[float]:
    f = row["features"]
    return [
        -float(row["bm25_rank"]),
        float(f["metric_exact_match"]),
        float(f["metric_normalized_match"]),
        float(f["period_any_match"]),
        float(f["required_period_coverage"]),
        float(f["period_in_table_header"]),
        float(f["metric_in_row_label"]),
        float(f["row_header_coherence"]),
        float(f["multi_period_coverage"]),
        float(f["operation_evidence_compatibility"]),
    ]


def fit_b2(rows: list[dict[str, Any]], gold: dict[str, str]) -> tuple[dict[str, Any], Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    x_values: list[list[float]] = []
    labels: list[int] = []
    by_query = groups(rows)
    gold_top50 = pairs = 0
    for qid in sorted(by_query):
        candidates = by_query[qid]
        positive = next(
            (row for row in candidates if str(row["candidate_context_id"]) == gold[qid]),
            None,
        )
        if positive is None:
            continue
        gold_top50 += 1
        negatives = [
            row for row in candidates
            if row is not positive and (
                int(row["bm25_rank"]) <= 10 or int(row["bm25_rank"]) in NEGATIVE_RANKS
            )
        ]
        p = np.asarray(vector(positive), dtype=float)
        for negative in negatives:
            difference = p - np.asarray(vector(negative), dtype=float)
            x_values.extend([difference.tolist(), (-difference).tolist()])
            labels.extend([1, 0])
            pairs += 1
    if not x_values:
        raise RuntimeError("b2_training_pairs_empty")
    model = LogisticRegression(
        C=1.0, fit_intercept=False, max_iter=200, penalty="l2",
        random_state=20250810, solver="lbfgs",
    )
    model.fit(np.asarray(x_values), np.asarray(labels))
    data = {
        "model_type": "linear_pairwise_logistic",
        "features": list(B2_FEATURES),
        "feature_transform": "bm25_rank -> -rank",
        "C": 1.0,
        "fit_intercept": False,
        "max_iter": 200,
        "penalty": "l2",
        "random_seed": 20250810,
        "solver": "lbfgs",
        "train_queries_total": len(by_query),
        "train_queries_with_gold_in_top50": gold_top50,
        "negative_ranks": {"all_top10": True, "fixed_remaining": list(NEGATIVE_RANKS)},
        "negative_pairs": pairs,
        "pair_rows": len(x_values),
        "coefficients": [float(v) for v in model.coef_[0]],
        "intercept": 0.0,
    }
    data["model_hash"] = sha256_obj(data)
    return data, model


def b2_order(rows: list[dict[str, Any]], model: Any) -> dict[str, list[str]]:
    result = {}
    for qid, candidates in groups(rows).items():
        scored = [
            (
                float(model.decision_function([vector(row)])[0]),
                int(row["bm25_rank"]),
                str(row["candidate_context_id"]),
            )
            for row in candidates
        ]
        result[qid] = [cid for _, _, cid in sorted(scored, key=lambda x: (-x[0], x[1], x[2]))]
    return result


def evaluate(
    qids: list[str],
    gold: dict[str, str],
    orders: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = orders["bm25"]
    metrics = {name: score(order, qids, gold) for name, order in orders.items()}
    moves = {
        name: movement(base, order, gold)
        for name, order in orders.items() if name != "bm25"
    }
    displacement_data = {
        name: displacement(base, order, qids)
        for name, order in orders.items() if name != "bm25"
    }
    return metrics, moves, displacement_data


def scoped_metrics(
    query_map: dict[str, dict[str, Any]],
    gold: dict[str, str],
    orders: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    result = {}
    for subset in SUBSETS:
        qids = sorted(
            qid for qid, row in query_map.items()
            if row["split"] == "dev" and row["subset"] == subset
        )
        result[subset] = {
            "query_count": len(qids),
            "methods": {
                name: score({qid: order[qid] for qid in qids}, qids, gold)
                for name, order in orders.items()
            },
        }
    return result


def query_type_metrics(
    query_map: dict[str, dict[str, Any]],
    gold: dict[str, str],
    orders: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    result = {}
    for operation in sorted({row["operation_intent"] for row in query_map.values()}):
        qids = sorted(
            qid for qid, row in query_map.items()
            if row["split"] == "dev" and row["operation_intent"] == operation
        )
        if qids:
            result[operation] = {
                "query_count": len(qids),
                "methods": {
                    name: score({qid: order[qid] for qid in qids}, qids, gold)
                    for name, order in orders.items()
                },
            }
    return result


def choose(
    metrics: dict[str, Any],
    moves: dict[str, Any],
    subset: dict[str, Any],
) -> dict[str, Any]:
    base = metrics["bm25"]["recall_pct"]["@5"]
    candidates = []
    for name in ("b1_a", "b1_b", "b2"):
        gains = [
            subset[s]["methods"][name]["recall_pct"]["@5"]
            - subset[s]["methods"]["bm25"]["recall_pct"]["@5"]
            for s in SUBSETS
        ]
        candidates.append({
            "method": name,
            "r_at_5": metrics[name]["recall_pct"]["@5"],
            "gain_pp": metrics[name]["recall_pct"]["@5"] - base,
            "net_top5_gain": moves[name]["net_top5_gain"],
            "min_subset_gain_pp": min(gains),
            "eligible_for_selection": (
                metrics[name]["recall_pct"]["@5"] > base
                and moves[name]["net_top5_gain"] > 0
                and min(gains) >= -5.0
            ),
        })
    eligible = [item for item in candidates if item["eligible_for_selection"]]
    best_b1 = max(
        (item for item in eligible if item["method"] in {"b1_a", "b1_b"}),
        key=lambda x: (x["r_at_5"], x["method"] == "b1_a"),
        default=None,
    )
    best = max(eligible, key=lambda x: x["r_at_5"], default=None)
    if best_b1 and best and best["method"] == "b2" and best["r_at_5"] - best_b1["r_at_5"] < 0.5:
        selected = best_b1
    else:
        selected = best
    return {
        "selected_method": selected["method"] if selected else None,
        "candidates": candidates,
        "selection_rule": {
            "min_dev_gain_pp": 0.0,
            "min_net_top5_gain": 1,
            "min_subset_gain_pp": -5.0,
            "prefer_b1_if_b2_gap_lt_pp": 0.5,
        },
    }


def main(dataset_root: Path, feature_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    query_map, train_rows, dev_rows, seal = load_inputs(feature_root)
    train_ids = sorted(qid for qid, row in query_map.items() if row["split"] == "train")
    dev_ids = sorted(qid for qid, row in query_map.items() if row["split"] == "dev")
    contract = {
        "gate": "T2-04B",
        "candidate_depth": 50,
        "candidate_source": "T2-01 frozen BM25 Top50",
        "feature_seal_sha256": EXPECTED_SEAL,
        "promotion_scopes": {"b1_a": "ranks 6-10", "b1_b": "ranks 6-20"},
        "promotion_position": "rank5_boundary",
        "max_promotions_per_query": 1,
        "tiers": {
            "A": ["strong_metric", "strong_period", "strong_table_structure"],
            "B": ["calculation_like", "strong_metric", "full_period", "operation_supported"],
            "C": ["direct_fact", "strong_metric", "strong_period"],
        },
        "hard_conflicts": ["required_period_coverage_zero", "metric_match_zero"],
        "non_eligible_order": "original BM25 relative order",
        "entity_primary_promotion_signal": False,
    }
    write_json(output_root / "b1-guarded-contract.json", contract)
    all_rows = train_rows + dev_rows
    base_all = bm25_order(all_rows)
    feature_all = feature_order(all_rows, "required_period_coverage")
    b1a_all, b1a_diag = guarded_order(all_rows, query_map, 10)
    b1b_all, b1b_diag = guarded_order(all_rows, query_map, 20)
    # This is the first Gold read, after the complete A.2 feature seal check.
    train_gold = load_gold(dataset_root, "train", set(train_ids))
    model_data, model = fit_b2(train_rows, train_gold)
    write_json(output_root / "b2-model.json", model_data)
    write_json(output_root / "b2-training-manifest.json", {
        "gate": "T2-04B",
        "training_split": "Primary Train",
        "training_queries": len(train_ids),
        "candidate_depth": 50,
        "feature_seal_sha256": EXPECTED_SEAL,
        "features": list(B2_FEATURES),
        "negative_sampling": {
            "top10_all": True,
            "fixed_remaining_ranks": list(NEGATIVE_RANKS),
            "random_seed": 20250810,
        },
        "dev_gold_used_during_fit": False,
        "test_gold_used": False,
        "convfinqa_gold_used": False,
        "model_hash": model_data["model_hash"],
    })
    # Dev Gold is read only after fitting and sealing the B2 model.
    dev_gold = load_gold(dataset_root, "dev", set(dev_ids))
    b2_dev = b2_order(dev_rows, model)
    orders = {
        "bm25": {qid: base_all[qid] for qid in dev_ids},
        "required_period_reference": {qid: feature_all[qid] for qid in dev_ids},
        "b1_a": {qid: b1a_all[qid] for qid in dev_ids},
        "b1_b": {qid: b1b_all[qid] for qid in dev_ids},
        "b2": {qid: b2_dev[qid] for qid in dev_ids},
    }
    metrics, moves, displacements = evaluate(dev_ids, dev_gold, orders)
    subset = scoped_metrics({qid: query_map[qid] for qid in dev_ids}, dev_gold, orders)
    write_json(output_root / "b1-dev-results.json", {
        "b1_a": metrics["b1_a"],
        "b1_b": metrics["b1_b"],
        "diagnostics": {"b1_a": b1a_diag, "b1_b": b1b_diag},
    })
    write_json(output_root / "b2-dev-results.json", {
        "metrics": metrics["b2"],
        "model_hash": model_data["model_hash"],
        "dev_gold_reads_after_fit": True,
    })
    write_json(output_root / "method-comparison.json", {
        "candidate_depth": 50,
        "feature_seal_sha256": EXPECTED_SEAL,
        "metrics": metrics,
        "primary_test_gold_reads": 0,
        "convfinqa_gold_reads": 0,
    })
    write_json(output_root / "rank-movement.json", {
        "baseline": "bm25", "candidate_depth": 50, "methods": moves,
    })
    write_json(output_root / "rank-displacement.json", {
        "baseline": "bm25", "candidate_depth": 50, "methods": displacements,
    })
    write_json(output_root / "subset-analysis.json", subset)
    write_json(output_root / "query-type-analysis.json", query_type_metrics(
        {qid: query_map[qid] for qid in dev_ids}, dev_gold, orders
    ))
    selection = choose(metrics, moves, subset)
    selected = selection["selected_method"]
    selected_hash = None
    manifest = {
        "gate": "T2-04B",
        "selected": bool(selected),
        "method": selected,
        "candidate_depth": 50,
        "feature_seal_sha256": EXPECTED_SEAL,
        "tie_break": "score_desc_then_original_bm25_rank_asc_then_context_id_asc",
        "b1_contract_sha256": sha256_file(output_root / "b1-guarded-contract.json"),
        "b2_model_hash": model_data["model_hash"],
        "primary_test_gold_reads": 0,
        "convfinqa_gold_reads": 0,
    }
    write_json(output_root / "selected-method.json", {
        "selected_method": selected,
        "selection": selection,
        "metrics": metrics.get(selected) if selected else None,
    })
    write_json(output_root / "selected-method-manifest.json", manifest)
    selected_hash = sha256_file(output_root / "selected-method-manifest.json")
    (output_root / "selected-method-sha256.txt").write_text(selected_hash + "\n", encoding="utf-8")
    selected_metrics = metrics.get(selected) if selected else None
    decision = {
        "gate": "T2-04B",
        "base_commit": "6981d70",
        "dataset_commit": DATASET_COMMIT,
        "primary_train_queries": COUNTS["train"],
        "primary_dev_queries": COUNTS["dev"],
        "primary_test_queries": COUNTS["test"],
        "feature_seal": EXPECTED_SEAL,
        "candidate_depth": 50,
        "retrieval_rerun": False,
        "llm_execution": False,
        "primary_test_gold_reads": 0,
        "convfinqa_gold_reads": 0,
        "bm25_dev_recall_at_5": metrics["bm25"]["recall_pct"]["@5"],
        "b1a_dev_recall_at_5": metrics["b1_a"]["recall_pct"]["@5"],
        "b1b_dev_recall_at_5": metrics["b1_b"]["recall_pct"]["@5"],
        "b2_dev_recall_at_5": metrics["b2"]["recall_pct"]["@5"],
        "selected_method": selected,
        "selected_dev_recall_at_5": selected_metrics["recall_pct"]["@5"] if selected_metrics else None,
        "selected_gain_pp": (
            selected_metrics["recall_pct"]["@5"] - metrics["bm25"]["recall_pct"]["@5"]
            if selected_metrics else None
        ),
        "rescued_at_5": moves[selected]["rescued_at_5"] if selected else None,
        "damaged_at_5": moves[selected]["damaged_at_5"] if selected else None,
        "structure_reranker_selected": bool(selected),
        "selected_method_hash": selected_hash if selected else None,
        "selection": selection,
        "primary_test_structure_scoring": False,
        "convfinqa_structure_scoring": False,
        "empty_questions_retained": True,
        "next_gate": "t2_04c_frozen_test_evaluation" if selected else "t2_04_method_reconsideration",
    }
    write_json(output_root / "decision.json", decision)
    write_json(output_root / "input-integrity.json", {
        "dataset_commit": DATASET_COMMIT,
        "feature_seal_sha256": EXPECTED_SEAL,
        "candidate_depth": 50,
        "train_queries": len(train_ids),
        "dev_queries": len(dev_ids),
        "candidate_mutation": 0,
        "retrieval_rerun": False,
        "llm_execution": False,
        "primary_test_gold_reads": 0,
        "convfinqa_gold_reads": 0,
    })
    (output_root / "README.md").write_text(
        "# T2-04B Guarded Financial Structure-Aware Reranker\n\n"
        "B1 uses one rank-5-boundary promotion slot per query; B2 is a fixed "
        "linear pairwise logistic diagnostic. Both consume only frozen BM25 "
        "Top50 and A.2 features on Primary Train/Dev. Primary Test and "
        "ConvFinQA Gold remain locked.\n",
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
