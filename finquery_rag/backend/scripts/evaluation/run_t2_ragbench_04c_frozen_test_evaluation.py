from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASET_COMMIT = "adf7fe1541ac37351ce1142544d8e3b43010ed92"
METHOD_HASH = "93a7dccd72a1c3d19effe58053942504adfb9b9ca2c45ed32525edfcadd4006e"
FEATURE_SEAL_SHA = "98204451ca98046fb7bed2338ad346f511b10f90195b6e7d78f084c52131641d"
BM25_PREDICTION_SHA = "227f0e560dc166113a75edc59baf9a8d50ac040520b38c81ecd0bb6e5e945d80"
QUERY_MANIFEST_SHA = "471e3bd1e6856af643403deaee93c54fc17aee26da072ebb3abd38265b20b82b"
PRIMARY_TEST_COUNT = 2291
CONVFINQA_COUNT = 3458
CANDIDATE_DEPTH = 50
SUBSETS = ("FinQA", "TAT-DQA", "ConvFinQA")
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


def write_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in rows:
                compressed.write(
                    (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
                )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_obj(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_a2_module() -> Any:
    path = Path(__file__).with_name("run_t2_ragbench_04a2_structure_signal_audit.py")
    spec = importlib.util.spec_from_file_location("t2_04a2_frozen_extractor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("a2_extractor_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_frozen_contracts(
    method_root: Path,
    feature_root: Path,
    retrieval_root: Path,
    protocol_root: Path,
) -> dict[str, Any]:
    selected = read_json(method_root / "selected-method.json")
    manifest_path = method_root / "selected-method-manifest.json"
    manifest = read_json(manifest_path)
    manifest_hash = sha256_obj(manifest)
    if manifest_hash != METHOD_HASH:
        raise RuntimeError("method_hash_mismatch")
    if (method_root / "selected-method-sha256.txt").read_text().strip() != METHOD_HASH:
        raise RuntimeError("method_hash_file_mismatch")
    if selected["feature"] != "required_period_coverage" or not selected["selected"]:
        raise RuntimeError("selected_method_contract")
    if selected["feature_seal"] != FEATURE_SEAL_SHA:
        raise RuntimeError("selected_method_feature_seal")
    if selected["ranking"] != [
        "required_period_coverage_desc",
        "original_bm25_rank_asc",
        "context_id_asc",
    ]:
        raise RuntimeError("selected_method_ranking_contract")
    if manifest["candidate_depth"] != CANDIDATE_DEPTH:
        raise RuntimeError("selected_method_candidate_depth")

    feature_seal_path = feature_root / "feature-seal.json"
    if sha256_file(feature_seal_path) != FEATURE_SEAL_SHA:
        raise RuntimeError("feature_seal_mismatch")
    feature_seal = read_json(feature_seal_path)
    if feature_seal["primary_test_gold_reads"] != 0 or feature_seal["convfinqa_gold_reads"] != 0:
        raise RuntimeError("feature_gold_lock")
    for name, expected in feature_seal["feature_files"].items():
        if sha256_file(feature_root / name) != expected:
            raise RuntimeError(f"feature_file_mutation:{name}")

    prediction_seal = read_json(retrieval_root / "prediction-seal.json")
    bm25_path = retrieval_root / "bm25-predictions.jsonl.gz"
    if sha256_file(bm25_path) != BM25_PREDICTION_SHA:
        raise RuntimeError("bm25_prediction_mutation")
    if prediction_seal["output_sha256"]["bm25"] != BM25_PREDICTION_SHA:
        raise RuntimeError("bm25_seal_mismatch")
    if prediction_seal["gold_scoring_reads_before_seal"] != 0:
        raise RuntimeError("bm25_gold_preseal")
    if sha256_file(retrieval_root / "query-manifest.jsonl.gz") != QUERY_MANIFEST_SHA:
        raise RuntimeError("query_manifest_mutation")
    protocol = read_json(protocol_root / "evaluation-protocol.json")
    if protocol["dataset_commit"] != DATASET_COMMIT:
        raise RuntimeError("dataset_commit_mismatch")
    return {
        "selected_method": selected,
        "method_manifest": manifest,
        "feature_seal": feature_seal,
        "bm25_prediction_sha": BM25_PREDICTION_SHA,
        "query_manifest_sha": QUERY_MANIFEST_SHA,
    }


def load_query_ids(protocol_root: Path) -> tuple[set[str], set[str]]:
    primary = {str(value) for value in read_json(protocol_root / "primary-test-query-ids.json")}
    conv = {str(value) for value in read_json(protocol_root / "convfinqa-transfer-query-ids.json")}
    if len(primary) != PRIMARY_TEST_COUNT or len(conv) != CONVFINQA_COUNT:
        raise RuntimeError("query_id_manifest_count")
    if primary & conv:
        raise RuntimeError("query_id_manifest_overlap")
    return primary, conv


def load_query_manifest(
    retrieval_root: Path, query_ids: set[str]
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_gz(retrieval_root / "query-manifest.jsonl.gz"):
        query_id = str(row["query_id"])
        if query_id not in query_ids:
            continue
        rows[query_id] = {
            "query_id": query_id,
            "subset": str(row["subset"]),
            "split": str(row["split"]),
            "query": row.get("query", ""),
            "company_name": row.get("company_name"),
        }
    if set(rows) != query_ids:
        raise RuntimeError("query_manifest_identity_mismatch")
    return rows


def load_bm25_top50(
    retrieval_root: Path, query_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    path = retrieval_root / "bm25-predictions.jsonl.gz"
    output: dict[str, list[dict[str, Any]]] = {}
    for row in read_gz(path):
        query_id = str(row["query_id"])
        if query_id not in query_ids:
            continue
        ranked = row.get("ranked_contexts") or []
        top = []
        for item in ranked[:CANDIDATE_DEPTH]:
            top.append(
                {
                    "context_id": str(item["context_id"]),
                    "bm25_rank": int(item["rank"]),
                    "bm25_score": item.get("score"),
                }
            )
        if len(top) != CANDIDATE_DEPTH or [item["bm25_rank"] for item in top] != list(range(1, 51)):
            raise RuntimeError(f"bm25_top50_contract:{query_id}")
        if len({item["context_id"] for item in top}) != CANDIDATE_DEPTH:
            raise RuntimeError(f"bm25_duplicate_candidate:{query_id}")
        output[query_id] = top
    if set(output) != query_ids:
        raise RuntimeError("bm25_query_identity_mismatch")
    return output


def load_context_structures(
    feature_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_gz(feature_root / "context-structure.jsonl.gz"):
        key = (str(row["subset"]), str(row["context_id"]))
        if key in output:
            raise RuntimeError(f"duplicate_context_structure:{key}")
        output[key] = row
    return output


def hydrate_missing_contexts(
    a2: Any,
    dataset_root: Path,
    contexts: dict[tuple[str, str], dict[str, Any]],
    needed: dict[str, set[str]],
) -> int:
    """Replay the sealed A.2 context extractor only for uncovered transfer contexts."""
    missing = {
        subset: set(context_ids)
        - {context_id for (stored_subset, context_id) in contexts if stored_subset == subset}
        for subset, context_ids in needed.items()
    }
    missing = {subset: ids for subset, ids in missing.items() if ids}
    if not missing:
        return 0
    paths = [
        ("FinQA", dataset_root / "data" / "FinQA" / "test" / "metadata.jsonl"),
        ("TAT-DQA", dataset_root / "data" / "TAT-DQA" / "test" / "metadata.jsonl"),
        ("ConvFinQA", dataset_root / "data" / "ConvFinQA" / "turn_0.jsonl"),
    ]
    for subset, path in paths:
        wanted = missing.get(subset, set())
        if not wanted:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                context_id = str(source["context_id"])
                if context_id not in wanted:
                    continue
                key = (subset, context_id)
                context_text = str(source.get("context") or "")
                existing = contexts.get(key)
                if existing is not None and existing["context_sha256"] != hashlib.sha256(
                    context_text.encode("utf-8")
                ).hexdigest():
                    raise RuntimeError(f"context_replay_conflict:{subset}:{context_id}")
                if existing is None:
                    contexts[key] = a2.extract_context(
                        context_id, subset, context_text, source.get("file_name")
                    )
    unresolved = {
        subset: ids
        - {context_id for (stored_subset, context_id) in contexts if stored_subset == subset}
        for subset, ids in missing.items()
    }
    unresolved = {subset: ids for subset, ids in unresolved.items() if ids}
    if unresolved:
        raise RuntimeError(f"candidate_context_replay_missing:{unresolved}")
    return sum(len(ids) for ids in missing.values())


def build_query_structure(a2: Any, row: dict[str, Any]) -> dict[str, Any]:
    # This is a replay of the sealed A.2 extractor, not a new feature contract.
    return a2.extract_query(row.get("query", ""), row.get("company_name"))


def pcr_predictions(
    a2: Any,
    queries: dict[str, dict[str, Any]],
    bm25: dict[str, list[dict[str, Any]]],
    contexts: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, list[str]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    bm25_order: dict[str, list[str]] = {}
    pcr_order: dict[str, list[str]] = {}
    query_structures: dict[str, dict[str, Any]] = {}
    for query_id in sorted(queries):
        query = queries[query_id]
        structure = build_query_structure(a2, query)
        query_structures[query_id] = structure
        candidates = bm25[query_id]
        baseline_ids = [item["context_id"] for item in candidates]
        ranked = []
        for item in candidates:
            context = contexts.get((query["subset"], item["context_id"]))
            if context is None:
                raise RuntimeError(f"candidate_context_missing:{query_id}:{item['context_id']}")
            features = a2.build_features(structure, context)
            coverage = float(features["required_period_coverage"])
            ranked.append({**item, "required_period_coverage": coverage})
        ranked.sort(
            key=lambda item: (
                -item["required_period_coverage"],
                item["bm25_rank"],
                item["context_id"],
            )
        )
        if {item["context_id"] for item in ranked} != set(baseline_ids):
            raise RuntimeError(f"pcr_candidate_mutation:{query_id}")
        bm25_order[query_id] = baseline_ids
        pcr_order[query_id] = [item["context_id"] for item in ranked]
        rows.append(
            {
                "query_id": query_id,
                "subset": query["subset"],
                "split": query["split"],
                "candidate_count": CANDIDATE_DEPTH,
                "ranked_contexts": [
                    {
                        "context_id": item["context_id"],
                        "pre_rerank_rank": item["bm25_rank"],
                        "required_period_coverage": item["required_period_coverage"],
                        "post_rank": rank,
                    }
                    for rank, item in enumerate(ranked, start=1)
                ],
            }
        )
    return rows, bm25_order, pcr_order, query_structures


def validate_no_period_invariant(
    query_structures: dict[str, dict[str, Any]],
    bm25: dict[str, list[str]],
    pcr: dict[str, list[str]],
) -> dict[str, Any]:
    no_period = [qid for qid, structure in query_structures.items() if not structure["periods"]]
    mismatches = [qid for qid in no_period if bm25[qid] != pcr[qid]]
    return {
        "query_count": len(no_period),
        "ranking_identical": not mismatches,
        "mismatch_query_ids": mismatches,
    }


def seal_predictions(
    output_root: Path,
    filename: str,
    rows: list[dict[str, Any]],
    method_hash: str,
    feature_seal: str,
    gold_reads_before_seal: int = 0,
) -> dict[str, Any]:
    path = output_root / filename
    write_gz(path, rows)
    if len(rows) == 0:
        raise RuntimeError("empty_prediction")
    expected = PRIMARY_TEST_COUNT if "primary" in filename else CONVFINQA_COUNT
    if len(rows) != expected:
        raise RuntimeError(f"prediction_count:{filename}")
    digest = sha256_file(path)
    seal = {
        "gate": "T2-04C",
        "sealed": True,
        "prediction_file": filename,
        "prediction_count": len(rows),
        "candidate_depth": CANDIDATE_DEPTH,
        "method_hash": method_hash,
        "feature_seal": feature_seal,
        "gold_reads_before_seal": gold_reads_before_seal,
        "retrieval_rerun": False,
        "method_mutation": False,
        "output_sha256": digest,
    }
    seal_name = filename.replace("predictions.jsonl.gz", "prediction-seal.json")
    write_json(output_root / seal_name, seal)
    return seal


def load_gold(
    dataset_root: Path,
    query_ids: set[str],
    subsets: tuple[str, ...],
    split: str,
) -> dict[str, str]:
    targets: dict[str, str] = {}
    for subset in subsets:
        if subset == "ConvFinQA":
            path = dataset_root / "data" / subset / "turn_0.jsonl"
        else:
            path = dataset_root / "data" / subset / split / "metadata.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = str(row["id"])
                if query_id in query_ids:
                    targets[query_id] = str(row["context_id"])
    if set(targets) != query_ids:
        raise RuntimeError("gold_identity_mismatch")
    return targets


def score(
    order: dict[str, list[str]],
    qids: list[str],
    gold: dict[str, str],
) -> dict[str, Any]:
    hits = {str(k): 0 for k in KS}
    mrr = ndcg5 = ndcg10 = 0.0
    for query_id in qids:
        try:
            rank = order[query_id].index(gold[query_id]) + 1
        except ValueError:
            rank = None
        for cutoff in KS:
            hits[str(cutoff)] += int(rank is not None and rank <= cutoff)
        if rank is not None and rank <= 5:
            mrr += 1.0 / rank
        if rank is not None and rank <= 5:
            ndcg5 += 1.0 / math.log2(rank + 1.0)
        if rank is not None and rank <= 10:
            ndcg10 += 1.0 / math.log2(rank + 1.0)
    count = len(qids)
    return {
        "count": count,
        "hits": hits,
        "recall": {f"@{k}": f"{hits[str(k)]}/{count}" for k in KS},
        "recall_pct": {f"@{k}": round(100.0 * hits[str(k)] / count, 6) for k in KS},
        "mrr_at_5_pct": round(100.0 * mrr / count, 6),
        "ndcg_at_5_pct": round(100.0 * ndcg5 / count, 6),
        "ndcg_at_10_pct": round(100.0 * ndcg10 / count, 6),
    }


def movement(
    baseline: dict[str, list[str]],
    pcr: dict[str, list[str]],
    gold: dict[str, str],
) -> dict[str, Any]:
    rescued = damaged = unchanged = 0
    for query_id, gold_id in gold.items():
        before = gold_id in baseline[query_id][:5]
        after = gold_id in pcr[query_id][:5]
        if after and not before:
            rescued += 1
        elif before and not after:
            damaged += 1
        else:
            unchanged += 1
    total = rescued + damaged
    return {
        "rescued_at_5": rescued,
        "damaged_at_5": damaged,
        "net_top5_gain": rescued - damaged,
        "unchanged_at_5": unchanged,
        "rescue_precision": round(rescued / total, 6) if total else None,
    }


def scoped_metrics(
    query_rows: dict[str, dict[str, Any]],
    qids: list[str],
    gold: dict[str, str],
    baseline: dict[str, list[str]],
    pcr: dict[str, list[str]],
) -> dict[str, Any]:
    output = {}
    for subset in ("FinQA", "TAT-DQA"):
        subset_ids = [qid for qid in qids if query_rows[qid]["subset"] == subset]
        output[subset] = {
            "query_count": len(subset_ids),
            "bm25": score(baseline, subset_ids, gold),
            "pcr_v1": score(pcr, subset_ids, gold),
            "movement": movement(
                {qid: baseline[qid] for qid in subset_ids},
                {qid: pcr[qid] for qid in subset_ids},
                {qid: gold[qid] for qid in subset_ids},
            ),
        }
    return output


def query_type_metrics(
    query_rows: dict[str, dict[str, Any]],
    query_structures: dict[str, dict[str, Any]],
    qids: list[str],
    gold: dict[str, str],
    baseline: dict[str, list[str]],
    pcr: dict[str, list[str]],
) -> dict[str, Any]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for query_id in qids:
        by_type[str(query_structures[query_id]["operation_intent"])].append(query_id)
    output = {}
    for operation in sorted(by_type):
        ids = sorted(by_type[operation])
        output[operation] = {
            "query_count": len(ids),
            "bm25_r_at_5_pct": score(baseline, ids, gold)["recall_pct"]["@5"],
            "pcr_v1_r_at_5_pct": score(pcr, ids, gold)["recall_pct"]["@5"],
            "gain_pp": round(
                score(pcr, ids, gold)["recall_pct"]["@5"]
                - score(baseline, ids, gold)["recall_pct"]["@5"],
                6,
            ),
            "movement": movement(
                {qid: baseline[qid] for qid in ids},
                {qid: pcr[qid] for qid in ids},
                {qid: gold[qid] for qid in ids},
            ),
        }
    return output


def period_cohort_metrics(
    query_structures: dict[str, dict[str, Any]],
    qids: list[str],
    gold: dict[str, str],
    baseline: dict[str, list[str]],
    pcr: dict[str, list[str]],
) -> dict[str, Any]:
    required = [qid for qid in qids if query_structures[qid]["periods"]]
    no_period = [qid for qid in qids if not query_structures[qid]["periods"]]
    no_period_identical = all(baseline[qid] == pcr[qid] for qid in no_period)
    return {
        "period_required": {
            "query_count": len(required),
            "bm25": score(baseline, required, gold),
            "pcr_v1": score(pcr, required, gold),
            "movement": movement(
                {qid: baseline[qid] for qid in required},
                {qid: pcr[qid] for qid in required},
                {qid: gold[qid] for qid in required},
            ),
        },
        "no_period_requirement": {
            "query_count": len(no_period),
            "ranking_identical": no_period_identical,
            "bm25": score(baseline, no_period, gold),
            "pcr_v1": score(pcr, no_period, gold),
        },
    }


def main(
    dataset_root: Path,
    retrieval_root: Path,
    feature_root: Path,
    method_root: Path,
    protocol_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    verify_frozen_contracts(method_root, feature_root, retrieval_root, protocol_root)
    primary_ids, conv_ids = load_query_ids(protocol_root)
    all_ids = primary_ids | conv_ids
    query_rows = load_query_manifest(retrieval_root, all_ids)
    if len(query_rows) != PRIMARY_TEST_COUNT + CONVFINQA_COUNT:
        raise RuntimeError("query_count_contract")
    if any(query_rows[qid]["subset"] == "ConvFinQA" and query_rows[qid]["split"] != "all" for qid in conv_ids):
        raise RuntimeError("conv_split_contract")
    if any(query_rows[qid]["subset"] not in {"FinQA", "TAT-DQA"} for qid in primary_ids):
        raise RuntimeError("primary_subset_contract")

    bm25 = load_bm25_top50(retrieval_root, all_ids)
    contexts = load_context_structures(feature_root)
    a2 = load_a2_module()
    needed_contexts: dict[str, set[str]] = defaultdict(set)
    for query_id, candidates in bm25.items():
        subset = query_rows[query_id]["subset"]
        needed_contexts[subset].update(item["context_id"] for item in candidates)
    sealed_context_count = len(contexts)
    replayed_context_count = hydrate_missing_contexts(
        a2, dataset_root, contexts, dict(needed_contexts)
    )
    write_json(
        output_root / "context-replay-audit.json",
        {
            "feature_contract": "T2-04A.2 sealed FinancialStructureExtractor",
            "sealed_context_structures": sealed_context_count,
            "replayed_context_structures": replayed_context_count,
            "gold_reads_before_prediction_seal": 0,
            "feature_definition_modified": False,
        },
    )
    primary_queries = {qid: query_rows[qid] for qid in primary_ids}
    conv_queries = {qid: query_rows[qid] for qid in conv_ids}
    primary_bm25 = {qid: bm25[qid] for qid in primary_ids}
    conv_bm25 = {qid: bm25[qid] for qid in conv_ids}

    primary_rows, primary_base, primary_pcr, primary_structures = pcr_predictions(
        a2, primary_queries, primary_bm25, contexts
    )
    conv_rows, conv_base, conv_pcr, conv_structures = pcr_predictions(
        a2, conv_queries, conv_bm25, contexts
    )
    primary_no_period = validate_no_period_invariant(primary_structures, primary_base, primary_pcr)
    conv_no_period = validate_no_period_invariant(conv_structures, conv_base, conv_pcr)

    # No Gold fields are loaded before both complete prediction seals are written.
    primary_seal = seal_predictions(
        output_root,
        "primary-test-predictions.jsonl.gz",
        primary_rows,
        METHOD_HASH,
        FEATURE_SEAL_SHA,
    )
    conv_seal = seal_predictions(
        output_root,
        "convfinqa-transfer-predictions.jsonl.gz",
        conv_rows,
        METHOD_HASH,
        FEATURE_SEAL_SHA,
    )

    primary_gold = load_gold(dataset_root, primary_ids, ("FinQA", "TAT-DQA"), "test")
    conv_gold = load_gold(dataset_root, conv_ids, ("ConvFinQA",), "turn_0")
    primary_qids = sorted(primary_ids)
    conv_qids = sorted(conv_ids)
    primary_metrics = {
        "bm25": score(primary_base, primary_qids, primary_gold),
        "pcr_v1": score(primary_pcr, primary_qids, primary_gold),
    }
    conv_metrics = {
        "bm25": score(conv_base, conv_qids, conv_gold),
        "pcr_v1": score(conv_pcr, conv_qids, conv_gold),
    }
    primary_movement = movement(primary_base, primary_pcr, primary_gold)
    conv_movement = movement(conv_base, conv_pcr, conv_gold)
    primary_subset = scoped_metrics(query_rows, primary_qids, primary_gold, primary_base, primary_pcr)
    primary_query_types = query_type_metrics(
        query_rows, primary_structures, primary_qids, primary_gold, primary_base, primary_pcr
    )
    conv_query_types = query_type_metrics(
        query_rows, conv_structures, conv_qids, conv_gold, conv_base, conv_pcr
    )
    primary_period = period_cohort_metrics(
        primary_structures, primary_qids, primary_gold, primary_base, primary_pcr
    )
    conv_period = period_cohort_metrics(conv_structures, conv_qids, conv_gold, conv_base, conv_pcr)

    write_json(output_root / "primary-test-metrics.json", primary_metrics)
    write_json(output_root / "primary-test-rank-movement.json", primary_movement)
    write_json(output_root / "primary-test-subset-analysis.json", primary_subset)
    write_json(output_root / "primary-test-query-type-analysis.json", primary_query_types)
    write_json(output_root / "primary-test-period-cohort.json", primary_period)
    write_json(output_root / "convfinqa-transfer-metrics.json", conv_metrics)
    write_json(output_root / "convfinqa-transfer-rank-movement.json", conv_movement)
    write_json(output_root / "convfinqa-transfer-query-type-analysis.json", conv_query_types)
    write_json(output_root / "convfinqa-transfer-period-cohort.json", conv_period)

    primary_gain = round(
        primary_metrics["pcr_v1"]["recall_pct"]["@5"]
        - primary_metrics["bm25"]["recall_pct"]["@5"],
        6,
    )
    conv_gain = round(
        conv_metrics["pcr_v1"]["recall_pct"]["@5"]
        - conv_metrics["bm25"]["recall_pct"]["@5"],
        6,
    )
    if primary_metrics["pcr_v1"]["hits"]["50"] != primary_metrics["bm25"]["hits"]["50"]:
        raise RuntimeError("primary_r50_invariant")
    if conv_metrics["pcr_v1"]["hits"]["50"] != conv_metrics["bm25"]["hits"]["50"]:
        raise RuntimeError("conv_r50_invariant")
    primary_validated = primary_gain > 0 and primary_movement["net_top5_gain"] > 0
    transfer_supported = conv_gain > 0 and conv_movement["net_top5_gain"] > 0
    summary = {
        "gate": "T2-04C",
        "dataset_commit": DATASET_COMMIT,
        "primary_test_queries": PRIMARY_TEST_COUNT,
        "bm25_recall_at_5": primary_metrics["bm25"]["recall_pct"]["@5"] / 100.0,
        "pcr_v1_recall_at_5": primary_metrics["pcr_v1"]["recall_pct"]["@5"] / 100.0,
        "gain_pp": primary_gain,
        "rescued_at_5": primary_movement["rescued_at_5"],
        "damaged_at_5": primary_movement["damaged_at_5"],
        "net_top5_gain": primary_movement["net_top5_gain"],
        "finqa_bm25_recall_at_5": primary_subset["FinQA"]["bm25"]["recall_pct"]["@5"] / 100.0,
        "finqa_pcr_recall_at_5": primary_subset["FinQA"]["pcr_v1"]["recall_pct"]["@5"] / 100.0,
        "tatdqa_bm25_recall_at_5": primary_subset["TAT-DQA"]["bm25"]["recall_pct"]["@5"] / 100.0,
        "tatdqa_pcr_recall_at_5": primary_subset["TAT-DQA"]["pcr_v1"]["recall_pct"]["@5"] / 100.0,
        "convfinqa_queries": CONVFINQA_COUNT,
        "convfinqa_bm25_recall_at_5": conv_metrics["bm25"]["recall_pct"]["@5"] / 100.0,
        "convfinqa_pcr_recall_at_5": conv_metrics["pcr_v1"]["recall_pct"]["@5"] / 100.0,
        "method_hash": METHOD_HASH,
        "feature_seal": FEATURE_SEAL_SHA,
        "primary_test_gold_preseal_reads": 0,
        "convfinqa_gold_preseal_reads": 0,
        "primary_no_period_cohort": primary_no_period,
        "convfinqa_no_period_cohort": conv_no_period,
        "sealed_context_structures": sealed_context_count,
        "replayed_context_structures": replayed_context_count,
        "external_structure_method_validated": primary_validated,
        "transfer_supported": transfer_supported,
    }
    write_json(output_root / "external-track-final-summary.json", summary)
    decision = {
        "gate": "T2-04C",
        "base_commit": "d674a3b",
        "method": "Financial Period-Consistency Reranker V1",
        "method_hash": METHOD_HASH,
        "feature_seal": FEATURE_SEAL_SHA,
        "candidate_depth": CANDIDATE_DEPTH,
        "candidate_universe_unchanged": True,
        "primary_test_prediction_sealed": True,
        "convfinqa_prediction_sealed": True,
        "primary_test_gold_reads_before_seal": 0,
        "convfinqa_gold_reads_before_seal": 0,
        "primary_r50_invariant": primary_metrics["pcr_v1"]["hits"]["50"] == primary_metrics["bm25"]["hits"]["50"],
        "convfinqa_r50_invariant": conv_metrics["pcr_v1"]["hits"]["50"] == conv_metrics["bm25"]["hits"]["50"],
        "external_structure_method_validated": primary_validated,
        "transfer_supported": transfer_supported,
        "decision": (
            "external_structure_method_validated"
            if primary_validated
            else "external_structure_method_not_validated_bm25_final_baseline"
        ),
        "next_gate": "external_track_complete",
        "primary_prediction_sha256": primary_seal["output_sha256"],
        "convfinqa_prediction_sha256": conv_seal["output_sha256"],
        "replayed_context_structures": replayed_context_count,
    }
    write_json(output_root / "decision.json", decision)
    (output_root / "README.md").write_text(
        "# T2-04C Frozen Primary Test and ConvFinQA Transfer\n\n"
        "PCR-V1 was applied one-shot to frozen BM25 Top50 candidates. "
        "Prediction artifacts were sealed before reading either test Gold.\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            main(
                args.dataset_root,
                args.retrieval_root,
                args.feature_root,
                args.method_root,
                args.protocol_root,
                args.output_root,
            ),
            sort_keys=True,
        )
    )
