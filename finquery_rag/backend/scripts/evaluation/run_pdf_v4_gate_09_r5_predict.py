#!/usr/bin/env python3
"""Zero-Gold Gate09 R5 Top10 semantic evidence-set prediction and seal."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.semantic_evidence_set import (  # noqa: E402
    ACCESS_TOP_K,
    MAX_EVIDENCE_ITEMS,
    build_access_universe,
    build_semantic_classes,
    match_slots,
    minimum_candidate_cover,
    operand_projection,
)

EVAL = ROOT / "artifacts/evaluation"
R33 = EVAL / "pdf-retrieval-v4-gate-08-r8-r3-3"
SE1_P0 = EVAL / "pdf-retrieval-v4-gate-08-r8-se1-p0"
QUERY_PLAN = EVAL / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
OUT = EVAL / "pdf-retrieval-v4-gate-09-r5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    main_path = R33 / "main_rerank_predictions.jsonl.gz"
    slot_path = R33 / "slot_rerank_predictions.jsonl.gz"
    registry_path = SE1_P0 / "candidate-semantic-fact-registry.jsonl.gz"
    r33_seal = json.loads((R33 / "prediction-seal.json").read_text(encoding="utf-8"))
    se1_seal = json.loads((SE1_P0 / "prediction-seal.json").read_text(encoding="utf-8"))
    if r33_seal["main_prediction_sha256"] != sha256(main_path) or r33_seal["slot_prediction_sha256"] != sha256(slot_path):
        raise RuntimeError("r3_3_ranking_seal_invalid")
    if se1_seal["registry_sha256"] != sha256(registry_path) or se1_seal["gold_reads_before_seal"] != 0:
        raise RuntimeError("se1_registry_seal_invalid")

    main_rankings = {str(row["case_id"]): row["ranked_candidates"] for row in read_jsonl(main_path)}
    slot_rankings: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for row in read_jsonl(slot_path):
        slot_rankings[str(row["case_id"])][str(row["slot_id"])] = row["ranked_candidates"]
    registry = {str(row["candidate_key"]): row for row in read_jsonl(registry_path)}
    query_payload = json.loads(QUERY_PLAN.read_text(encoding="utf-8"))
    plans = {str(row["case_id"]): row["plan"] for row in query_payload["plans"]}
    if set(main_rankings) != set(plans) or len(plans) != 72:
        raise RuntimeError("gate09_r5_case_contract_blocked")

    access_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    set_rows: list[dict[str, Any]] = []
    access_counts: list[int] = []
    ambiguity_counts: Counter[str] = Counter()
    calculation_ready = 0
    for case_id in sorted(plans):
        plan = plans[case_id]
        slots = list(plan.get("operand_slots") or [])
        is_multi = len(slots) > 1
        slot_local = slot_rankings.get(case_id, {})
        if is_multi and set(slot_local) != {str(slot["slot_id"]) for slot in slots}:
            raise RuntimeError(f"missing_frozen_slot_ranking:{case_id}")
        access = build_access_universe(plan, main_rankings[case_id], slot_local)
        access_keys = {item["candidate_key"] for item in access}
        if not access_keys <= set(registry):
            raise RuntimeError(f"candidate_outside_semantic_registry:{case_id}")
        semantic_classes = build_semantic_classes(access, registry)
        slot_matches = match_slots(plan, semantic_classes)
        cover = minimum_candidate_cover(slot_matches, semantic_classes, access)
        projection = operand_projection(plan, slot_matches, semantic_classes)
        selected = cover["selected_candidate_keys"]
        if len(selected) > MAX_EVIDENCE_ITEMS or not set(selected) <= access_keys:
            raise RuntimeError(f"evidence_budget_or_access_violation:{case_id}")
        for match in slot_matches:
            ambiguity_counts[match["slot_status"]] += 1
        calculation_ready += bool(projection["calculation_runtime_ready"])
        access_counts.append(len(access))
        access_rows.append(
            {
                "case_id": case_id,
                "access_mode": "U1_main_top10_plus_slot_top10" if is_multi else "U0_main_top10",
                "is_multi_slot": is_multi,
                "candidate_count": len(access),
                "candidates": access,
            }
        )
        class_rows.append({"case_id": case_id, "semantic_class_count": len(semantic_classes), "semantic_classes": semantic_classes})
        match_rows.append({"case_id": case_id, "slot_matches": slot_matches})
        projection_rows.append({"case_id": case_id, **projection})
        set_rows.append(
            {
                "case_id": case_id,
                "plan_id": plan["plan_id"],
                "access_mode": "U1" if is_multi else "U0",
                "required_slot_count": len(slots),
                "deterministic_slot_count": sum(item["slot_status"] == "deterministic" for item in slot_matches),
                "undercovered_slot_count": sum(item["slot_status"] == "undercovered" for item in slot_matches),
                "ambiguous_slot_count": sum(item["slot_status"] == "runtime_operand_ambiguity" for item in slot_matches),
                "evidence_set_status": "complete" if cover["complete"] else "blocked",
                "selected_semantic_fact_ids": cover["covered_semantic_fact_ids"],
                "selected_candidate_keys": selected,
                "evidence_item_count": cover["evidence_item_count"],
                "calculation_runtime_ready": projection["calculation_runtime_ready"],
            }
        )

    paths = {
        "access": OUT / "evidence-access-universe.jsonl.gz",
        "classes": OUT / "semantic-evidence-classes.jsonl.gz",
        "matches": OUT / "slot-semantic-matches.jsonl.gz",
        "projections": OUT / "operand-projections.jsonl.gz",
        "sets": OUT / "evidence-set-predictions.jsonl.gz",
    }
    write_jsonl_gz(paths["access"], access_rows)
    write_jsonl_gz(paths["classes"], class_rows)
    write_jsonl_gz(paths["matches"], match_rows)
    write_jsonl_gz(paths["projections"], projection_rows)
    write_jsonl_gz(paths["sets"], set_rows)
    protocol = {
        "gate": "pdf_retrieval_v4_gate_09_r5",
        "main_access_top_k": ACCESS_TOP_K,
        "slot_access_top_k": ACCESS_TOP_K,
        "max_evidence_items": MAX_EVIDENCE_ITEMS,
        "single_slot_route": "U0_main_top10",
        "multi_slot_route": "U1_main_top10_union_each_slot_top10",
        "slot_binding": "strict_semantic_fact_exact",
        "gold_reads_before_seal": 0,
        "strict_binding_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,
        "expected_value_reads_before_seal": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "embedding_calls": 0,
        "bridge_runs": 0,
        "query_plan_runs": 0,
        "candidate_mutation": 0,
        "semantic_registry_mutation": 0,
        "parameter_scan": False,
        "evidence_budget_scan": False,
        "production_writes": 0,
    }
    integrity = {
        "case_count": len(set_rows),
        "single_slot_case_count": sum(not row["is_multi_slot"] for row in access_rows),
        "multi_slot_case_count": sum(row["is_multi_slot"] for row in access_rows),
        "slot_ranking_count": sum(len(value) for value in slot_rankings.values()),
        "access_candidate_count_min": min(access_counts),
        "access_candidate_count_max": max(access_counts),
        "evidence_set_size_max": max(row["evidence_item_count"] for row in set_rows),
        "candidate_outside_frozen_rankings": 0,
        "candidate_mutation": 0,
        "semantic_registry_mutation": 0,
        "slot_status_counts": dict(sorted(ambiguity_counts.items())),
        "calculation_runtime_ready_preseal_count": calculation_ready,
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "input-integrity.json", integrity)
    write_json(
        OUT / "ambiguity-audit.json",
        {
            "slot_status_counts": dict(sorted(ambiguity_counts.items())),
            "runtime_operand_ambiguity_explicit": True,
            "rank_used_to_resolve_semantic_ambiguity": False,
        },
    )
    output_hashes = {name: sha256(path) for name, path in paths.items()}
    manifest = {
        "r3_3_main_prediction_sha256": sha256(main_path),
        "r3_3_slot_prediction_sha256": sha256(slot_path),
        "r3_3_prediction_seal_sha256": sha256(R33 / "prediction-seal.json"),
        "se1_registry_sha256": sha256(registry_path),
        "se1_registry_seal_sha256": sha256(SE1_P0 / "prediction-seal.json"),
        "query_plan_sha256": sha256(QUERY_PLAN),
        "semantic_set_source_sha256": sha256(ROOT / "src/pdf_retrieval_v4/semantic_evidence_set.py"),
        "prediction_source_sha256": sha256(Path(__file__)),
        "output_sha256": output_hashes,
    }
    write_json(OUT / "prediction-manifest.json", manifest)
    seal = {
        **protocol,
        **integrity,
        **manifest,
        "prediction_count": len(set_rows),
        "sealed": True,
        "production_switch_allowed": False,
    }
    write_json(OUT / "prediction-seal.json", seal)
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
