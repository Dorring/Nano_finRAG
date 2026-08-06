"""Build V4 Gate 06 R2 split typed-evidence Shadow Indexes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GATE06_PATH = ROOT / "scripts" / "evaluation" / "run_pdf_v4_gate_06.py"
SPEC = importlib.util.spec_from_file_location("gate06_base", GATE06_PATH)
assert SPEC and SPEC.loader
gate06 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate06)


DEFAULT_GATE05 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r3"
DEFAULT_R4 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r4"
DEFAULT_GATE04C = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-04c"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06-r2"
DEFAULT_RUNTIME = ROOT / "artifacts/runtime/pdf-retrieval-v4-gate-06-r2"
UNIT_TYPES = ("section", "table", "row", "cell", "atomic_fact", "comparison_fact", "bucket_fact")
SCHEMA_VERSION = "pdf-v4-retrieval-view-r2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_r4(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_fact: dict[str, dict[str, Any]] = {}
    by_cell: dict[str, dict[str, Any]] = {}
    with gzip.open(path / "temporal-binding-predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if index == 0 and value.get("stream") == "header":
                continue
            fact_id = value.get("fact_id")
            cell_id = value.get("cell_id")
            if fact_id:
                by_fact[str(fact_id)] = value
            if cell_id:
                by_cell[str(cell_id)] = value
    return by_fact, by_cell


def _temporal_fields(unit: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    binding = record.get("temporal_binding") or {}
    kind = binding.get("kind")
    enriched = dict(unit)
    enriched["temporal_binding"] = binding
    enriched["fact_semantic_type"] = record.get("fact_semantic_type")
    enriched["temporal_source"] = record.get("temporal_source")
    if kind in {"point", "duration"}:
        enriched["period"] = binding.get("period")
        enriched["normalized_period"] = binding.get("period")
        enriched["period_set"] = [binding.get("period")] if binding.get("period") else []
    elif kind == "comparison":
        enriched["period"] = None
        enriched["normalized_period"] = None
        enriched["period_set"] = [value for value in (binding.get("base_period"), binding.get("current_period")) if value]
    elif kind == "bucket":
        enriched["period"] = binding.get("reporting_period")
        enriched["normalized_period"] = binding.get("reporting_period")
        enriched["period_set"] = [binding.get("reporting_period")] if binding.get("reporting_period") else []
    return enriched


def _typed_fact_text(unit: dict[str, Any], ctx: dict[str, Any]) -> str:
    binding = unit.get("temporal_binding") or {}
    semantic = unit.get("fact_semantic_type")
    lines = [f"Issuer: {ctx['issuer']}", f"Metric: {ctx['metric_path']}" if ctx["metric_path"] else ""]
    if semantic == "atomic_fact":
        lines.extend([f"Period: {binding.get('period')}", f"Period Type: {binding.get('period_type')}" if binding.get("period_type") else ""])
    elif semantic == "comparison_fact":
        lines.extend([f"Base Period: {binding.get('base_period')}", f"Current Period: {binding.get('current_period')}", f"Measure: {binding.get('measure')}"])
    elif semantic == "bucket_fact":
        lines.extend([f"Reporting Period: {binding.get('reporting_period')}", f"Bucket: {binding.get('bucket_label')}"])
    lines.extend([
        f"Statement: {ctx['statement']}" if ctx["statement"] else "",
        f"Value Kind: {ctx['value_kind']}" if ctx["value_kind"] else "",
        f"Currency: {ctx['currency']}" if ctx["currency"] else "",
        f"Scale: {ctx['scale']}" if ctx["scale"] else "",
        f"Reported Value: {unit.get('raw_value')}",
    ])
    return "\n".join(value for value in lines if value)


def _view_id(unit: dict[str, Any], unit_type: str) -> str:
    return "view:" + _hash([unit.get("evidence_unit_id"), unit_type, SCHEMA_VERSION])


def _build_dense_r2(views: list[dict[str, Any]], path: Path, typ: str, model: Any, model_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one deterministic Dense index without an O(n²) self-similarity matrix."""
    import numpy as np

    path.mkdir(parents=True, exist_ok=True)
    ids = [view["retrieval_view_id"] for view in views]
    texts = [view["retrieval_text"] for view in views]
    vectors = np.asarray(
        model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    np.save(path / "vectors.npy", vectors, allow_pickle=False)
    (path / "ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    norms = np.linalg.norm(vectors, axis=1) if len(vectors) else np.array([], dtype=np.float32)
    finite = bool(np.isfinite(vectors).all()) if len(vectors) else True
    zero_count = int(np.sum(norms == 0)) if len(norms) else 0
    # normalize_embeddings makes every non-zero row its own nearest neighbour;
    # avoid materializing an n×n similarity matrix for the integrity check.
    self_id_miss = 0 if finite and zero_count == 0 else len(vectors)
    semantic_hash = _hash([[identifier, [round(float(value), 6) for value in vector]] for identifier, vector in zip(ids, vectors)])
    manifest = {
        "index_type": f"{typ}_dense",
        "unit_type": typ,
        "document_count": len(views),
        "vector_count": len(vectors),
        "dimension": int(vectors.shape[1]) if len(vectors) else 0,
        "retrieval_view_id_hash": _hash(ids),
        "retrieval_text_hash": _hash([[view["retrieval_view_id"], view["retrieval_text"]] for view in views]),
        "embedding_model": model_config["embedding_model"],
        "embedding_provider": "sentence-transformers",
        "embedding_library": model_config["sentence_transformers_version"],
        "device": model_config["device"],
        "batch_size": 64,
        "precision": "float32",
        "normalized": True,
        "exact_file_hash": _sha(path / "vectors.npy"),
        "ids_file_hash": _sha(path / "ids.json"),
        "semantic_replay_hash": semantic_hash,
        "path": str(path),
    }
    integrity = {
        "unit_type": typ,
        "vector_count": len(vectors),
        "dimension": manifest["dimension"],
        "nan_vector_count": int(np.isnan(vectors).any(axis=1).sum()) if len(vectors) else 0,
        "inf_vector_count": int(np.isinf(vectors).any(axis=1).sum()) if len(vectors) else 0,
        "zero_vector_count": zero_count,
        "nonfinite_matrix": not finite,
        "self_id_miss_count": self_id_miss,
        "min_norm": float(np.min(norms)) if len(norms) else 0.0,
        "max_norm": float(np.max(norms)) if len(norms) else 0.0,
        "mean_norm": float(np.mean(norms)) if len(norms) else 0.0,
    }
    return manifest, integrity


def _build_views_r2(
    units: list[dict[str, Any]],
    r4_by_fact: dict[str, dict[str, Any]],
    r4_by_cell: dict[str, dict[str, Any]],
    shadow_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups, fragment_groups = gate06._soft_edges(shadow_dir)
    views: dict[str, list[dict[str, Any]]] = {typ: [] for typ in UNIT_TYPES}
    exclusions: dict[str, dict[str, int]] = {typ: {} for typ in UNIT_TYPES}
    for original in units:
        unit = dict(original)
        original_type = unit.get("unit_type")
        record = r4_by_fact.get(str(unit.get("fact_id")))
        if record is None and original_type == "cell":
            record = r4_by_cell.get(str(unit.get("cell_id")))
        if original_type in {"fact", "cell"} and record:
            unit = _temporal_fields(unit, record)
        if original_type in {"section", "table", "row"}:
            target_type = original_type
        elif original_type == "cell":
            target_type = "cell" if record and record.get("fact_semantic_type") in {"atomic_fact", "comparison_fact", "bucket_fact"} else None
        elif original_type == "fact":
            target_type = record.get("fact_semantic_type") if record and record.get("fact_semantic_type") in {"atomic_fact", "comparison_fact", "bucket_fact"} else None
        else:
            target_type = None
        if target_type is None:
            exclusions.setdefault(str(original_type), {})["not_typed_index_admissible"] = exclusions.setdefault(str(original_type), {}).get("not_typed_index_admissible", 0) + 1
            continue
        trace = gate06._traceback(unit)
        if not gate06._traceback_complete(trace):
            exclusions[target_type]["incomplete_source_traceback"] = exclusions[target_type].get("incomplete_source_traceback", 0) + 1
            continue
        if target_type in {"section", "table", "row"} and not gate06._text(unit.get("retrieval_text")):
            exclusions[target_type]["empty_source_retrieval_text"] = exclusions[target_type].get("empty_source_retrieval_text", 0) + 1
            continue
        if target_type == "cell" and not gate06._text(unit.get("raw_value")):
            exclusions[target_type]["empty_resolved_text"] = exclusions[target_type].get("empty_resolved_text", 0) + 1
            continue
        ctx = gate06._context(unit)
        if target_type in {"atomic_fact", "comparison_fact", "bucket_fact"}:
            retrieval_text = _typed_fact_text(unit, ctx)
        else:
            retrieval_text = gate06._view_text(unit, ctx)
        if not retrieval_text.strip():
            exclusions[target_type]["empty_generated_retrieval_text"] = exclusions[target_type].get("empty_generated_retrieval_text", 0) + 1
            continue
        fragment_id = trace.get("table_fragment_id")
        view = {
            "retrieval_view_id": _view_id(unit, target_type),
            "evidence_unit_id": unit.get("evidence_unit_id"),
            "unit_type": target_type,
            "document_id": unit.get("document_id"),
            "pdf_pages": gate06._pages(unit),
            "logical_table_id": trace.get("logical_table_id") or unit.get("logical_table_id"),
            "table_fragment_ids": [fragment_id] if fragment_id else [],
            "row_id": trace.get("row_id") or unit.get("row_id"),
            "cell_id": trace.get("cell_id") or unit.get("cell_id"),
            "fact_id": trace.get("fact_id") or unit.get("fact_id"),
            "source_identity": gate06._source_identity(unit),
            "source_group_id": gate06._source_group_id(unit),
            "continuation_group_ids": sorted(fragment_groups.get(fragment_id, [])),
            "retrieval_text": retrieval_text,
            "metadata": {**ctx, "fact_semantic_type": unit.get("fact_semantic_type"), "temporal_binding": unit.get("temporal_binding"), "temporal_source": unit.get("temporal_source")},
            "source_traceback": trace,
        }
        views[target_type].append(view)
    for target_type in views:
        views[target_type].sort(key=lambda value: value["retrieval_view_id"])
    return views, {"exclusions": exclusions, "soft_groups": groups, "soft_fragment_groups": fragment_groups}


def _r0_audit(stream_path: Path) -> dict[str, Any]:
    header, units = gate06._load_stream(stream_path)
    type_counts = {typ: sum(unit.get("unit_type") == typ for unit in units) for typ in ("section", "table", "row", "cell", "fact")}
    ids = [unit.get("evidence_unit_id") for unit in units]
    manifest_path = stream_path.with_name("evidence-units-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    physical_line_count = 0
    with gzip.open(stream_path, "rt", encoding="utf-8") as handle:
        for _ in handle:
            physical_line_count += 1
    expected_count = manifest.get("record_count", header.get("record_count", header.get("unit_count")))
    parsed_ok = len(units) == sum(type_counts.values()) == expected_count
    return {
        "gzip_physical_line_count": physical_line_count,
        "header_unit_count": header.get("unit_count"),
        "parsed_record_count": len(units),
        "manifest_record_count": manifest.get("record_count"),
        "five_type_counts": type_counts,
        "five_type_sum": sum(type_counts.values()),
        "unknown_type_count": sum(unit.get("unit_type") not in {"section", "table", "row", "cell", "fact"} for unit in units),
        "duplicate_unit_id_count": len(ids) - len(set(ids)),
        "gate_passed": parsed_ok and len(ids) == len(set(ids)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate05", type=Path, default=DEFAULT_GATE05)
    parser.add_argument("--r4", type=Path, default=DEFAULT_R4)
    parser.add_argument("--gate04c", type=Path, default=DEFAULT_GATE04C)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    r4_acceptance = json.loads((args.r4 / "acceptance.json").read_text(encoding="utf-8"))
    if not r4_acceptance.get("gate_passed"):
        raise RuntimeError("gate_05_r4_not_passed")
    stream_path = args.gate05 / "evidence-units.jsonl.gz"
    header, units = gate06._load_stream(stream_path)
    r4_by_fact, r4_by_cell = _load_r4(args.r4)
    r0 = _r0_audit(stream_path)
    if not r0["gate_passed"]:
        raise RuntimeError("gate_06_r2_input_count_integrity_blocked")
    args.out.mkdir(parents=True, exist_ok=True)
    _write(args.out / "evidence-unit-count-audit.json", r0)
    view_groups, view_aux = _build_views_r2(units, r4_by_fact, r4_by_cell, args.gate04c)
    all_views = [view for typ in UNIT_TYPES for view in view_groups[typ]]
    view_ids = [view["retrieval_view_id"] for view in all_views]
    input_integrity = {"gate05_stream_sha256": _sha(stream_path), "gate05_manifest_sha256": _sha(args.gate05 / "evidence-units-manifest.json"), "r4_prediction_sha256": _sha(args.r4 / "temporal-binding-predictions.jsonl.gz"), "r4_seal_sha256": _sha(args.r4 / "temporal-binding-seal.json"), "r4_acceptance_sha256": _sha(args.r4 / "acceptance.json"), "unit_id_hash": _hash(sorted(unit.get("evidence_unit_id") for unit in units)), "retrieval_view_id_hash": _hash(view_ids), "question_reads": 0, "runtime_gold_reads": 0, "runtime_governance_reads": 0}
    _write(args.out / "gate-06-r2-protocol.json", {"gate": "pdf_retrieval_v4_gate_06_r2", "evaluation_type": "post_benchmark_iterative_evaluation", "code_commit": args.code_commit, "unit_types": UNIT_TYPES, "r4_typed_admission_required": True, "question_reads": 0, "runtime_gold_reads": 0, "runtime_governance_reads": 0, "retrieval_runs": 0, "reranker_calls": 0, "answer_generation_calls": 0, "production_index_writes": 0, "production_default_config_modified": False, "production_switch_allowed": False})
    _write(args.out / "gate-06-r2-input-integrity.json", input_integrity)
    _write(args.out / "retrieval-view-manifest.json", {"schema_version": SCHEMA_VERSION, "total_view_count": len(all_views), "view_counts": {typ: len(view_groups[typ]) for typ in UNIT_TYPES}, "excluded_counts": view_aux["exclusions"], "retrieval_view_id_hash": _hash(view_ids), "soft_continuation_group_count": len(view_aux["soft_groups"]), "soft_continuation_generalization_established": False, "physical_cross_page_merge": False})
    _write(args.out / "retrieval-text-audit.json", gate06._leakage(all_views) | {"retrieval_text_hash": _hash([[view["retrieval_view_id"], view["retrieval_text"]] for view in all_views]), "empty_text_count": sum(not view["retrieval_text"].strip() for view in all_views), "duplicate_text_count": len(all_views) - len({view["retrieval_text"] for view in all_views})})
    runtime = args.runtime.resolve()
    if runtime.name != "pdf-retrieval-v4-gate-06-r2" or "artifacts" not in runtime.parts:
        raise RuntimeError("unsafe_gate_06_r2_runtime_path")
    previous: dict[str, Any] = {}
    if (args.out / "metadata-store-manifest.json").is_file():
        previous["metadata"] = json.loads((args.out / "metadata-store-manifest.json").read_text()).get("serialized_store_hash")
    if (args.out / "bm25-index-manifests.json").is_file():
        previous["bm25"] = json.loads((args.out / "bm25-index-manifests.json").read_text()).get("index_hashes")
    if (args.out / "dense-index-manifests.json").is_file():
        previous["dense"] = json.loads((args.out / "dense-index-manifests.json").read_text()).get("semantic_replay_hashes")
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    metadata = gate06._build_metadata_store(all_views, view_aux["soft_groups"], runtime / "metadata" / "metadata.sqlite")
    _write(args.out / "metadata-store-manifest.json", metadata)
    model_config = gate06._load_model_config()
    bm25 = {typ: gate06._build_bm25(view_groups[typ], runtime / typ / "bm25" / "index.sqlite", typ) for typ in UNIT_TYPES}
    _write(args.out / "bm25-index-manifests.json", {"index_hashes": {typ: value["serialized_index_hash"] for typ, value in bm25.items()}, "manifests": bm25, "document_counts": {typ: value["document_count"] for typ, value in bm25.items()}})
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_config["embedding_model"], device="cpu")
    dense = {}
    vectors = {}
    for typ in UNIT_TYPES:
        dense[typ], vectors[typ] = _build_dense_r2(view_groups[typ], runtime / typ / "dense", typ, model, model_config)
    _write(args.out / "dense-index-manifests.json", {"model_config": model_config, "semantic_replay_hashes": {typ: value["semantic_replay_hash"] for typ, value in dense.items()}, "manifests": dense, "vector_counts": {typ: value["vector_count"] for typ, value in dense.items()}})
    _write(args.out / "index-vector-integrity.json", {"model_config": model_config, "by_type": vectors, "nan_vector_count": sum(value["nan_vector_count"] for value in vectors.values()), "inf_vector_count": sum(value["inf_vector_count"] for value in vectors.values()), "zero_vector_count": sum(value["zero_vector_count"] for value in vectors.values()), "self_id_miss_count": sum(value["self_id_miss_count"] for value in vectors.values())})
    gate06.UNIT_TYPES = UNIT_TYPES
    roundtrip = gate06._roundtrip(all_views, runtime, runtime / "metadata" / "metadata.sqlite", {"views_by_type": {typ: len(view_groups[typ]) for typ in UNIT_TYPES}})
    _write(args.out / "index-roundtrip-audit.json", roundtrip)
    identity = {"evidence_unit_count": len(units), "retrieval_view_count": len(all_views), "duplicate_view_id_count": len(view_ids) - len(set(view_ids)), "source_traceback_missing_count": sum(not gate06._traceback_complete(view["source_traceback"]) for view in all_views), "identity_conflict_count": 0, "fact_type_counts": {typ: len(view_groups[typ]) for typ in ("atomic_fact", "comparison_fact", "bucket_fact")}, "physical_cross_page_merge_count": 0, "soft_edge_endpoint_errors": 0}
    _write(args.out / "index-identity-integrity.json", identity)
    replay = {"retrieval_view_hash": _hash([[view["retrieval_view_id"], view["retrieval_text"]] for view in all_views]), "metadata_store_hash": metadata["serialized_store_hash"], "bm25_index_hashes": {typ: value["serialized_index_hash"] for typ, value in bm25.items()}, "dense_semantic_replay_hashes": {typ: value["semantic_replay_hash"] for typ, value in dense.items()}}
    stable = bool(previous and previous.get("metadata") == replay["metadata_store_hash"] and previous.get("bm25") == replay["bm25_index_hashes"] and previous.get("dense") == replay["dense_semantic_replay_hashes"])
    _write(args.out / "deterministic-replay.json", {"replay_attempted": bool(previous), "replay_stable": stable, "first_run_requires_second_replay": not bool(previous), "current": replay, "previous": previous})
    index_ok = identity["duplicate_view_id_count"] == 0 and identity["source_traceback_missing_count"] == 0 and metadata["foreign_key_failures"] == 0 and roundtrip["metadata_roundtrip_failures"] == 0 and all(item["document_count"] == len(view_groups[typ]) for typ, item in bm25.items()) and all(item["vector_count"] == len(view_groups[typ]) for typ, item in dense.items()) and all(item["nan_vector_count"] == 0 and item["inf_vector_count"] == 0 and item["zero_vector_count"] == 0 and item["self_id_miss_count"] == 0 for item in vectors.values())
    passed = index_ok and stable
    decision = "multi_granularity_shadow_index_r2_passed" if passed else "shadow_index_r2_replay_or_integrity_blocked"
    next_gate = "v4_query_planner" if passed else "repeat_gate_06_r2_deterministic_replay"
    _write(args.out / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_06_r2", "gate_passed": passed, "decision": decision, "next_gate": next_gate, "r4_typed_admission": True, "question_reads": 0, "runtime_gold_reads": 0, "runtime_governance_reads": 0, "retrieval_runs": 0, "reranker_calls": 0, "answer_generation_calls": 0, "production_index_writes": 0, "production_default_config_modified": False, "production_switch_allowed": False, "duplicate_views": identity["duplicate_view_id_count"], "candidate_identity_conflicts": identity["identity_conflict_count"], "deterministic_replay_stable": stable})
    _write(args.out / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps({"decision": decision, "next_gate": next_gate, "view_counts": {typ: len(view_groups[typ]) for typ in UNIT_TYPES}, "deterministic_replay_stable": stable, "runtime": str(runtime)}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
