#!/usr/bin/env python3
"""Build and seal the zero-Gold R8-SE1 candidate semantic-fact registry."""

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

from src.pdf_retrieval_v4.runtime_semantic_fact_identity import (  # noqa: E402
    IDENTITY_FIELDS,
    IDENTITY_SCHEMA,
    deduplicate_facts,
    expand_authoritative_evidence,
)

EVAL = ROOT / "artifacts/evaluation"
R33 = EVAL / "pdf-retrieval-v4-gate-08-r8-r3-3"
G03 = EVAL / "pdf-retrieval-v4-gate-03-r2"
G05 = EVAL / "pdf-retrieval-v4-gate-05-r5"
OUT = EVAL / "pdf-retrieval-v4-gate-08-r8-se1-p0"

CATALOG_FILES = {
    "atomic": "atomic-facts.jsonl",
    "comparison": "comparison-facts.jsonl",
    "bucket": "bucket-facts.jsonl",
    "matrix": "row-matrices.jsonl",
    "narrative": "narrative-evidence.jsonl",
}
ID_FIELDS = {
    "atomic": "semantic_fact_id",
    "comparison": "semantic_fact_id",
    "bucket": "semantic_fact_id",
    "matrix": "semantic_fact_id",
    "narrative": "semantic_evidence_id",
}


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
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, compresslevel=9, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    r33_prediction = R33 / "main_rerank_predictions.jsonl.gz"
    structured_path = G05 / "structured-views.jsonl"
    bridge_path = G05 / "bridge-results.jsonl"

    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_counts: dict[str, int] = {}
    for case in read_jsonl(r33_prediction):
        case_id = str(case["case_id"])
        candidates = case["ranked_candidates"]
        case_counts[case_id] = len(candidates)
        for item in candidates:
            occurrences[str(item["candidate_key"])].append(
                {"case_id": case_id, "pre_rerank_rank": int(item["pre_rerank_rank"])}
            )
    if len(case_counts) != 72 or sum(case_counts.values()) != 7200 or set(case_counts.values()) != {100}:
        raise RuntimeError("r3_3_top100_input_contract_blocked")
    candidate_keys = set(occurrences)

    structured: dict[str, dict[str, Any]] = {}
    for view in read_jsonl(structured_path):
        key = str(view["candidate_key"])
        if key in candidate_keys:
            structured[key] = view
    bridge: dict[str, dict[str, Any]] = {}
    for result in read_jsonl(bridge_path):
        key = str(result["candidate_key"])
        if key in candidate_keys:
            bridge[key] = result

    supported_prefixes = set(CATALOG_FILES)
    needed_ids = {
        str(evidence_id)
        for view in structured.values()
        for evidence_id in (
            list(view.get("semantic_evidence_ids") or [])
            + [fact.get("evidence_id") for fact in (view.get("facts") or []) if fact.get("evidence_id")]
        )
        if str(evidence_id).partition(":")[0] in supported_prefixes
    }
    needed_row_ids = {
        str(row_id)
        for view in structured.values()
        for row_id in (view.get("row_ids") or [])
    }
    authoritative: dict[str, dict[str, Any]] = {}
    evidence_by_row: dict[str, set[str]] = defaultdict(set)
    catalog_hashes: dict[str, str] = {}
    for prefix, filename in CATALOG_FILES.items():
        path = G03 / filename
        catalog_hashes[filename] = sha256(path)
        id_field = ID_FIELDS[prefix]
        for record in read_jsonl(path):
            evidence_id = str(record[id_field])
            row_id = str(record.get("row_id") or "")
            if evidence_id in needed_ids or row_id in needed_row_ids:
                authoritative[evidence_id] = record
                if row_id:
                    evidence_by_row[row_id].add(evidence_id)

    missing_authoritative = sorted(needed_ids - set(authoritative))
    if missing_authoritative:
        raise RuntimeError(f"missing_authoritative_evidence:{len(missing_authoritative)}")

    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    expansion_counts: Counter[str] = Counter()
    for candidate_key in sorted(candidate_keys):
        view = structured.get(candidate_key)
        bridge_result = bridge.get(candidate_key) or {}
        if view:
            structural_lineage_ids = sorted(
                {
                    str(value)
                    for value in view.get("semantic_evidence_ids") or []
                    if str(value).partition(":")[0] not in supported_prefixes
                }
            )
            evidence_ids = {
                str(value)
                for value in view.get("semantic_evidence_ids") or []
                if str(value).partition(":")[0] in supported_prefixes
            }
            evidence_ids.update(
                str(fact["evidence_id"])
                for fact in (view.get("facts") or [])
                if fact.get("evidence_id") and str(fact["evidence_id"]).partition(":")[0] in supported_prefixes
            )
            for row_id in view.get("row_ids") or []:
                evidence_ids.update(evidence_by_row.get(str(row_id), set()))
            evidence_ids = sorted(evidence_ids)
            expanded: list[dict[str, Any]] = []
            evidence_expansion: list[dict[str, Any]] = []
            narrative_lineage_ids: list[str] = []
            for evidence_id in evidence_ids:
                facts, status = expand_authoritative_evidence(evidence_id, authoritative[evidence_id])
                expanded.extend(facts)
                expansion_counts[status] += 1
                evidence_expansion.append(
                    {
                        "authoritative_evidence_id": evidence_id,
                        "semantic_equivalence_status": status,
                        "semantic_fact_ids": sorted({fact["semantic_fact_id"] for fact in facts}),
                    }
                )
                if status == "semantic_expansion_not_supported":
                    narrative_lineage_ids.append(evidence_id)
            facts = deduplicate_facts(expanded)
            context_status = "authoritative_structured" if facts else "structured_without_expandable_fact"
            row = {
                "candidate_key": candidate_key,
                "context_status": context_status,
                "bridge_grade": view.get("bridge_grade"),
                "document_id": view.get("document_id"),
                "semantic_evidence_ids": evidence_ids,
                "structural_lineage_ids": structural_lineage_ids,
                "semantic_fact_ids": [fact["semantic_fact_id"] for fact in facts],
                "semantic_facts": facts,
                "narrative_lineage_ids": sorted(narrative_lineage_ids),
                "evidence_expansion": evidence_expansion,
                "occurrences": sorted(occurrences[candidate_key], key=lambda item: (item["case_id"], item["pre_rerank_rank"])),
            }
        else:
            grade = str(bridge_result.get("grade") or "unmapped")
            context_status = "ambiguous_not_attached" if grade.startswith("B") else "unmapped"
            if str(bridge_result.get("failure_stage") or "") == "candidate_type_unsupported":
                context_status = "raw_only"
            row = {
                "candidate_key": candidate_key,
                "context_status": context_status,
                "bridge_grade": grade,
                "document_id": None,
                "semantic_evidence_ids": [],
                "structural_lineage_ids": [],
                "semantic_fact_ids": [],
                "semantic_facts": [],
                "narrative_lineage_ids": [],
                "evidence_expansion": [],
                "occurrences": sorted(occurrences[candidate_key], key=lambda item: (item["case_id"], item["pre_rerank_rank"])),
            }
        status_counts[row["context_status"]] += 1
        rows.append(row)

    registry_path = OUT / "candidate-semantic-fact-registry.jsonl.gz"
    write_jsonl_gz(registry_path, rows)
    schema = {
        "schema": IDENTITY_SCHEMA,
        "identity_function": "sha256(unit-separator-joined normalized fields in declared order)",
        "identity_fields": list(IDENTITY_FIELDS),
        "excluded_physical_fields": [
            "candidate_key", "pdf_page", "table_id", "table_fragment_id", "row_id", "evidence_id", "bbox", "retrieval_rank"
        ],
        "numeric_tolerance": False,
        "fuzzy_matching": False,
        "llm_equivalence": False,
        "row_matrix_expansion": "cell_level_dimension",
        "comparison_expansion": "exact_two_operands_or_fail_closed",
        "narrative_expansion": "strict_lineage_only",
    }
    write_json(OUT / "semantic-identity-schema.json", schema)
    integrity = {
        "gate": "pdf_retrieval_v4_gate_08_r8_se1_p0",
        "case_count": len(case_counts),
        "candidate_occurrence_count": sum(case_counts.values()),
        "candidate_occurrences_per_case": {"min": min(case_counts.values()), "max": max(case_counts.values())},
        "unique_candidate_count": len(candidate_keys),
        "registry_record_count": len(rows),
        "candidate_added": 0,
        "candidate_removed": 0,
        "candidate_mutation": 0,
        "needed_authoritative_evidence": len(needed_ids),
        "missing_authoritative_evidence": len(missing_authoritative),
        "context_status_counts": dict(sorted(status_counts.items())),
        "expansion_status_counts": dict(sorted(expansion_counts.items())),
        "gold_reads_before_seal": 0,
        "strict_source_binding_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,
        "expected_value_reads_before_seal": 0,
        "semantic_graph_runs": 0,
        "bridge_runs": 0,
        "retrieval_runs": 0,
        "embedding_calls": 0,
        "reranker_calls": 0,
    }
    write_json(OUT / "semantic-identity-integrity.json", integrity)
    manifest = {
        "r3_3_commit": "c8a4ef33103e40625ed7e0853d4c93cc7f6b18cd",
        "r3_3_main_prediction_sha256": sha256(r33_prediction),
        "r3_3_prediction_seal_sha256": sha256(R33 / "prediction-seal.json"),
        "structured_views_sha256": sha256(structured_path),
        "bridge_results_sha256": sha256(bridge_path),
        "gate03_catalog_sha256": catalog_hashes,
        "semantic_identity_source_sha256": sha256(ROOT / "src/pdf_retrieval_v4/runtime_semantic_fact_identity.py"),
        "registry_builder_source_sha256": sha256(Path(__file__)),
        "registry_sha256": sha256(registry_path),
    }
    write_json(OUT / "prediction-manifest.json", manifest)
    seal = {
        **integrity,
        **manifest,
        "registry_sha256": sha256(registry_path),
        "sealed": True,
        "post_benchmark_diagnostic": True,
        "historical_strict_physical_source_recall_at_5": "43/80",
    }
    write_json(OUT / "prediction-seal.json", seal)
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
