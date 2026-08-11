"""NF-OPT-18 R0: offline internal reranker representation audit.

This module intentionally has no model, retrieval, embedding, or Gold-scoring
dependencies.  It only reads sealed artifacts, reconstructs the representation
that was actually serialized to the internal reranker, and writes diagnostics.
"""

from __future__ import annotations

import argparse
import ast
import collections
import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


GATE = "NF-OPT-18-R0"
MODEL_ID = "Qwen/Qwen3-Reranker-4B"
MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
STRICT_BINDING_COUNT = 80
TOP100_EXPECTED = 68
TOP5_EXPECTED = 43
R3_COMMIT = "1e741b907fd88aaa2326dfa18cc400bbf41c074e"
INTERNAL_CALC_COMMIT = "237ccff"
BOUNDED_TOP100_SHA = "ced014c357a3c8862a5ae3723a4c618556156542fd63aa2a8fd4bf766b60f01a"

FIELDS = [
    "document_identity",
    "company_entity",
    "file_name",
    "page",
    "section",
    "table_title",
    "raw_text",
    "table_context",
    "row_label",
    "column_header",
    "metric",
    "metric_path",
    "period",
    "temporal_axis",
    "scale",
    "currency",
    "cell_value",
    "logical_table",
    "neighbor_text",
    "structured_semantic_evidence",
]

DIFF_DIMENSIONS = [
    "candidate_granularity",
    "narrative_context",
    "table_context",
    "document_identity",
    "company_entity",
    "section_table_title",
    "metric",
    "metric_path",
    "row_label",
    "column_header",
    "period",
    "scale",
    "currency",
    "neighbor_evidence",
    "operation_context",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).casefold()
    value = re.sub(r"[\u2010-\u2015\-_/|:;,()\[\]{}]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def token_set(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(value)))


def pctl(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def distribution(values: list[float | int]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values if value is not None]
    return {
        "count": len(numeric),
        "p50": pctl(numeric, 0.50),
        "p90": pctl(numeric, 0.90),
        "p95": pctl(numeric, 0.95),
        "p99": pctl(numeric, 0.99),
        "max": max(numeric) if numeric else None,
    }


def status(value: Any, source: str | None = None, *, partial: bool = False) -> dict[str, Any]:
    if value is not None and str(value).strip():
        return {"status": "partial" if partial else "present", "source": source, "included_in_model_input": True}
    return {"status": "absent", "source": None, "included_in_model_input": False}


def parse_document_view(view: str) -> dict[str, Any]:
    """Parse only fields visibly serialized in the frozen document_view string."""
    lines = view.splitlines()
    typed: dict[str, Any] = {}
    source_lines: list[str] = []
    in_source = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_source = False
            continue
        if stripped == "Source:":
            in_source = True
            continue
        if in_source:
            source_lines.append(line)
        match = re.match(r"^(Document|Page|Statement|Section|Table|Metric Path|Metric|Row|Column Header|Column|Header|Type|Period|Segment|Bucket|Value|Scale|Currency|Logical Table):\s*(.*)$", stripped, re.I)
        if match:
            key = normalize_text(match.group(1)).replace(" ", "_")
            typed[key] = match.group(2).strip()

    source = "\n".join(source_lines).strip()
    block_match = re.search(r"(?:^|\n)Block Type:\s*([^\n]+)", view, re.I)
    block_type = block_match.group(1).strip().casefold() if block_match else ""
    evidence_type = typed.get("type", "").casefold()
    raw_period = typed.get("period", "")
    raw_has_period = bool(re.search(r"\b(?:FY\s*)?20\d{2}\b|\bQ[1-4]\b", source, re.I))
    raw_has_currency = bool(re.search(r"\$|\b(?:USD|EUR|GBP|JPY|CNY)\b", source, re.I))
    raw_has_scale = bool(re.search(r"\b(?:in|dollars in)\s+(?:millions?|thousands?|billions?)\b", source, re.I))
    raw_has_number = bool(re.search(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?", source))
    table = block_type in {"table", "table_row", "table_fragment", "row"} or evidence_type in {"logical_table", "semantic_row"} or "|" in source
    raw_text = source if source else re.sub(r"\s+", " ", view).strip()

    values: dict[str, dict[str, Any]] = {
        "document_identity": status(typed.get("document"), "[DOCUMENT]/Document"),
        "company_entity": status(typed.get("document"), "Document identifier", partial=True),
        "file_name": status(None),
        "page": status(typed.get("page"), "[CONTENT]/Page"),
        "section": status(typed.get("section"), "[STRUCTURE]/Section"),
        "table_title": status(typed.get("table"), "[STRUCTURE]/Table"),
        "raw_text": status(raw_text, "[CONTENT]/Source"),
        "table_context": status(table, "Block Type/Evidence Type"),
        "row_label": status(typed.get("row"), "[STRUCTURE]/Row") if typed.get("row") else status(source if block_type == "table_row" else None, "[CONTENT]/Source", partial=True),
        "column_header": status(typed.get("column_header") or typed.get("column") or typed.get("header"), "[STRUCTURE]/Header"),
        "metric": status(typed.get("metric") or typed.get("metric_path"), "[STRUCTURE]/Metric"),
        "metric_path": status(typed.get("metric_path"), "[STRUCTURE]/Metric Path"),
        "period": status(raw_period, "[EVIDENCE]/Period") if raw_period else status(source if raw_has_period else None, "[CONTENT]/Source", partial=True),
        "temporal_axis": status(raw_period, "[EVIDENCE]/Period") if raw_period else status(None),
        "scale": status(typed.get("scale"), "[EVIDENCE]/Scale") if typed.get("scale") else status(source if raw_has_scale else None, "[CONTENT]/Source", partial=True),
        "currency": status(typed.get("currency"), "[EVIDENCE]/Currency") if typed.get("currency") else status(source if raw_has_currency else None, "[CONTENT]/Source", partial=True),
        "cell_value": status(typed.get("value"), "[EVIDENCE]/Value") if typed.get("value") else status(source if raw_has_number else None, "[CONTENT]/Source", partial=True),
        "logical_table": status(typed.get("logical_table"), "[STRUCTURE]/Logical Table"),
        "neighbor_text": status(None),
        "structured_semantic_evidence": status(evidence_type, "[EVIDENCE]/Type"),
    }
    for key in FIELDS:
        values.setdefault(key, status(None))
    return {
        "fields": values,
        "typed": typed,
        "source": source,
        "raw_text": raw_text,
        "block_type": block_type,
        "evidence_type": evidence_type,
        "table": table,
        "document_id": typed.get("document"),
    }


def extract_instruction(source: str) -> str | None:
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RERANK_INSTRUCTION" for t in node.targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    return value
    except (SyntaxError, ValueError):
        pass
    match = re.search(r"RERANK_INSTRUCTION\s*=\s*[\"'](.+?)[\"']", source, re.S)
    return match.group(1) if match else None


def field_summary(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    counts = collections.Counter(record["parsed"]["fields"][field]["status"] for record in records)
    total = len(records)
    return {
        "count": total,
        "status_counts": dict(counts),
        "coverage": (counts["present"] + counts["partial"]) / total if total else 0.0,
        "typed_present": counts["present"],
    }


def graph_available(record: dict[str, Any]) -> dict[str, bool]:
    evidence = record.get("authoritative_evidence") or []
    if not evidence:
        return {field: False for field in FIELDS}
    item = evidence[0]
    context = item.get("context") or {}
    payload = item.get("semantic_payload") or {}
    metadata = record.get("metadata") or {}
    def any_value(*values: Any) -> bool:
        return any(value is not None and value != "" and value != [] for value in values)
    return {
        "document_identity": any_value(item.get("document_id"), context.get("document_id"), payload.get("document_id")),
        "company_entity": False,
        "file_name": False,
        "page": any_value(context.get("pdf_page"), payload.get("pdf_page"), metadata.get("pdf_page")),
        "section": any_value(context.get("section_path")),
        "table_title": any_value(context.get("table_title")),
        "raw_text": any_value(metadata.get("retrieval_text"), record.get("raw_text")),
        "table_context": any_value(context.get("table_fragment_id"), metadata.get("logical_table_ids")),
        "row_label": any_value(context.get("raw_row_label"), payload.get("raw_label")),
        # Gate03 represents column headers/axes through normalized period-bearing
        # dimensions even when the authoritative attachment has no literal
        # ``column_header`` key.  This is Graph availability, not serializer
        # consumption; the frozen document_view has no typed header line.
        "column_header": any_value(
            context.get("column_header"),
            context.get("headers"),
            context.get("period"),
            context.get("periods"),
            payload.get("column_header"),
            payload.get("period"),
            payload.get("periods"),
            metadata.get("periods"),
        ),
        "metric": any_value(context.get("metric_path"), metadata.get("metric_paths")),
        "metric_path": any_value(context.get("metric_path"), metadata.get("metric_paths")),
        "period": any_value(context.get("period"), context.get("periods"), metadata.get("periods"), payload.get("period"), payload.get("periods")),
        "temporal_axis": any_value(context.get("temporal_axis"), metadata.get("temporal_types"), payload.get("temporal_axis")),
        "scale": any_value(context.get("scale"), context.get("scale_unit"), metadata.get("scale"), payload.get("scale")),
        "currency": any_value(context.get("currency"), context.get("currency_code"), metadata.get("currency"), payload.get("currency")),
        "cell_value": any_value(context.get("value"), context.get("cell_value"), payload.get("value"), payload.get("normalized_value")),
        "logical_table": any_value(context.get("logical_table_id"), metadata.get("logical_table_ids")),
        "neighbor_text": any_value(context.get("neighbor_text"), payload.get("neighbor_text")),
        "structured_semantic_evidence": True,
    }


def compare_dimension(gold: dict[str, Any], competitor: dict[str, Any], dimension: str) -> str:
    mapping = {
        "metric": "metric",
        "period": "period",
        "table": "table_title",
        "statement": "statement",
        "row": "row_label",
        "header": "column_header",
        "scale": "scale",
        "currency": "currency",
    }
    if dimension == "statement":
        a = gold["parsed"]["typed"].get("statement", "")
        b = competitor["parsed"]["typed"].get("statement", "")
    else:
        key = mapping.get(dimension, dimension)
        a = gold["parsed"]["typed"].get(key, "") if key in {"metric", "metric_path", "period", "table_title", "row_label", "column_header", "scale", "currency"} else ""
        b = competitor["parsed"]["typed"].get(key, "") if key in {"metric", "metric_path", "period", "table_title", "row_label", "column_header", "scale", "currency"} else ""
        if not a and key in gold["parsed"]["fields"] and gold["parsed"]["fields"][key]["status"] == "partial":
            a = gold["parsed"]["source"]
        if not b and key in competitor["parsed"]["fields"] and competitor["parsed"]["fields"][key]["status"] == "partial":
            b = competitor["parsed"]["source"]
    a = normalize_text(a)
    b = normalize_text(b)
    if not a and not b:
        return "unknown"
    if not a or not b:
        return "partial"
    if a == b:
        return "no"
    return "yes"


def ambiguity_record(gold: dict[str, Any], competitor: dict[str, Any]) -> dict[str, Any]:
    comparisons = {key: compare_dimension(gold, competitor, key) for key in ["metric", "period", "table", "row", "header", "scale", "currency", "statement"]}
    discriminators = sum(value == "yes" for value in comparisons.values())
    if discriminators == 0:
        level = "high"
    elif discriminators <= 2:
        level = "medium"
    else:
        level = "low"
    return {"comparisons": comparisons, "discriminating_dimensions": discriminators, "representation_ambiguity": level}


def classify_granularity(parsed: dict[str, Any]) -> str:
    block = parsed.get("block_type", "")
    evidence = parsed.get("evidence_type", "")
    if evidence in {"semantic_row", "narrative_evidence"}:
        return "semantic_evidence"
    if block in {"table", "table_fragment"}:
        return "table_fragment"
    if block in {"table_row", "row", "row_fragment"}:
        return "row_fragment"
    if block in {"cell", "cell_fragment"}:
        return "cell_fragment"
    if block in {"text", "paragraph", "paragraph_chunk"}:
        return "paragraph_chunk"
    if parsed.get("table") and parsed.get("source"):
        return "mixed_fragment"
    return "unknown"


def query_plan_info(view: str) -> dict[str, Any]:
    task = re.search(r"Task:\s*([^\n]+)", view)
    operation = re.search(r"Operation:\s*([^\n]+)", view)
    slots = re.findall(r"-\s*Slot\s+\d+:\s*([^\n]+)", view)
    return {
        "task": task.group(1).strip() if task else None,
        "operation": operation.group(1).strip() if operation else None,
        "slot_count": len(slots),
        "slots": slots,
        "is_calculation": bool(re.search(r"calculation|growth|difference|ratio|percentage|sum|average", view, re.I)),
    }


def count_graph_records(graph_dir: Path) -> dict[str, Any]:
    files = {
        "tables": graph_dir / "logical-tables.jsonl",
        "rows": graph_dir / "semantic-rows.jsonl",
        "atomic_facts": graph_dir / "atomic-facts.jsonl",
        "comparison_facts": graph_dir / "comparison-facts.jsonl",
        "bucket_facts": graph_dir / "bucket-facts.jsonl",
        "row_matrices": graph_dir / "row-matrices.jsonl",
        "axis_bindings": graph_dir / "axis-bindings.jsonl",
        "currency_resolutions": graph_dir / "currency-resolutions.jsonl",
        "scale_resolutions": graph_dir / "scale-resolutions.jsonl",
    }
    counts: dict[str, Any] = {}
    for key, path in files.items():
        counts[key] = sum(1 for _ in read_jsonl(path)) if path.exists() else None
    cell_count = 0
    cell_method = "row_matrices.cells"
    if files["axis_bindings"].exists():
        cell_count = counts["axis_bindings"] or 0
        cell_method = "axis-bindings records (cell/axis identity)"
    if not cell_count and files["row_matrices"].exists():
        for record in read_jsonl(files["row_matrices"]):
            found = False
            for key in ("cells", "cell_values", "values", "matrix", "dimensions"):
                value = record.get(key)
                if isinstance(value, list):
                    cell_count += len(value)
                    found = True
                    break
            if not found:
                for value in record.values():
                    if isinstance(value, list) and value and isinstance(value[0], dict) and any("value" in item for item in value):
                        cell_count += len(value)
                        found = True
                        break
    if not cell_count:
        cell_method = "atomic+comparison+bucket fallback"
        cell_count = sum(counts.get(key) or 0 for key in ("atomic_facts", "comparison_facts", "bucket_facts"))
    counts["cell_count"] = cell_count
    counts["cell_count_method"] = cell_method
    corpus_rules = {
        "metric_path": [("metric-paths", "metric_path"), ("row-matrices", "metric_path"), ("atomic-facts", "metric_path"), ("bucket-facts", "metric_path"), ("comparison-facts", "metric_path")],
        "row_label": [("metric-paths", "raw_row_label")],
        "column_header": [("axis-bindings", "normalized_period")],
        "period": [("axis-bindings", "normalized_period"), ("atomic-facts", "normalized_period")],
        "scale": [("row-matrices", "scale"), ("atomic-facts", "scale"), ("bucket-facts", "scale"), ("scale-resolutions", "scale")],
        "currency": [("row-matrices", "currency_code"), ("atomic-facts", "currency_code"), ("bucket-facts", "currency_code"), ("currency-resolutions", "currency_code")],
        "table_title": [("logical-tables", "table_title")],
        "logical_table": [("logical-tables", "table_fragment_id"), ("row-matrices", "table_fragment_id")],
    }
    corpus_fields: dict[str, dict[str, Any]] = {}
    for field, rules in corpus_rules.items():
        assets: dict[str, int] = {}
        total = 0
        for asset, key in rules:
            path = graph_dir / f"{asset}.jsonl"
            if not path.exists():
                continue
            count = sum(1 for record in read_jsonl(path) if record.get(key) not in (None, "", [], {}))
            assets[asset] = count
            total += count
        corpus_fields[field] = {"asset_counts": assets, "nonempty_record_count_sum": total}
    counts["corpus_field_availability"] = corpus_fields
    return counts


def load_t2_contract(backend_root: Path) -> dict[str, Any]:
    t2_dir = backend_root / "artifacts/evaluation/t2-ragbench-05-qwen3-strong-reranker"
    frozen = read_json(t2_dir / "frozen-contract.json")
    runtime = read_json(t2_dir / "runtime-equivalence.json") if (t2_dir / "runtime-equivalence.json").exists() else {}
    common = next(iter(backend_root.glob("**/t2_ragbench_03_common.py")), None)
    source_hash = sha256_file(common) if common else None
    return {
        "source_artifacts": {
            "frozen_contract": str(t2_dir / "frozen-contract.json"),
            "runtime_equivalence": str(t2_dir / "runtime-equivalence.json"),
            "common_code": str(common) if common else None,
            "common_code_sha256": source_hash,
        },
        "model": {
            "model_id": frozen.get("model_id", MODEL_ID),
            "revision": frozen.get("model_revision", MODEL_REVISION),
            "instruction": frozen.get("instruction"),
            "instruction_sha256": frozen.get("instruction_sha256"),
            "max_length": frozen.get("max_length"),
            "dtype": frozen.get("dtype"),
            "batch_size": frozen.get("batch_size"),
            "scoring": frozen.get("scoring"),
        },
        "candidate_source": frozen.get("candidate_source", "frozen T2-01 BM25 Top50"),
        "candidate_depth": frozen.get("candidate_depth", 50),
        "query": {
            "representation": "original question formatted with company_name : question",
            "instruction": "frozen T2-05 instruction",
            "query_rewrite": False,
        },
        "context": {
            "representation": "original whole-context text/table string",
            "whole_context_supplied": True,
            "narrative_context": "present",
            "table_context": "present when present in original context",
            "entity_information": "raw context/query availability",
            "metric_phrase": "raw context availability",
            "period": "raw context availability",
            "typed_semantic_fields": "not separately injected",
            "candidate_granularity": "whole_context_document",
        },
        "runtime": runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    backend_root = args.backend_root.resolve()
    evaluation_root = backend_root / "artifacts/evaluation"
    output = (args.output or evaluation_root / "nf-opt-18-r0-reranker-representation-audit").resolve()
    output.mkdir(parents=True, exist_ok=True)

    p0_dir = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3-p0"
    r33_dir = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3"
    r31a_dir = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-1a"
    r2a_dir = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r2a-2"
    strict_dir = evaluation_root / "pdf-retrieval-v4-strict-source-contract"
    graph_dir = evaluation_root / "pdf-retrieval-v4-gate-03-r2"

    input_paths = [
        p0_dir / "queryplan-rerank-input-views.jsonl.gz",
        r33_dir / "main_rerank_predictions.jsonl.gz",
        r31a_dir / "top100-authoritative-context-v2.jsonl.gz",
        r2a_dir / "bounded-top100-predictions.jsonl.gz",
    ]
    input_hashes = {str(path.relative_to(evaluation_root)): sha256_file(path) for path in input_paths if path.exists()}

    views = read_jsonl(input_paths[0])
    predictions = read_jsonl(input_paths[1])
    authoritative = read_jsonl(input_paths[2])
    bounded_top100 = read_jsonl(input_paths[3])
    view_map: dict[tuple[str, str], dict[str, Any]] = {}
    query_views: dict[str, str] = {}
    for case in views:
        case_id = case["case_id"]
        query_views[case_id] = case.get("main_query_view", "")
        for candidate in case.get("candidates", []):
            parsed = parse_document_view(candidate.get("document_view", ""))
            view_map[(case_id, candidate["candidate_key"])] = {
                **candidate,
                "parsed": parsed,
                "query_plan": query_plan_info(case.get("main_query_view", "")),
            }

    prediction_map: dict[str, dict[str, Any]] = {row["case_id"]: row for row in predictions}
    authoritative_map: dict[tuple[str, str], dict[str, Any]] = {}
    for case in authoritative:
        for candidate in case.get("candidates", []):
            authoritative_map[(case["case_id"], candidate["candidate_key"])] = candidate
    p0_candidate_sets = {
        case["case_id"]: {candidate["candidate_key"] for candidate in case.get("candidates", [])}
        for case in views
    }
    bounded_candidate_sets = {
        case["case_id"]: {candidate["candidate_key"] for candidate in case.get("candidates", [])}
        for case in bounded_top100
    }
    bounded_identity_mismatches = sum(
        p0_candidate_sets.get(case_id) != bounded_candidate_sets.get(case_id)
        or len(p0_candidate_sets.get(case_id, set())) != 100
        or len(bounded_candidate_sets.get(case_id, set())) != 100
        for case_id in p0_candidate_sets
    )
    if bounded_identity_mismatches:
        raise RuntimeError(f"frozen Top100 identity mismatch in {bounded_identity_mismatches} cases")
    bounded_seal_path = r2a_dir / "prediction-seal.json"
    bounded_seal = read_json(bounded_seal_path) if bounded_seal_path.exists() else {}

    # Registry/seal is deliberately written before strict Gold bindings are opened.
    seal_payload = {
        "gate": GATE,
        "gold_reads_before_seal": 0,
        "retrieval_rerun": False,
        "model_execution": False,
        "input_hashes": input_hashes,
        "view_occurrences": len(view_map),
        "prediction_cases": len(predictions),
        "candidate_mutation": 0,
        "bounded_top100_prediction_sha256": bounded_seal.get("prediction_sha256"),
        "bounded_top100_identity_mismatches": bounded_identity_mismatches,
        "sealed_sha256": sha256_bytes(canonical_json({"input_hashes": input_hashes, "view_occurrences": len(view_map)}).encode()),
    }
    write_json(output / "internal-representation-seal.json", seal_payload)

    bindings = read_jsonl(strict_dir / "strict-gold-source-bindings.jsonl")
    if len(bindings) != STRICT_BINDING_COUNT:
        raise RuntimeError(f"strict binding count mismatch: {len(bindings)}")

    # Internal model contract recovered from actual frozen code + artifacts.
    serializer_path = backend_root / "src/pdf_retrieval_v4/structure_aware_rerank_view.py"
    serializer_source = serializer_path.read_text(encoding="utf-8")
    instruction = extract_instruction(serializer_source)
    all_records = list(view_map.values())
    overall_fields = {field: field_summary(all_records, field) for field in FIELDS}
    internal_contract = {
        "gate": GATE,
        "source": {
            "frozen_input_views": str(input_paths[0]),
            "prediction_artifact": str(input_paths[1]),
            "serializer_code": str(serializer_path),
            "serializer_code_sha256": sha256_file(serializer_path),
            "model_manifest": str(evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-2/model-manifest.json"),
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "max_length": 8192,
            "batch_size": 1,
            "scoring": "last-position full-vocabulary causal-lm yes/no logits; log_softmax(no, yes) yes probability",
        },
        "instruction": {
            "text": instruction,
            "sha256": sha256_bytes(instruction.encode()) if instruction else None,
            "source": "RERANK_INSTRUCTION in structure_aware_rerank_view.py",
        },
        "query_serialization": {
            "template": "[QUESTION] raw_question + [QUERY PLAN] Task/Operation/Statement Hint/Required evidence slots",
            "statement_hint": "optional; frozen R3.3 manifest reports nonempty=0",
            "gold_independent": True,
        },
        "candidate_source": "bounded Top100 frozen R8-R2A.2 / R3.3 input views",
        "candidate_type": "mixed PDF-derived chunks and structured evidence views",
        "candidate_serialization": {
            "template": "[DOCUMENT] + [STRUCTURE] + [EVIDENCE] + [CONTENT]",
            "fields": {
                field: {
                    **summary,
                    "status": "present" if summary["coverage"] >= 0.999999 else ("partial" if summary["coverage"] > 0 else "absent"),
                    "source": "frozen R3.3 document_view actual-input scan",
                    "included_in_model_input": summary["coverage"] > 0,
                }
                for field, summary in overall_fields.items()
            },
            "actual_field_status_rule": "present=typed serialized field; partial=only visible in raw Source; absent=not in model input",
        },
        "model_execution": False,
        "retrieval_rerun": False,
        "production_switch_allowed": False,
    }
    write_json(output / "internal-reranker-contract.json", internal_contract)

    t2_contract = load_t2_contract(backend_root)
    write_json(output / "t2-reranker-contract.json", t2_contract)

    # Candidate inventory split by strict physical source bindings (not semantic identity).
    gold_keys = {(binding["case_id"], binding["candidate_key"]) for binding in bindings}
    gold_records = [record for key, record in view_map.items() if key in gold_keys]
    non_gold_records = [record for key, record in view_map.items() if key not in gold_keys]
    def inventory(records: list[dict[str, Any]], strict_binding_count: int | None = None) -> dict[str, Any]:
        return {
            "candidate_occurrences": len(records),
            "unique_candidate_keys": len({record["candidate_key"] for record in records}),
            "strict_source_binding_occurrences": strict_binding_count,
            "field_coverage": {field: field_summary(records, field) for field in FIELDS},
            "context_status": dict(collections.Counter(record.get("context_status") for record in records)),
            "granularity": dict(collections.Counter(classify_granularity(record["parsed"]) for record in records)),
        }
    write_json(output / "internal-candidate-representation-inventory.json", {
        "candidate_universe": "R3.3 frozen 72x100 views",
        "overall": inventory(all_records),
        "gold_candidates": inventory(gold_records, len(bindings)),
        "non_gold_candidates": inventory(non_gold_records),
        "strict_gold_identity": "(case_id, source_index, candidate_key); physical source binding only",
    })

    graph_rows = []
    for key, record in view_map.items():
        graph_rows.append({
            "case_id": key[0],
            "candidate_key": key[1],
            "available": graph_available(authoritative_map.get(key, {})),
            "consumed": {field: record["parsed"]["fields"][field]["status"] in {"present", "partial"} for field in FIELDS},
        })
    graph_consumption: dict[str, Any] = {
        "graph_corpus_counts": count_graph_records(graph_dir),
        "occurrence_count": len(graph_rows),
        "fields": {},
        "gold_reads_before_feature_or_representation_seal": 0,
    }
    for field in FIELDS:
        available_count = sum(row["available"][field] for row in graph_rows)
        consumed_count = sum(row["consumed"][field] for row in graph_rows)
        consumed_on_available = sum(row["available"][field] and row["consumed"][field] for row in graph_rows)
        graph_consumption["fields"][field] = {
            "available_count": available_count,
            "consumed_count": consumed_count,
            "consumed_on_available_count": consumed_on_available,
            "consumption_rate_over_available": consumed_on_available / available_count if available_count else 0.0,
            "consumed_status_counts": dict(collections.Counter(view_map[(row["case_id"], row["candidate_key"])] ["parsed"]["fields"][field]["status"] for row in graph_rows)),
        }
    write_json(output / "semantic-graph-consumption.json", graph_consumption)

    # Reconstruct strict cohorts from the sealed R3.3 ranking.
    cohort_rows: list[dict[str, Any]] = []
    for binding in bindings:
        case_id = binding["case_id"]
        candidate_key = binding["candidate_key"]
        ranked = prediction_map[case_id]["ranked_candidates"]
        rank_by_key = {item["candidate_key"]: item["post_rerank_rank"] for item in ranked}
        rank = rank_by_key.get(candidate_key)
        cohort = "C2_top100_absent" if rank is None or rank > 100 else ("C0_top5_success" if rank <= 5 else "C1_top100_present_top5_miss")
        cohort_rows.append({
            "case_id": case_id,
            "source_index": binding["source_index"],
            "gold_candidate_key": candidate_key,
            "gold_post_rerank_rank": rank,
            "cohort": cohort,
            "document_id": binding.get("document_id"),
            "evidence_id": binding.get("evidence_id"),
        })
    cohort_counts = collections.Counter(row["cohort"] for row in cohort_rows)
    if sum(cohort_counts.values()) != STRICT_BINDING_COUNT:
        raise RuntimeError("cohort union does not cover all strict bindings")
    write_json(output / "failure-cohorts.json", {
        "definition": {
            "C0_top5_success": "Gold post_rerank_rank <= 5",
            "C1_top100_present_top5_miss": "Gold post_rerank_rank 6..100",
            "C2_top100_absent": "Gold not present in frozen Top100",
        },
        "counts": dict(cohort_counts),
        "rows": cohort_rows,
        "candidate_mutation": 0,
    })

    record_by_key = view_map
    cohort_ambiguity: dict[str, list[dict[str, Any]]] = {"C0_top5_success": [], "C1_top100_present_top5_miss": []}
    compression_rows: list[dict[str, Any]] = []
    for row in cohort_rows:
        if row["cohort"] == "C2_top100_absent":
            continue
        case_id = row["case_id"]
        gold = record_by_key.get((case_id, row["gold_candidate_key"]))
        top5 = [item for item in prediction_map[case_id]["ranked_candidates"] if item["post_rerank_rank"] <= 5]
        competitors = [item for item in top5 if item["candidate_key"] != row["gold_candidate_key"]]
        competitor = record_by_key.get((case_id, competitors[0]["candidate_key"])) if competitors else None
        if not gold or not competitor:
            ambiguity = {"comparisons": {}, "discriminating_dimensions": 0, "representation_ambiguity": "unknown"}
            tags = ["other"]
        else:
            ambiguity = ambiguity_record(gold, competitor)
            tags: list[str] = []
            parsed = gold["parsed"]
            if parsed["fields"]["table_context"]["status"] == "absent":
                tags.append("missing_table_context")
            if parsed["fields"]["table_title"]["status"] == "absent":
                tags.append("missing_table_title")
            if parsed["fields"]["row_label"]["status"] == "absent":
                tags.append("missing_row_label")
            if parsed["fields"]["column_header"]["status"] == "absent":
                tags.append("missing_column_header")
            if parsed["fields"]["metric"]["status"] == "absent":
                tags.append("missing_metric_context")
            if parsed["fields"]["metric_path"]["status"] == "absent":
                tags.append("missing_metric_path")
            if parsed["fields"]["period"]["status"] == "absent":
                tags.append("missing_period_context")
            if parsed["fields"]["scale"]["status"] == "absent":
                tags.append("missing_scale")
            if parsed["fields"]["currency"]["status"] == "absent":
                tags.append("missing_currency")
            if parsed["fields"]["neighbor_text"]["status"] == "absent":
                tags.append("missing_neighbor_context")
            if classify_granularity(parsed) in {"row_fragment", "cell_fragment", "semantic_evidence"} and not parsed["fields"]["neighbor_text"]["included_in_model_input"]:
                tags.append("candidate_fragment_too_local")
            if parsed["table"] and not parsed["fields"]["column_header"]["included_in_model_input"] and len(re.findall(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?", parsed["source"])) >= 3:
                tags.append("ambiguous_numeric_fragment")
            available = graph_available(authoritative_map.get((case_id, row["gold_candidate_key"]), {}))
            if any(available[field] and parsed["fields"][field]["status"] == "absent" for field in ["table_title", "row_label", "column_header", "metric_path", "period", "scale", "currency", "logical_table"]):
                tags.append("structure_available_but_not_serialized")
            overlap = token_set(parsed["source"]) & token_set(competitor["parsed"]["source"])
            union = token_set(parsed["source"]) | token_set(competitor["parsed"]["source"])
            if union and len(overlap) / len(union) >= 0.8:
                tags.append("gold_and_negative_text_near_duplicate")
            plan = gold.get("query_plan", {})
            if plan.get("is_calculation"):
                tags.append("calculation_operand_competition")
            if plan.get("slot_count", 0) > 1:
                tags.append("multi_slot_competition")
            if ambiguity["comparisons"].get("statement") == "yes":
                tags.append("same_metric_wrong_statement")
            if ambiguity["comparisons"].get("period") == "yes":
                tags.append("wrong_period")
            if not any(value == "yes" for value in ambiguity["comparisons"].values()):
                tags.append("generic_financial_similarity")
            if not tags:
                tags.append("other")
        ambiguity_row = {"case_id": case_id, "source_index": row["source_index"], **ambiguity}
        cohort_ambiguity[row["cohort"]].append(ambiguity_row)
        if row["cohort"] == "C1_top100_present_top5_miss":
            compression_rows.append({
                **row,
                "top5_candidate_ids": [item["candidate_key"] for item in top5],
                "gold_representation": {"candidate_key": row["gold_candidate_key"], **gold["parsed"]} if gold else {},
                "top_competitor_representations": [{"candidate_key": competitor["candidate_key"], **competitor["parsed"]}] if competitor else [],
                "gold_vs_top_competitor": ambiguity,
                "diagnostic_tags": sorted(set(tags)),
            })
    write_jsonl_gz(output / "compression-failure-analysis.jsonl.gz", compression_rows)

    ambiguity_summary: dict[str, Any] = {"rule": "0 discriminating key dimensions=high; 1-2=medium; >=3=low; key dimensions metric/period/table/row/header/scale/currency/statement"}
    for cohort, rows in cohort_ambiguity.items():
        counts = collections.Counter(row["representation_ambiguity"] for row in rows)
        ambiguity_summary[cohort] = {"count": len(rows), "distribution": dict(counts), "high_rate": counts["high"] / len(rows) if rows else 0.0, "rows": rows}
    c0_high = ambiguity_summary["C0_top5_success"]["high_rate"]
    c1_high = ambiguity_summary["C1_top100_present_top5_miss"]["high_rate"]
    write_json(output / "representation-ambiguity.json", ambiguity_summary)

    # Token-length audit uses sealed internal prediction metadata, not model execution.
    internal_lengths = [item.get("final_token_count") for row in predictions for item in row.get("ranked_candidates", []) if isinstance(item.get("final_token_count"), (int, float))]
    internal_raw_lengths = [item.get("raw_token_count") for row in predictions for item in row.get("ranked_candidates", []) if isinstance(item.get("raw_token_count"), (int, float))]
    t2_pred = next(iter((evaluation_root / "t2-ragbench-05-qwen3-strong-reranker").glob("predictions.jsonl.gz")), None)
    t2_lengths: list[float] = []
    t2_raw_lengths: list[float] = []
    if t2_pred:
        for row in read_jsonl(t2_pred):
            for item in row.get("ranked_candidates", row.get("ranked_contexts", row.get("candidates", []))):
                before = item.get("token_count_before_truncation", item.get("token_count_before"))
                after = item.get("token_count_after_truncation", item.get("token_count_after"))
                if isinstance(before, (int, float)):
                    t2_raw_lengths.append(before)
                if isinstance(after, (int, float)):
                    t2_lengths.append(after)
    write_json(output / "token-length-comparison.json", {
        "internal": {"candidate_raw_tokens": distribution(internal_raw_lengths), "candidate_final_tokens": distribution(internal_lengths), "query_tokens": {"status": "not present in frozen R3.3 prediction artifact"}, "total_tokens": {"status": "not separately present"}},
        "t2": {"candidate_or_pair_tokens_before_truncation": distribution(t2_raw_lengths), "candidate_or_pair_tokens_after_truncation": distribution(t2_lengths), "source": str(t2_pred) if t2_pred else None, "query_tokens": {"status": "not separately present"}},
        "interpretation": "token length is diagnostic only; representation content is audited independently",
    })

    # Granularity overall and by strict cohort.
    granularity_by_cohort: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in cohort_rows:
        record = record_by_key.get((row["case_id"], row["gold_candidate_key"]))
        if record:
            granularity_by_cohort[row["cohort"]][classify_granularity(record["parsed"])] += 1
    write_json(output / "candidate-granularity-analysis.json", {
        "overall": dict(collections.Counter(classify_granularity(record["parsed"]) for record in all_records)),
        "C0": dict(granularity_by_cohort["C0_top5_success"]),
        "C1": dict(granularity_by_cohort["C1_top100_present_top5_miss"]),
        "C2": dict(granularity_by_cohort["C2_top100_absent"]),
        "candidate_granularity_contract": "whole-context T2 documents versus mixed PDF-derived paragraph/table/row/semantic candidates internally",
    })

    packet_design = {
        "name": "Evidence Packet V1",
        "execution": "design_only_not_run",
        "fields": [
            "[Document] company/file/page",
            "[Location] section/table title",
            "[Metric] metric path/row label",
            "[Period] column header/normalized period",
            "[Unit] currency/scale",
            "[Evidence] raw cell/row/chunk text",
            "[Local Context] limited parent table and neighboring row context",
        ],
        "canonical_serialization": "Document -> Location -> Metric -> Period -> Unit -> Evidence -> Local Context",
        "local_context_rule": "same row and same table header plus a fixed limited neighboring-row/parent-block context; no page-wide expansion",
        "constraints": {
            "gold_independent": True,
            "candidate_identity_preserving": True,
            "deterministic": True,
            "same_format_for_all_candidates": True,
            "no_expected_answer": True,
            "no_gold_metric_injection": True,
            "no_expected_period_injection": True,
            "no_expected_value_injection": True,
        },
        "query": "original question only; no Gold or reference answer",
    }
    write_json(output / "evidence-packet-v1-design.json", packet_design)
    write_json(output / "r1-experiment-contract.json", {
        "gate": "NF-OPT-18-R1",
        "execution": "not_run",
        "only_variable": "old candidate serialization versus Evidence Packet V1",
        "frozen": {"candidate_universe": "same bounded Top100", "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "instruction_sha256": internal_contract["instruction"]["sha256"], "candidate_ids": "exactly frozen Top100"},
        "metrics": ["strict physical R@1/3/5/10/20/50/100", "semantic fact R@5/10", "multi-evidence access", "calculation operand access", "old_hit_to_packet_miss", "old_miss_to_packet_hit"],
        "stop_rule": {"material_gain": ">=5 strict source hits and damage <= rescue", "insufficient": "gain <3 sources or damage >= rescue", "marginal": "3-4 source gain"},
    })

    structure_not_serialized_overall = 0
    structure_not_serialized_c1 = 0
    for row in graph_rows:
        parsed = view_map[(row["case_id"], row["candidate_key"])]
        if any(row["available"][field] and parsed["parsed"]["fields"][field]["status"] == "absent" for field in ["table_title", "row_label", "column_header", "metric_path", "period", "scale", "currency", "logical_table"]):
            structure_not_serialized_overall += 1
    c1_keys = {(row["case_id"], row["gold_candidate_key"]) for row in cohort_rows if row["cohort"] == "C1_top100_present_top5_miss"}
    structure_not_serialized_c1 = sum(1 for key in c1_keys if any(graph_available(authoritative_map.get(key, {}))[field] and view_map[key]["parsed"]["fields"][field]["status"] == "absent" for field in ["table_title", "row_label", "column_header", "metric_path", "period", "scale", "currency", "logical_table"]))

    decision = {
        "gate": GATE,
        "base_commit": R3_COMMIT,
        "internal_calculation_frozen_commit": INTERNAL_CALC_COMMIT,
        "model_execution": False,
        "retrieval_rerun": False,
        "candidate_mutation": 0,
        "bounded_top100_prediction_sha256": bounded_seal.get("prediction_sha256"),
        "bounded_top100_sha256_matches_frozen": bounded_seal.get("prediction_sha256") == BOUNDED_TOP100_SHA,
        "bounded_top100_identity_mismatches": bounded_identity_mismatches,
        "semantic_fact_identity_modified": False,
        "binder_modified": False,
        "calculator_run": False,
        "production_switch_allowed": False,
        "internal_strict_sources": STRICT_BINDING_COUNT,
        "current_top100_hits": cohort_counts["C0_top5_success"] + cohort_counts["C1_top100_present_top5_miss"],
        "current_qwen_top5_hits": cohort_counts["C0_top5_success"],
        "top100_present_top5_miss": cohort_counts["C1_top100_present_top5_miss"],
        "top100_absent": cohort_counts["C2_top100_absent"],
        "semantic_graph_available_fields": sorted(set(
            [field for field in FIELDS if graph_consumption["fields"][field]["available_count"] > 0]
            + [field for field, value in graph_consumption["graph_corpus_counts"].get("corpus_field_availability", {}).items() if value["nonempty_record_count_sum"] > 0]
        )),
        "reranker_consumed_fields": [field for field in FIELDS if graph_consumption["fields"][field]["consumed_count"] > 0],
        "structure_available_but_not_serialized_count": structure_not_serialized_overall,
        "structure_available_but_not_serialized_c1_count": structure_not_serialized_c1,
        "c0_high_ambiguity_rate": c0_high,
        "c1_high_ambiguity_rate": c1_high,
        "ambiguity_high_rate_delta": c1_high - c0_high,
        "representation_gap_supported": bool(c1_high - c0_high >= 0.10 and structure_not_serialized_c1 > 0),
        "evidence_packet_v1_allowed": bool(c1_high - c0_high >= 0.10 and structure_not_serialized_c1 > 0),
        "next_gate": "nf_opt_18_r1_evidence_packet_shadow_rerank" if (c1_high - c0_high >= 0.10 and structure_not_serialized_c1 > 0) else "nf_opt_18_method_reconsideration",
        "input_sha256": input_hashes,
        "strict_binding_reads_after_representation_seal": len(bindings),
    }
    write_json(output / "decision.json", decision)
    readme = f"""# NF-OPT-18 R0 鈥?Internal Reranker Representation Audit\n\nPure post-seal/offline audit. Model execution and retrieval rerun are false.\n\n- Strict physical bindings: {STRICT_BINDING_COUNT}\n- Top100 present: {decision['current_top100_hits']}\n- Qwen Top5: {decision['current_qwen_top5_hits']}\n- C1 Top100-present/Top5-miss: {decision['top100_present_top5_miss']}\n- C2 Top100-absent: {decision['top100_absent']}\n- C0 high ambiguity: {c0_high:.4f}\n- C1 high ambiguity: {c1_high:.4f}\n- Structure available but not serialized (overall/C1): {structure_not_serialized_overall}/{structure_not_serialized_c1}\n\nDecision: `representation_gap_supported={decision['representation_gap_supported']}`, `evidence_packet_v1_allowed={decision['evidence_packet_v1_allowed']}`. Next gate: `{decision['next_gate']}`. Production switch remains false.\n\nStrict source identity is used for all cohorts. Semantic Fact identity is diagnostic only and never replaces physical Gold.\n"""
    stale_companion = output / "README.md.json"
    if stale_companion.exists():
        stale_companion.unlink()
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": decision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
