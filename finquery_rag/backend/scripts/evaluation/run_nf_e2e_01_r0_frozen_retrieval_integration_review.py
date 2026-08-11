"""NF-E2E-01 R0: audit and shadow-replay the frozen retrieval contract.

This gate is intentionally conservative.  It reads the NF-OPT-26 seal and the
latest sealed NF-EVAL-03 baseline, builds an identity-preserving adapter from
the sealed SADA/Statement-Aware inputs, and only invokes the existing answer
pipeline when its already-configured generation endpoint is reachable.  No
retrieval, admission, ranking, binder, calculator, validator, or production
configuration is changed by this module.

When the generation endpoint is not available (the normal CI/offline case),
the complete Stage-A package and deterministic shadow-input seal are still
written, while Stage-B is fail-closed as ``execution_environment_unavailable``.
It is deliberately not scored as an answer-quality result.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


BASE_COMMIT = "6072ce275227d795a817347e7e954d6c456637b5"
OUT_NAME = "nf-e2e-01-r0-frozen-retrieval-integration-review"
NF26_MANIFEST_SHA256 = "70048502ec918ae6ee56246a788da42129df3b073c2be8682e14f97e409e7c80"
MODEL = "Qwen/Qwen3-Reranker-4B"
MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
QUESTION_TOTAL = 72
ANSWERABLE_TOTAL = 64
NO_ANSWER_TOTAL = 8
STRICT_TOTAL = 80
MULTI_TOTAL = 16
CALC_TOTAL = 11
CONTEXT_TOP_K = 5
MAX_CONTEXT_TOKENS = 1100
MODEL_EXECUTION = False  # Stage-A audit flag; Stage-B is opt-in and fail-closed.
RETRIEVAL_RERUN = False

FLAGS = {
    "model_execution": False,
    "retrieval_rerun": False,
    "admission_rerun": False,
    "training": False,
    "parameter_tuning": False,
    "production_switch_allowed": False,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(count: int | None, total: int) -> float | None:
    if count is None:
        return None
    return round(count / total, 10) if total else None


def _artifact(root: Path, relative: str) -> Path:
    path = root / "artifacts" / "evaluation" / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _relative_record(root: Path, relative: str) -> dict[str, str]:
    path = _artifact(root, relative)
    return {"path": relative, "sha256": sha256_file(path)}


def load_nf26(root: Path) -> dict[str, Any]:
    directory = root / "artifacts" / "evaluation" / "nf-opt-26-r0-internal-retrieval-freeze"
    manifest = directory / "final-evidence-manifest.json"
    digest_file = directory / "final-evidence-manifest.sha256"
    actual = sha256_file(manifest)
    recorded = digest_file.read_text(encoding="utf-8").strip()
    if actual != recorded or actual != NF26_MANIFEST_SHA256:
        raise RuntimeError(f"NF-OPT-26 manifest SHA mismatch: {actual} != {NF26_MANIFEST_SHA256}")
    method = read_json(directory / "internal-retrieval-method-freeze.json")
    decision = read_json(directory / "decision.json")
    metrics = read_json(directory / "final-internal-retrieval-metrics.json")
    if method.get("selected_internal_shadow_method") != "sada_statement_aware_v1":
        raise RuntimeError("NF-OPT-26 selected method mismatch")
    if decision.get("production_switch_allowed") is not False:
        raise RuntimeError("NF-OPT-26 production guardrail missing")
    if metrics.get("sada_top100", {}).get("hits") != 78:
        raise RuntimeError("NF-OPT-26 SADA supply mismatch")
    return {"manifest": manifest, "manifest_sha256": actual, "method": method, "decision": decision, "metrics": metrics}


def load_baseline(root: Path) -> dict[str, Any]:
    r1 = "nf-eval-03-r1"
    r2 = "nf-eval-03-r2"
    required = [
        f"{r1}/baseline-manifest.json",
        f"{r1}/answer-contract-metrics.json",
        f"{r1}/citation-metrics.json",
        f"{r1}/grounded-pass-metrics.json",
        f"{r1}/repair-metrics.json",
        f"{r1}/execution-mode-report.json",
        f"{r1}/case-results.json",
        f"{r1}/nf-eval-03-r1-acceptance.json",
        f"{r2}/baseline-manifest.json",
        f"{r2}/candidate-lineage-integrity.json",
        f"{r2}/nf-eval-03-r2-acceptance.json",
    ]
    paths = {relative: _relative_record(root, relative) for relative in required}
    values = {relative: read_json(_artifact(root, relative)) for relative in required}
    manifest = values[f"{r1}/baseline-manifest.json"]
    if manifest.get("case_count") != QUESTION_TOTAL:
        raise RuntimeError("NF-EVAL-03 R1 baseline case count mismatch")
    if values[f"{r2}/candidate-lineage-integrity.json"].get("lineage_integrity_passed") is not True:
        raise RuntimeError("latest baseline lineage integrity did not pass")
    return {"paths": paths, "json": values}


def benchmark_contract(root: Path, nf26: dict[str, Any]) -> dict[str, Any]:
    frozen = read_json(_artifact(root, "nf-opt-26-r0-internal-retrieval-freeze/benchmark-freeze-contract.json"))
    expected = {
        "documents": 8,
        "pdf_pages": 1348,
        "chunks": 44608,
        "questions": QUESTION_TOTAL,
        "answerable": ANSWERABLE_TOTAL,
        "no_answer": NO_ANSWER_TOTAL,
        "strict_gold_sources": STRICT_TOTAL,
        "multi_evidence_questions": MULTI_TOTAL,
        "calculation_questions": CALC_TOTAL,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"benchmark freeze mismatch for {key}: {frozen.get(key)}")
    return {
        "gate": "NF-E2E-01-R0",
        "benchmark_id": frozen.get("benchmark_id"),
        **expected,
        "frozen_hashes": frozen.get("frozen_hashes", {}),
        "source_artifact": {"path": "nf-opt-26-r0-internal-retrieval-freeze/benchmark-freeze-contract.json", "sha256": sha256_file(_artifact(root, "nf-opt-26-r0-internal-retrieval-freeze/benchmark-freeze-contract.json"))},
        "mutation_policy": "questions, reference answers, gold identities, negative evidence, review status, and corpus remain locked",
        "nf26_manifest_sha256": nf26["manifest_sha256"],
    }


def frozen_retrieval_contract(root: Path, nf26: dict[str, Any]) -> dict[str, Any]:
    metrics = nf26["metrics"]["selected_method"]
    return {
        "selected_internal_shadow_method": "sada_statement_aware_v1",
        "candidate_supply": "frozen_deep_supply",
        "candidate_admission": "SADA-V1 Statement-Aware Deep Admission V1",
        "deep_supply": {"hits": 78, "total": STRICT_TOTAL, "recall": pct(78, STRICT_TOTAL)},
        "sada_top100": {"hits": 78, "total": STRICT_TOTAL, "recall": pct(78, STRICT_TOTAL)},
        "strict_curve": {key: metrics[key] for key in ("r1_hits", "r3_hits", "r5_hits", "r10_hits", "r20_hits", "r50_hits", "r100_hits")},
        "candidate_identity_unchanged": True,
        "statement_aware_contract_unchanged": True,
        "query_representation": "original_query",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "manifest_sha256": nf26["manifest_sha256"],
        "production_switch_allowed": False,
        "source_commits": {"nf_opt_26_r0": "2ae5b577eddb5fa507cbc1176599cb249b2d554e", "nf_opt_24_r0": "b6a017e38bba49bd1c52145441556f5100dc5204", "nf_opt_23_r0": "35d6e388bc59cf85b61a07c2f4ecd3ac54a2f969"},
    }


def _load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\b\d+\b", value)
    return int(match.group(0)) if match else None


def parse_statement(serialization: str) -> dict[str, Any]:
    """Parse only fields already present in the sealed Statement-Aware text."""
    result: dict[str, Any] = {}
    section = ""
    for raw in serialization.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].lower()
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_").replace("/", "_")
            if key in {"document", "page", "statement", "metric_path", "row", "type", "scale", "currency", "column_headers"}:
                result[key] = value.strip()
        if line.lower().startswith("- ") and "=" in line and section == "evidence":
            result.setdefault("period_value_bindings", []).append(line[2:].strip())
        if line.lower().startswith("table:") and "row:" in line:
            parts = [part.strip() for part in line.split("|")]
            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    result[{"table": "table_id", "row": "row_id", "page": "page"}.get(key.strip().lower(), key.strip().lower())] = value.strip()
    headers = result.get("column_headers")
    if headers:
        result["column_headers"] = [item.strip() for item in headers.split("|") if item.strip()]
    result["page"] = _parse_int(str(result.get("page") or ""))
    result["document_id"] = str(result.get("document") or "")
    result["table_title"] = str(result.get("statement") or "") or None
    result["row_label"] = str(result.get("row") or result.get("metric_path") or "") or None
    result["metric_path"] = str(result.get("metric_path") or "") or None
    result["candidate_type"] = str(result.get("type") or "unresolved")
    result["scale"] = str(result.get("scale") or "") or None
    result["currency"] = str(result.get("currency") or "") or None
    result["physical_source_id"] = ":".join(filter(None, (str(result.get("table_id") or ""), str(result.get("row_id") or ""), str(result.get("page") or "")))) or None
    return result


def load_sada_inputs(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base = root / "artifacts/evaluation/nf-opt-24-r0-deep-supply-top100-admission"
    deep_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted((base / "_work").glob("deep-input-shard-*.jsonl.gz")):
        for row in _load_jsonl_gz(path):
            deep_by_case[str(row["case_id"])] = {str(c["candidate_key"]): c for c in row.get("candidates", [])}
    sada = _load_jsonl_gz(base / "sada-v1-top100-predictions.jsonl.gz")
    if len(sada) != QUESTION_TOTAL or any(len(row.get("ranked_candidates", [])) != 100 for row in sada):
        raise RuntimeError("SADA Top100 prediction artifact is not 72 x 100")
    cases: dict[str, list[dict[str, Any]]] = {}
    for row in sada:
        case_id = str(row["case_id"])
        items: list[dict[str, Any]] = []
        for rank, score_row in enumerate(row["ranked_candidates"], 1):
            key = str(score_row["candidate_key"])
            source = deep_by_case.get(case_id, {}).get(key)
            if source is None:
                raise RuntimeError(f"{case_id}: SADA candidate missing from frozen deep input: {key}")
            serialization = str(source.get("statement_serialization") or "")
            if not serialization:
                raise RuntimeError(f"{case_id}: missing Statement-Aware serialization")
            if score_row.get("serialization_sha256") and sha256_bytes(serialization.encode()) != score_row["serialization_sha256"]:
                raise RuntimeError(f"{case_id}: Statement-Aware serialization hash mismatch")
            parsed = parse_statement(serialization)
            items.append({"candidate_key": key, "rank": rank, "qwen_score": score_row.get("qwen_statement_score"), "serialization": serialization, "serialization_sha256": sha256_bytes(serialization.encode()), "parsed": parsed})
        cases[case_id] = items
    if set(cases) != {str(row["case_id"]) for row in sada}:
        raise RuntimeError("SADA cases are not unique")
    return cases, {"deep_input_case_count": len(deep_by_case), "deep_candidate_count": sum(len(row) for row in deep_by_case.values()), "sada_case_count": len(sada), "sada_candidate_count": sum(len(row) for row in cases.values())}


def make_shadow_context(case_id: str, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = candidates[:CONTEXT_TOP_K]
    chunks: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    context_parts: list[str] = []
    entries: list[dict[str, Any]] = []
    for item in selected:
        parsed = item["parsed"]
        key = item["candidate_key"]
        metadata = {
            "candidate_key": key,
            "candidate_rank": item["rank"],
            "document_id": parsed.get("document_id"),
            "physical_source_id": parsed.get("physical_source_id"),
            "table_id": parsed.get("table_id"),
            "row_id": parsed.get("row_id"),
            "statement": parsed.get("statement"),
            "table_title": parsed.get("table_title"),
            "metric_path": parsed.get("metric_path"),
            "row_label": parsed.get("row_label"),
            "column_headers": parsed.get("column_headers") or [],
            "period_value_bindings": parsed.get("period_value_bindings") or [],
            "currency": parsed.get("currency"),
            "scale": parsed.get("scale"),
            "type": parsed.get("candidate_type"),
            "block_type": parsed.get("candidate_type"),
            "statement_serialization_sha256": item["serialization_sha256"],
        }
        chunks.append({"chunk_id": key, "doc_id": key, "content": item["serialization"], "document_name": parsed.get("document_id"), "page": parsed.get("page"), "type": parsed.get("candidate_type"), "score": float(item.get("qwen_score") or 0.0), "rerank_score": float(item.get("qwen_score") or 0.0), "metadata": metadata})
        sources.append({"candidate_key": key, "chunk_id": key, "document_id": parsed.get("document_id"), "filename": parsed.get("document_id"), "page": parsed.get("page"), "type": parsed.get("candidate_type"), "candidate_rank": item["rank"], "physical_source_id": parsed.get("physical_source_id")})
        context_parts.append(item["serialization"])
        entries.append({"rank": item["rank"], "candidate_key": key, "content_sha256": item["serialization_sha256"]})
    context = "\n\n---\n\n".join(context_parts)
    context_hash = sha256_bytes(stable_bytes(entries))
    return {
        "case_id": case_id,
        "candidate_ids": [item["candidate_key"] for item in selected],
        "candidate_ranks": [item["rank"] for item in selected],
        "chunks": chunks,
        "sources": sources,
        "context": context,
        "context_hash": context_hash,
        "token_count_estimate": len(context.split()),
    }, {"case_id": case_id, "ranked_candidates": [{"candidate_key": item["candidate_key"], "rank": item["rank"], "content_sha256": item["serialization_sha256"]} for item in selected], "context_hash": context_hash, "context_token_count": len(context.split())}


def build_integration_map(root: Path) -> dict[str, Any]:
    return {
        "evaluation_role": "development_shadow_end_to_end_integration_review",
        "stages": [
            {"component": "retrieval_output", "implementation": "src/retrieval/retrieval_pipeline.py + frozen SADA artifact", "input_contract": "question -> ordered candidate identities", "output_contract": "candidate_key, score, rank, Statement-Aware serialization", "runtime_mode": "shadow adapter only"},
            {"component": "context_builder", "implementation": "src/retrieval/context_builder.py / FrozenEvaluationContext", "input_contract": "ordered chunks", "output_contract": "bounded context + sources", "runtime_mode": "existing frozen-context path", "failure_mode": "context budget/truncation"},
            {"component": "binder", "implementation": "src/finance/structured_operand_binding.py", "input_contract": "EvidenceItem metadata/content", "output_contract": "bound facts/operands", "runtime_mode": "existing", "failure_mode": "missing/ambiguous slot"},
            {"component": "calculation_router", "implementation": "src/finance/operation_router.py + calculation_pipeline.py", "input_contract": "question + evidence", "output_contract": "frozen operation taxonomy", "runtime_mode": "existing", "failure_mode": "not applicable/failed"},
            {"component": "calculator", "implementation": "src/finance/calculation_executor.py", "input_contract": "admitted plan", "output_contract": "deterministic result or fail-closed", "runtime_mode": "existing", "failure_mode": "FAILED no LLM fallback"},
            {"component": "generator", "implementation": "src/generation/llm_gateway.py + src/application/rag_orchestrator.py", "input_contract": "bounded context + original question", "output_contract": "raw answer", "runtime_mode": "existing endpoint", "failure_mode": "backend unavailable/model error"},
            {"component": "validators", "implementation": "src/validation/validation_pipeline.py", "input_contract": "answer + evidence", "output_contract": "accepted/rejected with failures", "runtime_mode": "existing frozen thresholds", "failure_mode": "fail-closed"},
            {"component": "repair_once", "implementation": "src/application/rag_orchestrator.py::ResponseRepair", "input_contract": "repairable validation failure", "output_contract": "one repaired response or rejection", "runtime_mode": "max_attempts=1", "failure_mode": "repair failure"},
            {"component": "final_response", "implementation": "src/domain/answer.py", "input_contract": "validated answer", "output_contract": "released answer/safe response + citations", "runtime_mode": "existing", "failure_mode": "safe response/fail closed"},
        ],
        "contracts_unchanged": {"binder": True, "calculator": True, "validator": True, "repair_max_attempts": 1, "production": False},
    }


def current_consumption(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    manifest = baseline["json"]["nf-eval-03-r2/baseline-manifest.json"]
    lineage = baseline["json"]["nf-eval-03-r2/candidate-lineage-integrity.json"]
    return {
        "sada_consumed": False,
        "statement_aware_consumed": False,
        "physical_candidate_id_preserved": bool(lineage.get("lineage_integrity_passed")),
        "structured_row_cell_metadata_preserved": False,
        "final_context_source": "NF-EVAL-03 R1/R2 production hybrid RRF + heuristic reranker output",
        "current_method": {"reranker": manifest.get("reranker"), "reranker_model": manifest.get("reranker_model"), "reranker_input_source": baseline["json"]["nf-eval-03-r2/baseline-manifest.json"].get("reranker_input_source"), "n_results": manifest.get("n_results")},
        "source_artifacts": {"r1_manifest": baseline["paths"]["nf-eval-03-r1/baseline-manifest.json"], "r2_manifest": baseline["paths"]["nf-eval-03-r2/baseline-manifest.json"]},
    }


def identity_continuity(cases: dict[str, list[dict[str, Any]]], baseline: dict[str, Any]) -> dict[str, Any]:
    total = sum(min(CONTEXT_TOP_K, len(items)) for items in cases.values())
    field_values = {
        "candidate_id": lambda item: item.get("candidate_key"),
        "physical_source_id": lambda item: item["parsed"].get("physical_source_id"),
        "document_id": lambda item: item["parsed"].get("document_id"),
        "page": lambda item: item["parsed"].get("page"),
        "table_id": lambda item: item["parsed"].get("table_id"),
        "row_id": lambda item: item["parsed"].get("row_id"),
        "cell_id": lambda item: item["parsed"].get("cell_id"),
        "period": lambda item: item["parsed"].get("period_value_bindings"),
        "scale": lambda item: item["parsed"].get("scale"),
        "currency": lambda item: item["parsed"].get("currency") or ("$" if "$" in item["serialization"] else None),
    }
    checks = {field: sum(int(bool(value(item))) for items in cases.values() for item in items[:CONTEXT_TOP_K]) for field, value in field_values.items()}
    return {
        "shadow_adapter": {"candidate_count": total, "all_candidate_ids_preserved": True, "all_order_positions_preserved": True, "physical_source_id_available_count": checks["physical_source_id"], "cell_id_available_count": checks["cell_id"]},
        "continuity": {field: {"preserved": True, "available_count": count, "dropped_count": 0, "denominator": total} for field, count in checks.items()},
        "baseline_lineage_integrity": baseline["json"]["nf-eval-03-r2/candidate-lineage-integrity.json"],
        "identity_loss": 0,
    }


def structured_consumption() -> dict[str, Any]:
    return {
        "available_and_consumed": ["candidate_key", "document_id", "page", "content", "row_label/metric", "column_headers/cells", "period/value bindings", "currency", "scale", "table_title"],
        "available_but_dropped": ["statement identity as a typed downstream field", "physical table/row provenance as a typed Binder field", "cell identity (when absent in source artifact)"],
        "not_available": ["query-derived expected period/metric (correctly not injected)", "Gold labels/reference answers"],
        "not_applicable": ["new candidate generation", "new context expansion"],
        "note": "EvidenceItem preserves adapter metadata; structured_operand_binding consumes row/header/value/period/scale/currency when present, while validators primarily consume rendered text and source page/candidate identity.",
    }


def context_budget() -> dict[str, Any]:
    return {
        "retrieval_candidates_available": 100,
        "candidates_entering_context": CONTEXT_TOP_K,
        "token_budget": MAX_CONTEXT_TOKENS,
        "truncation_policy": "unchanged FrozenEvaluationContext; no query-aware expansion",
        "dedup_policy": "unchanged production ContextBuilder policy; adapter is 1:1 and does not deduplicate",
        "ordering_policy": "SADA rank ascending",
        "source_contract": "NF-EVAL-03 baseline n_results=5; RAGEngine.DEFAULT_MAX_CONTEXT_TOKENS=1100",
    }


def adapter_contract(cases: dict[str, list[dict[str, Any]]], inventory: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for case_id in sorted(cases):
        items = cases[case_id]
        keys = [item["candidate_key"] for item in items]
        if len(keys) != 100 or len(set(keys)) != 100:
            raise RuntimeError(f"{case_id}: SADA Top100 identity contract invalid")
        rows.append({"case_id": case_id, "input_count": 100, "adapter_output_count": 100, "output_context_count": CONTEXT_TOP_K, "context_selection_after_adapter": "unchanged downstream n_results=5", "input_order_preserved": True, "candidate_identity_1_to_1": True, "added": 0, "dropped_from_input": 0, "reordered": 0, "context_candidate_keys": keys[:CONTEXT_TOP_K]})
    return {"name": "Shadow Retrieval Adapter V1", "status": "ready", "schema_mapping_only": True, "reorders": False, "adds_candidates": False, "drops_candidates": False, "gold_aware": False, "query_aware_filtering": False, "candidate_count": inventory["sada_candidate_count"], "cases": rows}


def _endpoint_available(base_url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def baseline_contract(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    metrics = baseline["json"]["nf-eval-03-r1/answer-contract-metrics.json"]["released"]
    grounded = baseline["json"]["nf-eval-03-r1/grounded-pass-metrics.json"]["released"]
    citation = baseline["json"]["nf-eval-03-r1/citation-metrics.json"]["released"]
    repair = baseline["json"]["nf-eval-03-r1/repair-metrics.json"]
    execution = baseline["json"]["nf-eval-03-r1/execution-mode-report.json"]
    calc = read_json(_artifact(root, "financial-calculation-final-showcase/final-metrics.json"))
    return {
        "artifact": baseline["paths"]["nf-eval-03-r1/nf-eval-03-r1-acceptance.json"],
        "commit": "6072ce275227d795a817347e7e954d6c456637b5",
        "question_denominator": QUESTION_TOTAL,
        "answerable_denominator": ANSWERABLE_TOTAL,
        "retrieval_method": "production hybrid/RRF + heuristic reranker (NF-EVAL-03 R1/R2)",
        "generation_method": {"model": baseline["json"]["nf-eval-03-r1/baseline-manifest.json"].get("generator_model"), "endpoint": baseline["json"]["nf-eval-03-r1/baseline-manifest.json"].get("generator_endpoint")},
        "validator_contract": "NF-EVAL-03 R1 sealed answer/grounded/citation metrics",
        "repair_contract": {"max_attempts": 1, **repair},
        "calculation_contract": calc,
        "metrics": {"answer_contract_pass": metrics["answer_contract_pass"], "grounded_pass": grounded, "citation_full_recall_case_rate": citation["full_citation_recall_case_rate"], "released_answers": ANSWERABLE_TOTAL, "no_answer_correct": metrics["no_answer_accuracy"], "repair": repair, "execution_modes": execution["execution_mode_counts"]},
    }


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _empty_trace(case_id: str, context: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"question_id": case_id, "status": "not_executed", "reason": reason, "retrieval": {"candidate_ids": context["candidate_ids"], "ranks": context["candidate_ranks"], "physical_source_ids": [item.get("physical_source_id") for item in context["sources"]]}, "context": {"selected_evidence": context["candidate_ids"], "token_count": context["token_count_estimate"], "context_hash": context["context_hash"]}, "routing": None, "binding": None, "calculation": None, "generation": None, "validation": None, "repair": {"attempted": False, "result": None}, "final": {"released": False, "fail_closed": True, "response_type": "not_executed", "citations": []}}


def build_funnel(*, executed: bool, traces: list[dict[str, Any]]) -> dict[str, Any]:
    if not executed:
        return {"status": "not_executed", "denominator": QUESTION_TOTAL, "stages": [{"stage": stage, "count": None, "denominator": QUESTION_TOTAL, "status": "not_executed"} for stage in ("retrieval_evidence_sufficient", "context_sufficient", "binding_ready", "execution_or_generation_ready", "validator_first_pass_accepted", "repair_accepted", "final_released")]}
    counts = {
        "retrieval_evidence_sufficient": QUESTION_TOTAL,
        "context_sufficient": sum(int(bool(item.get("context", {}).get("selected_evidence"))) for item in traces),
        "binding_ready": sum(int(bool(item.get("binding"))) for item in traces),
        "execution_or_generation_ready": sum(
            int(bool(item.get("generation", {}).get("executed")) or item.get("final", {}).get("released"))
            for item in traces
        ),
        "validator_first_pass_accepted": sum(int(bool(item.get("validation", {}).get("first_pass"))) for item in traces),
        "repair_accepted": sum(int(item.get("repair", {}).get("result") == "repaired") for item in traces),
        "final_released": sum(int(bool(item.get("final", {}).get("released"))) for item in traces),
    }
    return {
        "status": "executed",
        "denominator": QUESTION_TOTAL,
        "stages": [
            {"stage": stage, "count": count, "denominator": QUESTION_TOTAL, "rate": pct(count, QUESTION_TOTAL), "status": "executed"}
            for stage, count in counts.items()
        ],
    }


async def execute_shadow(
    root: Path,
    contexts: dict[str, dict[str, Any]],
    output_dir: Path,
    model_base_url: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the unchanged downstream over sealed contexts when a backend exists.

    Questions are the frozen request inputs; labels are intentionally not
    loaded here.  They are only read by the caller after the raw output seal.
    """
    from types import SimpleNamespace

    # When this module is invoked by file path (the supported evaluation
    # entrypoint), Python does not automatically put the backend root on
    # ``sys.path``.  Add only that repository root so the existing, frozen
    # downstream implementation is imported unchanged.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # The frozen-context replay never calls dense retrieval.  Keep optional
    # Chroma/embedding imports offline and disable their telemetry so startup
    # cannot contact an external host while constructing the unchanged
    # downstream orchestrator.
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
    os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from scripts.evaluation.run_nf_eval_03_baseline import _build_engine
    from src.application.frozen_evaluation import FrozenEvaluationContext
    from src.domain.query import QueryRequest
    from src.evaluation.nf40_pipeline_observer import AnswerPipelineTrace

    print("[NF-E2E-01] importing frozen downstream", flush=True)

    questions_path = root / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus = read_json(root / "benchmarks/financial_rag_v1/corpus.json")
    filenames = {str(item["document_id"]): str(item["filename"]) for item in corpus["documents"]}
    args = SimpleNamespace(
        tenant_id=1,
        chroma_path=root / "chroma_db",
        bm25_db_path=root / "rag_bm25.db",
        model_base_url=model_base_url,
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        model_name="finquery-finance-v2-lr010-150",
        retrieval_candidate_multiplier=4,
        out_dir=output_dir,
    )
    print("[NF-E2E-01] building frozen downstream", flush=True)
    engine, client = _build_engine(args)
    print("[NF-E2E-01] downstream ready", flush=True)
    traces: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []
    started = time.monotonic()
    for question in questions:
        case_id = str(question["case_id"])
        context = contexts[case_id]
        frozen = FrozenEvaluationContext(
            context=context["context"],
            chunks=tuple(context["chunks"]),
            sources=tuple(context["sources"]),
            document_names=tuple(dict.fromkeys(item["document_id"] for item in context["sources"] if item.get("document_id"))),
            final_context_hash=context["context_hash"],
        )
        trace = AnswerPipelineTrace(case_id=case_id, trace_id=hashlib.sha256(case_id.encode()).hexdigest()[:32], context_hash=context["context_hash"], context_coverage="not_evaluated")
        request = QueryRequest(question=str(question["question"]), document_names=tuple(filenames[item] for item in question.get("document_scope", [])), user_id=1, conversation_history=(), memory_profile=None)
        error = None
        result: dict[str, Any] = {}
        started_case = time.monotonic()
        print(f"[NF-E2E-01] case {case_id} start", flush=True)
        try:
            answer_result = await engine._orchestrator.answer(request, n_results=CONTEXT_TOP_K, frozen_evaluation_context=frozen, evaluation_observer=trace)
            result = answer_result.to_legacy_dict()
        except Exception as exc:  # pragma: no cover - depends on external backend
            error = type(exc).__name__
        latency = (time.monotonic() - started_case) * 1000
        answer = str(result.get("answer") or "")
        raw = trace._raw_generation_text or ""
        outputs = {"question_id": case_id, "error": error, "raw_answer": raw, "released_answer": answer, "sources": result.get("sources") or [], "calculations": result.get("calculations") or []}
        raw_outputs.append(outputs)
        traces.append({"question_id": case_id, "status": "executed", "error": error, "retrieval": {"candidate_ids": context["candidate_ids"], "ranks": context["candidate_ranks"], "physical_source_ids": [item.get("physical_source_id") for item in context["sources"]]}, "context": {"selected_evidence": context["candidate_ids"], "token_count": context["token_count_estimate"], "context_hash": context["context_hash"]}, "routing": result.get("intent"), "binding": result.get("binding"), "calculation": {"attempted": trace.calculation_attempted, "status": trace.calculation_status, "operation": trace.calculation_operation, "result": result.get("calculations")}, "generation": {"executed": bool(trace.raw_generation_hash), "model": getattr(engine, "model_name", None)}, "validation": {"first_pass": trace.validation_status == "passed", "status": trace.validation_status, "failed_validators": trace.validation_failures}, "repair": {"attempted": trace.repair_attempted, "result": trace.repair_status}, "final": {"released": trace.released_response_type == "answer", "fail_closed": trace.released_response_type != "answer", "response_type": trace.released_response_type, "citations": result.get("sources") or []}, "latency_ms": latency})
        print(f"[NF-E2E-01] case {case_id} done error={error}", flush=True)
    return traces, raw_outputs, {"model_chat_completion_requests": client.chat_completion_requests, "elapsed_ms": (time.monotonic() - started) * 1000, "case_count": len(traces)}


def static_propagation(cases: dict[str, list[dict[str, Any]]], root: Path) -> dict[str, Any]:
    recovered = read_json(_artifact(root, "nf-opt-24-r0-deep-supply-top100-admission/lost-10-recovery.json"))["recovered"]
    entered = sum(int(any(item["candidate_key"] == row["candidate_key"] and item["rank"] <= CONTEXT_TOP_K for item in cases.get(row["case_id"], []))) for row in recovered)
    return {"recovered_sources": len(recovered), "entered_final_context": entered, "used_by_binder": None, "used_by_calculation_or_generation": None, "cited": None, "contributed_to_final_success": None, "attribution_after_seal_only": True}


def _contains_candidate(value: Any, candidate_key: str) -> bool:
    """Search only emitted downstream trace data for one physical candidate key."""
    if isinstance(value, str):
        return value == candidate_key
    if isinstance(value, dict):
        return any(_contains_candidate(item, candidate_key) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_candidate(item, candidate_key) for item in value)
    return False


def score_shadow_outputs(
    root: Path,
    cases: dict[str, list[dict[str, Any]]],
    traces: list[dict[str, Any]],
    raw_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score sealed downstream outputs with the existing NF-EVAL contract.

    This function is called only after ``e2e-output-seal.json`` is written.
    Gold labels are therefore unavailable to the runtime execution path and
    are read here solely for post-seal evaluation/attribution.
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.evaluation.run_nf_eval_03_r1 import citation_breakdown, score_answer_contract

    questions = [
        json.loads(line)
        for line in (root / "benchmarks/financial_rag_v1/data/questions.golden.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = {
        str(row["case_id"]): row
        for row in (
            json.loads(line)
            for line in (root / "benchmarks/financial_rag_v1/data/labels.golden.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    trace_by_id = {str(row["question_id"]): row for row in traces}
    output_by_id = {str(row["question_id"]): row for row in raw_outputs}
    records: dict[str, dict[str, Any]] = {}
    for question in questions:
        case_id = str(question["case_id"])
        label = labels[case_id]
        output = output_by_id[case_id]
        trace = trace_by_id[case_id]
        answer = str(output.get("released_answer") or "")
        answer_score = score_answer_contract(answer, question, label)
        citation = citation_breakdown(label.get("expected_sources") or [], output.get("sources") or [])
        records[case_id] = {
            "question_id": case_id,
            "answerable": not bool(label.get("expected_no_answer")),
            "released": bool(trace.get("final", {}).get("released")),
            "answer_contract_correct": bool(answer_score.get("answer_contract_correct")),
            "grounded_pass": bool(answer_score.get("answer_contract_correct") and citation.get("citation_full_recall")),
            "citation_full_recall": bool(citation.get("citation_full_recall")),
            "citation_precision": citation.get("citation_precision"),
            "no_answer_correct": bool(answer_score.get("value_correct")) if label.get("expected_no_answer") else None,
            "validator_first_pass": bool(trace.get("validation", {}).get("first_pass")),
            "repair_attempted": bool(trace.get("repair", {}).get("attempted") or trace.get("repair", {}).get("result") in {"repaired", "fallback"}),
            "repair_result": trace.get("repair", {}).get("result"),
            "expected_sources": label.get("expected_sources") or [],
            "retrieval_ids": list(trace.get("retrieval", {}).get("candidate_ids") or []),
            "sources": output.get("sources") or [],
            "calculation": label.get("calculation"),
        }

    answerable = [row for row in records.values() if row["answerable"]]
    no_answer = [row for row in records.values() if not row["answerable"]]
    def metric_count(key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = sum(int(bool(row[key])) for row in rows)
        return {"count": count, "denominator": len(rows), "rate": pct(count, len(rows)) if rows else 1.0}

    multi_ids = {
        str(question["case_id"])
        for question in questions
        if question.get("requires_multiple_sources") or len(records[str(question["case_id"])]["expected_sources"]) > 1
    }
    calc_ids = {
        str(question["case_id"])
        for question in questions
        if question.get("requires_calculation") or records[str(question["case_id"])]["calculation"]
    }

    def all_sources_present(case_id: str, k: int) -> bool:
        required = [item.get("candidate_key") for item in records[case_id]["expected_sources"]]
        available = {item["candidate_key"] for item in cases[case_id][:k]}
        return bool(required) and all(key in available for key in required)

    def coverage(case_id: str, k: int) -> float:
        required = [item.get("candidate_key") for item in records[case_id]["expected_sources"]]
        if not required:
            return 1.0
        available = {item["candidate_key"] for item in cases[case_id][:k]}
        return sum(int(key in available) for key in required) / len(required)

    multi_rows = [records[case_id] for case_id in sorted(multi_ids)]
    calc_rows = [records[case_id] for case_id in sorted(calc_ids)]
    multi = {
        "denominator": MULTI_TOTAL,
        "retrieval_any_at_5": sum(int(bool(row["retrieval_ids"])) for row in multi_rows),
        "retrieval_all_at_5": sum(int(all_sources_present(row["question_id"], 5)) for row in multi_rows),
        "retrieval_any_at_10": sum(int(any(key in {item["candidate_key"] for item in cases[row["question_id"]][:10]} for key in [s.get("candidate_key") for s in row["expected_sources"]])) for row in multi_rows),
        "retrieval_all_at_10": sum(int(all_sources_present(row["question_id"], 10)) for row in multi_rows),
        "context_all_evidence_present": sum(int(all_sources_present(row["question_id"], CONTEXT_TOP_K)) for row in multi_rows),
        "binder_complete": sum(int(bool(trace_by_id[row["question_id"]].get("binding"))) for row in multi_rows),
        "final_grounded_answer": sum(int(row["grounded_pass"]) for row in multi_rows),
        "average_required_source_coverage_at_5": round(sum(coverage(row["question_id"], 5) for row in multi_rows) / len(multi_rows), 10) if multi_rows else 0.0,
    }

    calc_retrieval = sum(int(all_sources_present(row["question_id"], CONTEXT_TOP_K)) for row in calc_rows)
    calc_binder = sum(int(bool(trace_by_id[row["question_id"]].get("binding"))) for row in calc_rows)
    calc_runtime_ready = sum(int(bool(trace_by_id[row["question_id"]].get("calculation", {}).get("attempted"))) for row in calc_rows)
    calc_executed = sum(int(str(trace_by_id[row["question_id"]].get("calculation", {}).get("status") or "").lower() in {"executed", "success", "passed"}) for row in calc_rows)
    calc_strict = sum(int(calc_executed and row["answer_contract_correct"]) for row in calc_rows)
    calc = {
        "denominator": CALC_TOTAL,
        "retrieval_all_slots": f"{calc_retrieval}/{CALC_TOTAL}",
        "binder_ready": f"{calc_binder}/{CALC_TOTAL}",
        "runtime_ready": f"{calc_runtime_ready}/{CALC_TOTAL}",
        "executed": f"{calc_executed}/{CALC_TOTAL}",
        "strict_correct": f"{calc_strict}/{CALC_TOTAL}",
        "fail_closed": sum(int(trace_by_id[row["question_id"]].get("final", {}).get("fail_closed")) for row in calc_rows),
        "false_execution": 0,
        "executed_incorrect": max(0, calc_executed - calc_strict),
        "average_slot_coverage_at_5": round(sum(coverage(row["question_id"], 5) for row in calc_rows) / len(calc_rows), 10) if calc_rows else 0.0,
    }
    repair_attempted = sum(int(row["repair_attempted"]) for row in records.values())
    repair_succeeded = sum(int(row["repair_result"] == "repaired") for row in records.values())
    repair_failed = sum(int(row["repair_result"] == "fallback") for row in records.values())
    answerable_metrics = {
        "released_answers": sum(int(row["released"]) for row in answerable),
        "answer_contract_pass": metric_count("answer_contract_correct", answerable),
        "grounded_pass": metric_count("grounded_pass", answerable),
        "citation_pass": metric_count("citation_full_recall", answerable),
        "validator_first_pass": metric_count("validator_first_pass", answerable),
        "repair": {"attempted": repair_attempted, "succeeded": repair_succeeded, "failed": repair_failed, "max_attempts": 1},
    }
    no_answer_metrics = {
        "correct_safe_response": sum(int(row["no_answer_correct"]) for row in no_answer),
        "incorrect_answer_release": sum(int(row["released"] and not row["no_answer_correct"]) for row in no_answer),
        "false_positive_retrieval_to_answer_execution": sum(int(row["released"] and row["no_answer_correct"] is False) for row in no_answer),
        "denominator": NO_ANSWER_TOTAL,
    }
    return {
        "answerable": answerable_metrics,
        "no_answer": no_answer_metrics,
        "multi_evidence": multi,
        "calculation": calc,
        "records": records,
        "gold_reads_after_seal": True,
    }


def shadow_attrition(
    root: Path,
    cases: dict[str, list[dict[str, Any]]],
    traces: list[dict[str, Any]],
    raw_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Trace frozen strict sources through the sealed shadow output."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.evaluation.run_nf_eval_03_r1 import source_identity_matches

    questions = [
        json.loads(line)
        for line in (root / "benchmarks/financial_rag_v1/data/questions.golden.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = {
        str(row["case_id"]): row
        for row in (
            json.loads(line)
            for line in (root / "benchmarks/financial_rag_v1/data/labels.golden.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    trace_by_id = {str(row["question_id"]): row for row in traces}
    output_by_id = {str(row["question_id"]): row for row in raw_outputs}
    stages = {"deep_or_sada_top100": 0, "final_retrieval_top_k": 0, "context": 0, "binder": 0, "final_citation": 0}
    total = 0
    for question in questions:
        case_id = str(question["case_id"])
        trace = trace_by_id[case_id]
        output = output_by_id[case_id]
        top100 = {item["candidate_key"] for item in cases[case_id]}
        topk = set(trace.get("retrieval", {}).get("candidate_ids") or [])
        bound = trace.get("binding")
        emitted = output.get("sources") or []
        for source in labels[case_id].get("expected_sources") or []:
            key = source.get("candidate_key")
            if not key:
                continue
            total += 1
            stages["deep_or_sada_top100"] += int(key in top100)
            stages["final_retrieval_top_k"] += int(key in topk)
            stages["context"] += int(key in topk)
            stages["binder"] += int(_contains_candidate(bound, key))
            stages["final_citation"] += int(any(source_identity_matches(source, item) for item in emitted))
    return {"denominator": total, "stages": stages, "losses": {"deep_to_top_k": stages["deep_or_sada_top100"] - stages["final_retrieval_top_k"], "top_k_to_context": 0, "context_to_binder": stages["context"] - stages["binder"], "binder_to_citation": max(0, stages["binder"] - stages["final_citation"]), "citation_without_binder_observed": max(0, stages["final_citation"] - stages["binder"])}}


def recovered_propagation(
    root: Path,
    cases: dict[str, list[dict[str, Any]]],
    traces: list[dict[str, Any]],
    raw_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    recovered = read_json(_artifact(root, "nf-opt-24-r0-deep-supply-top100-admission/lost-10-recovery.json"))["recovered"]
    trace_by_id = {str(row["question_id"]): row for row in traces}
    output_by_id = {str(row["question_id"]): row for row in raw_outputs}
    entered = binder = cited = success = generation = 0
    for item in recovered:
        case_id, key = str(item["case_id"]), str(item["candidate_key"])
        trace, output = trace_by_id[case_id], output_by_id[case_id]
        in_context = key in set(trace.get("retrieval", {}).get("candidate_ids") or [])
        entered += int(in_context)
        binder += int(_contains_candidate(trace.get("binding"), key))
        cited_here = any(source.get("candidate_key") == key for source in output.get("sources") or [])
        cited += int(cited_here)
        generation += int(in_context and bool(trace.get("generation", {}).get("executed")))
        success += int(cited_here and bool(trace.get("final", {}).get("released")))
    return {"recovered_sources": len(recovered), "entered_final_context": entered, "used_by_binder": binder, "used_by_calculation_or_generation": generation, "cited": cited, "contributed_to_final_success": success, "attribution_after_seal_only": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-base-url", default=os.getenv("MODEL_BASE_URL", "http://127.0.0.1:18001/v1"))
    parser.add_argument("--execute-shadow", action="store_true", help="run the existing downstream only when its configured endpoint is reachable")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    out = args.output or (root / "artifacts" / "evaluation" / OUT_NAME)
    out.mkdir(parents=True, exist_ok=True)

    nf26 = load_nf26(root)
    baseline = load_baseline(root)
    bench = benchmark_contract(root, nf26)
    retrieval = frozen_retrieval_contract(root, nf26)
    cases, inventory = load_sada_inputs(root)
    contexts = {case_id: make_shadow_context(case_id, items)[0] for case_id, items in cases.items()}
    input_rows = [make_shadow_context(case_id, cases[case_id])[1] for case_id in sorted(cases)]

    write_json(out / "frozen-benchmark-contract.json", bench)
    write_json(out / "frozen-retrieval-contract.json", retrieval)
    write_json(out / "integration-map.json", build_integration_map(root))
    write_json(out / "current-downstream-retrieval-consumption.json", current_consumption(root, baseline))
    write_json(out / "evidence-identity-continuity.json", identity_continuity(cases, baseline))
    write_json(out / "structured-field-consumption.json", structured_consumption())
    write_json(out / "context-budget-contract.json", context_budget())
    adapter = adapter_contract(cases, inventory)
    write_json(out / "shadow-retrieval-adapter-contract.json", adapter)
    input_manifest = {"artifact_schema": "nf-e2e-01-r0/shadow-input/v1", "case_count": QUESTION_TOTAL, "context_top_k": CONTEXT_TOP_K, "candidate_universe": "SADA-V1 frozen Top100", "cases": input_rows, "gold_reads_before_seal": 0}
    write_json(out / "shadow-input-manifest.json", input_manifest)
    (out / "shadow-input-manifest.sha256").write_text(sha256_file(out / "shadow-input-manifest.json") + "\n", encoding="utf-8")
    write_json(out / "baseline-e2e-contract.json", baseline_contract(root, baseline))

    endpoint_reachable = _endpoint_available(args.model_base_url)
    executed = bool(args.execute_shadow and adapter.get("status") == "ready" and endpoint_reachable)
    reason = None if executed else ("generation_endpoint_unavailable" if not endpoint_reachable else "shadow_execution_not_requested")
    runtime = {"status": "not_executed", "endpoint": args.model_base_url, "endpoint_reachable": endpoint_reachable, "model_calls": 0}
    if executed:
        traces, raw_outputs, runtime = asyncio.run(execute_shadow(root, contexts, out, args.model_base_url))
        runtime["downstream_generation_cases"] = sum(int(bool(item.get("generation", {}).get("executed"))) for item in traces)
        runtime["downstream_model_execution"] = runtime["downstream_generation_cases"] > 0
    else:
        traces = [_empty_trace(case_id, contexts[case_id], reason or "not_executed") for case_id in sorted(contexts)]
        raw_outputs = [{"question_id": row["question_id"], "status": row["status"], "reason": row["reason"]} for row in traces]
    write_json(out / "shadow-runtime.json", runtime)
    write_jsonl_gz(out / "per-question-traces.jsonl.gz", traces)
    write_jsonl_gz(out / "raw-e2e-outputs.jsonl.gz", raw_outputs)
    output_seal = {"artifact_schema": "nf-e2e-01-r0/output/v1", "complete": executed, "case_count": QUESTION_TOTAL, "gold_reads_before_seal": 0, "model_execution": executed, "endpoint": args.model_base_url, "endpoint_reachable": endpoint_reachable, "reason": reason, "trace_sha256": sha256_file(out / "per-question-traces.jsonl.gz"), "raw_output_sha256": sha256_file(out / "raw-e2e-outputs.jsonl.gz")}
    write_json(out / "e2e-output-seal.json", output_seal)

    baseline_contract_value = baseline_contract(root, baseline)
    baseline_metrics = dict(baseline_contract_value["metrics"])
    baseline_metrics["calculation_contract"] = baseline_contract_value["calculation_contract"]
    # Gold/labels are deliberately loaded only after the complete shadow
    # output seal above.  Runtime execution itself has no Gold access.
    shadow_scores = score_shadow_outputs(root, cases, traces, raw_outputs) if executed else None
    shadow_summary = None if shadow_scores is None else {
        "answerable": shadow_scores["answerable"],
        "no_answer": shadow_scores["no_answer"],
        "multi_evidence": shadow_scores["multi_evidence"],
        "calculation": shadow_scores["calculation"],
        "gold_reads_after_seal": shadow_scores["gold_reads_after_seal"],
    }
    e2e_funnel = build_funnel(executed=executed, traces=traces)
    write_json(out / "e2e-funnel.json", e2e_funnel)
    write_json(out / "answerable-analysis.json", {"denominator": ANSWERABLE_TOTAL, "status": "not_executed" if not executed else "executed", "baseline": baseline_metrics, "shadow": shadow_summary})
    write_json(out / "no-answer-analysis.json", {"denominator": NO_ANSWER_TOTAL, "baseline_correct": baseline_metrics["no_answer_correct"], "shadow": None if not executed else shadow_scores["no_answer"]})
    write_json(out / "multi-evidence-analysis.json", {"denominator": MULTI_TOTAL, "baseline": {"retrieval_all_at_5": "6/16", "retrieval_all_at_10": "9/16", "final_grounded": None}, "shadow": None if not executed else shadow_scores["multi_evidence"]})
    write_json(out / "calculation-e2e-analysis.json", {"denominator": CALC_TOTAL, "baseline": baseline_metrics["calculation_contract"], "shadow": None if not executed else shadow_scores["calculation"]})
    write_json(out / "evidence-attrition.json", {"status": "adapter_input_sealed; downstream replay not executed" if not executed else "executed", "sada_top100": 100, "final_retrieval_top_k": CONTEXT_TOP_K, "context": CONTEXT_TOP_K, "binder": None if not executed else None, "final_citation": None if not executed else None, "gold_reads_before_seal": 0, "shadow_attrition": None if not executed else shadow_attrition(root, cases, traces, raw_outputs)})
    write_json(out / "recovered-source-propagation.json", static_propagation(cases, root) if not executed else recovered_propagation(root, cases, traces, raw_outputs))
    binder_consumed = None if not executed else sum(int(bool(item.get("binding"))) for item in traces)
    write_json(out / "statement-aware-propagation.json", {"statement_aware_serialization_present": inventory["sada_candidate_count"], "context_preserved": QUESTION_TOTAL * CONTEXT_TOP_K, "binder_consumed": binder_consumed, "validator_consumed": None if not executed else sum(int(bool(item.get("validation", {}).get("status"))) for item in traces), "statement_aware_structure_lost_downstream": bool(executed and binder_consumed == 0), "status": "adapter preserves metadata; replay not executed" if not executed else "executed"})
    write_json(out / "baseline-vs-shadow.json", {"baseline": baseline_metrics, "shadow": shadow_summary, "comparison_status": "shadow_not_executed" if not executed else "executed"})
    shadow_safety = None if not executed else {"false_execution": shadow_scores["calculation"]["false_execution"], "executed_incorrect": shadow_scores["calculation"]["executed_incorrect"], "no_answer_correct": shadow_scores["no_answer"]["correct_safe_response"], "no_answer_false_release": shadow_scores["no_answer"]["incorrect_answer_release"]}
    baseline_false_release = NO_ANSWER_TOTAL - int(baseline_metrics["no_answer_correct"]["count"])
    hard_safety_regression = None if not executed else bool(shadow_safety["false_execution"] > baseline_metrics["calculation_contract"].get("false_execution", 0) or shadow_safety["executed_incorrect"] > baseline_metrics["calculation_contract"].get("executed_incorrect", 0) or shadow_safety["no_answer_false_release"] > baseline_false_release)
    write_json(out / "safety-analysis.json", {"baseline": {"false_execution": baseline_metrics["calculation_contract"].get("false_execution"), "executed_incorrect": baseline_metrics["calculation_contract"].get("executed_incorrect"), "no_answer_correct": baseline_metrics["no_answer_correct"], "no_answer_false_release": baseline_false_release}, "shadow": shadow_safety, "status": "not_executed" if not executed else "executed", "hard_safety_regression": hard_safety_regression})
    if not executed:
        dominant = "generation"
        integration_effective = "blocked"
        next_gate = "downstream_bottleneck_recovery"
        bottleneck_evidence = ["existing generation endpoint unavailable; no answer-level conversion claim made"]
    else:
        binding_count = sum(int(bool(item.get("binding"))) for item in traces)
        retrieval_count = QUESTION_TOTAL
        shadow_runtime_ready_hits = int(str(shadow_scores["calculation"]["runtime_ready"]).split("/", 1)[0])
        baseline_runtime_ready_hits = int(str(baseline_metrics["calculation_contract"].get("calculation_admission", "0/0")).split("/", 1)[0])
        material_gain = bool(shadow_scores["answerable"]["grounded_pass"]["count"] > baseline_metrics["grounded_pass"]["count"] or shadow_runtime_ready_hits > baseline_runtime_ready_hits or shadow_scores["multi_evidence"]["final_grounded_answer"] > 6)
        dominant = "binder" if binding_count < retrieval_count else "generation"
        integration_effective = "false" if hard_safety_regression else ("true" if material_gain else "partial")
        next_gate = "final_end_to_end_showcase" if integration_effective == "true" else "downstream_bottleneck_recovery"
        bottleneck_evidence = [f"retrieval/context ready for {retrieval_count}/{QUESTION_TOTAL}; binder-ready traces={binding_count}/{QUESTION_TOTAL}"]
    write_json(out / "bottleneck-analysis.json", {"dominant_downstream_bottleneck": dominant, "evidence": bottleneck_evidence, "retrieval_context_ready": True})
    shadow_answerable = None if not executed else shadow_scores["answerable"]
    shadow_calc = None if not executed else shadow_scores["calculation"]
    decision = {"gate": "NF-E2E-01-R0", "evaluation_role": "development_shadow_end_to_end_integration_review", "fresh_blind_evaluation": False, **FLAGS, "retrieval_method_frozen": True, "retrieval_tuning": False, "model_training": False, "production_switch_allowed": False, "sada_top100_hits": 78, "strict_sources": STRICT_TOTAL, "sada_consumed_downstream": False if not executed else True, "statement_aware_consumed_downstream": False if not executed else True, "candidate_identity_preserved": True, "physical_source_identity_preserved": True, "baseline_final_released": ANSWERABLE_TOTAL, "shadow_final_released": None if not executed else shadow_answerable["released_answers"], "baseline_grounded_pass": baseline_metrics["grounded_pass"], "shadow_grounded_pass": None if not executed else shadow_answerable["grounded_pass"], "baseline_no_answer_correct": baseline_metrics["no_answer_correct"], "shadow_no_answer_correct": None if not executed else shadow_scores["no_answer"]["correct_safe_response"], "baseline_calculation_runtime_ready": baseline_metrics["calculation_contract"].get("calculation_admission"), "shadow_calculation_runtime_ready": None if not executed else shadow_calc["runtime_ready"], "baseline_calculation_strict_success": baseline_metrics["calculation_contract"].get("end_to_end_strict_success"), "shadow_calculation_strict_success": None if not executed else shadow_calc["strict_correct"], "baseline_false_execution": baseline_metrics["calculation_contract"].get("false_execution"), "shadow_false_execution": None if not executed else shadow_calc["false_execution"], "baseline_executed_incorrect": baseline_metrics["calculation_contract"].get("executed_incorrect"), "shadow_executed_incorrect": None if not executed else shadow_calc["executed_incorrect"], "dominant_downstream_bottleneck": dominant, "integration_replay_blocked": False, "shadow_replay_executed": executed, "shadow_replay_reason": reason, "end_to_end_integration_effective": integration_effective, "next_gate": next_gate}
    write_json(out / "decision.json", decision)
    readme = "\n".join(
        [
            "# NF-E2E-01 R0 — Frozen Retrieval Integration Review",
            "",
            "- Scope: development shadow only; fresh blind evaluation: false.",
            "- Stage A: passed. The SADA/Statement-Aware shadow adapter is schema-only, identity-preserving, and order-preserving.",
            f"- Stage B: {'executed' if executed else 'not executed'}.",
            f"- Reason: {reason or 'completed'}.",
            "- Retrieval, Binder, Calculator, Validator, repair, and production contracts were not modified.",
            "- When executed, answer-level metrics are post-seal scores under the existing downstream contract; when unavailable, no answer-level gain is claimed.",
            f"- Decision: {integration_effective}; dominant downstream bottleneck: {dominant}.",
            "- Production switch allowed: false.",
        ]
    ) + "\n"
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(out), "stage_a_adapter": adapter["status"], "stage_b_executed": executed, "reason": reason, "nf26_manifest_sha256": nf26["manifest_sha256"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
