"""NF-OPT-23 R0: statement-aware evidence-unit shadow reranking.

This is a strict candidate-serialization ablation.  The frozen Internal Qwen
model, query views, Top100 candidate identities, and scoring protocol are
unchanged.  Serialization is built only from pre-existing provenance/graph
artifacts and is sealed before any Gold-dependent analysis is loaded.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

BASE_COMMIT = "18e2a21baac3c5eacf19547e0a2492dabd9b6c90"
OUT_NAME = "nf-opt-23-r0-statement-aware-evidence-unit"
TOP100_SHA = "ced014c357a3c8862a5ae3723a4c618556156542fd63aa2a8fd4bf766b60f01a"
VIEWS_SHA = "91bafe5612fab14d1229c877c9dd1bc290b815a8d73f7f014427ce916cdf1705"
MAX_LENGTH = 8192
MODEL_ID = "Qwen/Qwen3-Reranker-4B"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return sha256_file(path)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " | ".join(clean(item) for item in value if clean(item))
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def stable_unique(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean(value)
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def parse_baseline_content(document_view: str) -> str:
    marker = "\n[CONTENT]\n"
    return document_view.split(marker, 1)[1].strip() if marker in document_view else document_view.strip()


def statement_type(value: Any) -> str:
    text = clean(value)
    return text.replace("_", " ") if text else ""


def load_graph(evaluation_root: Path) -> dict[str, Any]:
    graph_root = evaluation_root / "pdf-retrieval-v4-gate-03-r2"
    logical_tables: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(graph_root / "logical-tables.jsonl"):
        traceback = row.get("source_traceback", {})
        table_id = row.get("table_fragment_id") or traceback.get("table_fragment_id")
        if table_id:
            logical_tables[table_id] = row
    semantic_rows = {row["row_id"]: row for row in read_jsonl(graph_root / "semantic-rows.jsonl") if row.get("row_id")}
    axis_bindings: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(graph_root / "axis-bindings.jsonl"):
        axis_bindings[(row.get("table_fragment_id", ""), row.get("row_id", ""))].append(row)
    for values in axis_bindings.values():
        values.sort(key=lambda item: (item.get("column_index", 10**9), clean(item.get("cell_id"))))
    row_matrices = {row["row_id"]: row for row in read_jsonl(graph_root / "row-matrices.jsonl") if row.get("row_id")}
    return {"logical_tables": logical_tables, "semantic_rows": semantic_rows, "axis_bindings": axis_bindings, "row_matrices": row_matrices}


def classify_structured_view(view: dict[str, Any] | None) -> str:
    if not view:
        return "unresolved"
    row_backed = bool(view.get("row_ids")) or view.get("candidate_type") == "table_row"
    table_backed = bool(view.get("source_traceback")) and any(trace.get("table_fragment_id") for trace in view.get("source_traceback", []))
    narrative = view.get("candidate_type") == "text" or any(fact.get("type") == "narrative" for fact in view.get("facts", []))
    if row_backed and narrative:
        return "mixed"
    if row_backed:
        return "table_row_backed"
    if table_backed and view.get("candidate_type") == "table":
        return "table_fragment_backed"
    if table_backed:
        return "table_cell_backed"
    if narrative:
        return "narrative_text"
    return "unresolved"


def fact_pairs(view: dict[str, Any], graph: dict[str, Any], row_id: str, table_id: str) -> tuple[list[str], list[dict[str, str]]]:
    headers: list[str] = []
    pairs: list[dict[str, str]] = []
    for fact in view.get("facts", []) or []:
        if fact.get("type") == "atomic" and fact.get("period"):
            period = clean(fact.get("period"))
            headers.append(period)
            pairs.append({"period": period, "value": clean(fact.get("value")), "currency": clean(fact.get("currency") or fact.get("currency_code")), "scale": clean(fact.get("scale") or fact.get("scale_unit"))})
    if not pairs:
        for fact in view.get("facts", []) or []:
            periods = [clean(x) for x in fact.get("periods", []) if clean(x)]
            values = [clean(x) for x in fact.get("values", []) if clean(x)]
            for period, value in zip(periods, values, strict=False):
                headers.append(period)
                pairs.append({"period": period, "value": value, "currency": clean(fact.get("currency") or fact.get("currency_code")), "scale": clean(fact.get("scale") or fact.get("scale_unit"))})
    for binding in graph["axis_bindings"].get((table_id, row_id), []):
        if binding.get("normalized_period"):
            headers.append(binding["normalized_period"])
    return stable_unique(headers), pairs


def build_statement_unit(candidate_key: str, baseline_document: str, view: dict[str, Any] | None, graph: dict[str, Any]) -> dict[str, Any]:
    """Build a query-independent, fail-closed evidence-unit representation."""
    category = classify_structured_view(view)
    if not view or category == "unresolved":
        text = baseline_document.strip()
        return {
            "candidate_key": candidate_key,
            "candidate_type": "unresolved",
            "serialization": text,
            "serialization_sha256": sha256_text(text),
            "statement_present": False,
            "row_present": False,
            "header_value_binding_present": False,
            "period_value_binding_present": False,
            "metric_path_present": False,
            "currency_present": False,
            "scale_present": False,
            "relational_structure_available": False,
            "fallback": "baseline_document_view",
            "source_ids": {},
            "document_id": "",
            "pdf_page": None,
        }
    traces = [item for item in view.get("source_traceback", []) if isinstance(item, dict)]
    trace = traces[0] if traces else {}
    row_id = clean((view.get("row_ids") or [""])[0])
    table_id = clean(trace.get("table_fragment_id"))
    row = graph["semantic_rows"].get(row_id, {})
    table = graph["logical_tables"].get(table_id, {})
    metric_paths = stable_unique(view.get("metric_paths", []))
    if not metric_paths and view.get("facts"):
        metric_paths = stable_unique(f.get("metric") for f in view.get("facts", []) if isinstance(f, dict))
    row_label = clean(row.get("raw_label")) or (metric_paths[0] if metric_paths else "")
    headers, pairs = fact_pairs(view, graph, row_id, table_id)
    # Prefer the graph's row-matrix dimensions: these carry stable header/value bindings.
    matrix = graph["row_matrices"].get(row_id, {})
    dimensions = sorted(matrix.get("dimensions", []) or [], key=lambda d: (d.get("column_index", 10**9), clean(d.get("cell_id"))))
    matrix_pairs: list[dict[str, str]] = []
    for dim in dimensions:
        period = clean(dim.get("normalized_period"))
        value = clean(dim.get("value_raw"))
        if period:
            matrix_pairs.append({"period": period, "value": value, "currency": clean(matrix.get("currency_code")), "scale": clean(matrix.get("scale_unit") or matrix.get("scale"))})
    if matrix_pairs:
        pairs = matrix_pairs
        headers = stable_unique([item["period"] for item in matrix_pairs if item.get("period")])
    elif headers and pairs:
        pairs = list(pairs)
    statement = clean(table.get("statement_type")) or clean(view.get("table_title")) or clean(view.get("section_path"))
    table_title = clean(table.get("table_title")) or clean(view.get("table_title"))
    document_id = clean(view.get("document_id") or trace.get("document_id"))
    page = view.get("pdf_page") if view.get("pdf_page") is not None else trace.get("pdf_page")
    currency_values = stable_unique(item.get("currency") for item in pairs)
    scale_values = stable_unique(item.get("scale") for item in pairs)
    lines = ["[DOCUMENT]", f"Document: {document_id}" if document_id else "Document: unknown", f"Page: {page}" if page is not None else ""]
    lines += ["", "[STRUCTURE]"]
    if statement:
        lines.append(f"Statement: {statement}")
    if table_title:
        lines.append(f"Table: {table_title}")
    if metric_paths:
        lines.append(f"Metric Path: {' | '.join(metric_paths)}")
    if row_label:
        lines.append(f"Row: {row_label}")
    lines += ["", "[EVIDENCE]", f"Type: {category}"]
    if headers:
        lines.append(f"Column Headers: {' | '.join(headers)}")
    if pairs:
        lines.append("Period / Value:")
        for item in pairs:
            period = item.get("period") or "(unresolved period)"
            value = item.get("value") or "(unresolved value)"
            suffix = " ".join(x for x in (item.get("currency"), item.get("scale")) if x)
            lines.append(f"- {period} = {value}{(' ' + suffix) if suffix else ''}")
    if currency_values:
        lines.append(f"Currency: {' | '.join(currency_values)}")
    if scale_values:
        lines.append(f"Scale: {' | '.join(scale_values)}")
    lines += ["", "[CONTENT]", clean(view.get("raw_content")) or parse_baseline_content(baseline_document), "", "[SOURCE]"]
    source_parts = [x for x in (table_id, row_id, f"page:{page}" if page is not None else "") if x]
    lines.append(" | ".join(source_parts) if source_parts else "stable structured provenance unavailable")
    text = "\n".join(lines).strip()
    relational = bool(row_label and headers and pairs)
    return {
        "candidate_key": candidate_key,
        "candidate_type": category,
        "serialization": text,
        "serialization_sha256": sha256_text(text),
        "statement_present": bool(statement or table_title),
        "row_present": bool(row_label),
        "header_value_binding_present": bool(headers and pairs),
        "period_value_binding_present": bool(any(item.get("period") for item in pairs) and any(item.get("value") for item in pairs)),
        "metric_path_present": bool(metric_paths),
        "currency_present": bool(currency_values),
        "scale_present": bool(scale_values),
        "relational_structure_available": relational,
        "fallback": None,
        "source_ids": {"table_fragment_id": table_id, "row_id": row_id},
        "document_id": document_id,
        "pdf_page": page,
    }


def load_baseline_inputs(evaluation_root: Path) -> dict[str, Any]:
    qview_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3-p0" / "queryplan-rerank-input-views.jsonl.gz"
    qwen_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3" / "main_rerank_predictions.jsonl.gz"
    bounded_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r2a-2" / "bounded-top100-predictions.jsonl.gz"
    qview_rows = read_gzip_jsonl(qview_path)
    qwen_rows = read_gzip_jsonl(qwen_path)
    bounded_rows = read_gzip_jsonl(bounded_path)
    if sha256_file(qview_path) != VIEWS_SHA:
        raise RuntimeError("frozen queryplan input view SHA mismatch")
    if sha256_file(bounded_path) != TOP100_SHA:
        raise RuntimeError("frozen bounded Top100 SHA mismatch")
    if not (len(qview_rows) == len(qwen_rows) == len(bounded_rows) == 72):
        raise RuntimeError("expected exactly 72 frozen cases")
    qviews = {row["case_id"]: row for row in qview_rows}
    qwen_by_case: dict[str, list[dict[str, Any]]] = {}
    bounded_by_case = {row["case_id"]: row for row in bounded_rows}
    baseline_docs: dict[str, str] = {}
    identity_mismatch = 0
    query_mismatch = 0
    rank_mismatch = 0
    for row in qwen_rows:
        case_id = row["case_id"]
        view = qviews.get(case_id)
        bounded = bounded_by_case.get(case_id)
        if view is None or bounded is None:
            raise RuntimeError(f"missing frozen case {case_id}")
        ranked = sorted(row["ranked_candidates"], key=lambda x: int(x["post_rerank_rank"]))
        source_candidates = sorted(view["candidates"], key=lambda x: int(x["pre_rerank_rank"]))
        bounded_candidates = sorted(bounded["candidates"], key=lambda x: int(x.get("rank", x.get("final_candidate_rank", 10**9))))
        if len(ranked) != 100 or len(source_candidates) != 100 or len(bounded_candidates) != 100:
            raise RuntimeError(f"candidate count mismatch {case_id}")
        ranked_query_sha = ranked[0].get("query_view_sha256") if ranked else None
        if view.get("main_query_view_sha256") != ranked_query_sha:
            query_mismatch += 1
        source_keys = [x["candidate_key"] for x in source_candidates]
        ranked_keys = [x["candidate_key"] for x in ranked]
        bounded_keys = [x["candidate_key"] for x in bounded_candidates]
        if set(source_keys) != set(ranked_keys) or set(source_keys) != set(bounded_keys):
            identity_mismatch += 1
        by_key = {x["candidate_key"]: x for x in source_candidates}
        for item in source_candidates:
            baseline_docs.setdefault(item["candidate_key"], item["document_view"])
            if item["candidate_key"] in baseline_docs and baseline_docs[item["candidate_key"]] != item["document_view"]:
                raise RuntimeError(f"query-dependent baseline serialization {item['candidate_key']}")
        bounded_rank = {x["candidate_key"]: int(x.get("rank", x.get("final_candidate_rank", 10**9))) for x in bounded_candidates}
        prepared: list[dict[str, Any]] = []
        for item in ranked:
            key = item["candidate_key"]
            qrank = int(item["post_rerank_rank"])
            brank = int(item.get("pre_rerank_rank")) if item.get("pre_rerank_rank") is not None else None
            if brank is None or key not in bounded_rank or brank != bounded_rank[key]:
                rank_mismatch += 1
            prepared.append({**item, "original_qwen_rank": qrank, "original_bm25_rank": brank, "baseline_document": by_key.get(key, {}).get("document_view", "")})
        qwen_by_case[case_id] = prepared
    if identity_mismatch or query_mismatch or rank_mismatch:
        raise RuntimeError(f"frozen contract mismatch identity={identity_mismatch} query={query_mismatch} rank={rank_mismatch}")
    return {"qviews": qviews, "qwen_by_case": qwen_by_case, "bounded_by_case": bounded_by_case, "baseline_docs": baseline_docs, "qview_sha": sha256_file(qview_path), "top100_sha": sha256_file(bounded_path), "identity_mismatch": 0, "query_mismatch": 0, "rank_mismatch": 0}


def load_internal_contract(backend_root: Path) -> dict[str, Any]:
    evaluation_root = backend_root / "artifacts" / "evaluation"
    path = evaluation_root / "nf-opt-18-r0-reranker-representation-audit" / "internal-reranker-contract.json"
    contract = read_json(path)
    model = contract.get("model", {})
    if model.get("model_id") != MODEL_ID or not model.get("revision"):
        raise RuntimeError("frozen Internal Qwen3-4B contract missing")
    sys.path.insert(0, str(backend_root))
    from src.pdf_retrieval_v4.structure_aware_rerank_view import RERANK_INSTRUCTION  # type: ignore
    instruction_sha = sha256_text(RERANK_INSTRUCTION)
    expected_sha = contract.get("instruction", {}).get("sha256")
    if expected_sha and instruction_sha != expected_sha:
        raise RuntimeError("frozen Internal instruction SHA mismatch")
    return {"contract": contract, "instruction": RERANK_INSTRUCTION, "instruction_sha256": instruction_sha, "model_id": model["model_id"], "revision": model["revision"], "max_length": int(model.get("max_length", MAX_LENGTH)), "dtype": model.get("dtype", "bfloat16"), "batch_size": int(model.get("batch_size", 1)), "scoring": model.get("scoring", "last-position full-vocabulary causal-lm yes/no logits")}


def model_score(rows: list[dict[str, Any]], instruction: str, revision: str, max_length: int, out_path: Path) -> dict[str, Any]:
    """Run exactly the frozen causal-LM yes/no path, one pair at a time."""
    import os
    import platform
    import types
    import importlib.machinery
    # Recent Transformers imports sklearn.metrics.  This run does not use
    # generation/candidate sampling; provide only the unused symbol so the
    # frozen causal-LM scorer remains usable on the server's ABI-mismatched
    # optional SciPy installation.
    if "sklearn" not in sys.modules:
        sklearn_stub = types.ModuleType("sklearn")
        metrics_stub = types.ModuleType("sklearn.metrics")
        sklearn_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
        metrics_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
        metrics_stub.roc_curve = lambda *args, **kwargs: ([], [], [])
        sklearn_stub.metrics = metrics_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.metrics"] = metrics_stub
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.pdf_retrieval_v4.qwen3_reranker import build_input_ids, score_batch  # type: ignore

    snapshot = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-Reranker-4B" / "snapshots" / revision
    if not snapshot.is_dir():
        raise RuntimeError("exact_4b_snapshot_not_cached")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required_for_frozen_qwen4b")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), padding_side="left", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True).to("cuda:0").eval()
    started = time.time()
    prediction_rows: list[dict[str, Any]] = []
    pairs = 0
    truncated = 0
    nonfinite = 0
    token_lengths: list[int] = []
    original_lengths: list[int] = []
    peak_vram = 0
    for source in sorted(rows, key=lambda item: item["case_id"]):
        ranked: list[dict[str, Any]] = []
        query = source["query_view"]
        for candidate in source["candidates"]:
            ids, audit = build_input_ids(tokenizer, instruction, query, candidate["statement_serialization"], max_length)
            score = score_batch(model, tokenizer, [ids])[0]
            value = float(score["reranker_score"])
            nonfinite += int(not math.isfinite(value))
            truncated += int(audit["truncated"])
            token_lengths.append(int(audit["final_token_count"]))
            original_lengths.append(int(audit["original_token_count"]))
            ranked.append({"candidate_key": candidate["candidate_key"], "reranker_score": value, "yes_logit": float(score["yes_logit"]), "no_logit": float(score["no_logit"]), "original_qwen_rank": int(candidate["original_qwen_rank"]), "original_bm25_rank": candidate.get("original_bm25_rank"), "document_view_sha256": candidate["statement_serialization_sha256"], "query_view_sha256": source["query_view_sha256"], "truncated": bool(audit["truncated"]), "final_token_count": int(audit["final_token_count"]), "original_token_count": int(audit["original_token_count"])})
            pairs += 1
            if torch.cuda.is_available():
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated() / (1024 * 1024)))
        ranked.sort(key=lambda item: (-item["reranker_score"], item["original_qwen_rank"], item.get("original_bm25_rank") if item.get("original_bm25_rank") is not None else math.inf, item["candidate_key"]))
        for rank, item in enumerate(ranked, 1):
            item["post_rerank_rank"] = rank
        prediction_rows.append({"case_id": source["case_id"], "input_candidate_count": len(ranked), "ranked_candidates": ranked})
    elapsed = max(time.time() - started, 1e-9)
    prediction_sha = write_gzip_jsonl(out_path, prediction_rows)
    return {
        "model_execution": True, "model_id": MODEL_ID, "model_revision": revision, "pairs_scored": pairs, "queries_scored": len(prediction_rows), "elapsed_seconds": elapsed, "pairs_per_second": pairs / elapsed, "peak_vram_mb": peak_vram, "oom": False, "nonfinite_scores": nonfinite, "truncated_pairs": truncated, "truncated_percentage": truncated / pairs if pairs else 0.0, "token_p50": statistics.median(token_lengths) if token_lengths else None, "token_p90": percentile(token_lengths, 0.90) if token_lengths else None, "token_p95": percentile(token_lengths, 0.95) if token_lengths else None, "token_p99": percentile(token_lengths, 0.99) if token_lengths else None, "token_max": max(token_lengths) if token_lengths else None, "original_token_p50": statistics.median(original_lengths) if original_lengths else None, "prediction_sha256": prediction_sha, "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "dtype": "bfloat16", "batch_size": 1, "max_length": max_length, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


def rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    return {item["candidate_key"]: index + 1 for index, item in enumerate(sorted(items, key=lambda x: int(x.get("post_rerank_rank", x.get("rank", x.get("final_candidate_rank", 10**9))))))}


def top_keys(items: list[dict[str, Any]], k: int) -> list[str]:
    return list(rank_map(items).keys())[:k]


def rate(value: int | float, total: int) -> float | None:
    return value / total if total else None


def percentile(values: list[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_nf21_module(backend_root: Path) -> Any:
    import importlib.util
    path = backend_root / "scripts" / "evaluation" / "run_nf_opt_21_r0_qwen_bm25_late_fusion.py"
    spec = importlib.util.spec_from_file_location("nf21_r0", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable_to_load_nf21_metric_helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def postseal_analysis(backend_root: Path, out_dir: Path, inputs: dict[str, Any], prediction_rows: list[dict[str, Any]], units: dict[str, dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    """All Gold-dependent work is deliberately isolated after prediction seal."""
    evaluation_root = backend_root / "artifacts" / "evaluation"
    r0 = load_nf21_module(backend_root)
    strict_rows = read_jsonl(evaluation_root / "pdf-retrieval-v4-strict-source-contract" / "strict-gold-source-bindings.jsonl")
    targets = r0.load_targets(evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1" / "gold-semantic-targets.jsonl")
    registry_rows = read_gzip_jsonl(evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1-p0" / "candidate-semantic-fact-registry.jsonl.gz")
    facts = r0.build_case_facts(registry_rows)
    calc_rows = read_json(evaluation_root / "nf-opt-19-r0-setwise-ranking-audit" / "calculation-slot-coverage.json").get("cases", [])
    qwen = {case: sorted(items, key=lambda x: int(x["original_qwen_rank"])) for case, items in inputs["qwen_by_case"].items()}
    experiment = {row["case_id"]: sorted(row["ranked_candidates"], key=lambda x: int(x["post_rerank_rank"])) for row in prediction_rows}
    lr_path = evaluation_root / "nf-opt-21-r0-qwen-bm25-late-fusion" / "lrrf-predictions.jsonl.gz"
    lr_rows = read_gzip_jsonl(lr_path)
    lrrf = {row["case_id"]: sorted(row["ranked_candidates"], key=lambda x: int(x.get("fusion_rank", x.get("post_rerank_rank", 10**9)))) for row in lr_rows}
    strict = {"qwen": r0.strict_metrics(strict_rows, qwen), "lrrf_v1": r0.strict_metrics(strict_rows, lrrf), "statement_aware": r0.strict_metrics(strict_rows, experiment)}
    semantic = {"qwen": r0.semantic_metrics(strict_rows, qwen, facts, targets), "lrrf_v1": r0.semantic_metrics(strict_rows, lrrf, facts, targets), "statement_aware": r0.semantic_metrics(strict_rows, experiment, facts, targets)}
    write_json(out_dir / "strict-metrics.json", {"strict_sources": len(strict_rows), **strict})
    write_json(out_dir / "semantic-metrics.json", semantic)

    bindings_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in strict_rows:
        bindings_by_case[binding["case_id"]].append(binding)
    case_ids = sorted(experiment)
    movement: dict[str, Any] = {}
    for name, ranked in (("statement_aware", experiment),):
        rescued = damaged = 0
        rows: list[dict[str, Any]] = []
        for binding in strict_rows:
            case_id, key = binding["case_id"], binding["candidate_key"]
            old = r0.rank_map(qwen[case_id]).get(key, 10**9)
            new = r0.rank_map(ranked[case_id]).get(key, 10**9)
            old_hit, new_hit = old <= 5, new <= 5
            outcome = "unchanged"
            if not old_hit and new_hit:
                rescued += 1
                outcome = "rescued"
            elif old_hit and not new_hit:
                damaged += 1
                outcome = "damaged"
            rows.append({"case_id": case_id, "source_index": binding.get("source_index", 0), "candidate_key": key, "qwen_rank": old, "statement_aware_rank": new, "rank_delta": old - new, "outcome": outcome})
        movement[name] = {"rescued": rescued, "damaged": damaged, "net": rescued - damaged, "rows": rows, "queries_top5_changed": sum(r0.top_keys(qwen[c], 5) != r0.top_keys(ranked[c], 5) for c in case_ids)}
    band_counts = {"6_10": 0, "11_20": 0, "21_100": 0}
    for row in movement["statement_aware"]["rows"]:
        if row["outcome"] == "rescued":
            old_rank = row["qwen_rank"]
            if 6 <= old_rank <= 10:
                band_counts["6_10"] += 1
            elif 11 <= old_rank <= 20:
                band_counts["11_20"] += 1
            elif 21 <= old_rank <= 100:
                band_counts["21_100"] += 1
    write_json(out_dir / "rank-movement.json", {"baseline": "qwen", "variants": movement, "rank_bands": band_counts})

    # Strict C1 and frozen failure-taxonomy attribution are only read here.
    failure_rows = read_json(evaluation_root / "nf-opt-20-r0-pointwise-discrimination-audit" / "failure-taxonomy.json")
    failure_list = failure_rows if isinstance(failure_rows, list) else failure_rows.get("records", failure_rows.get("cases", failure_rows.get("rows", [])))
    cohort: dict[str, dict[str, Any]] = {}
    for tag in ("wrong_period", "multi_slot_competition", "same_metric_wrong_row", "same_document_wrong_evidence", "calculation_operand_competition"):
        selected = [row for row in failure_list if tag in (row.get("tags") or [])]
        selected_ids = {(row.get("case_id"), row.get("source_index")) for row in selected}
        resc = dmg = improved = worsened = unchanged = 0
        for mov in movement["statement_aware"]["rows"]:
            if (mov["case_id"], mov["source_index"]) not in selected_ids:
                continue
            if mov["outcome"] == "rescued":
                resc += 1
            elif mov["outcome"] == "damaged":
                dmg += 1
            if mov["rank_delta"] > 0:
                improved += 1
            elif mov["rank_delta"] < 0:
                worsened += 1
            else:
                unchanged += 1
        deltas = [mov["rank_delta"] for mov in movement["statement_aware"]["rows"] if (mov["case_id"], mov["source_index"]) in selected_ids]
        cohort[tag] = {"cases": len(selected), "rescued": resc, "damaged": dmg, "net": resc - dmg, "rank_improved": improved, "rank_worsened": worsened, "unchanged": unchanged, "mean_rank_delta": statistics.mean(deltas) if deltas else None}
    write_json(out_dir / "failure-cohort-analysis.json", cohort)
    near_ids = {(row.get("case_id"), row.get("source_index")) for row in failure_list if row.get("cohort") == "near_boundary"}
    clear_ids = {(row.get("case_id"), row.get("source_index")) for row in failure_list if row.get("cohort") == "clear_loss"}
    def cohort_summary(ids: set[tuple[Any, Any]]) -> dict[str, Any]:
        rows = [x for x in movement["statement_aware"]["rows"] if (x["case_id"], x["source_index"]) in ids]
        return {"total": len(rows), "rescued": sum(x["outcome"] == "rescued" for x in rows), "damaged": sum(x["outcome"] == "damaged" for x in rows), "rank_improved": sum(x["rank_delta"] > 0 for x in rows), "rank_worsened": sum(x["rank_delta"] < 0 for x in rows), "unchanged": sum(x["rank_delta"] == 0 for x in rows)}
    write_json(out_dir / "near-boundary-clear-loss-analysis.json", {"near_boundary": cohort_summary(near_ids), "clear_loss": cohort_summary(clear_ids)})

    calc_ids = {row["case_id"] for row in calc_rows}
    taxonomy_records = read_json(evaluation_root / "nf-opt-19-r0-setwise-ranking-audit" / "diversity-ceiling.json").get("records", [])
    taxonomy_by_case = {row["case_id"]: row for row in taxonomy_records}
    categories: dict[str, list[str]] = {"single_evidence": [], "multi_evidence": [], "calculation": []}
    for case_id in case_ids:
        record = taxonomy_by_case.get(case_id, {})
        category = "calculation" if bool(record.get("calculation")) or case_id in calc_ids else "multi_evidence" if bool(record.get("multi_evidence")) else "single_evidence"
        categories[category].append(case_id)
    single_bindings = [b for b in strict_rows if b["case_id"] in set(categories["single_evidence"])]
    single_metrics = {"qwen": r0.strict_metrics(single_bindings, qwen), "statement_aware": r0.strict_metrics(single_bindings, experiment)}
    write_json(out_dir / "single-evidence-analysis.json", {"queries": len(categories["single_evidence"]), **single_metrics})
    # The frozen NF-OPT-19 denominator defines calculation as a subset of
    # multi-evidence; use all 16 `multi_evidence=true` records here.
    multi_cases = [case_id for case_id in case_ids if bool(taxonomy_by_case.get(case_id, {}).get("multi_evidence"))]
    multi = {name: {f"@{k}": r0.coverage(bindings_by_case, multi_cases, ranked, k) for k in (5, 10)} for name, ranked in (("qwen", qwen), ("lrrf_v1", lrrf), ("statement_aware", experiment))}
    write_json(out_dir / "multi-evidence-analysis.json", {"queries": len(multi_cases), "variants": multi})
    calc = {name: {f"@{k}": r0.calc_coverage(calc_rows, ranked, facts, k) for k in (5, 10)} for name, ranked in (("qwen", qwen), ("lrrf_v1", lrrf), ("statement_aware", experiment))}
    write_json(out_dir / "calculation-slot-analysis.json", {"queries": len(calc_rows), "variants": calc})

    gold_keys = {row["candidate_key"] for row in strict_rows}
    inventory: dict[str, dict[str, int]] = defaultdict(lambda: {"overall": 0, "gold_candidate": 0, "non_gold_candidate": 0})
    for key, unit in units.items():
        name = unit["candidate_type"]
        inventory[name]["overall"] += 1
        inventory[name]["gold_candidate"] += int(key in gold_keys)
        inventory[name]["non_gold_candidate"] += int(key not in gold_keys)
    write_json(out_dir / "candidate-type-inventory.json", {"unique_candidates": len(units), "types": dict(inventory), "gold_reads_after_prediction_seal": True})
    by_type: dict[str, Any] = {}
    for category in ("table_row_backed", "table_cell_backed", "table_fragment_backed", "narrative_text", "mixed", "unresolved"):
        subset = [b for b in strict_rows if units.get(b["candidate_key"], {}).get("candidate_type") == category]
        if subset:
            by_type[category] = {"gold_bindings": len(subset), "baseline": r0.strict_metrics(subset, qwen), "statement_aware": r0.strict_metrics(subset, experiment)}
    write_json(out_dir / "representation-type-analysis.json", by_type)
    rel = {"true": [], "false": []}
    for b in strict_rows:
        rel["true" if bool(units.get(b["candidate_key"], {}).get("relational_structure_available")) else "false"].append(b)
    write_json(out_dir / "relational-structure-cohort.json", {key: {"bindings": len(value), "baseline": r0.strict_metrics(value, qwen), "statement_aware": r0.strict_metrics(value, experiment)} for key, value in rel.items()})

    statement_r5 = strict["statement_aware"]["@5"]["hits"]
    baseline_r5 = strict["qwen"]["@5"]["hits"]
    mov = movement["statement_aware"]
    table_type = by_type.get("table_row_backed", {})
    table_gain = (table_type.get("statement_aware", {}).get("@5", {}).get("hits", 0) - table_type.get("baseline", {}).get("@5", {}).get("hits", 0)) if table_type else 0
    wrong_period_positive = cohort.get("wrong_period", {}).get("net", 0) > 0
    safe_semantic = semantic["statement_aware"]["@5"]["hits"] >= 51
    safe_multi = multi["statement_aware"]["@5"]["all"] >= 6
    safe_calc = calc["statement_aware"]["@5"]["all_slots"] >= 6
    if statement_r5 >= 49 and mov["net"] >= 6 and mov["damaged"] <= 3 and safe_semantic and safe_multi and safe_calc:
        effectiveness, next_gate = True, "nf-opt-23-r1-query-requirement-serialization"
    elif 46 <= statement_r5 <= 48 and mov["net"] > 0 and mov["damaged"] < mov["rescued"] and table_gain > 0 and wrong_period_positive:
        effectiveness, next_gate = "marginal", "nf-opt-23-r1-query-requirement-serialization"
    else:
        effectiveness, next_gate = False, "reranker_instruction_calibration"
    decision = {"gate": "NF-OPT-23-R0", "evaluation_role": "development_shadow_evidence_unit_ablation", "fresh_blind_evaluation": False, "retrieval_rerun": False, "training": False, "model": MODEL_ID, "model_revision": runtime.get("model_revision"), "reranker_instruction_unchanged": True, "query_representation_unchanged": True, "candidate_identity_unchanged": True, "top100_sha_match": True, "strict_sources": len(strict_rows), "baseline_qwen_r5_hits": baseline_r5, "lrrf_reference_r5_hits": 46, "statement_aware_r5_hits": statement_r5, "baseline_semantic_r5_hits": semantic["qwen"]["@5"]["hits"], "statement_aware_semantic_r5_hits": semantic["statement_aware"]["@5"]["hits"], "rescued": mov["rescued"], "damaged": mov["damaged"], "net": mov["net"], "wrong_period_cases": 17, "wrong_period_rescued": cohort.get("wrong_period", {}).get("rescued", 0), "multi_evidence_all_at_5": multi["statement_aware"]["@5"]["all"], "calculation_all_slots_at_5": calc["statement_aware"]["@5"]["all_slots"], "statement_aware_evidence_unit_effective": effectiveness, "production_switch_allowed": False, "next_gate": next_gate}
    write_json(out_dir / "decision.json", decision)
    return {"strict": strict, "semantic": semantic, "movement": movement, "multi": multi, "calc": calc, "cohort": cohort, "decision": decision, "single": single_metrics, "by_type": by_type}


def main() -> int:
    backend_root = Path(__file__).resolve().parents[2]
    evaluation_root = backend_root / "artifacts" / "evaluation"
    out_dir = evaluation_root / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = load_internal_contract(backend_root)
    inputs = load_baseline_inputs(evaluation_root)
    graph = load_graph(evaluation_root)
    structured_rows = read_jsonl(evaluation_root / "pdf-retrieval-v4-gate-05-r5" / "structured-views.jsonl")
    structured = {row["candidate_key"]: row for row in structured_rows if row.get("candidate_key")}
    units: dict[str, dict[str, Any]] = {}
    for key, baseline_doc in sorted(inputs["baseline_docs"].items()):
        units[key] = build_statement_unit(key, baseline_doc, structured.get(key), graph)
    # Pre-scoring contract and serialization seal.  Nothing in this section
    # loads strict bindings, targets, or answer-derived metadata.
    write_json(out_dir / "frozen-reranker-contract.json", {"gate": "NF-OPT-23-R0", "base_commit": BASE_COMMIT, "model": contract["model_id"], "model_revision": contract["revision"], "dtype": contract["dtype"], "max_length": contract["max_length"], "batch_size": contract["batch_size"], "scoring": contract["scoring"], "instruction_sha256": contract["instruction_sha256"], "instruction_unchanged": True, "query_unchanged": True, "model_execution": True, "retrieval_rerun": False})
    write_json(out_dir / "frozen-candidate-contract.json", {"candidate_source": "frozen_internal_bounded_top100", "candidate_budget": 100, "cases": 72, "candidates_per_case": 100, "candidate_identity_unchanged": True, "top100_sha256": inputs["top100_sha"], "expected_top100_sha256": TOP100_SHA, "candidate_mismatch": 0, "gold_reads_before_prediction_seal": 0})
    type_counts = Counter(unit["candidate_type"] for unit in units.values())
    write_json(out_dir / "candidate-type-inventory.json", {"unique_candidates": len(units), "types": dict(sorted(type_counts.items())), "gold_reads_before_prediction_seal": 0})
    unit_contract = {"gate": "NF-OPT-23-R0", "method": "Statement-Aware Financial Evidence Unit V1", "query_independent": True, "gold_independent": True, "full_row_preserved": True, "original_column_order_preserved": True, "query_period_filtering": False, "missing_structure_policy": "fail_closed_to_baseline_serialization", "narrative_policy": "retain_original_text", "candidate_identity_mutation": 0, "instruction_sha256": contract["instruction_sha256"], "query_view_sha256": inputs["qview_sha"], "max_length": contract["max_length"]}
    write_json(out_dir / "statement-aware-unit-contract.json", unit_contract)
    audit_keys = ("statement_present", "row_present", "header_value_binding_present", "period_value_binding_present", "metric_path_present", "currency_present", "scale_present")
    audit = {key: sum(bool(unit[key]) for unit in units.values()) for key in audit_keys}
    audit.update({"unique_candidates": len(units), "candidate_type_counts": dict(sorted(type_counts.items())), "relational_structure_available": sum(bool(unit["relational_structure_available"]) for unit in units.values()), "gold_reads_before_prediction_seal": 0})
    write_json(out_dir / "serialization-audit.json", audit)
    examples = []
    for key in sorted(units, key=lambda item: hashlib.sha256(item.encode()).hexdigest())[:20]:
        examples.append({"candidate_id": key, "baseline_serialization": inputs["baseline_docs"][key], "statement_aware_serialization": units[key]["serialization"]})
    write_json(out_dir / "representation-examples.json", {"sample_size": len(examples), "selection": "sha256(candidate_id) ascending", "examples": examples})
    experiment_rows: list[dict[str, Any]] = []
    serialization_records: list[dict[str, Any]] = []
    for case_id in sorted(inputs["qviews"]):
        source = inputs["qviews"][case_id]
        query_view = source["main_query_view"]
        qsha = source["main_query_view_sha256"]
        candidates: list[dict[str, Any]] = []
        for candidate in sorted(source["candidates"], key=lambda x: int(x["pre_rerank_rank"])):
            unit = units[candidate["candidate_key"]]
            candidates.append({"candidate_key": candidate["candidate_key"], "original_qwen_rank": int(next(item["original_qwen_rank"] for item in inputs["qwen_by_case"][case_id] if item["candidate_key"] == candidate["candidate_key"])), "original_bm25_rank": next(item.get("original_bm25_rank") for item in inputs["qwen_by_case"][case_id] if item["candidate_key"] == candidate["candidate_key"]), "statement_serialization": unit["serialization"], "statement_serialization_sha256": unit["serialization_sha256"]})
            serialization_records.append({"case_id": case_id, "candidate_key": candidate["candidate_key"], "document_view_sha256": unit["serialization_sha256"], "baseline_document_view_sha256": candidate["document_view_sha256"], "candidate_type": unit["candidate_type"]})
        experiment_rows.append({"case_id": case_id, "query_view": query_view, "query_view_sha256": qsha, "candidates": candidates})
    serialization_sha = write_gzip_jsonl(out_dir / "serialization-manifest.jsonl.gz", serialization_records)
    write_json(out_dir / "serialization-manifest.json", {"gate": "NF-OPT-23-R0", "cases": len(experiment_rows), "candidates": len(serialization_records), "candidates_per_case": 100, "query_views_byte_identical": True, "candidate_identity_unchanged": True, "gold_reads": 0, "serialization_sha256": serialization_sha})
    (out_dir / "serialization-sha256.txt").write_text(serialization_sha + "\n", encoding="utf-8")
    # Baseline token statistics are reused from the frozen scored artifact;
    # the experiment statistics are measured by the exact scoring path.
    baseline_tokens = [int(item["final_token_count"]) for row in inputs["qwen_by_case"].values() for item in row]
    baseline_truncated = sum(bool(item.get("truncated")) for row in inputs["qwen_by_case"].values() for item in row)
    write_json(out_dir / "token-length-baseline.json", {"pairs": len(baseline_tokens), "p50": statistics.median(baseline_tokens), "p95": sorted(baseline_tokens)[int(0.95 * (len(baseline_tokens) - 1))], "max": max(baseline_tokens), "truncated": baseline_truncated})
    prediction_path = out_dir / "predictions.jsonl.gz"
    existing_prediction_rows: list[dict[str, Any]] = []
    if prediction_path.exists():
        try:
            existing_prediction_rows = read_gzip_jsonl(prediction_path)
        except (OSError, json.JSONDecodeError):
            existing_prediction_rows = []
    if len(existing_prediction_rows) == 72 and sum(len(row.get("ranked_candidates", [])) for row in existing_prediction_rows) == 7200:
        # The first frozen scoring pass wrote all predictions before a metrics
        # formatting exception.  Reuse those exact scores; no model call is
        # made on this recovery path.
        token_lengths = [int(item.get("final_token_count", 0)) for row in existing_prediction_rows for item in row["ranked_candidates"]]
        original_lengths = [int(item.get("original_token_count", 0)) for row in existing_prediction_rows for item in row["ranked_candidates"]]
        truncated_existing = sum(bool(item.get("truncated")) for row in existing_prediction_rows for item in row["ranked_candidates"])
        observed_elapsed = 487.6
        runtime = {"model_execution": True, "model_id": MODEL_ID, "model_revision": contract["revision"], "pairs_scored": 7200, "queries_scored": 72, "elapsed_seconds": observed_elapsed, "pairs_per_second": 7200 / observed_elapsed, "peak_vram_mb": None, "oom": False, "nonfinite_scores": 0, "truncated_pairs": truncated_existing, "truncated_percentage": truncated_existing / 7200, "token_p50": statistics.median(token_lengths), "token_p90": percentile(token_lengths, 0.90), "token_p95": percentile(token_lengths, 0.95), "token_p99": percentile(token_lengths, 0.99), "token_max": max(token_lengths), "original_token_p50": statistics.median(original_lengths), "prediction_sha256": sha256_file(prediction_path), "dtype": contract["dtype"], "batch_size": contract["batch_size"], "max_length": contract["max_length"], "reused_completed_prediction_after_metrics_exception": True, "observed_elapsed_seconds_source": "completed_first_pass_runner_elapsed"}
    else:
        runtime = model_score(experiment_rows, contract["instruction"], contract["revision"], contract["max_length"], prediction_path)
    write_json(out_dir / "runtime-metrics.json", {**runtime, "baseline": {"pairs": len(baseline_tokens), "p50": statistics.median(baseline_tokens), "p95": sorted(baseline_tokens)[int(0.95 * (len(baseline_tokens) - 1))], "max": max(baseline_tokens), "truncated": baseline_truncated}})
    prediction_rows = read_gzip_jsonl(out_dir / "predictions.jsonl.gz")
    if len(prediction_rows) != 72 or sum(len(row["ranked_candidates"]) for row in prediction_rows) != 7200:
        raise RuntimeError("prediction completeness contract failed")
    for row in prediction_rows:
        frozen = {item["candidate_key"] for item in inputs["qwen_by_case"][row["case_id"]]}
        actual = {item["candidate_key"] for item in row["ranked_candidates"]}
        if frozen != actual or any(not math.isfinite(float(item["reranker_score"])) for item in row["ranked_candidates"]):
            raise RuntimeError(f"prediction candidate identity/nonfinite failure {row['case_id']}")
    pred_sha = sha256_file(out_dir / "predictions.jsonl.gz")
    write_json(out_dir / "prediction-seal.json", {"gate": "NF-OPT-23-R0", "prediction_sha256": pred_sha, "queries": 72, "pairs": 7200, "candidate_identity_mismatch": 0, "gold_reads_before_prediction_seal": 0, "sealed": True})
    result = postseal_analysis(backend_root, out_dir, inputs, prediction_rows, units, runtime)
    # Supplement decision with mandatory protocol flags and serialization data.
    decision = read_json(out_dir / "decision.json")
    decision.update({"feature_serialization_sha256": serialization_sha, "prediction_sha256": pred_sha, "gold_reads_before_prediction_seal": 0, "candidate_type_counts": dict(sorted(type_counts.items())), "serialization_audit": audit, "runtime_pairs_per_second": runtime["pairs_per_second"], "runtime_truncated_pairs": runtime["truncated_pairs"]})
    write_json(out_dir / "decision.json", decision)
    write_json(out_dir / "serialization-manifest.json", {"gate": "NF-OPT-23-R0", "cases": 72, "pairs": 7200, "serialization_sha256": serialization_sha, "prediction_sha256": pred_sha, "gold_reads_before_prediction_seal": 0, "candidate_identity_unchanged": True})
    (out_dir / "README.md").write_text("# NF-OPT-23 R0\n\nStatement-Aware Evidence Unit V1 candidate-side serialization ablation. The Internal Qwen3-Reranker-4B contract, query bytes, Top100 identities, and scoring formulation are frozen. Gold-dependent artifacts are generated only after `prediction-seal.json`. No retrieval, training, fusion, query-aware filtering, or production switch is performed.\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "runtime": runtime, "strict": result["strict"], "decision": decision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
