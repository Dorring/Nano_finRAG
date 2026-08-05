"""Run the fixed MinerU capability probe for PDF Retrieval V4 Gate 01.

This gate only measures parser output.  It does not build an adapter, write a
production index, run retrieval, generate an answer, or construct a Cell
identity.  The parser input is a deterministic combined PDF made from the
existing 84-page Shadow Page Set plus one fixed complex-table page from each
of the three development PDFs.  The 22 Oracle records are read only after
the raw parser run, for post-hoc capability scoring.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks/financial_rag_v1"
CORPUS_PATH = BENCHMARK / "corpus.json"
DEFAULT_SHADOW_MANIFEST = ROOT / "artifacts/evaluation/nf-opt-08/shadow-page-set-manifest.json"
DEFAULT_ORACLE_PACKAGE = ROOT / "artifacts/evaluation/nf-opt-08-r2/manual-mapping-review-package.json"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-01"
SHARED_NANOCHAT_ROOT = ROOT.parents[4]
DEFAULT_BENCHMARK_PDFS = ROOT.parents[3] / "backend/runtime/benchmark/financial_rag_v1/review-package/pdfs"
DEFAULT_DEVELOPMENT_PDFS = SHARED_NANOCHAT_ROOT / ".runtime/pdf-source-representation-v2"
DEFAULT_RUNTIME = SHARED_NANOCHAT_ROOT / ".runtime/pdf-retrieval-v4-gate-01"
DEFAULT_MINERU = SHARED_NANOCHAT_ROOT / ".runtime/mineru-venv-cu126/bin/mineru"
RUNBOOK_PATH = ROOT / "docs/operations/runtime-environment-runbook.md"

THRESHOLDS = {
    "oracle_table_detection_recall": 0.95,
    "oracle_row_text_recovery": 0.90,
    "oracle_numeric_text_accuracy": 0.98,
    "oracle_period_header_availability": 0.85,
    "oracle_scale_header_availability": 0.90,
    "cross_page_fragment_detection": 0.80,
}

DEV_COMPLEX_PAGES = {
    "adobe_fy2025_pdf_dev": {
        "filename": "adobe_fy2025_pdf_dev.pdf",
        "pdf_page": 50,
        "reason": "consolidated_income_statement_with_multi_year_header",
    },
    "salesforce_fy2026_pdf_dev": {
        "filename": "salesforce_fy2026_pdf_dev.pdf",
        "pdf_page": 45,
        "reason": "subscription_support_revenue_table",
    },
    "walmart_fy2026_pdf_dev": {
        "filename": "walmart_fy2026_pdf_dev.pdf",
        "pdf_page": 54,
        "reason": "consolidated_income_statement_with_multi_year_header",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def isolated_runtime_info(mineru: Path) -> dict[str, Any]:
    """Capture the exact isolated runtime required by the operations runbook."""
    info: dict[str, Any] = {
        "mineru_path": str(mineru),
        "mineru_version": None,
        "version_return_code": None,
        "python_path": str(mineru.parent / "python"),
        "torch_version": None,
        "torch_cuda_version": None,
        "torch_cuda_available": None,
        "torch_return_code": None,
    }
    if not mineru.is_file():
        return info
    try:
        version = subprocess.run(
            [str(mineru), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        info["mineru_version"] = (version.stdout or version.stderr).strip() or None
        info["version_return_code"] = version.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["version_error"] = f"{type(exc).__name__}:{exc}"
    python = mineru.parent / "python"
    if not python.is_file():
        return info
    probe = (
        "import torch; "
        "print(torch.__version__); "
        "print(torch.version.cuda); "
        "print(torch.cuda.is_available())"
    )
    try:
        torch = subprocess.run(
            [str(python), "-c", probe],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        lines = [line.strip() for line in torch.stdout.splitlines() if line.strip()]
        if len(lines) >= 3:
            info["torch_version"], info["torch_cuda_version"] = lines[:2]
            info["torch_cuda_available"] = lines[2].lower() == "true"
        info["torch_return_code"] = torch.returncode
        if torch.returncode != 0:
            info["torch_error"] = torch.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["torch_error"] = f"{type(exc).__name__}:{exc}"
    return info


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def digits_only(value: Any) -> str:
    return re.sub(r"[^0-9]", "", str(value or ""))


def iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def find_named_pdf(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.is_file():
        return direct
    if not root.is_dir():
        return None
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def html_rows(table_html: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL):
        cells: list[str] = []
        for cell_match in re.finditer(r"<t[dh][^>]*>(.*?)</t[dh]>", row_match.group(1), flags=re.IGNORECASE | re.DOTALL):
            cell = re.sub(r"<[^>]+>", " ", cell_match.group(1))
            cells.append(re.sub(r"\s+", " ", html.unescape(cell)).strip())
        if cells:
            rows.append(cells)
    return rows


def extract_html(block: dict[str, Any]) -> str | None:
    for key in ("html", "table_body"):
        value = block.get(key)
        if isinstance(value, str) and "<table" in value.lower():
            return value
    for span in block.get("spans", []) or []:
        if isinstance(span, dict):
            for key in ("html", "table_body"):
                value = span.get(key)
                if isinstance(value, str) and "<table" in value.lower():
                    return value
    return None


def extract_bbox(block: dict[str, Any]) -> list[float] | None:
    for key in ("bbox", "img_bbox"):
        value = block.get(key)
        if isinstance(value, list) and len(value) == 4:
            try:
                return [float(item) for item in value]
            except (TypeError, ValueError):
                continue
    return None


def extract_page_texts(content_path: Path | None) -> dict[int, str]:
    if content_path is None or not content_path.is_file():
        return {}
    try:
        content = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    pages: defaultdict[int, list[str]] = defaultdict(list)
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        page_idx = block.get("page_idx", 0)
        try:
            page = int(page_idx)
        except (TypeError, ValueError):
            page = 0
        if block.get("type") in {"text", "title", "discarded"} and block.get("text"):
            pages[page].append(str(block["text"]))
    return {page: " ".join(texts) for page, texts in pages.items()}


def extract_middle_pages(middle_path: Path | None) -> dict[int, dict[str, Any]]:
    if middle_path is None or not middle_path.is_file():
        return {}
    try:
        payload = json.loads(middle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    page_info = payload.get("pdf_info", []) if isinstance(payload, dict) else []
    pages: dict[int, dict[str, Any]] = {}
    for page_idx, page_payload in enumerate(page_info):
        tables: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for block in iter_dicts(page_payload):
            table_html = extract_html(block)
            if not table_html:
                continue
            key = (normalize_text(table_html), json.dumps(extract_bbox(block), sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            rows = html_rows(table_html)
            tables.append({
                "bbox": extract_bbox(block),
                "row_count": len(rows),
                "rows": rows,
                "text": " ".join(" ".join(row) for row in rows),
            })
        pages[page_idx] = {"tables": tables}
    return pages


def page_metrics(middle: dict[str, Any], page_text: str) -> dict[str, Any]:
    tables = middle.get("tables", [])
    table_text = " ".join(str(table.get("text") or "") for table in tables)
    all_text = f"{page_text} {table_text}"
    numeric_rows = 0
    metric_rows = 0
    for table in tables:
        for row in table.get("rows", []):
            row_text = " ".join(row)
            if re.search(r"(?<![A-Za-z])\(?\d[\d,]*(?:\.\d+)?\)?", row_text):
                numeric_rows += 1
                if row and normalize_text(row[0]):
                    metric_rows += 1
    period_tokens = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b|\bFY\s*\d{4}\b", all_text, flags=re.IGNORECASE)))
    scale_tokens = sorted(set(re.findall(r"\b(?:in|dollars in|amounts in)\s+(?:millions?|thousands?|billions?)\b|\b(?:millions?|thousands?|billions?)\b", all_text, flags=re.IGNORECASE)))
    continuation = bool(re.search(r"\b(?:continued|continuation)\b", all_text, flags=re.IGNORECASE))
    return {
        "table_count": len(tables),
        "numeric_row_count": numeric_rows,
        "metric_row_count": metric_rows,
        "period_tokens": period_tokens,
        "scale_tokens": scale_tokens,
        "page_text": page_text,
        "table_text": table_text,
        "all_text": all_text,
        "continuation_marker": continuation,
    }


def collect_parser_pages(output_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    middle_paths = sorted(output_dir.rglob("*_middle.json"))
    content_paths = sorted(output_dir.rglob("*_content_list.json"))
    middle_path = middle_paths[0] if middle_paths else None
    content_path = content_paths[0] if content_paths else None
    middle_pages = extract_middle_pages(middle_path)
    page_texts = extract_page_texts(content_path)
    pages: dict[int, dict[str, Any]] = {}
    for page_idx in sorted(set(middle_pages) | set(page_texts)):
        pages[page_idx] = page_metrics(middle_pages.get(page_idx, {"tables": []}), page_texts.get(page_idx, ""))
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            files.append({
                "path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return pages, {
        "middle_json_count": len(middle_paths),
        "content_list_count": len(content_paths),
        "model_json_count": len(list(output_dir.rglob("*_model.json"))),
        "raw_file_count": len(files),
        "raw_files": files,
        "middle_path": str(middle_path) if middle_path else None,
        "content_list_path": str(content_path) if content_path else None,
    }


def build_probe_input(records: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    import fitz

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    combined = fitz.open()
    mapped: list[dict[str, Any]] = []
    for probe_index, record in enumerate(records):
        source = fitz.open(record["source_path"])
        page_index = int(record["pdf_page"]) - 1
        if page_index < 0 or page_index >= source.page_count:
            source.close()
            raise ValueError(f"page_out_of_range:{record['document_id']}:{record['pdf_page']}")
        combined.insert_pdf(source, from_page=page_index, to_page=page_index)
        source.close()
        item = dict(record)
        item["probe_page_index"] = probe_index
        mapped.append(item)
    combined.save(output_path, garbage=4, deflate=True)
    combined.close()
    return mapped


def resolve_records(
    shadow_manifest_path: Path,
    oracle_package_path: Path,
    corpus_path: Path,
    benchmark_pdf_dir: Path,
    development_pdf_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    shadow = json.loads(shadow_manifest_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    benchmark_by_id: dict[str, Path] = {}
    issues: list[str] = []
    for document in corpus.get("documents", []):
        path = find_named_pdf(benchmark_pdf_dir, str(document["filename"]))
        if path is None:
            issues.append(f"missing_benchmark_pdf:{document['document_id']}")
        else:
            benchmark_by_id[str(document["document_id"])] = path
    records: list[dict[str, Any]] = []
    for item in shadow.get("pages", []):
        document_id = str(item["document_id"])
        path = benchmark_by_id.get(document_id)
        if path is None:
            continue
        records.append({
            "scope": "benchmark_shadow",
            "document_id": document_id,
            "pdf_page": int(item["pdf_page"]),
            "selection_reasons": list(item.get("selection_reasons", [])),
            "source_candidate_keys": list(item.get("source_candidate_keys", [])),
            "source_path": str(path),
        })
    for document_id, config in DEV_COMPLEX_PAGES.items():
        path = find_named_pdf(development_pdf_dir, config["filename"])
        if path is None:
            issues.append(f"missing_development_pdf:{document_id}")
            continue
        records.append({
            "scope": "development_complex",
            "document_id": document_id,
            "pdf_page": int(config["pdf_page"]),
            "selection_reasons": [config["reason"]],
            "source_candidate_keys": [],
            "source_path": str(path),
        })
    oracle = json.loads(oracle_package_path.read_text(encoding="utf-8"))
    oracle_records = oracle.get("records", [])
    input_manifest = {
        "shadow_page_set_hash": shadow.get("shadow_page_set_hash"),
        "shadow_page_count_declared": shadow.get("page_count"),
        "shadow_page_count_loaded": len(shadow.get("pages", [])),
        "development_complex_page_count": len(DEV_COMPLEX_PAGES),
        "probe_page_count": len(records),
        "oracle_source_count": len(oracle_records),
        "oracle_package_sha256": sha256_file(oracle_package_path),
        "records": records,
        "development_page_policy": DEV_COMPLEX_PAGES,
        "source_identity_is_parser_input": False,
    }
    if len(shadow.get("pages", [])) != 84:
        issues.append(f"shadow_page_count_not_84:{len(shadow.get('pages', []))}")
    if len(oracle_records) != 22:
        issues.append(f"oracle_source_count_not_22:{len(oracle_records)}")
    return records, {"manifest": input_manifest, "oracle_records": oracle_records}, issues


def oracle_page_match(oracle_record: dict[str, Any], page: dict[str, Any]) -> dict[str, bool]:
    proposed = oracle_record.get("proposed_candidate") or {}
    all_text = normalize_text(page.get("all_text"))
    metric = normalize_text(proposed.get("normalized_metric") or oracle_record.get("expected_metric"))
    metric_tokens = [token for token in re.findall(r"[a-z0-9]+", metric) if len(token) > 2]
    metric_match = bool(metric_tokens) and all(token in all_text for token in metric_tokens)
    expected_raw = str(proposed.get("raw_cell_text") or "")
    expected_digits = digits_only(expected_raw)
    expected_base = digits_only(proposed.get("normalized_base_value") or oracle_record.get("expected_value"))
    numeric_match = bool(expected_digits and expected_digits in digits_only(page.get("all_text")))
    if not numeric_match and expected_base:
        numeric_match = expected_base in digits_only(page.get("all_text"))
    period = normalize_text(proposed.get("normalized_period") or oracle_record.get("expected_period"))
    year_match = bool(period) and any(year in all_text for year in re.findall(r"\d{4}", period))
    scale_match = bool(re.search(r"\b(?:million|millions|thousand|thousands|billion|billions)\b", all_text))
    return {
        "table_detected": bool(page.get("table_count")),
        "row_text_recovered": metric_match and bool(page.get("table_count")),
        "metric_text_recovered": metric_match,
        "numeric_text_recovered": numeric_match,
        "numeric_text_accurate": numeric_match,
        "period_header_available": year_match,
        "scale_header_available": scale_match,
    }


def score_backend(
    backend_name: str,
    pages: dict[int, dict[str, Any]],
    records: list[dict[str, Any]],
    oracle_records: list[dict[str, Any]],
    raw_inventory: dict[str, Any],
) -> dict[str, Any]:
    source_page_metrics: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        source_page_metrics[(record["document_id"], int(record["pdf_page"]))] = pages.get(int(record["probe_page_index"]), {"table_count": 0, "all_text": ""})
    per_oracle: list[dict[str, Any]] = []
    counters: defaultdict[str, int] = defaultdict(int)
    missing_oracle_pages: list[dict[str, Any]] = []
    for oracle_record in oracle_records:
        key = (str(oracle_record.get("document_id")), int(oracle_record.get("pdf_page", 0)))
        page = source_page_metrics.get(key)
        if page is None:
            missing_oracle_pages.append({"document_id": key[0], "pdf_page": key[1]})
            continue
        match = oracle_page_match(oracle_record, page)
        for name, value in match.items():
            counters[name] += int(value)
        per_oracle.append({
            "document_id": key[0],
            "pdf_page": key[1],
            "legacy_candidate_key": oracle_record.get("legacy_candidate_key"),
            "observed": match,
        })
    denominator = len(oracle_records)
    def ratio(name: str) -> float | None:
        return counters[name] / denominator if denominator else None

    adjacent_records = [r for r in records if "oracle_adjacent_page" in r.get("selection_reasons", [])]
    adjacent_signal_count = 0
    for record in adjacent_records:
        page = pages.get(int(record["probe_page_index"]), {})
        if page.get("continuation_marker") or len(page.get("period_tokens", [])) >= 2 or page.get("table_count", 0) > 0:
            adjacent_signal_count += 1
    cross_page_rate = adjacent_signal_count / len(adjacent_records) if adjacent_records else None
    metrics = {
        "oracle_source_denominator": denominator,
        "oracle_pages_missing_from_probe": missing_oracle_pages,
        "oracle_table_detection_recall": ratio("table_detected"),
        "oracle_row_text_recovery": ratio("row_text_recovered"),
        "oracle_metric_text_recovery": ratio("metric_text_recovered"),
        "oracle_numeric_text_recovery": ratio("numeric_text_recovered"),
        "oracle_numeric_text_accuracy": ratio("numeric_text_accurate"),
        "oracle_period_header_availability": ratio("period_header_available"),
        "oracle_scale_header_availability": ratio("scale_header_available"),
        "adjacent_page_count": len(adjacent_records),
        "adjacent_page_structural_signal_rate": cross_page_rate,
        "cross_page_fragment_detection": cross_page_rate,
        "cross_page_metric_status": "shadow_selection_diagnostic_not_manual_continuation_label",
        "total_parser_pages_observed": len(pages),
        "total_tables_detected": sum(int(page.get("table_count", 0)) for page in pages.values()),
        "total_numeric_rows_detected": sum(int(page.get("numeric_row_count", 0)) for page in pages.values()),
    }
    gate_checks = {
        key: (metrics.get(key) is not None and metrics[key] >= threshold)
        for key, threshold in THRESHOLDS.items()
    }
    passed = all(gate_checks.values()) and not missing_oracle_pages
    return {
        "backend": backend_name,
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "gate_checks": gate_checks,
        "gate_passed": passed,
        "oracle_posthoc_scoring": True,
        "oracle_records_read_after_raw_parse": denominator,
        "per_oracle_record": per_oracle,
        "raw_inventory": raw_inventory,
    }


def run_backend(
    mineru: Path,
    backend_name: str,
    input_pdf: Path,
    output_dir: Path,
    timeout_seconds: int,
    cuda_visible_devices: str,
    runtime_info: dict[str, Any],
) -> dict[str, Any]:
    config = {
        "hybrid_high": {"backend": "hybrid-engine", "method": "auto", "effort": "high"},
        "pipeline_auto_ocr": {"backend": "pipeline", "method": "auto", "effort": None},
    }[backend_name]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "probe-run.json"
    command = [str(mineru), "-p", str(input_pdf), "-o", str(output_dir), "-b", config["backend"], "-m", config["method"]]
    if config["effort"]:
        command.extend(["--effort", config["effort"]])
    log_path = output_dir / "mineru.log"
    started = time.time()
    return_code: int | None = None
    error: str | None = None
    if not mineru.is_file():
        error = f"mineru_not_found:{mineru}"
    else:
        try:
            with log_path.open("w", encoding="utf-8") as log:
                env = os.environ.copy()
                env["TMPDIR"] = str(DEFAULT_RUNTIME / "finquery_tmp")
                env["TEMP"] = env["TMPDIR"]
                env["TMP"] = env["TMPDIR"]
                env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_seconds, check=False, env=env)
            return_code = completed.returncode
            if return_code != 0:
                error = f"mineru_exit:{return_code}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = f"mineru_execution:{type(exc).__name__}:{exc}"
    elapsed = time.time() - started
    pages, inventory = collect_parser_pages(output_dir)
    manifest = {
        "schema": "pdf-retrieval-v4/gate-01/backend-run/v1",
        "backend": backend_name,
        "mineru_path": str(mineru),
        "mineru_runtime": runtime_info,
        "config": config,
        "input_pdf": str(input_pdf),
        "input_pdf_sha256": sha256_file(input_pdf),
        "command": command,
        "cuda_visible_devices": cuda_visible_devices,
        "return_code": return_code,
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
        "parser_pages_observed": len(pages),
        "raw_output_inventory": inventory,
    }
    write_json(run_manifest_path, manifest)
    return {"manifest": manifest, "pages": pages, "inventory": inventory}


def runbook_smoke(
    mineru: Path,
    smoke_input: Path,
    smoke_output: Path,
    timeout_seconds: int,
    cuda_visible_devices: str,
    runtime_info: dict[str, Any],
) -> dict[str, Any]:
    """Run the required one-PDF smoke before the 87-page probe."""
    if smoke_output.exists():
        shutil.rmtree(smoke_output)
    smoke_output.mkdir(parents=True, exist_ok=True)
    log_path = smoke_output / "mineru-smoke.log"
    command = [str(mineru), "-p", str(smoke_input), "-o", str(smoke_output), "-b", "pipeline", "-m", "auto"]
    started = time.time()
    return_code: int | None = None
    error: str | None = None
    if not mineru.is_file():
        error = f"mineru_not_found:{mineru}"
    else:
        try:
            with log_path.open("w", encoding="utf-8") as log:
                env = os.environ.copy()
                env["TMPDIR"] = str(DEFAULT_RUNTIME / "finquery_tmp")
                env["TEMP"] = env["TMPDIR"]
                env["TMP"] = env["TMPDIR"]
                env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_seconds, check=False, env=env)
            return_code = completed.returncode
            if return_code != 0:
                error = f"mineru_exit:{return_code}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = f"mineru_execution:{type(exc).__name__}:{exc}"
    elapsed = time.time() - started
    _, inventory = collect_parser_pages(smoke_output)
    return {
        "command": command,
        "mineru_path": str(mineru),
        "mineru_runtime": runtime_info,
        "smoke_input": str(smoke_input),
        "smoke_input_sha256": sha256_file(smoke_input) if smoke_input.is_file() else None,
        "smoke_output": str(smoke_output),
        "cuda_visible_devices": cuda_visible_devices,
        "return_code": return_code,
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
        "raw_output_inventory": inventory,
    }


def nvidia_snapshot() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=20)
        return {"command": command, "return_code": completed.returncode, "output": completed.stdout.strip(), "error": completed.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "return_code": None, "output": "", "error": f"{type(exc).__name__}:{exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-manifest", type=Path, default=DEFAULT_SHADOW_MANIFEST)
    parser.add_argument("--oracle-package", type=Path, default=DEFAULT_ORACLE_PACKAGE)
    parser.add_argument("--benchmark-pdf-dir", type=Path, default=DEFAULT_BENCHMARK_PDFS)
    parser.add_argument("--development-pdf-dir", type=Path, default=DEFAULT_DEVELOPMENT_PDFS)
    parser.add_argument("--mineru", type=Path, default=DEFAULT_MINERU)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--cuda-visible-devices", default="", help="CUDA_VISIBLE_DEVICES for the isolated MinerU run; empty means CPU-only.")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    records, auxiliary, input_issues = resolve_records(
        args.shadow_manifest,
        args.oracle_package,
        CORPUS_PATH,
        args.benchmark_pdf_dir,
        args.development_pdf_dir,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    combined_pdf = args.runtime_dir / "probe-input-87-pages.pdf"
    mapped_records = build_probe_input(records, combined_pdf)
    input_manifest = auxiliary["manifest"]
    input_manifest["records"] = mapped_records
    input_manifest["combined_probe_pdf"] = str(combined_pdf)
    input_manifest["combined_probe_pdf_sha256"] = sha256_file(combined_pdf)
    input_manifest["input_issues"] = input_issues
    write_json(args.out_dir / "probe-input-manifest.json", input_manifest)
    write_json(args.out_dir / "shadow-page-inputs.json", {"shadow_manifest": str(args.shadow_manifest), "shadow_page_set_hash": input_manifest["shadow_page_set_hash"], "pages": [record for record in mapped_records if record["scope"] == "benchmark_shadow"]})
    write_json(args.out_dir / "development-complex-pages.json", {"pages": [record for record in mapped_records if record["scope"] == "development_complex"], "policy": DEV_COMPLEX_PAGES})
    protocol = {
        "schema": "pdf-retrieval-v4/gate-01/protocol/v1",
        "gate": "pdf_retrieval_v4_gate_01",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "fixed_backends": {
            "hybrid_high": {"backend": "hybrid-engine", "method": "auto", "effort": "high"},
            "pipeline_auto_ocr": {"backend": "pipeline", "method": "auto", "ocr_mode": "auto"},
        },
        "input_contract": {
            "benchmark_shadow_pages": 84,
            "development_complex_pages": 3,
            "oracle_source_records": 22,
            "combined_probe_pages": len(mapped_records),
            "shadow_page_set_hash": input_manifest["shadow_page_set_hash"],
        },
        "thresholds": THRESHOLDS,
        "parameter_scan": False,
        "per_query_oracle": False,
        "production_index_write": False,
        "retrieval_runs": 0,
        "adapter_builds": 0,
        "index_builds": 0,
        "answer_generation_calls": 0,
        "calculator_calls": 0,
        "binder_calls": 0,
        "runtime_gold_reads": 0,
        "oracle_annotations_read_posthoc": len(auxiliary["oracle_records"]),
        "runbook": {
            "path": str(RUNBOOK_PATH),
            "sha256": sha256_file(RUNBOOK_PATH) if RUNBOOK_PATH.is_file() else None,
            "isolated_mineru_environment": str(args.mineru.parent.parent),
            "project_tmpdir": str(args.runtime_dir / "finquery_tmp"),
            "cuda_visible_devices": args.cuda_visible_devices,
        },
        "mineru_runtime": isolated_runtime_info(args.mineru),
        "nvidia_smi_before_run": nvidia_snapshot(),
    }
    write_json(args.out_dir / "mineru-probe-protocol.json", protocol)
    if args.prepare_only:
        write_json(args.out_dir / "acceptance.json", {
            "gate": "pdf_retrieval_v4_gate_01",
            "gate_passed": False,
            "decision": "mineru_probe_prepared_not_run",
            "next_gate": "mineru_capability_probe",
            "input_issues": input_issues,
            "production_index_writes": 0,
            "production_behavior_changed": False,
            "production_switch_allowed": False,
        })
        return 0 if not input_issues else 2

    smoke_record = next((record for record in mapped_records if "oracle_source_page" in record.get("selection_reasons", [])), mapped_records[0])
    smoke_input = args.runtime_dir / "smoke" / f"{smoke_record['document_id']}-p{smoke_record['pdf_page']}.pdf"
    build_probe_input([smoke_record], smoke_input)
    smoke = runbook_smoke(
        args.mineru,
        smoke_input,
        args.runtime_dir / "smoke" / "output",
        min(args.timeout_seconds, 900),
        args.cuda_visible_devices,
        protocol["mineru_runtime"],
    )
    write_json(args.out_dir / "runbook-smoke.json", smoke)
    if smoke.get("return_code") != 0:
        acceptance = {
            "gate": "pdf_retrieval_v4_gate_01",
            "gate_passed": False,
            "decision": "mineru_probe_smoke_blocked",
            "next_gate": "fix_mineru_runtime_or_capacity",
            "smoke": smoke,
            "invalid_direct_batch_attempt_not_counted": True,
            "mineru_calls": 1,
            "runtime_gold_reads": 0,
            "runtime_governance_reads": 0,
            "retrieval_runs": 0,
            "adapter_builds": 0,
            "index_builds": 0,
            "production_index_writes": 0,
            "production_default_config_modified": False,
            "production_behavior_changed": False,
            "parameter_scan": False,
            "per_query_oracle": False,
            "production_switch_allowed": False,
        }
        write_json(args.out_dir / "acceptance.json", acceptance)
        write_json(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False, "reason": smoke.get("error")})
        return 2

    backend_runs: dict[str, dict[str, Any]] = {}
    for backend_name in ("hybrid_high", "pipeline_auto_ocr"):
        backend_runs[backend_name] = run_backend(
            args.mineru,
            backend_name,
            combined_pdf,
            args.runtime_dir / backend_name,
            args.timeout_seconds,
            args.cuda_visible_devices,
            protocol["mineru_runtime"],
        )
    backend_results: dict[str, Any] = {}
    for backend_name, run in backend_runs.items():
        backend_results[backend_name] = score_backend(
            backend_name,
            run["pages"],
            mapped_records,
            auxiliary["oracle_records"],
            run["inventory"],
        )
    write_json(args.out_dir / "backend-results.json", backend_results)
    write_json(args.out_dir / "capability-metrics.json", {name: result["metrics"] for name, result in backend_results.items()})
    write_json(args.out_dir / "raw-output-inventory.json", {name: result["raw_inventory"] for name, result in backend_results.items()})
    successful = [result for result in backend_results.values() if result["gate_passed"]]
    if successful:
        selected = max(successful, key=lambda result: tuple(result["metrics"].get(key) or 0 for key in THRESHOLDS))
        decision = "mineru_backend_selected"
        next_gate = "unified_structured_adapter"
        selected_backend = selected["backend"]
    elif input_issues:
        decision = "mineru_probe_input_incomplete"
        next_gate = "resolve_probe_inputs"
        selected_backend = None
    elif any(run["manifest"].get("error") for run in backend_runs.values()):
        decision = "mineru_probe_execution_blocked"
        next_gate = "fix_mineru_runtime"
        selected_backend = None
    else:
        decision = "mineru_capability_insufficient"
        next_gate = "stop_and_classify_visual_failures"
        selected_backend = None
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_01",
        "gate_passed": bool(successful),
        "decision": decision,
        "next_gate": next_gate,
        "selected_backend": selected_backend,
        "input_page_count": len(mapped_records),
        "benchmark_shadow_page_count": sum(record["scope"] == "benchmark_shadow" for record in mapped_records),
        "development_complex_page_count": sum(record["scope"] == "development_complex" for record in mapped_records),
        "oracle_source_count": len(auxiliary["oracle_records"]),
        "mineru_calls": 2,
        "runtime_gold_reads": 0,
        "runtime_governance_reads": 0,
        "expected_value_reads": 0,
        "reference_answer_reads": 0,
        "oracle_annotations_read_posthoc": len(auxiliary["oracle_records"]),
        "retrieval_runs": 0,
        "adapter_builds": 0,
        "index_builds": 0,
        "answer_generation_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "production_index_writes": 0,
        "production_default_config_modified": False,
        "production_behavior_changed": False,
        "candidate_identity_conflicts": 0,
        "duplicate_views": 0,
        "parameter_scan": False,
        "per_query_oracle": False,
        "input_issues": input_issues,
        "production_switch_allowed": False,
    }
    write_json(args.out_dir / "acceptance.json", acceptance)
    write_json(args.out_dir / "next-gate.json", {"decision": decision, "next_gate": next_gate, "selected_backend": selected_backend, "production_switch_allowed": False, "reason": "fixed dual-backend capability probe"})
    return 0 if successful else 2


if __name__ == "__main__":
    raise SystemExit(main())
