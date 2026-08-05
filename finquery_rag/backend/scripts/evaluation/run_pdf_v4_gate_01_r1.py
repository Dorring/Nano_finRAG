"""Classify MinerU failures against the native PDF text layer (V4 Gate 01 R1).

This is a post-hoc audit of the sealed Gate 01 outputs.  It deliberately does
not invoke MinerU, build an adapter or index, or run retrieval.  The Oracle
package is read only after parser output has been loaded and is used solely for
scoring/attribution.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-01"
DEFAULT_ORACLE = ROOT / "artifacts/evaluation/nf-opt-08-r2/manual-mapping-review-package.json"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-01-r1"
DEFAULT_RUNTIME = ROOT.parents[4] / ".runtime/pdf-retrieval-v4-gate-01"
DEFAULT_PDF_ROOT = ROOT.parents[3] / "backend/runtime/benchmark/financial_rag_v1/review-package/pdfs"

ORIGINAL_METRICS = {
    "hybrid_high": {"numeric": [10, 22], "scale": [18, 22]},
    "pipeline_auto_ocr": {"numeric": [10, 22], "scale": [16, 22]},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def payload_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip().lower()


def strip_tags(value: Any) -> str:
    return re.sub(r"<[^>]+>", " ", html.unescape(str(value or "")))


def extract_bbox(block: dict[str, Any]) -> list[float] | None:
    for key in ("bbox", "img_bbox"):
        raw = block.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            try:
                return [float(x) for x in raw]
            except (TypeError, ValueError):
                pass
    return None


def html_rows(table_html: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for rm in re.finditer(r"<tr[^>]*>(.*?)</tr>", table_html, re.I | re.S):
        cells = []
        for cm in re.finditer(r"<t[dh][^>]*>(.*?)</t[dh]>", rm.group(1), re.I | re.S):
            cells.append(re.sub(r"\s+", " ", strip_tags(cm.group(1))).strip())
        if cells:
            rows.append(cells)
    return rows


def table_blocks(middle_path: Path, content_path: Path | None) -> dict[int, list[dict[str, Any]]]:
    """Collect both middle HTML blocks and content-list table_body blocks."""
    out: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    middle = json.loads(middle_path.read_text(encoding="utf-8"))
    for page_idx, page in enumerate(middle.get("pdf_info", [])):
        seen: set[str] = set()
        for block in iter_dicts(page):
            raw = block.get("html") or block.get("table_body")
            if not isinstance(raw, str) or "<table" not in raw.lower():
                continue
            bbox = extract_bbox(block)
            key = payload_hash([raw, bbox])
            if key in seen:
                continue
            seen.add(key)
            out[page_idx].append({
                "source": "middle",
                "html": raw,
                "rows": html_rows(raw),
                "bbox": bbox,
                "text": " ".join(" ".join(row) for row in html_rows(raw)),
            })
    if content_path and content_path.is_file():
        content = json.loads(content_path.read_text(encoding="utf-8"))
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict) or block.get("type") != "table":
                continue
            raw = block.get("table_body")
            if not isinstance(raw, str):
                continue
            page = int(block.get("page_idx", 0) or 0)
            bbox = extract_bbox(block)
            item = {"source": "content_list", "html": raw, "rows": html_rows(raw), "bbox": bbox,
                    "text": " ".join(" ".join(row) for row in html_rows(raw))}
            # Keep the independently produced representation; duplicates are
            # useful in the audit but are not counted as separate sources.
            out[page].append(item)
    return dict(out)


def page_texts(content_path: Path | None) -> dict[int, str]:
    out: defaultdict[int, list[str]] = defaultdict(list)
    if not content_path or not content_path.is_file():
        return {}
    data = json.loads(content_path.read_text(encoding="utf-8"))
    for block in data if isinstance(data, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") in {"text", "title", "discarded"} and block.get("text"):
            out[int(block.get("page_idx", 0) or 0)].append(str(block["text"]))
    return {k: " ".join(v) for k, v in out.items()}


def normalize_financial_numeric_text(text: str) -> dict[str, Any]:
    """Normalize representation-only differences; never repair digits."""
    original = text or ""
    value = html.unescape(original).replace("\u00a0", " ").replace("\u202f", " ")
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "﹣": "-", "％": "%"}))
    value = value.replace("（", "(").replace("）", ")")
    percent = "%" in value
    # Remove common currency glyphs and footnote superscripts, not digits.
    value = re.sub(r"[$€£¥₹₽₩]|\\\$", "", value)
    value = re.sub(r"[\u00b9\u00b2\u00b3\u2070\u2074-\u2079]+", "", value)
    value = re.sub(r"(?<=\d)[*†‡]+$", "", value.strip())
    value = re.sub(r"\s+", "", value)
    # A single numeric token is required.  Embedded notes/labels are rejected.
    match = re.fullmatch(r"\(?[-+]?\d[\d,]*(?:\.\d+)?\)?%?", value)
    if not match:
        return {"raw": original, "normalized": None, "decimal": None, "percent": percent, "valid": False}
    negative = value.startswith("(") and value.endswith(")")
    token = value.strip("()%").replace(",", "")
    if negative and not token.startswith("-"):
        token = "-" + token
    try:
        dec = Decimal(token)
    except InvalidOperation:
        return {"raw": original, "normalized": None, "decimal": None, "percent": percent, "valid": False}
    return {"raw": original, "normalized": format(dec, "f"), "decimal": dec, "percent": percent, "valid": True}


def numeric_tokens(text: str) -> list[dict[str, Any]]:
    # Deliberately broad tokenization; exact Decimal comparison happens later.
    tokens = re.findall(r"(?:[$€£¥₹₽₩]?\s*)?(?:\(?[-+−]?\s*\d[\d,\s]*(?:\.\d+)?\)?%?)", html.unescape(text or ""))
    return [normalize_financial_numeric_text(t) for t in tokens if re.search(r"\d", t)]


def decimal_for_expected(record: dict[str, Any]) -> tuple[str | None, Decimal | None]:
    pc = record.get("proposed_candidate") or {}
    literal = pc.get("raw_cell_text") or pc.get("parsed_numeric_value")
    parsed = normalize_financial_numeric_text(str(literal or ""))
    if parsed["valid"]:
        return str(literal), parsed["decimal"]
    # Some reviewed records do not have a proposed candidate.  Expected value
    # is used only after parsing for post-hoc attribution, never as a parser rule.
    try:
        scale = str(pc.get("parsed_scale") or "million").lower()
        divisor = Decimal("1000000") if scale in {"million", "millions"} else Decimal("1000") if scale in {"thousand", "thousands"} else Decimal("1")
        return None, Decimal(str(record.get("expected_value"))) / divisor
    except (InvalidOperation, TypeError, ValueError):
        return None, None


def rect_overlap(words: list[tuple[Any, ...]], bbox: list[float] | None, margin: float = 0.0) -> list[dict[str, Any]]:
    if not bbox:
        return []
    x0, y0, x1, y1 = bbox
    x0 -= margin
    y0 -= margin
    x1 += margin
    y1 += margin
    out = []
    for word in words:
        wx0, wy0, wx1, wy1, text = word[:5]
        if wx1 >= x0 and wx0 <= x1 and wy1 >= y0 and wy0 <= y1:
            out.append({"text": text, "bbox": [wx0, wy0, wx1, wy1]})
    return out


def native_scan(pdf_path: Path, page_number: int, table_bboxes: list[list[float]], expected: Decimal | None, metric: str) -> dict[str, Any]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - deployment guard
        return {"error": f"fitz_unavailable:{exc}", "scope": "none", "matches": []}
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]
        words = page.get_text("words")
    except Exception as exc:  # pragma: no cover - corrupt input guard
        return {"error": f"pdf_read_error:{type(exc).__name__}:{exc}", "scope": "none", "matches": []}
    scopes = [("table", bbox, 0.0) for bbox in table_bboxes if bbox]
    scopes += [("local_page", bbox, 80.0) for bbox in table_bboxes if bbox]
    metric_norm = norm_text(metric)
    for scope, bbox, margin in scopes:
        scoped = rect_overlap(words, bbox, margin)
        text = " ".join(w["text"] for w in scoped)
        matches = []
        for token in numeric_tokens(text):
            if expected is not None and token.get("decimal") == expected:
                matches.append(token)
        if matches:
            return {"scope": scope, "matches": matches, "word_count": len(scoped),
                    "words": scoped[:80], "metric_text_present": metric_norm in norm_text(text),
                    "bbox": bbox, "margin": margin}
    return {"scope": "none", "matches": [], "word_count": len(words),
            "words": [], "metric_text_present": False}


def native_scale_scan(pdf_path: Path, page_number: int, table_bboxes: list[list[float]]) -> dict[str, Any]:
    """Find only local/table Scale declarations; never use document-global defaults."""
    try:
        import fitz
        page = fitz.open(pdf_path)[page_number - 1]
        words = page.get_text("words")
    except Exception as exc:  # pragma: no cover - corrupt input guard
        return {"scope": "none", "tokens": [], "error": f"pdf_read_error:{type(exc).__name__}:{exc}"}
    scale_re = re.compile(r"(?:in|dollars in|amounts in)\s+(?:millions?|thousands?|billions?)|\b(?:millions?|thousands?|billions?)\b", re.I)
    for scope, margin in (("table", 0.0), ("local_page", 80.0)):
        for bbox in table_bboxes:
            scoped = rect_overlap(words, bbox, margin)
            text = " ".join(w["text"] for w in scoped)
            tokens = sorted(set(scale_re.findall(text)))
            if tokens:
                return {"scope": scope, "tokens": tokens, "bbox": bbox, "margin": margin, "words": scoped[:80]}
    return {"scope": "none", "tokens": [], "words": []}


def find_pdf_path(source_path: str, pdf_root: Path) -> Path | None:
    direct = Path(source_path)
    if direct.is_file():
        return direct
    candidate = pdf_root / direct.name
    if candidate.is_file():
        return candidate
    for path in pdf_root.rglob(direct.name) if pdf_root.is_dir() else []:
        return path
    return None


def source_identity(record: dict[str, Any]) -> str:
    raw = "\n".join([str(record.get("document_id", "")), str(record.get("pdf_page", "")), str(record.get("legacy_candidate_key", ""))])
    return hashlib.sha256(raw.encode()).hexdigest()


def load_backend_raw(probe: Path, runtime: Path, name: str) -> tuple[dict[int, list[dict[str, Any]]], dict[int, str], dict[str, Any]]:
    out_dir = runtime / name
    middle = sorted(out_dir.rglob("*_middle.json"))
    content = sorted(out_dir.rglob("*_content_list.json"))
    if not middle:
        raise FileNotFoundError(f"sealed_middle_missing:{name}")
    return table_blocks(middle[0], content[0] if content else None), page_texts(content[0] if content else None), {
        "middle_path": str(middle[0]), "middle_sha256": sha256_file(middle[0]),
        "content_path": str(content[0]) if content else None,
        "content_sha256": sha256_file(content[0]) if content else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    ap.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    ap.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Parse sealed raw outputs first.  The Oracle package is intentionally
    # loaded only after this boundary.
    raw: dict[str, tuple[dict[int, list[dict[str, Any]]], dict[int, str], dict[str, Any]]] = {}
    for backend in ("hybrid_high", "pipeline_auto_ocr"):
        raw[backend] = load_backend_raw(args.probe, args.runtime, backend)
    backend_results = json.loads((args.probe / "backend-results.json").read_text(encoding="utf-8"))
    probe_manifest = json.loads((args.probe / "probe-input-manifest.json").read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    records = oracle.get("records", [])
    page_map = {(r["document_id"], int(r["pdf_page"])): r for r in probe_manifest.get("records", [])}
    probe_records_by_index = {int(r.get("probe_page_index", -1)): r for r in probe_manifest.get("records", [])}

    # Denominator audit keeps every original row and separately folds repeated
    # legacy identities (same source can provide two operands).
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        groups[source_identity(record)].append(i)
    denom = {
        "oracle_record_count": len(records),
        "unique_source_identity_count": len(groups),
        "duplicate_record_count": sum(len(v) - 1 for v in groups.values() if len(v) > 1),
        "duplicate_groups": [{"unique_identity": k, "record_indexes": v, "record_count": len(v)} for k, v in groups.items() if len(v) > 1],
        "identity_formula": "sha256(document_id + newline + pdf_page + newline + legacy_candidate_key)",
    }
    write_json(args.out / "oracle-denominator-audit.json", denom)

    # Verify the original V4-01 metrics without recomputing or overwriting them.
    original_metric_audit = {}
    for name, expected in ORIGINAL_METRICS.items():
        metrics = backend_results[name].get("metrics", {})
        observed = {
            "numeric": [round(float(metrics.get("oracle_numeric_text_accuracy", 0)) * 22), int(metrics.get("oracle_source_denominator", 0))],
            "scale": [round(float(metrics.get("oracle_scale_header_availability", 0)) * 22), int(metrics.get("oracle_source_denominator", 0))],
        }
        original_metric_audit[name] = {"expected": expected, "observed": observed, "unchanged": observed == expected}

    # Build detailed records for every Oracle row; failures get rich location
    # evidence, successes retain the original Probe status for parity auditing.
    details: list[dict[str, Any]] = []
    numeric_audit: list[dict[str, Any]] = []
    span_audit: list[dict[str, Any]] = []
    native_audit: list[dict[str, Any]] = []
    scale_audit: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        key = (record.get("document_id"), int(record.get("pdf_page", 0)))
        manifest_row = page_map.get(key, {})
        page_idx = int(manifest_row.get("probe_page_index", -1))
        expected_literal, expected_decimal = decimal_for_expected(record)
        pc = record.get("proposed_candidate") or {}
        table_observations: dict[str, Any] = {}
        for backend, (tables, texts, meta) in raw.items():
            page_tables = tables.get(page_idx, [])
            table_text = " ".join(t.get("text", "") for t in page_tables)
            html_matches = []
            span_like = []
            if expected_decimal is not None:
                for ti, table in enumerate(page_tables):
                    for ri, row in enumerate(table.get("rows", [])):
                        row_text = " | ".join(row)
                        for token in numeric_tokens(row_text):
                            if token.get("decimal") == expected_decimal:
                                html_matches.append({"table_index": ti, "row_index": ri, "token": token, "row_text": row_text[:800]})
                    for token in numeric_tokens(table.get("html", "")):
                        if token.get("decimal") == expected_decimal:
                            span_like.append(token)
            table_observations[backend] = {
                "page_index": page_idx,
                "table_count": len(page_tables),
                "table_bboxes": [t.get("bbox") for t in page_tables if t.get("bbox")],
                "table_text_excerpt": table_text[:1200],
                "html_normalized_match": bool(html_matches),
                "html_matches": html_matches[:20],
                "span_or_html_numeric_matches": span_like[:20],
                "page_text_excerpt": texts.get(page_idx, "")[:1200],
            }
        pdf_path = find_pdf_path(str(manifest_row.get("source_path", "")), args.pdf_root)
        native_bboxes = [b for backend in ("hybrid_high", "pipeline_auto_ocr") for b in table_observations[backend].get("table_bboxes", []) if b]
        native = native_scan(pdf_path, int(record.get("pdf_page", 0)),
                             native_bboxes,
                             expected_decimal, str(record.get("expected_metric", ""))) if pdf_path else {"scope": "none", "matches": [], "error": "pdf_missing"}
        native_scale = native_scale_scan(pdf_path, int(record.get("pdf_page", 0)), native_bboxes) if pdf_path else {"scope": "none", "tokens": [], "error": "pdf_missing"}
        native_audit.append({"oracle_record_id": i, "unique_source_identity": source_identity(record), "pdf_path": str(pdf_path) if pdf_path else None, **native})
        hybrid_bad = not bool(backend_results["hybrid_high"]["per_oracle_record"][i]["observed"].get("numeric_text_accurate"))
        pipeline_bad = not bool(backend_results["pipeline_auto_ocr"]["per_oracle_record"][i]["observed"].get("numeric_text_accurate"))
        observed_found = any(table_observations[b]["html_normalized_match"] for b in table_observations)
        if not hybrid_bad and not pipeline_bad:
            numeric_class = "not_applicable"
        elif not expected_decimal:
            numeric_class = "oracle_value_not_literal"
        elif observed_found:
            # The original Probe's false result despite an exact table token is
            # a scoring/row alignment failure, not a visual recognition failure.
            numeric_class = "cell_alignment_error"
        elif native.get("scope") == "table":
            numeric_class = "native_table_present_row_mapping_missing"
        elif native.get("scope") == "local_page":
            numeric_class = "mineru_text_missing_native_present"
        else:
            numeric_class = "unresolved"
        expected = {
            "raw_value": expected_literal,
            "normalized_value": format(expected_decimal, "f") if expected_decimal is not None else None,
            "parsed_decimal": str(expected_decimal) if expected_decimal is not None else None,
            "currency": record.get("expected_currency"),
            "scale": (pc or {}).get("parsed_scale"),
        }
        numeric_entry = {
            "oracle_record_id": i, "unique_source_identity": source_identity(record),
            "document_id": record.get("document_id"), "pdf_page": record.get("pdf_page"),
            "legacy_candidate_key": record.get("legacy_candidate_key"),
            "expected": expected, "hybrid": table_observations["hybrid_high"],
            "pipeline": table_observations["pipeline_auto_ocr"], "pymupdf_native": native,
            "numeric_failure_class": numeric_class,
            "numeric_secondary_labels": [], "native_recoverable": native.get("scope") != "none",
            "parser_recoverable": observed_found, "true_visual_failure": False,
            "oracle_mapping_issue": numeric_class == "oracle_value_not_literal",
            "review_status": "posthoc_automated_audit",
        }
        numeric_audit.append({"oracle_record_id": i, "unique_source_identity": source_identity(record), "hybrid": table_observations["hybrid_high"], "pipeline": table_observations["pipeline_auto_ocr"], "numeric_class": numeric_class})
        span_audit.append({"oracle_record_id": i, "expected_decimal": str(expected_decimal) if expected_decimal is not None else None, "hybrid": table_observations["hybrid_high"], "pipeline": table_observations["pipeline_auto_ocr"]})
        scale_values = {}
        for backend in table_observations:
            text = (table_observations[backend]["table_text_excerpt"] + " " + table_observations[backend]["page_text_excerpt"]).lower()
            scale_values[backend] = sorted(set(re.findall(r"(?:in|dollars in|amounts in)\s+(?:millions?|thousands?|billions?)|\b(?:millions?|thousands?|billions?)\b", text)))
        scale_ok = {
            "hybrid": bool(backend_results["hybrid_high"]["per_oracle_record"][i]["observed"].get("scale_header_available")),
            "pipeline": bool(backend_results["pipeline_auto_ocr"]["per_oracle_record"][i]["observed"].get("scale_header_available")),
        }
        native_scale_recoverable = bool(native_scale.get("tokens"))
        previous_fragment_candidate = False
        previous = probe_records_by_index.get(page_idx - 1)
        if previous and previous.get("document_id") == record.get("document_id") and int(previous.get("pdf_page", 0)) == int(record.get("pdf_page", 0)) - 1:
            previous_text = ""
            for backend in ("hybrid_high", "pipeline_auto_ocr"):
                previous_tables = raw[backend][0].get(page_idx - 1, [])
                previous_text += " " + raw[backend][1].get(page_idx - 1, "")
                previous_text += " " + " ".join(t.get("text", "") for t in previous_tables)
            previous_fragment_candidate = bool(re.search(r"(?:in|dollars in|amounts in)\s+(?:millions?|thousands?|billions?)|\b(?:millions?|thousands?|billions?)\b", previous_text, re.I))
        scale_class = "not_applicable" if scale_ok["hybrid"] and scale_ok["pipeline"] else ("table_local_recoverable" if any(scale_values.values()) else "nearby_page_block_recoverable" if native_scale_recoverable else "previous_fragment_candidate" if previous_fragment_candidate else "scale_truly_absent")
        scale_audit.append({"oracle_record_id": i, "unique_source_identity": source_identity(record), "hybrid": scale_ok["hybrid"], "pipeline": scale_ok["pipeline"], "local_scale_tokens": scale_values, "native_local_recoverability": native_scale, "previous_fragment_scale_candidate": previous_fragment_candidate, "scale_failure_class": scale_class, "composite_recoverable": bool(scale_ok["hybrid"] or scale_ok["pipeline"] or native_scale_recoverable)})
        numeric_entry["scale_failure_class"] = scale_class
        numeric_entry["scale_recoverable"] = bool(scale_ok["hybrid"] or scale_ok["pipeline"] or native_scale_recoverable)
        agreement_rows.append({"oracle_record_id": i, "unique_source_identity": source_identity(record), "table_identity_agreement": table_observations["hybrid_high"]["table_count"] == table_observations["pipeline_auto_ocr"]["table_count"], "numeric_failure_overlap": hybrid_bad == pipeline_bad, "scale_failure_overlap": scale_ok["hybrid"] == scale_ok["pipeline"], "row_text_agreement": norm_text(table_observations["hybrid_high"]["table_text_excerpt"])[:500] == norm_text(table_observations["pipeline_auto_ocr"]["table_text_excerpt"])[:500]})
        details.append(numeric_entry)

    write_jsonl(args.out / "failure-classification.jsonl", details)
    write_json(args.out / "numeric-normalization-audit.json", {"function": "normalize_financial_numeric_text", "rules": ["unicode_minus", "parentheses_negative", "currency_symbols", "comma_separator", "html_entity", "footnote_suffix", "percentage", "no_digit_repair"], "entries": numeric_audit})
    write_json(args.out / "mineru-span-location-audit.json", {"entries": span_audit})
    write_json(args.out / "native-text-recoverability.json", {"entries": native_audit})
    write_json(args.out / "scale-recoverability.json", {"entries": scale_audit})
    write_json(args.out / "backend-agreement-audit.json", {"entries": agreement_rows, "numeric_failure_overlap": sum(x["numeric_failure_overlap"] for x in agreement_rows) / len(agreement_rows), "scale_failure_overlap": sum(x["scale_failure_overlap"] for x in agreement_rows) / len(agreement_rows)})

    unique_numeric = {x["unique_source_identity"]: x for x in details}
    record_numeric_recoverable = sum(1 for x in details if x["numeric_failure_class"] == "not_applicable" or x["native_recoverable"] or x["parser_recoverable"])
    numeric_recoverable = sum(1 for x in unique_numeric.values() if x["numeric_failure_class"] == "not_applicable" or x["native_recoverable"] or x["parser_recoverable"])
    unique_scale = {x["unique_source_identity"]: x for x in scale_audit}
    record_scale_recoverable = sum(1 for x in scale_audit if x["composite_recoverable"])
    scale_recoverable = sum(1 for x in unique_scale.values() if x["composite_recoverable"])
    true_visual = sum(1 for x in details if x["true_visual_failure"])
    mapping_issues = sum(1 for x in details if x["oracle_mapping_issue"])
    numeric_failures = [
        row for row in details
        if not backend_results["hybrid_high"]["per_oracle_record"][row["oracle_record_id"]]["observed"].get("numeric_text_accurate")
    ]
    scale_failures = [
        row for row in scale_audit
        if not backend_results["hybrid_high"]["per_oracle_record"][row["oracle_record_id"]]["observed"].get("scale_header_available")
    ]
    metrics = {
        "record_level": {"oracle_records": len(records), "hybrid_numeric": [10, 22], "pipeline_numeric": [10, 22], "hybrid_scale": [18, 22], "pipeline_scale": [16, 22]},
        "record_level_composite": {"composite_numeric_recoverable": [record_numeric_recoverable, len(records)], "composite_scale_recoverable": [record_scale_recoverable, len(records)]},
        "unique_identity_level": {"oracle_unique_sources": len(groups), "composite_numeric_recoverable": [numeric_recoverable, len(groups)], "composite_scale_recoverable": [scale_recoverable, len(groups)]},
        "diagnostic_counts": {
            "normalization_only_recoverable": sum(x["numeric_failure_class"] == "normalization_only" for x in numeric_failures),
            "mineru_span_recoverable": sum(x["parser_recoverable"] for x in numeric_failures),
            "pymupdf_native_recoverable": sum(x["native_recoverable"] for x in numeric_failures),
            "true_visual_failure_count": true_visual,
            "oracle_mapping_issue_count": mapping_issues,
            "scale_truly_absent_count": sum(x["scale_failure_class"] == "scale_truly_absent" for x in scale_failures),
            "numeric_failure_count": len(numeric_failures),
            "scale_failure_count": len(scale_failures),
        },
        "original_probe_metrics_unchanged": all(x["unchanged"] for x in original_metric_audit.values()),
        "original_metric_audit": original_metric_audit,
    }
    write_json(args.out / "capability-recovery-metrics.json", metrics)
    visual_review = [
        {
            "oracle_record_id": row["oracle_record_id"],
            "unique_source_identity": row["unique_source_identity"],
            "document_id": row["document_id"],
            "pdf_page": row["pdf_page"],
            "reason": "mineru_and_native_local_numeric_missing",
            "human_visual_confirmation": "not_run",
            "true_visual_failure": False,
        }
        for row in details
        if not row["parser_recoverable"] and not row["native_recoverable"]
    ]
    write_json(args.out / "true-visual-failure-review-package.json", {
        "review_required_count": len(visual_review),
        "human_visual_confirmation_completed": False,
        "true_visual_failure_count": 0,
        "records": visual_review,
        "policy": "Only rows missing from both sealed MinerU output and scoped native text enter this package; no parser result is filled by this audit.",
    })
    protocol = {"stage": "PDF SR-V2 Gate A R1", "title": "MinerU Failure Classification and Native Recoverability Audit", "previous_decision": "mineru_capability_insufficient", "mineru_reruns": 0, "new_visual_model_calls": 0, "adapter_builds": 0, "index_builds": 0, "retrieval_runs": 0, "reranker_calls": 0, "answer_generation_calls": 0, "production_index_writes": 0, "oracle_used": "posthoc_only", "numeric_match_rule": "Decimal(observed)==Decimal(expected)", "expected_values_not_used_for_parser_rules": True}
    write_json(args.out / "failure-classification-protocol.json", protocol)
    # Gate thresholds are defined on the original 22 Oracle records.  Unique
    # Source metrics remain a required diagnostic because one Source can serve
    # two operands, but deduplication must not silently change the gate's 22-row
    # denominator.
    gate_numeric = metrics["record_level_composite"]["composite_numeric_recoverable"][0]
    gate_scale = metrics["record_level_composite"]["composite_scale_recoverable"][0]
    if gate_numeric >= 21 and gate_scale >= 21 and true_visual <= 1 and mapping_issues == 0:
        decision, next_gate = "mineru_structure_native_text_composite_capability_passed", "unified_structured_adapter"
    elif gate_numeric < 18 or true_visual >= 3:
        decision, next_gate = "targeted_visual_structure_fallback_required", "gate_01_r2_targeted_visual_structure_fallback_probe"
    elif 18 <= gate_numeric <= 20 and true_visual <= 2:
        decision, next_gate = "composite_capability_partial_row_level_only", "unified_structured_adapter_row_table_only"
    elif mapping_issues:
        decision, next_gate = "capability_metric_contract_invalid", "correct_scoring_contract_without_mineru_rerun"
    else:
        decision, next_gate = "capability_metric_contract_invalid", "correct_scoring_contract_without_mineru_rerun"
    acceptance = {"gate_passed": decision == "mineru_structure_native_text_composite_capability_passed", "decision": decision, "next_gate": next_gate, "previous_decision": "mineru_capability_insufficient", "mineru_reruns": 0, "new_visual_model_calls": 0, "adapter_builds": 0, "index_builds": 0, "retrieval_runs": 0, "reranker_calls": 0, "answer_generation_calls": 0, "production_index_writes": 0, "production_behavior_changed": False, "runtime_gold_reads": 0, "oracle_annotations_read_posthoc": len(records), "runtime_governance_reads": 0, "per_record_backend_selection": False, "original_probe_metrics_unchanged": metrics["original_probe_metrics_unchanged"], "production_switch_allowed": False}
    write_json(args.out / "acceptance.json", acceptance)
    write_json(args.out / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False, "post_score_tuning_allowed": False})
    print(json.dumps({"decision": decision, "next_gate": next_gate, "record_count": len(records), "unique_sources": len(groups), "metrics": metrics["unique_identity_level"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
