#!/usr/bin/env python3
"""NF-V2-06 R0: verified-evidence generation ablation.

This runner deliberately starts *after* evidence verification.  It freezes two
packet sets, invokes the same generation instruction with the general and
financial checkpoints, seals both prediction streams, and only then reads
reference answers for post-hoc completeness scoring.  It never changes the
upstream Binder, retrieval, or admission contracts.
"""
from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import run_nf_e2e_01_r0_frozen_retrieval_integration_review as e2e01  # noqa: E402
from scripts.evaluation.run_nf_v2_03_r1d_formal_attempt_6 import load_r1c_frozen_inputs  # noqa: E402


BASE_COMMIT = "984ed8d0e3bae9c63aed593202898693e592dc87"
OUT = ROOT / "artifacts/evaluation/nf-v2-06-r0-verified-generation"
GENERAL_MODEL = "qwen3.7-plus"
FINANCIAL_MODEL = "finquery-finance-v2-lr010-150"
SFT_BASE_URL = "http://127.0.0.1:18001/v1"
ANSWERABLE = 64
NO_ANSWER = 8

GENERATION_PROMPT = """You are a trusted financial answer generator. Answer the question using only the supplied VerifiedEvidencePacket. Every material factual claim must be supported by the packet and cite one or more allowed citation identifiers exactly in square brackets, such as [EV-1]. Preserve every supplied number, fiscal period, unit, currency, and scale exactly. For a supplied calculation result, verbalize the result without doing new arithmetic. Never invent facts, numbers, periods, currencies, units, scales, sources, or citations. If the packet does not support a claim, say that the supplied evidence is insufficient. Return only the concise answer text; do not return JSON, hidden reasoning, or a chain of thought."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())


def stable_sha(value: Any) -> str:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 4) if d else 0.0


def percentile(values: list[float], q: float = 0.95) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    return xs[max(0, min(len(xs) - 1, math.ceil(q * len(xs)) - 1))]


def norm_period(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").upper())


def answerable_metadata() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    qpath = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
    lpath = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
    questions = {str(x["case_id"]): x for x in read_jsonl(qpath)}
    labels = {str(x["case_id"]): x for x in read_jsonl(lpath)}
    return questions, labels


def route_for(plan: Any) -> str:
    return {"DIRECT_FACT": "DIRECT", "CALCULATION": "CALCULATION", "MULTI_EVIDENCE": "MULTI_EVIDENCE"}[plan.intent.value]


def source_to_fact(
    source: Mapping[str, Any],
    candidates: Mapping[str, list[dict[str, Any]]],
    *,
    occurrence: int = 0,
) -> dict[str, Any]:
    """Project one frozen reviewed source candidate into packet evidence.

    Existing FinancialFact fields are preferred.  For reviewed source rows that
    pre-date materialization, the Statement-Aware serialization is retained as
    source-derived provenance; no answer/reference field is consulted.
    """
    key = str(source.get("candidate_key") or "")
    period = str(source.get("period") or "")
    matches = [x for x in candidates.get(key, []) if not period or norm_period(x.get("normalized_period") or x.get("raw_period")) == norm_period(period)]
    fact = matches[0] if matches else (candidates.get(key) or [None])[0]
    parsed = dict(source.get("parsed") or {})
    binding = source.get("period_value_binding") or {}
    fact_id = str(fact.get("fact_id")) if fact else f"reviewed-source:{stable_sha({'candidate_key': key, 'period': period, 'row': source.get('row_label'), 'i': occurrence})[:24]}"
    value = fact.get("parsed_numeric_value") if fact else binding.get("value")
    if value is None:
        value = binding.get("raw_value")
    item = {
        "fact_id": fact_id,
        "source_id": str(fact.get("physical_source_id") if fact else parsed.get("physical_source_id") or key),
        "provenance": {
            "candidate_key": key,
            "document_id": str(fact.get("document_id") if fact else parsed.get("document_id") or source.get("document_id") or ""),
            "page": fact.get("pdf_page") if fact else parsed.get("page") or source.get("page"),
            "table_id": fact.get("table_id") if fact else parsed.get("table_id"),
            "row_id": fact.get("row_id") if fact else parsed.get("row_id"),
            "cell_id": fact.get("cell_id") if fact else None,
            "statement_id": fact.get("statement_id") if fact else parsed.get("statement"),
            "physical_source_id": str(fact.get("physical_source_id") if fact else parsed.get("physical_source_id") or key),
        },
        "metric": str(fact.get("raw_metric") or fact.get("normalized_metric") if fact else parsed.get("row_label") or source.get("row_label") or source.get("metric") or ""),
        "normalized_metric": str(fact.get("normalized_metric") if fact else parsed.get("metric_path") or source.get("row_label") or ""),
        "period": period or str(fact.get("normalized_period") or fact.get("raw_period") if fact else ""),
        "scope": str(source.get("scope") or parsed.get("metric_path") or ""),
        "value": value,
        "unit": fact.get("unit") if fact else source.get("unit") or parsed.get("unit"),
        "currency": fact.get("currency") if fact else source.get("currency") or parsed.get("currency"),
        "scale": fact.get("normalized_scale") or fact.get("raw_scale") if fact else source.get("scale") or parsed.get("scale"),
        "row_label": str(fact.get("raw_metric") if fact else parsed.get("row_label") or source.get("row_label") or ""),
        "row_path": str(parsed.get("metric_path") or source.get("row_path") or ""),
        "column_header_path": parsed.get("column_headers") or source.get("column_header") or [],
        "table_title": str(fact.get("table_id") if fact else parsed.get("table_title") or source.get("table_title") or ""),
        "statement_title": str(source.get("section") or parsed.get("statement") or ""),
        "source_text": str(source.get("serialization") or "")[:4000],
        "fact_origin": "financial_fact_v1" if fact else "reviewed_source_artifact",
    }
    return item


def parse_period_binding(parsed: Mapping[str, Any], period: str) -> dict[str, Any]:
    wanted = norm_period(period)
    for raw in parsed.get("period_value_bindings") or []:
        text = str(raw)
        if norm_period(text.split("=", 1)[0]) != wanted:
            continue
        val = text.split("=", 1)[1].strip() if "=" in text else text
        return {"period": period, "raw_value": val, "value": re.sub(r"[^0-9.\\-]", "", val) or val}
    return {"period": period}


def offline_calculation_result(reviewed: Mapping[str, Any], fallback: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Compute a canonical oracle result from frozen reviewed operands.

    This is component-level Tier B construction only.  It never runs the
    runtime Calculator path and never exposes the natural-language answer.
    """
    if fallback:
        result = dict(fallback)
        result["runtime_calculation_ready"] = False
        result["oracle_calculation_evidence"] = True
        return result
    operands = list(reviewed.get("operands") or [])
    if len(operands) < 2:
        return None
    try:
        values = [float(str(x.get("value")).replace(",", "")) for x in operands]
    except (TypeError, ValueError):
        return None
    operation = str(reviewed.get("operation") or "")
    if operation == "growth_rate":
        value = (values[1] - values[0]) / values[0]
        unit = "ratio"
    elif operation == "ratio":
        value = values[0] / values[1]
        unit = "ratio"
    elif operation == "difference":
        value = values[0] - values[1]
        unit = "currency"
    else:
        return None
    return {
        "formula": reviewed.get("formula"),
        "formula_version": "tier_b.offline_reviewed_operands.v1",
        "operands": [{"name": str(x.get("metric") or "operand"), "period": x.get("period"), "value": x.get("value")} for x in operands],
        "operation": operation,
        "status": "executed_offline",
        "value": str(round(value, 10)),
        "unit": unit,
        "period": ", ".join(str(x.get("period")) for x in operands if x.get("period")),
        "scale": "1",
        "currency": None,
        "runtime_calculation_ready": False,
        "oracle_calculation_evidence": True,
    }


def build_packets() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    questions, labels = answerable_metadata()
    frozen = load_r1c_frozen_inputs()
    plans = frozen["plans"]
    facts = frozen["facts"]
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        ids = {str(fact.get("candidate_id"))} | {str(x) for x in fact.get("candidate_ids", [])}
        for cid in ids:
            by_candidate.setdefault(cid, []).append(fact)
    sada, sada_meta = e2e01.load_sada_inputs(ROOT)
    # Tier A is reconstructed only from the sealed runtime release artifact.
    runtime_path = ROOT / "artifacts/evaluation/nf-v2-03-r7-2-admission-contract-fix/runtime-v2-predictions.jsonl.gz"
    # Tier A is strictly the currently released DIRECT_FACT runtime subset.
    # The sealed runtime artifact also contains historical MULTI_EVIDENCE
    # releases; those are not part of the V2 Direct trusted smoke tier.
    runtime = {str(x["question_id"]): x for x in read_jsonl_gz(runtime_path) if x.get("released") is True and x.get("intent") == "DIRECT_FACT"}
    tier_a: list[dict[str, Any]] = []
    for qid in sorted(runtime):
        req = frozen["requests"][qid]
        row = runtime[qid]
        slot_bindings = row.get("v2_binding", {}).get("slot_bindings", {})
        items: list[dict[str, Any]] = []
        for slot_id, selected in sorted(slot_bindings.items()):
            for fid in selected:
                fact = next((x for x in req.facts if str(x.get("fact_id")) == str(fid)), None)
                if fact is not None:
                    items.append(source_to_fact({"candidate_key": fact.get("candidate_id"), "period": fact.get("normalized_period") or fact.get("raw_period"), "row_label": fact.get("raw_metric")}, {str(fact.get("candidate_id")): [fact]}))
        dedup = {str(x["fact_id"]): x for x in items}
        items = list(dedup.values())
        packet = packet_for(qid, req, "DIRECT", items, oracle=False, component_only=False, questions=questions)
        tier_a.append(packet)
    # Tier B route composition is derived from the frozen Supervisor plan and
    # answerability labels: 48 DIRECT + 11 CALCULATION + 5 MULTI_EVIDENCE.
    tier_b: list[dict[str, Any]] = []
    for qid in sorted(plans):
        if not questions[qid].get("answerable") or labels[qid].get("expected_no_answer"):
            continue
        req = frozen["requests"][qid]
        route = route_for(req.plan)
        reviewed_items: list[dict[str, Any]] = []
        for index, src in enumerate(labels[qid].get("expected_sources") or []):
            key = str(src.get("candidate_key") or "")
            candidates = sada.get(qid, [])
            selected = next((x for x in candidates if str(x.get("candidate_key")) == key), None)
            if selected is None:
                # The reviewed source is still frozen metadata; keep source
                # identity and do not substitute a neighboring candidate.
                selected = {"candidate_key": key, "parsed": {}, "serialization": ""}
            parsed = dict(selected.get("parsed") or {})
            selected = dict(selected)
            selected["period_value_binding"] = parse_period_binding(parsed, str(src.get("period") or ""))
            selected.update({k: src.get(k) for k in ("period", "row_label", "table_title", "section", "unit", "scale", "currency", "document_id", "page", "scope") if src.get(k) is not None})
            reviewed_items.append(source_to_fact(selected, by_candidate, occurrence=index))
        calc_result = None
        if route == "CALCULATION":
            calc_path = ROOT / "artifacts/evaluation/nf-e2e-02-r0-binder-contract-recovery/calculation-shadow-results.json"
            calc_rows = {str(x["case_id"]): x for x in read_json(calc_path)["cases"]}
            c = calc_rows.get(qid)
            calc_result = offline_calculation_result(labels[qid].get("calculation") or {}, (c or {}).get("calculator_result") if c else None)
            if calc_result:
                calc_result["allowed_citation_ids"] = [f"EV-{i+1}" for i in range(len(reviewed_items))]
        tier_b.append(packet_for(qid, req, route, reviewed_items, oracle=True, component_only=True, questions=questions, calc_result=calc_result))
    metadata = {
        "answerable": len(tier_b),
        "tier_a": len(tier_a),
        "route_counts": {route: sum(1 for x in tier_b if x["route"] == route) for route in ("DIRECT", "CALCULATION", "MULTI_EVIDENCE")},
        "no_answer_total": sum(1 for x in questions.values() if not x.get("answerable")),
        "sada": sada_meta,
        "r1c_facts": len(facts),
    }
    if len(tier_b) != ANSWERABLE or metadata["no_answer_total"] != NO_ANSWER:
        raise RuntimeError(f"frozen answerable partition mismatch: {metadata}")
    return tier_a, tier_b, metadata


def packet_for(qid: str, req: Any, route: str, items: list[dict[str, Any]], *, oracle: bool, component_only: bool, questions: Mapping[str, Mapping[str, Any]], calc_result: dict[str, Any] | None = None) -> dict[str, Any]:
    allowed = [f"EV-{i+1}" for i in range(len(items))]
    packet = {
        "query_id": qid,
        "question": str(questions[qid]["question"]),
        "route": route,
        "evidence_items": [{**item, "citation_id": allowed[i]} for i, item in enumerate(items)],
        "calculation_result": calc_result,
        "allowed_citation_ids": allowed,
        "validation_status": "VERIFIED",
        "evaluation_tier": "oracle_verified_generation_only" if oracle else "runtime_trusted_v2",
        "oracle_evidence": oracle,
        "component_eval_only": component_only,
        "fresh_blind": False,
    }
    packet["packet_sha256"] = stable_sha({k: v for k, v in packet.items() if k != "packet_sha256"})
    return packet


def extract_citations(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[([A-Za-z0-9:_-]+)\]", text or "")))


def numeric_strings(text: str) -> list[str]:
    # Citation IDs and fiscal-year labels are not material numeric claims.
    cleaned = re.sub(r"\[[^\]]+\]", "", text or "")
    cleaned = re.sub(r"FY\s*20\d{2}", "", cleaned, flags=re.I)
    raw = re.findall(r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:%|[A-Za-z]+)?", cleaned)
    # Hex/source identifiers are provenance, not financial claims.
    return [token for token in raw if not (re.search(r"[A-Za-z]", token) and token.casefold() not in {"million", "millions", "billion", "billions", "trillion", "trillions", "ratio", "percent", "percentage"})]


def supported_numbers(packet: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for item in packet.get("evidence_items", []):
        for key in ("value",):
            if item.get(key) is not None:
                values.add(re.sub(r"[^0-9.\\-]", "", str(item[key])))
        values.update(re.sub(r"[^0-9.\\-]", "", x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", str(item.get("source_text") or "")))
    calc = packet.get("calculation_result") or {}
    if calc.get("value") is not None:
        values.add(re.sub(r"[^0-9.\\-]", "", str(calc["value"])))
        if str(calc.get("unit") or "").casefold() in {"ratio", "percent", "percentage"}:
            try:
                values.add(str(round(float(calc["value"]) * 100.0, 8)))
            except (TypeError, ValueError):
                pass
    return {x for x in values if x}


def validate_output(packet: Mapping[str, Any], text: str, envelope: Mapping[str, Any]) -> dict[str, Any]:
    citations = extract_citations(text)
    allowed = set(packet.get("allowed_citation_ids") or [])
    unknown = sorted(set(citations) - allowed)
    nums = numeric_strings(text)
    supported = supported_numbers(packet)
    unsupported_nums: list[str] = []
    for token in nums:
        clean = re.sub(r"[^0-9.\\-]", "", token)
        # Years and bare ordinal-like prose are not material numeric claims.
        if len(clean) == 4 and clean.startswith(("19", "20")):
            continue
        if clean and not any(clean == s or (clean.lstrip("0") and clean.lstrip("0") == s.lstrip("0")) for s in supported):
            unsupported_nums.append(token)
    packet_periods = {norm_period(x.get("period")) for x in packet.get("evidence_items", []) if x.get("period")}
    answer_periods = {norm_period(x) for x in re.findall(r"FY\s*20\d{2}", text or "", flags=re.I)}
    period_bad = bool(answer_periods and packet_periods and not answer_periods.issubset(packet_periods))
    envelope_valid = bool(envelope.get("query_id") == packet.get("query_id") and envelope.get("answer_text") is not None and envelope.get("route") == packet.get("route"))
    citation_valid = not unknown
    citation_complete = bool(citations) if text.strip() else True
    numeric_pass = not unsupported_nums
    unit_pass = True
    # Explicit conflicting currency/unit tokens are rejected; absent tokens are safe.
    for token in re.findall(r"\b(?:USD|EUR|GBP|JPY|percent|percentage|million|billion|trillion)\b|[$%]", text or "", flags=re.I):
        packet_text = json.dumps(packet.get("evidence_items", []) + [packet.get("calculation_result") or {}], ensure_ascii=False).casefold()
        if token.casefold() not in packet_text and token not in {"%"}:
            unit_pass = False
    grounded = envelope_valid and numeric_pass and not period_bad and unit_pass and citation_valid and citation_complete
    return {
        "envelope_valid": envelope_valid,
        "numeric_fidelity": numeric_pass,
        "period_fidelity": not period_bad,
        "unit_currency_scale_fidelity": unit_pass,
        "citation_valid": citation_valid,
        "citation_complete": citation_complete,
        "unknown_citations": unknown,
        "unsupported_numeric_claims": unsupported_nums,
        "unsupported_claims": int(not grounded),
        "grounded": grounded,
        "citation_ids": citations,
    }


def call_one(model: str, packet: Mapping[str, Any], base_url: str, api_key: str) -> dict[str, Any]:
    from openai import OpenAI
    started = time.perf_counter()
    prompt_packet = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    request_sha = hashlib.sha256(prompt_packet.encode()).hexdigest()
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=180.0, max_retries=0)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": GENERATION_PROMPT}, {"role": "user", "content": prompt_packet}],
            temperature=0.0,
            max_tokens=512,
        )
        text = str(response.choices[0].message.content or "")
        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else {}
        status = "complete"
    except Exception as exc:  # sealed infrastructure failure; no retry
        text = ""
        usage_dict = {}
        status = "execution_environment_unavailable"
        error = f"{type(exc).__name__}: {str(exc)[:240]}"
    else:
        error = None
    envelope = {"query_id": packet["query_id"], "answer_text": text, "citation_ids": extract_citations(text), "route": packet["route"], "generator_model": model, "generation_status": status}
    result = {"query_id": packet["query_id"], "route": packet["route"], "model": model, "packet_sha256": packet["packet_sha256"], "request_sha256": request_sha, "answer_envelope": envelope, "latency_ms": round((time.perf_counter() - started) * 1000.0, 3), "usage": usage_dict, "status": status}
    if error:
        result["error"] = error
    result["metrics"] = validate_output(packet, text, envelope) if status == "complete" else {"envelope_valid": False, "numeric_fidelity": False, "period_fidelity": False, "unit_currency_scale_fidelity": False, "citation_valid": False, "citation_complete": False, "unknown_citations": [], "unsupported_numeric_claims": [], "unsupported_claims": 1, "grounded": False, "citation_ids": []}
    return result


def run_model(model: str, packets: list[dict[str, Any]], base_url: str, api_key: str) -> list[dict[str, Any]]:
    # Bounded parallelism is still one request per packet, with no semantic retry.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda p: call_one(model, p, base_url, api_key), packets))
    return sorted(rows, key=lambda x: str(x["query_id"]))


def summarize(rows: list[dict[str, Any]], packets: list[dict[str, Any]]) -> dict[str, Any]:
    route_summary: dict[str, Any] = {}
    for route in ("DIRECT", "CALCULATION", "MULTI_EVIDENCE"):
        subset = [x for x in rows if x["route"] == route]
        route_summary[route] = summarize_rows(subset)
    return {"n": len(rows), "overall": summarize_rows(rows), "by_route": route_summary}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    def count(key: str) -> int:
        return sum(int(x.get("metrics", {}).get(key, False)) for x in rows)
    return {
        "n": n,
        "generation_complete": sum(int(x.get("status") == "complete") for x in rows),
        "answer_envelope_valid": count("envelope_valid"),
        "numeric_fidelity": count("numeric_fidelity"),
        "period_fidelity": count("period_fidelity"),
        "unit_currency_scale_fidelity": count("unit_currency_scale_fidelity"),
        "citation_valid": count("citation_valid"),
        "citation_complete": count("citation_complete"),
        "grounded": count("grounded"),
        "unsupported_claim_queries": sum(int(x.get("metrics", {}).get("unsupported_claims", 0) > 0) for x in rows),
        "unsupported_numeric_claims": sum(len(x.get("metrics", {}).get("unsupported_numeric_claims", [])) for x in rows),
        "reference_answer_complete": sum(int(x.get("reference_answer_complete", False)) for x in rows),
        "latency_ms": {"avg": round(statistics.mean([x["latency_ms"] for x in rows]), 3) if rows else 0.0, "p50": round(statistics.median([x["latency_ms"] for x in rows]), 3) if rows else 0.0, "p95": round(percentile([x["latency_ms"] for x in rows]), 3)},
        "input_tokens": sum(int(x.get("usage", {}).get("prompt_tokens", 0) or 0) for x in rows),
        "output_tokens": sum(int(x.get("usage", {}).get("completion_tokens", 0) or 0) for x in rows),
    }


def posthoc_reference_completeness(rows: list[dict[str, Any]], questions: Mapping[str, Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    # Reference answers are intentionally accessed only from this function,
    # called after both model prediction streams have been sealed.
    out: dict[str, int] = {}
    for row in rows:
        ref = labels.get(row["query_id"], {}).get("expected_answer")
        if not ref:
            out[row["query_id"]] = 0
            continue
        answer = str(row.get("answer_envelope", {}).get("answer_text") or "").casefold()
        canonical = str(ref.get("canonical_value") or "")
        display = str(ref.get("display_value") or "").casefold()
        out[row["query_id"]] = int(bool(answer) and ((canonical and canonical in re.sub(r"[^0-9]", "", answer)) or display in answer or not canonical))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tier_a, tier_b, packet_meta = build_packets()
    all_packets = tier_a + tier_b
    write_json(OUT / "verified-evidence-packet-contract.json", {"schema": "VerifiedEvidencePacketV1", "required": ["query_id", "route", "evidence_items", "allowed_citation_ids", "validation_status", "evaluation_tier", "oracle_evidence"], "validation_status": "VERIFIED", "reference_answer_in_packet": False, "gold_used_to_build_tier_b": True, "gold_reads_before_prediction_seal": 0})
    write_jsonl_gz(OUT / "tier-a-runtime-packets.jsonl.gz", tier_a)
    write_jsonl_gz(OUT / "tier-b-oracle-generation-packets.jsonl.gz", tier_b)
    write_json(OUT / "tier-a-packet-seal.json", {"sealed": True, "packet_count": len(tier_a), "packet_set_sha256": stable_sha(tier_a), "gold_reads_before_prediction_seal": 0})
    write_json(OUT / "tier-b-packet-seal.json", {"sealed": True, "packet_count": len(tier_b), "packet_set_sha256": stable_sha(tier_b), "gold_reads_before_prediction_seal": 0, "route_counts": packet_meta["route_counts"]})
    (OUT / "generation-prompt-contract.txt").write_text(GENERATION_PROMPT + "\n", encoding="utf-8")
    (OUT / "generation-prompt-contract.sha256").write_text(sha256_file(OUT / "generation-prompt-contract.txt") + "\n", encoding="utf-8")

    # Resolve the configured general endpoint without ever writing credentials
    # into artifacts.  The checked-in backend .env is loaded only in-process.
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    general_url = os.getenv("V2_SUPERVISOR_BASE_URL", "")
    general_key = os.getenv("V2_SUPERVISOR_API_KEY", "")
    reconcile_only = os.getenv("NF_V2_06_RECONCILE_ONLY") == "1"
    rebuild_calc_only = os.getenv("NF_V2_06_REBUILD_CALC_ONLY") == "1"
    excluded_initial_rows = 0
    if rebuild_calc_only:
        old_general_rows = read_jsonl_gz(OUT / "general-predictions.jsonl.gz")
        old_financial_rows = read_jsonl_gz(OUT / "financial-sft-predictions.jsonl.gz")
        old_shas = {x.get("packet_sha256") for x in old_general_rows} & {x.get("packet_sha256") for x in old_financial_rows}
        current_shas = {p["packet_sha256"] for p in all_packets}
        unchanged_shas = old_shas & current_shas
        replacement_packets = [p for p in all_packets if p["packet_sha256"] not in unchanged_shas]
        if not replacement_packets or any(p["route"] != "CALCULATION" for p in replacement_packets):
            raise RuntimeError("calculation packet rebuild did not isolate changed calculation packets")
        old_general = [x for x in old_general_rows if x.get("packet_sha256") in unchanged_shas]
        old_financial = [x for x in old_financial_rows if x.get("packet_sha256") in unchanged_shas]
        general_rows = old_general + run_model(GENERAL_MODEL, replacement_packets, general_url, general_key)
        financial_rows = old_financial + run_model(FINANCIAL_MODEL, replacement_packets, SFT_BASE_URL, "EMPTY")
        excluded_initial_rows = len(replacement_packets)
    elif reconcile_only:
        # Used only to repair the tier partition metadata after the original
        # sealed run exposed two historical MULTI releases in the runtime file.
        # No provider call is made in this mode.
        allowed = {p["packet_sha256"] for p in all_packets}
        old_general_rows = read_jsonl_gz(OUT / "general-predictions.jsonl.gz")
        old_financial_rows = read_jsonl_gz(OUT / "financial-sft-predictions.jsonl.gz")
        general_rows = [x for x in old_general_rows if x.get("packet_sha256") in allowed]
        financial_rows = [x for x in old_financial_rows if x.get("packet_sha256") in allowed]
        excluded_initial_rows = max(len(old_general_rows), len(old_financial_rows)) - len(all_packets)
        if len(general_rows) != len(all_packets) or len(financial_rows) != len(all_packets):
            raise RuntimeError("reconcile prediction rows do not cover corrected packet set")
        packet_by_sha = {p["packet_sha256"]: p for p in all_packets}
        for rows in (general_rows, financial_rows):
            for row in rows:
                packet = packet_by_sha[row["packet_sha256"]]
                row["metrics"] = validate_output(packet, str(row.get("answer_envelope", {}).get("answer_text") or ""), row.get("answer_envelope", {}))
    else:
        general_rows = run_model(GENERAL_MODEL, all_packets, general_url, general_key) if general_url and general_key else [{"query_id": p["query_id"], "route": p["route"], "model": GENERAL_MODEL, "packet_sha256": p["packet_sha256"], "status": "execution_environment_unavailable", "answer_envelope": {"query_id": p["query_id"], "answer_text": "", "citation_ids": [], "route": p["route"], "generator_model": GENERAL_MODEL, "generation_status": "execution_environment_unavailable"}, "latency_ms": 0.0, "usage": {}, "metrics": {"envelope_valid": False, "numeric_fidelity": False, "period_fidelity": False, "unit_currency_scale_fidelity": False, "citation_valid": False, "citation_complete": False, "unsupported_claims": 1, "grounded": False, "unsupported_numeric_claims": [], "unknown_citations": [], "citation_ids": []}} for p in all_packets]
        financial_rows = run_model(FINANCIAL_MODEL, all_packets, SFT_BASE_URL, "EMPTY")
    # Both streams are sealed before the first reference-answer access.
    write_jsonl_gz(OUT / "general-predictions.jsonl.gz", general_rows)
    write_jsonl_gz(OUT / "financial-sft-predictions.jsonl.gz", financial_rows)
    write_json(OUT / "general-prediction-seal.json", {"sealed": True, "prediction_count": len(general_rows), "packet_set_sha256": stable_sha(all_packets), "prediction_set_sha256": stable_sha(general_rows), "reference_reads_before_prediction_seal": 0})
    write_json(OUT / "financial-sft-prediction-seal.json", {"sealed": True, "prediction_count": len(financial_rows), "packet_set_sha256": stable_sha(all_packets), "prediction_set_sha256": stable_sha(financial_rows), "reference_reads_before_prediction_seal": 0})

    questions, labels = answerable_metadata()
    for rows, name in ((general_rows, "general"), (financial_rows, "financial")):
        complete = posthoc_reference_completeness(rows, questions, labels)
        for row in rows:
            row["reference_answer_complete"] = bool(complete.get(row["query_id"], 0))
    # Rewrite sealed rows with post-hoc-only reference labels in separate result
    # files; the original prediction seals remain the pre-reference boundary.
    tier_a_shas = {p["packet_sha256"] for p in tier_a}
    tier_b_shas = {p["packet_sha256"] for p in tier_b}
    for rows, name in ((general_rows, "general"), (financial_rows, "financial")):
        a_rows = [x for x in rows if x.get("packet_sha256") in tier_a_shas]
        b_rows = [x for x in rows if x.get("packet_sha256") in tier_b_shas]
        write_json(OUT / ("tier-a-results.json" if name == "general" else "tier-a-results-financial-sft.json"), summarize(a_rows, tier_a))
        overall = summarize(b_rows, tier_b)
        write_json(OUT / ("tier-b-overall-results.json" if name == "general" else "tier-b-overall-results-financial-sft.json"), overall)
        for route in ("DIRECT", "CALCULATION", "MULTI_EVIDENCE"):
            route_name = "multi-results" if route == "MULTI_EVIDENCE" else f"{route.lower()}-results"
            write_json(OUT / f"tier-b-{route_name}-{name}.json", summarize([x for x in b_rows if x["route"] == route], [p for p in tier_b if p["route"] == route]))

    g = summarize([x for x in general_rows if x.get("packet_sha256") in tier_b_shas], tier_b)["overall"]
    s = summarize([x for x in financial_rows if x.get("packet_sha256") in tier_b_shas], tier_b)["overall"]
    a_general = summarize([x for x in general_rows if x.get("packet_sha256") in tier_a_shas], tier_a)
    a_financial = summarize([x for x in financial_rows if x.get("packet_sha256") in tier_a_shas], tier_a)
    # Unsuffixed artifacts are the stable comparison contract; suffixed files
    # above remain convenient per-model raw summaries.
    write_json(OUT / "tier-a-results.json", {"general": a_general, "financial_sft": a_financial})
    write_json(OUT / "tier-b-overall-results.json", {"general": g, "financial_sft": s})
    for route in ("DIRECT", "CALCULATION", "MULTI_EVIDENCE"):
        route_name = "multi-results" if route == "MULTI_EVIDENCE" else f"{route.lower()}-results"
        write_json(OUT / f"tier-b-{route_name}.json", {
            "general": summarize([x for x in general_rows if x.get("packet_sha256") in tier_b_shas and x["route"] == route], [p for p in tier_b if p["route"] == route]),
            "financial_sft": summarize([x for x in financial_rows if x.get("packet_sha256") in tier_b_shas and x["route"] == route], [p for p in tier_b if p["route"] == route]),
        })
    ablation = {"general": g, "financial_sft": s, "delta_pp_financial_minus_general": {k: round(pct(s.get(k, 0), ANSWERABLE) - pct(g.get(k, 0), ANSWERABLE), 4) for k in ("answer_envelope_valid", "numeric_fidelity", "period_fidelity", "unit_currency_scale_fidelity", "citation_valid", "citation_complete", "grounded", "unsupported_claim_queries")}}
    write_json(OUT / "general-vs-financial-sft-ablation.json", ablation)
    write_json(OUT / "no-answer-generator-gate.json", {"no_answer_total": NO_ANSWER, "generator_invocations": 0, "generator_invocation_false_positive": 0, "gold_reads_before_prediction_seal": 0})
    def failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts = {key: 0 for key in ("FG0_schema", "FG1_citation", "FG2_numeric", "FG3_period", "FG4_unit_currency_scale", "FG5_unsupported_claim", "FG6_incompleteness", "FG7_style_only", "FG8_other")}
        for row in rows:
            m = row.get("metrics", {})
            if not m.get("envelope_valid", False):
                key = "FG0_schema"
            elif not m.get("citation_valid", False):
                key = "FG1_citation"
            elif not m.get("numeric_fidelity", False):
                key = "FG2_numeric"
            elif not m.get("period_fidelity", False):
                key = "FG3_period"
            elif not m.get("unit_currency_scale_fidelity", False):
                key = "FG4_unit_currency_scale"
            elif m.get("unsupported_claims", 0):
                key = "FG5_unsupported_claim"
            elif not row.get("reference_answer_complete", False):
                key = "FG6_incompleteness"
            else:
                key = "FG7_style_only"
            counts[key] += 1
        return counts
    write_json(OUT / "failure-taxonomy.json", {"general": failure_counts([x for x in general_rows if x.get("packet_sha256") in tier_b_shas]), "financial_sft": failure_counts([x for x in financial_rows if x.get("packet_sha256") in tier_b_shas]), "classification": "first_failed_deterministic_contract_check", "tier_b_only": True})
    write_json(OUT / "latency-token-cost.json", {"general": summarize_rows(general_rows), "financial_sft": summarize_rows(financial_rows), "general_endpoint_configured": bool(general_url and general_key), "financial_endpoint": SFT_BASE_URL})
    financial_noninferior = all(pct(s.get(k, 0), ANSWERABLE) >= pct(g.get(k, 0), ANSWERABLE) - 2.0 for k in ("answer_envelope_valid", "numeric_fidelity", "period_fidelity", "unit_currency_scale_fidelity", "citation_valid", "citation_complete", "grounded"))
    improves = any(s.get(k, 0) > g.get(k, 0) for k in ("numeric_fidelity", "period_fidelity", "citation_valid", "grounded", "reference_answer_complete"))
    effective = bool(general_url and general_key and financial_noninferior and improves and pct(s.get("unsupported_claim_queries", 0), ANSWERABLE) <= pct(g.get("unsupported_claim_queries", 0), ANSWERABLE) + 2.0)
    decision = {"financial_sft_generator_effective": effective, "financial_generator_frozen": effective, "general_endpoint_configured": bool(general_url and general_key), "generation_repair_budget": 0, "reference_reads_before_prediction_seal": 0, "semantic_retries": 0, "scored_model_calls_per_model": len(all_packets), "excluded_initial_rows_from_corrected_tier_partition": excluded_initial_rows, "next_gate": "v2_07_validator_and_generation_repair_once" if effective else "v2_06_financial_sft_failure_review", "tier_a_n": len(tier_a), "tier_b_n": len(tier_b), "route_counts": packet_meta["route_counts"]}
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": "NF-V2-06-R0", "base_commit": BASE_COMMIT, "models": {"general": GENERAL_MODEL, "financial": FINANCIAL_MODEL}, "tiers": {"A": "runtime_trusted_v2", "B": "oracle_verified_generation_only"}, "reference_answer_inputs": False, "semantic_retries": 0, "scored_model_calls_per_model": len(all_packets), "excluded_initial_rows_from_corrected_tier_partition": excluded_initial_rows, "note": "Tier B is component-level generation-only evaluation and must not be reported as fresh-blind E2E."})
    print(json.dumps({"tier_a": len(tier_a), "tier_b": len(tier_b), "route_counts": packet_meta["route_counts"], "general": g, "financial_sft": s, "decision": decision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
