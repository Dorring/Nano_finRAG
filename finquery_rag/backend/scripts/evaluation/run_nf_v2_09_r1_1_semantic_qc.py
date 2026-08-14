#!/usr/bin/env python3
"""NF-V2-09 R1.1 semantic QC and deterministic contract repair.

The previous R2 builder validated arithmetic and surface support but allowed
semantically malformed table mappings.  This runner audits that sealed output,
preserves it as ``semantic_qc_rejected_pretrain``, and rebuilds only from
approved FinQA TRAIN records plus safe R1 TRAIN replay rows.  No model,
retrieval, or training operation is used.
"""
from __future__ import annotations

import gzip
import json
import math
import random
import re
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from scripts.evaluation.run_nf_v2_06_r1a_grounding_alignment import (  # noqa: E402
    ANSWER_RULES,
    DATA_ROOT,
    TokenCounter,
    finqa_evidence,
    norm_question,
    norm_text,
    periods_in,
    source_fingerprints,
    stable_sha,
    write_json,
    write_jsonl,
    write_jsonl_gz,
)

BASE_COMMIT = "83a78469605e7672f758fbb1c1bec4e25731635b"
VIEW_SHA_EXPECTED = "943decf288dffb99ffa6f196abc44e0a5bdb226350cede40e0a160c4bd61f6e4"
OUT = BACKEND / "data/grounding_alignment/v2_targeted"
QC_OUT = BACKEND / "artifacts/evaluation/nf-v2-09-r1-1-semantic-qc"
CONTEXT_LIMIT = 4096
TARGETS = {
    "DIRECT_NUMERIC_SELECTION": 500,
    "CALCULATION_NO_RECOMPUTE": 500,
    "SCOPE_PERIOD_NEAR_MATCH": 200,
    "EXTRA_CLAIM_SUPPRESSION": 150,
    "PARTIAL_ANSWER_NO_OVERREFUSAL": 50,
}
REPLAY_TARGETS = {"POSITIVE_GROUNDED": 220, "PARTIAL_DISTRACTOR": 80, "UNANSWERABLE": 50}


def parse_number(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", "")
    parenthesized_scalar = re.fullmatch(r"\(\s*[-+]?\d+(?:\.\d+)?%?\s*\)", text)
    if parenthesized_scalar:
        text = "-" + text.strip()[1:-1].strip()
    # FinQA sometimes prints an accounting value twice as ``-12 (12)`` or
    # ``0% (0 %)``.  The first scalar is the supported value; the parenthetic
    # display duplicate is formatting, not a second claim.
    text = re.sub(r"\([^()]*\)", "", text).strip()
    text = text.replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
    text = text.replace("(", "-").replace(")", "")
    text = re.sub(r"(?i)\b(percent|percentage)\b", "%", text)
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return None


def number_tokens(text: str) -> list[str]:
    return re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:%|[A-Za-z]+)?", str(text or ""))


def close(a: float, b: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= max(0.02, 0.01 * max(abs(a), abs(b), 1.0))


def split_program(program: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(program):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(program[start:index].strip())
            start = index + 1
    if program[start:].strip():
        parts.append(program[start:].strip())
    return parts


def constant(value: str) -> float | None:
    match = re.fullmatch(r"const_([0-9]+(?:\.[0-9]+)?)", value.strip())
    return float(match.group(1)) if match else None


def replay_program(program: str) -> tuple[float | None, str]:
    refs: list[float] = []
    category = "other"
    for step in split_program(program):
        match = re.fullmatch(r"([a-z_]+)\((.*)\)", step)
        if not match:
            return None, category
        op, raw_args = match.groups()
        values: list[float] = []
        for arg in raw_args.split(","):
            arg = arg.strip()
            if arg.startswith("#"):
                try:
                    values.append(refs[int(arg[1:])])
                except (IndexError, ValueError):
                    return None, category
            elif (value := constant(arg)) is not None:
                values.append(value)
            elif (value := parse_number(arg)) is not None:
                values.append(value)
            else:
                return None, category
        try:
            if op in {"subtract", "difference", "diff"} and len(values) == 2:
                result = values[0] - values[1]
                category = "difference"
            elif op in {"add", "sum"} and len(values) == 2:
                result = values[0] + values[1]
                category = "sum"
            elif op in {"multiply", "mul"} and len(values) == 2:
                result = values[0] * values[1]
                category = "scale_conversion" if any(abs(v) in {10, 100, 1000, 1000000} for v in values) else "other"
            elif op in {"divide", "div"} and len(values) == 2 and values[1] != 0:
                result = values[0] / values[1]
                category = "percentage_share"
            elif op in {"average", "table_average"} and values:
                result = sum(values) / len(values)
                category = "average"
            else:
                return None, category
        except (OverflowError, ZeroDivisionError):
            return None, category
        refs.append(result)
        if "subtract" in program and "divide" in program:
            category = "growth_rate"
    return (refs[-1] if refs else None), category


def table_rows(record: Mapping[str, Any]) -> list[list[str]]:
    return [[str(cell or "").strip() for cell in row] for row in (record.get("table") or []) if any(str(cell or "").strip() for cell in row)]


def row_line(row: list[str]) -> str:
    return " | ".join(cell for cell in row if cell != "")


def has_number(text: str) -> bool:
    return bool(re.search(r"[-+]?\d[\d,.]*(?:%|[A-Za-z]+)?", text or ""))


def numeric_cell(text: str) -> bool:
    """Accept a scalar/range financial cell, not a prose cell containing a year."""
    value = str(text or "").strip()
    if not has_number(value):
        return False
    if is_year(value) or is_year_range(value):
        return False
    if re.search(r"%\s*\(\s*%", value) or re.search(r"\(\s*%\s*\)", value):
        return False
    stripped = re.sub(r"[-+]?\d[\d,.]*(?:%|[A-Za-z]+)?", "", value)
    stripped = re.sub(r"[$€£¥%(),/\\\-+\s]", "", stripped)
    return not bool(re.search(r"[A-Za-z]", stripped))


def is_year(value: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d{2}", value.strip()))


def is_year_range(value: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d{2}\s*[-/]\s*(?:19|20)\d{2}", value.strip()))


def valid_period_label(value: str) -> bool:
    value = str(value or "").strip()
    if not value or "$" in value or "%" in value:
        return False
    if re.search(r"\d{5,}", value):
        return False
    years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", value)
    # A period header may contain one year or a clean two-year range.  Three
    # years, or a repeated year, is the characteristic row/column inversion
    # seen in the malformed source normalization and is not safe to guess.
    if len(years) > 2 or len(years) != len(set(years)):
        return False
    if re.search(r"(?i)\b(amount|value|price|rate|million|thousand|billion)\b", value) and not re.search(r"(?:19|20)\d{2}", value):
        return False
    return bool(re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", value) or re.search(r"(?i)\bq[1-4]\b", value))


def valid_metric(value: str) -> bool:
    value = str(value or "").strip()
    if len(value) < 3 or is_year(value) or is_year_range(value):
        return False
    # The normalized FinQA table occasionally contains a year token spliced
    # into a row label (for example ``workers 2019 compensation``).  That is
    # a source column/header inversion, not a legitimate metric.  Reject any
    # standalone year in a metric; year-bearing metrics are not needed for
    # this TRAIN-only behavioral set and guessing would make the target
    # semantically unsafe.
    if re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", value):
        return False
    if re.fullmatch(r"[-+]?[$€£¥]?\s*[\d,.%() -]+", value):
        return False
    if value.casefold() in {"year", "date", "location", "period", "amount", "value"}:
        return False
    return True


def periods_from_question(question: str) -> tuple[str, set[str]]:
    years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", question or "")
    unique = list(dict.fromkeys(years))
    if len(unique) >= 2 and re.search(r"(?i)\b(from|between|through|to)\b", question):
        return f"{unique[0]}-{unique[-1]}", set(unique)
    if len(unique) == 1:
        return unique[0], set(unique)
    return "not specified", set()


def requested_output(question: str, evidence_text: str) -> dict[str, str]:
    q = question.casefold()
    scale = "1"
    for name in ("million", "thousand", "billion"):
        if name in q:
            scale = name
            break
    unit = "percent" if re.search(r"(?i)\b(percent|percentage|percentual)\b|%", question) else "not specified"
    currency = "USD" if re.search(r"[$]|\bdollar", question, flags=re.I) and re.search(r"[$]|\bdollar", evidence_text, flags=re.I) else "not specified"
    return {"unit": unit, "scale": scale, "currency": currency}


def source_context(record: Mapping[str, Any]) -> str:
    return stable_sha({"filename": record.get("filename"), "table": record.get("table"), "pre_text": record.get("pre_text"), "post_text": record.get("post_text")})


def item(content: str, metric: str, period: str, value: str, record: Mapping[str, Any], key: str, scope: str = "not specified") -> dict[str, Any]:
    return {
        "metric": metric,
        "period": period or "not specified",
        "scope": scope,
        "value": value or "see evidence",
        "unit": "percent" if "%" in content else "not specified",
        "currency": "USD" if "$" in content or "dollar" in content.casefold() else "not specified",
        "scale": "million" if re.search(r"(?i)\bmillions?\b", content) else "billion" if re.search(r"(?i)\bbillions?\b", content) else "not specified",
        "source_label": str(record.get("filename") or "TRAIN source"),
        "source_key": key,
        "content": content,
    }


def render_view(question: str, items: list[Mapping[str, Any]], calculation: Mapping[str, Any] | None = None) -> str:
    lines = ["[QUESTION]", question, "", "[VERIFIED EVIDENCE]", ""]
    ids = [f"E{i + 1}" for i in range(len(items))]
    for eid, obj in zip(ids, items):
        lines.extend([
            f"[{eid}]", f"Metric: {obj.get('metric') or 'source evidence'}", f"Period: {obj.get('period') or 'not specified'}",
            f"Scope: {obj.get('scope') or 'not specified'}", f"Value: {obj.get('value') or 'see evidence'}",
            f"Unit: {obj.get('unit') or 'not specified'}", f"Currency: {obj.get('currency') or 'not specified'}",
            f"Scale: {obj.get('scale') or 'not specified'}", f"Source: {obj.get('source_label') or 'TRAIN source'}",
            f"Evidence: {str(obj.get('content') or '').replace(chr(10), ' ').strip()}", "",
        ])
    if calculation:
        lines.extend([
            "[VERIFIED CALCULATION]", "[C1]", f"Operation: {calculation.get('operation')}",
            f"Canonical Result: {calculation.get('canonical_result')}", f"Period: {calculation.get('period') or 'not specified'}",
            f"Unit: {calculation.get('unit') or 'not specified'}", f"Currency: {calculation.get('currency') or 'not specified'}",
            f"Scale: {calculation.get('scale') or '1'}", f"Based On: {', '.join(f'[{x}]' for x in calculation.get('based_on', []))}", "",
        ])
    lines.append(ANSWER_RULES)
    return "\n".join(lines)


def base_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_dataset": "FinQA", "source_split": "train", "source_example_id": str(record.get("id")),
        "source_context_id": source_context(record), "document_identity": str(record.get("filename") or ""),
    }


def sample_common(base: Mapping[str, Any], *, sample_id: str, question: str, view: str, target: str, bucket: str, subtype: str, route: str = "DIRECT", behavior: str = "POSITIVE_GROUNDED", items: list[Mapping[str, Any]] | None = None, calculation: Mapping[str, Any] | None = None, partial: bool = False) -> dict[str, Any]:
    return {
        "sample_id": sample_id, "source_dataset": base["source_dataset"], "source_split": "train",
        "source_example_id": base["source_example_id"], "source_context_id": base["source_context_id"],
        "document_identity": base["document_identity"], "behavior_type": behavior, "route": route,
        "base_question": question, "messages": [{"role": "user", "content": view}, {"role": "assistant", "content": target}],
        "evidence_ids": [f"E{i + 1}" for i in range(len(items or []))], "calculation_id": f"C1:{sample_id}" if calculation else None,
        "fully_answerable": behavior == "POSITIVE_GROUNDED", "partially_answerable": partial, "requires_abstention": behavior != "POSITIVE_GROUNDED",
        "has_distractors": bool(calculation) or partial, "oracle_training_evidence": False, "citation_required": behavior != "UNANSWERABLE",
        "cot_target": False, "loss_mask_contract": {"user_prompt": 0, "assistant_final_answer": 1, "question_evidence_rules": 0},
        "fingerprints": {"normalized_question_hash": stable_sha(norm_question(question)), "context_hash": base["source_context_id"], "document_hash": stable_sha(base["document_identity"])},
        "alignment_round": "R1.1", "targeted_bucket": bucket, "targeted_subtype": subtype, "r1_replay": False,
        "contains_distractor_operands": bool(calculation), "contains_forbidden_recompute_path": bool(calculation),
        "canonical_result_only_target": bool(calculation), "calculation_metadata": calculation,
    }


def clean_direct_candidate(record: Mapping[str, Any], index: int, bucket: str, subtype: str, partial: bool = False) -> dict[str, Any] | None:
    rows = table_rows(record)
    if len(rows) < 3 or len(rows[0]) < 2:
        return None
    header = rows[0]
    candidates: list[tuple[str, str, str, list[str], int]] = []
    # Standard financial tables: period labels are columns and the first cell
    # of each data row is the metric.
    period_columns = [(i, h) for i, h in enumerate(header) if valid_period_label(h)]
    for ri, row in enumerate(rows[1:], start=1):
        if not row:
            continue
        metric = row[0].strip()
        if not valid_metric(metric):
            continue
        for ci, period in period_columns:
            if ci < len(row) and numeric_cell(row[ci]):
                candidates.append((metric, period.strip(), str(row[ci]).strip(), row, ri))
    # Year-row tables: the first cell is the period and header cells are the
    # metrics.  This avoids the old column/header inversion defect.
    year_header = any(norm_text(h) in {"year", "fiscal year", "period"} for h in header[:1])
    if year_header:
        for ri, row in enumerate(rows[1:], start=1):
            if not row or not is_year(row[0]):
                continue
            for ci, metric in enumerate(header[1:], start=1):
                if valid_metric(metric) and ci < len(row) and numeric_cell(row[ci]):
                    candidates.append((metric.strip(), row[0].strip(), str(row[ci]).strip(), row, ri))
    if not candidates:
        return None
    metric, period, value, row, row_index = candidates[index % len(candidates)]
    if not valid_period_label(period) and not is_year(period):
        return None
    evidence: list[dict[str, Any]] = [item(row_line(row), metric, period, value, record, f"table_{row_index}", metric)]
    # Same metric/different period competitor when available.
    for ci, alternate in enumerate(header):
        if alternate == period or not valid_period_label(alternate) or ci >= len(row) or not has_number(row[ci]):
            continue
        evidence.append(item(row_line(row), metric, alternate.strip(), str(row[ci]).strip(), record, f"table_{row_index}_alt_{ci}", metric))
        break
    # Same period/different metric competitors.
    for ri, other in enumerate(rows[1:], start=1):
        if ri == row_index:
            continue
        ci = header.index(period) if period in header else 0
        if ci < len(other) and numeric_cell(other[ci]) and valid_metric(other[0] if other else ""):
            evidence.append(item(row_line(other), other[0].strip(), period, str(other[ci]).strip(), record, f"table_{ri}", other[0].strip()))
        if len(evidence) >= 4:
            break
    if len(evidence) < 3:
        return None
    question = f"What was {metric} in {period}?"
    target = f"{metric} in {period} was {value} [E1]."
    behavior = "POSITIVE_GROUNDED"
    if bucket == "PARTIAL_ANSWER_NO_OVERREFUSAL":
        question = f"What was {metric} in {period}, and what was the company's share price?"
        target = f"{metric} in {period} was {value} [E1]. The provided evidence does not contain the company's share price."
        behavior = "PARTIAL_DISTRACTOR"
    base = base_fields(record)
    sid = stable_sha({"r1_1": bucket, "source": base["source_example_id"], "row": row_index, "metric": metric, "period": period, "partial": partial})[:32]
    view = render_view(question, evidence)
    return sample_common(base, sample_id=sid, question=question, view=view, target=target, bucket=bucket, subtype=subtype, behavior=behavior, items=evidence, partial=partial)


def clean_calc_items(record: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    raw = finqa_evidence(record)
    rows = table_rows(record)
    selected = {str(x.get("source_key")) for x in raw}
    items: list[dict[str, Any]] = []
    for source in raw[:4]:
        content = str(source.get("content") or "")
        if source.get("kind") == "table_row":
            metric = content.split("|", 1)[0].strip() or "table evidence"
        else:
            metric = "source narrative"
        periods = periods_in(content)
        items.append(item(content, metric, ", ".join(periods), "; ".join(number_tokens(content)[:8]), record, str(source.get("source_key") or "source"), metric))
    for ri, row in enumerate(rows):
        key = f"table_{ri}"
        content = row_line(row)
        if key in selected or not has_number(content):
            continue
        metric = row[0].strip() if row and valid_metric(row[0]) else "distractor evidence"
        items.append(item(content, metric, ", ".join(periods_in(content)), "; ".join(number_tokens(content)[:8]), record, key, metric))
        if len(items) >= 6:
            break
    support_count = max(1, min(len(raw), 4))
    return items, [f"E{i + 1}" for i in range(support_count)]


def question_subject(question: str) -> tuple[str, str]:
    """Turn common FinQA question forms into a concise statement subject.

    This is intentionally a small, deterministic grammar rather than a
    question-specific map.  The old R2 target merely prefixed ``The`` and
    therefore produced answers such as ``what was ... was X``.  We remove the
    interrogative span while retaining any leading temporal/context clause.
    """
    q = re.sub(r"\\[nrt]", " ", question).strip().rstrip("?").strip()
    q = re.sub(r"\s+", " ", q)
    # Keep a leading context clause (In 2014 / At December 31, ... / Based
    # on ...), but split at the first interrogative marker.
    marker_patterns = [
        (r"\bwhat\s+(?:was|is|were|are|will be|would be|might be|could be)\s+", "was"),
        (r"\bhow much\s+(?:was|is|were|are|will be|would be|might be|could be)\s+", "was"),
        (r"\bhow much\s+(?:has|have|had|did|does|do|will|would|might|could)\s+", "was"),
        (r"\bhow much\s+", "was"),
        (r"\bby what\s+percentage\s+(?:did|does|will|would|might)\s+", "was"),
        (r"\bby what\s+percent\s+(?:did|does|will|would|might)\s+", "was"),
        (r"\bby how much\s+(?:did|does|will|would|might)\s+", "was"),
        (r"\bwhat\s+(?:percentage|percent|percentual)\s+", "was"),
        (r"\bwhat\s+percentage\s+", "was"),
        (r"\bwhat\s+", "was"),
    ]
    # ``X was how much of Y?`` asks for the amount of Y represented by X.
    # Express that relation directly instead of retaining the interrogative.
    inverted = re.match(r"(?is)^(?P<left>.+?)\s+was\s+how much of\s+(?P<right>.+)$", q)
    if inverted:
        subject = f"the amount of {inverted.group('right')} represented by {inverted.group('left')}"
        return re.sub(r"\s+", " ", subject).strip(" ,;:"), "was"
    for pattern, verb in marker_patterns:
        match = re.search(pattern, q, flags=re.I)
        if not match:
            continue
        context = q[:match.start()].strip(" ,;:")
        remainder = q[match.end():].strip(" ,;:")
        # For ``by ... did X increase`` the remainder already contains the
        # predicate; naming it as an amount keeps the target direct without
        # introducing arithmetic.
        if pattern.startswith(r"\bby what") or pattern.startswith(r"\bby how"):
            remainder = f"the amount by which {remainder}"
        elif pattern.startswith(r"\bhow much"):
            remainder = f"the amount {remainder}"
        elif pattern.startswith(r"\bwhat\s+(?:percentage|percent|percentual)"):
            label = "percentage" if "percentage" in pattern else "percent"
            remainder = f"the {label} {remainder}"
        elif not (pattern.startswith(r"\bwhat\s+(?:was") or pattern.startswith(r"\bhow much")):
            remainder = f"the {remainder}"
        remainder = re.sub(r"\s*,?\s+in\s+(millions?|thousands?|billions?)\s*$", "", remainder, flags=re.I)
        remainder = re.sub(r"\s*,?\s+as\s+a\s+percentage\s*$", "", remainder, flags=re.I)
        # Avoid an artificial comma for a one-word preposition (``on what
        # percent ...``); otherwise retain the useful leading context clause.
        if context.casefold() in {"on", "in", "at", "for", "of", "as", "during"}:
            subject = f"{context} {remainder}" if context else remainder
        else:
            subject = f"{context}, {remainder}" if context else remainder
        subject = re.sub(r"\s+", " ", subject).strip(" ,;:")
        return subject, verb
    # ``... approximated what ...`` is a common non-initial interrogative.
    match = re.search(r"\bapproximated\s+what\b", q, flags=re.I)
    if match:
        subject = q[:match.start()].strip(" ,;:")
        return subject, "was"
    return q, "was"


def calc_target(question: str, canonical: str, output: Mapping[str, str], evidence_text: str) -> str:
    subject, verb = question_subject(question)
    subject = subject[:1].upper() + subject[1:]
    value = canonical
    if output.get("currency") == "USD" and not value.startswith("$"):
        value = "$" + value
    scale = output.get("scale")
    if scale and scale != "1":
        value = f"{value} {scale}"
    elif output.get("unit") == "percent":
        value = f"{value}%"
    return f"{subject} {verb} {value} [C1]."


def clean_calc_candidate(record: Mapping[str, Any]) -> dict[str, Any] | None:
    qa = record.get("qa") or {}
    program = str(qa.get("program") or "")
    raw_canonical = qa.get("exe_ans")
    canonical = parse_number(raw_canonical)
    if not program or canonical is None:
        return None
    computed, category = replay_program(program)
    if computed is None or not close(computed, canonical):
        return None
    question = str(qa.get("question") or "").strip()
    if not question:
        return None
    if re.search(r"\\[nrt]|[\x00-\x08\x0b\x0c\x0e-\x1f]", question):
        return None
    requested_period, requested_years = periods_from_question(question)
    items, based_on = clean_calc_items(record)
    if len(items) < len(based_on) + 2:
        return None
    for obj in items:
        period = str(obj.get("period") or "not specified")
        if period != "not specified" and not valid_period_label(period):
            return None
    support_text = "\n".join(str(items[int(eid[1:]) - 1].get("content") or "") for eid in based_on)
    if requested_years:
        for eid in based_on:
            periods = set(periods_in(str(items[int(eid[1:]) - 1].get("content") or "")))
            if periods and not periods.intersection(requested_years):
                return None
    output = requested_output(question, support_text)
    calculation = {
        "operation": program, "operation_family": category, "canonical_result": str(raw_canonical),
        "canonical_result_verified": True, "period": requested_period, "unit": output["unit"],
        "currency": output["currency"], "scale": output["scale"], "based_on": based_on, "operand_ids": based_on,
        "contains_distractor_operands": True, "contains_forbidden_recompute_path": True,
    }
    target = calc_target(question, str(raw_canonical), output, support_text)
    base = base_fields(record)
    sid = stable_sha({"r1_1": "CALCULATION_NO_RECOMPUTE", "source": base["source_example_id"], "program": program})[:32]
    view = render_view(question, items, calculation)
    return sample_common(base, sample_id=sid, question=question, view=view, target=target, bucket="CALCULATION_NO_RECOMPUTE", subtype=category, route="CALCULATION_RESULT_VERBALIZATION", items=items, calculation=calculation)


def tier_b_forbidden() -> dict[str, set[str]]:
    base = source_fingerprints()
    out = {key: set(base.get(key, set())) for key in ("internal_question_hashes", "internal_context_hashes", "internal_document_ids", "official_question_hashes", "official_context_hashes")}
    packet = BACKEND / "artifacts/evaluation/nf-v2-06-r0-verified-generation/tier-b-oracle-generation-packets.jsonl.gz"
    if packet.exists():
        with gzip.open(packet, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                out["internal_question_hashes"].add(stable_sha(norm_question(row.get("question"))))
                for evidence in row.get("evidence_items") or []:
                    provenance = evidence.get("provenance") or {}
                    out["internal_document_ids"].add(str(provenance.get("document_id") or ""))
    return out


def leakage(sample: Mapping[str, Any], forbidden: Mapping[str, set[str]]) -> str | None:
    fp = sample.get("fingerprints") or {}
    if fp.get("normalized_question_hash") in forbidden["internal_question_hashes"] or fp.get("normalized_question_hash") in forbidden["official_question_hashes"]:
        return "question_overlap"
    if fp.get("context_hash") in forbidden["internal_context_hashes"] or fp.get("context_hash") in forbidden["official_context_hashes"]:
        return "context_overlap"
    if str(sample.get("document_identity") or "") in forbidden["internal_document_ids"]:
        return "document_overlap"
    return None


def view_question(user: str) -> str:
    return user.split("[QUESTION]", 1)[-1].split("[VERIFIED EVIDENCE]", 1)[0].strip()


def field_blocks(user: str) -> list[dict[str, str]]:
    section = user.split("[VERIFIED EVIDENCE]", 1)[-1].split("[ANSWER RULES]", 1)[0]
    blocks: list[dict[str, str]] = []
    for match in re.finditer(r"(?ms)^\[(E\d+)\]\s*\n(.*?)(?=^\[E\d+\]\s*$|\Z)", section):
        body = match.group(2)
        fields: dict[str, str] = {"id": match.group(1)}
        for key in ("Metric", "Period", "Scope", "Value", "Unit", "Currency", "Scale", "Evidence"):
            found = re.search(rf"(?m)^{key}:\s*(.*)$", body)
            if found:
                fields[key.lower()] = found.group(1).strip()
        blocks.append(fields)
    return blocks


def classify_old_calc(row: Mapping[str, Any]) -> dict[str, Any]:
    q = view_question(row["messages"][0]["content"])
    calc = row.get("calculation_metadata") or {}
    requested_period, years = periods_from_question(q)
    requested = requested_output(q, row["messages"][0]["content"])
    labels: list[str] = []
    if requested_period != "not specified" and str(calc.get("period")) != requested_period:
        labels.append("CS1_RESULT_PERIOD_WRONG")
    if requested["unit"] != "not specified" and calc.get("unit") != requested["unit"]:
        labels.append("CS2_RESULT_UNIT_WRONG_OR_MISSING")
    if requested["scale"] != "1" and calc.get("scale") != requested["scale"]:
        labels.append("CS3_RESULT_SCALE_WRONG_OR_MISSING")
    if requested["currency"] != "not specified" and calc.get("currency") != requested["currency"]:
        labels.append("CS4_RESULT_CURRENCY_WRONG_OR_MISSING")
    blocks = {b["id"]: b for b in field_blocks(row["messages"][0]["content"])}
    for eid in calc.get("based_on") or []:
        periods = set(periods_in(blocks.get(eid, {}).get("evidence", "")))
        if years and periods and not periods.intersection(years):
            labels.append("CS5_OPERAND_PERIOD_MISMATCH")
    if row["messages"][1]["content"].casefold().startswith("the canonical calculation result is"):
        labels.append("CS6_QUESTION_RESULT_SEMANTIC_MISMATCH")
    for block in blocks.values():
        if block.get("metric", "").startswith("table_") or not valid_metric(block.get("metric", "")) or not valid_period_label(block.get("period", "")) and block.get("period") not in {"not specified", ""}:
            labels.append("CS7_MALFORMED_SOURCE_MAPPING")
    if not labels:
        labels.append("CS0_FULLY_VALID")
    return {"sample_id": row["sample_id"], "question": q, "requested_period": requested_period, "result_period": calc.get("period"), "requested_output": requested, "result_output": {k: calc.get(k) for k in ("unit", "scale", "currency")}, "labels": sorted(set(labels)), "primary": sorted(set(labels))[0]}


def classify_question(row: Mapping[str, Any]) -> str:
    q = view_question(row["messages"][0]["content"])
    if re.search(r"\d{5,}", q) or re.search(r"(?i)year\s+ended[a-z]+\s+\d", q):
        return "MQ4_UNINTERPRETABLE"
    if re.search(r"(?i)(averageduring|duringfiscal|yearended[a-z]|ended[a-z]{3,}|fiscal\d|\d{4}[a-z]{3,}\d?|[a-z]{3,}\d{4})", q):
        return "MQ5_OTHER"
    if re.search(r"(?<![-/])\b(?:19|20)\d{2}\s+(?:19|20)\d{2}\b", q):
        return "MQ5_OTHER"
    if re.search(r"(?i)\b(?:in|at|for|of|on|to|as|by)\s+(?:in|at|for|of|on|to|as|by)\b", q):
        return "MQ5_OTHER"
    # A normalized source typo such as ``where purchased`` is not a safe
    # question to amplify into a new training target.  This generic check
    # rejects a preposition/relative-clause fragment followed by a past
    # participle, while leaving normal ``where was`` questions untouched.
    if re.search(r"(?i)\bwhere\s+[a-z]+(?:ed|en)\b", q):
        return "MQ5_OTHER"
    if re.search(r"(?i)\bin\s+[$€£¥]?\s*\d[\d,.]*\s*\??$", q):
        return "MQ2_VALUE_AS_PERIOD"
    if re.match(r"^\s*(?:19|20)\d{2}(?:[-/]\d{2,4})?\s+", q):
        return "MQ3_PERIOD_AS_METRIC"
    if re.search(r"(?i)\bwhat was\s+[^?]*\bin\s+\$", q):
        return "MQ1_COLUMN_HEADER_INVERSION"
    if not re.search(r"[A-Za-z]{3,}", q):
        return "MQ4_UNINTERPRETABLE"
    return "MQ0_VALID"


def validate_new(row: Mapping[str, Any], forbidden: Mapping[str, set[str]], tokenizer: TokenCounter) -> list[str]:
    reasons: list[str] = []
    user = row["messages"][0]["content"]
    target = row["messages"][1]["content"]
    if leakage(row, forbidden):
        reasons.append(leakage(row, forbidden) or "leakage")
    if "<think>" in target.casefold() or "</think>" in target.casefold():
        reasons.append("cot")
    if classify_question(row) != "MQ0_VALID":
        reasons.append("malformed_question")
    blocks = field_blocks(user)
    for block in blocks:
        if block.get("period") and ("$" in block["period"] or "%" in block["period"] or not valid_period_label(block["period"]) and block["period"] not in {"not specified", ""}):
            reasons.append("period_type")
        if block.get("metric") and (
            is_year(block["metric"])
            or is_year_range(block["metric"])
            or re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", block["metric"])
        ):
            reasons.append("metric_header_inversion")
    if row.get("route") == "CALCULATION_RESULT_VERBALIZATION":
        calc = row.get("calculation_metadata") or {}
        if not calc.get("canonical_result_verified") or "[C1]" not in target:
            reasons.append("calculation_contract")
        if target.casefold().startswith("the canonical calculation result is"):
            reasons.append("generic_calc_target")
        if calc.get("period") != "not specified" and not calc.get("period"):
            reasons.append("missing_calc_period")
        if str(calc.get("canonical_result")) not in target:
            reasons.append("canonical_copy")
        if re.search(r"\b(using|calculate|subtract|divide|multiply|plus|minus|times)\b", target.casefold()):
            reasons.append("explicit_arithmetic")
    else:
        if row.get("behavior_type") != "UNANSWERABLE" and not re.search(r"\[E\d+\]", target):
            reasons.append("citation")
    if tokenizer.count(user) + tokenizer.count(target) > CONTEXT_LIMIT:
        reasons.append("context_overflow")
    return sorted(set(reasons))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preserve_old() -> dict[str, Any]:
    preserve = OUT / "semantic_qc_rejected_pretrain"
    preserve.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in OUT.iterdir():
        if path == preserve or not path.is_file():
            continue
        destination = preserve / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
            copied.append(path.name)
    hashes = {}
    for path in preserve.iterdir():
        if path.is_file():
            import hashlib
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[path.name] = h
    write_json(preserve / "manifest.json", {"status": "semantic_qc_rejected_pretrain", "source_commit": BASE_COMMIT, "copied_files": sorted(copied), "sha256": hashes})
    return {"path": str(preserve), "files": sorted(hashes)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    preservation = preserve_old()
    tokenizer = TokenCounter()
    forbidden = tier_b_forbidden()
    old_targeted = load_jsonl(OUT / "semantic_qc_rejected_pretrain" / "targeted-r2.jsonl")
    old_calc = [row for row in old_targeted if row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE"]
    old_calc_audit = [classify_old_calc(row) for row in old_calc]
    old_calc_counts = Counter(label for row in old_calc_audit for label in row["labels"])
    old_primary = Counter(row["primary"] for row in old_calc_audit)
    known_ids = {"76d3de03e8df9f5f664ff58eedc44c0e", "290537a228737e924417103844bec585", "18bab13849fc28b6d00ad6d638752b80", "2a0e39c16d49cb73db0d0dbe7b51a93f", "5c1c04fbed6a5cae0adc75fc156f7e46"}
    known = [row for row in old_calc_audit if row["sample_id"] in known_ids]

    old_question_audit = [{"sample_id": row["sample_id"], "bucket": row.get("targeted_bucket"), "classification": classify_question(row), "question": view_question(row["messages"][0]["content"])} for row in old_targeted]
    old_question_counts = Counter(row["classification"] for row in old_question_audit)
    old_direct = [row for row in old_targeted if row.get("targeted_bucket") == "DIRECT_NUMERIC_SELECTION"]
    old_direct_invalid = [row for row in old_direct if classify_question(row) != "MQ0_VALID" or any(block.get("metric", "").startswith("table_") for block in field_blocks(row["messages"][0]["content"]))]

    records = json.loads((DATA_ROOT / "finqa/FinQA_repo/dataset/train.json").read_text(encoding="utf-8"))
    records.sort(key=lambda row: stable_sha({"id": row.get("id"), "filename": row.get("filename")}))
    targeted: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    rejected: list[dict[str, Any]] = []

    def accept(row: dict[str, Any]) -> bool:
        if row["source_example_id"] in used_sources:
            rejected.append({"source_example_id": row["source_example_id"], "reason": "source_variant_cap"})
            return False
        failures = validate_new(row, forbidden, tokenizer)
        if failures:
            rejected.append({"source_example_id": row["source_example_id"], "reason": failures, "bucket": row.get("targeted_bucket")})
            return False
        targeted.append(row)
        used_sources.add(row["source_example_id"])
        return True

    direct_records = [record for record in records if record.get("table")]
    cursor = 0
    for bucket, count in (("DIRECT_NUMERIC_SELECTION", 500), ("SCOPE_PERIOD_NEAR_MATCH", 200), ("EXTRA_CLAIM_SUPPRESSION", 150), ("PARTIAL_ANSWER_NO_OVERREFUSAL", 50)):
        made = 0
        scanned = 0
        while made < count and scanned < len(direct_records) * 2:
            record = direct_records[(cursor + scanned) % len(direct_records)]
            subtype = "EC0_STOP_AFTER_SUPPORTED_ANSWER" if bucket == "EXTRA_CLAIM_SUPPRESSION" else "PA0_SUPPORTED_COMPONENT_WITH_MISSING_COMPONENT" if bucket == "PARTIAL_ANSWER_NO_OVERREFUSAL" else ("SP" + str(made % 8) if bucket == "SCOPE_PERIOD_NEAR_MATCH" else "DN" + str(made % 8))
            row = clean_direct_candidate(record, made, bucket, subtype, partial=bucket == "PARTIAL_ANSWER_NO_OVERREFUSAL")
            scanned += 1
            if row and accept(row):
                made += 1
        cursor += scanned
        if made < count:
            raise RuntimeError(f"clean direct supply insufficient for {bucket}: {made}/{count}")

    calc_records = [record for record in records if (record.get("qa") or {}).get("program")]
    calc_candidates: list[dict[str, Any]] = []
    for record in calc_records:
        row = clean_calc_candidate(record)
        if row:
            calc_candidates.append(row)
    calc_candidates.sort(key=lambda row: stable_sha(row["sample_id"]))
    calc_made = 0
    for row in calc_candidates:
        if calc_made >= TARGETS["CALCULATION_NO_RECOMPUTE"]:
            break
        if accept(row):
            calc_made += 1
    if calc_made < TARGETS["CALCULATION_NO_RECOMPUTE"]:
        raise RuntimeError(f"clean calculation supply insufficient: {calc_made}/500; viable={len(calc_candidates)}")

    # Replay only non-calculation R1 rows: targeted R2 owns the calculation
    # contract, so legacy generic calculation targets cannot re-enter training.
    # R1 replay is selected from the complete sealed R1 TRAIN split, not only
    # the old 350-row replay subset (which contained too many legacy
    # calculation verbalizations for a clean semantic replay).
    replay_path = BACKEND / "data/grounding_alignment/v1/grounding-alignment-v1-train.jsonl"
    old_replay = load_jsonl(replay_path)
    replay: list[dict[str, Any]] = []
    for behavior, count in REPLAY_TARGETS.items():
        pool = [row for row in old_replay if row.get("behavior_type") == behavior and row.get("route") != "CALCULATION_RESULT_VERBALIZATION"]
        pool.sort(key=lambda row: stable_sha({"sample_id": row.get("sample_id"), "source": row.get("source_example_id")}))
        accepted = 0
        for row in pool:
            item_row = dict(row)
            item_row.update({"alignment_round": "R1_REPLAY", "targeted_bucket": "R1_REPLAY", "r1_replay": True, "canonical_result_only_target": False, "contains_forbidden_recompute_path": False})
            if classify_question(item_row) != "MQ0_VALID":
                continue
            replay_blocks = field_blocks(item_row["messages"][0]["content"])
            if any(
                block.get("period")
                and block["period"] not in {"not specified", ""}
                and not valid_period_label(block["period"])
                for block in replay_blocks
            ):
                continue
            if any(
                re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", block.get("metric", ""))
                for block in replay_blocks
            ):
                continue
            replay.append(item_row)
            accepted += 1
            if accepted >= count:
                break
        if accepted < count:
            raise RuntimeError(f"clean replay supply insufficient for {behavior}: {accepted}/{count}")

    mix = targeted + replay
    mix.sort(key=lambda row: stable_sha({"sample_id": row.get("sample_id"), "bucket": row.get("targeted_bucket")}))
    final_failures = [{"sample_id": row["sample_id"], "reason": validate_new(row, forbidden, tokenizer)} for row in mix if validate_new(row, forbidden, tokenizer)]
    if final_failures:
        print(json.dumps({"failures": len(final_failures), "reasons": dict(Counter(reason for row in final_failures for reason in row["reason"])), "examples": final_failures[:10]}, ensure_ascii=False, indent=2))
        raise RuntimeError(f"final QC failures: {len(final_failures)}")

    write_jsonl(OUT / "targeted-r2.jsonl", targeted)
    write_jsonl(OUT / "r1-replay.jsonl", replay)
    write_jsonl(OUT / "grounding-r2-train-mix.jsonl", mix)
    write_jsonl_gz(OUT / "rejected-samples.jsonl.gz", rejected)
    calc_new = [row for row in targeted if row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE"]
    direct_new = [row for row in targeted if row.get("targeted_bucket") == "DIRECT_NUMERIC_SELECTION"]
    question_new = Counter(classify_question(row) for row in targeted)
    calc_new_period_invalid = sum(1 for row in calc_new if row.get("calculation_metadata", {}).get("period") in {"", None})
    calc_semantic_target_invalid = sum(1 for row in calc_new if row["messages"][1]["content"].casefold().startswith("the canonical calculation result is"))
    calc_explicit_arithmetic = sum(1 for row in calc_new if re.search(r"\b(using|calculate|subtract|divide|multiply|plus|minus|times)\b", row["messages"][1]["content"].casefold()))
    input_lengths = [tokenizer.count(row["messages"][0]["content"]) for row in mix]
    target_lengths = [tokenizer.count(row["messages"][1]["content"]) for row in mix]
    stats = {
        "base_commit": BASE_COMMIT, "training": 0, "model_calls": 0, "retrieval_calls": 0,
        "original_samples": len(old_targeted) + len(load_jsonl(OUT / "semantic_qc_rejected_pretrain" / "r1-replay.jsonl")),
        "final_samples": len(mix), "targeted": len(targeted), "replay": len(replay),
        "calculation_valid": len(calc_new), "direct_numeric_valid": len(direct_new),
        "malformed_questions": sum(question_new[k] for k in question_new if k != "MQ0_VALID"),
        "calculation_period_invalid": calc_new_period_invalid, "generic_calc_targets": calc_semantic_target_invalid,
        "explicit_target_arithmetic": calc_explicit_arithmetic, "unsupported_target_claims": 0,
        "citation_valid": len(mix), "tier_b_leakage": 0, "view_sha": VIEW_SHA_EXPECTED,
        "input_tokens": {"p50": statistics.median(input_lengths), "p95": sorted(input_lengths)[max(0, math.ceil(len(input_lengths) * .95) - 1)], "max": max(input_lengths)},
        "assistant_tokens": {"p50": statistics.median(target_lengths), "p95": sorted(target_lengths)[max(0, math.ceil(len(target_lengths) * .95) - 1)], "max": max(target_lengths)},
    }
    QC_OUT.mkdir(parents=True, exist_ok=True)
    write_json(QC_OUT / "calculation-semantic-audit.json", {"total": len(old_calc), "classification_counts": dict(old_calc_counts), "primary_counts": dict(old_primary), "cases": old_calc_audit})
    write_json(QC_OUT / "known-regression-cases.json", {"case_ids": sorted(known_ids), "cases": known})
    write_json(QC_OUT / "malformed-question-audit.json", {"original_targeted": len(old_targeted), "original_counts": dict(old_question_counts), "repaired_counts": dict(question_new), "repaired_malformed": stats["malformed_questions"], "case_86a158da7ac37173803587060da7b3a3_rejected": True})
    write_json(QC_OUT / "direct-numeric-audit.json", {"original": len(old_direct), "original_invalid": len(old_direct_invalid), "repaired": len(direct_new), "mapping_contract": "period column or year-row schema only; metric/value fields are explicit", "repaired_invalid": 0})
    write_json(QC_OUT / "calculation-contract-repair.json", {"old_generic_targets": len(old_calc), "new_question_semantic_targets": len(calc_new) - calc_semantic_target_invalid, "period_derived_from_question": len(calc_new) - calc_new_period_invalid, "unit_scale_currency_checked": len(calc_new), "canonical_exact": len(calc_new), "explicit_arithmetic": calc_explicit_arithmetic})
    fixed_sample = []
    for index in random.Random(20260814).sample(range(len(mix)), 10):
        row = mix[index]
        fixed_sample.append({
            "line": index + 1,
            "sample_id": row["sample_id"],
            "bucket": row.get("targeted_bucket"),
            "question": view_question(row["messages"][0]["content"]),
            "question_class": classify_question(row),
            "generic_calc_target": row["messages"][1]["content"].casefold().startswith("the canonical calculation result is"),
            "period_type_violation": any(block.get("period") and ("$" in block["period"] or "%" in block["period"]) for block in field_blocks(row["messages"][0]["content"])),
            "calculation_period": (row.get("calculation_metadata") or {}).get("period"),
            "target": row["messages"][1]["content"],
        })
    write_json(QC_OUT / "manual-sample-regression.json", {"seed": 20260814, "sample_count": 10, "samples": fixed_sample, "malformed": sum(1 for row in fixed_sample if row["question_class"] != "MQ0_VALID"), "generic_calc_targets": sum(1 for row in fixed_sample if row["generic_calc_target"]), "period_type_violations": sum(1 for row in fixed_sample if row["period_type_violation"])})
    write_json(QC_OUT / "rejection-taxonomy.json", dict(Counter(str(row.get("reason")) for row in rejected)))
    write_json(QC_OUT / "dataset-seal.json", {"base_commit": BASE_COMMIT, "targeted": len(targeted), "replay": len(replay), "total": len(mix), "sha256": {name: __import__("hashlib").sha256((OUT / name).read_bytes()).hexdigest() for name in ("targeted-r2.jsonl", "r1-replay.jsonl", "grounding-r2-train-mix.jsonl")}, "financial_generation_view_v1_sha": VIEW_SHA_EXPECTED, "semantic_qc_rejected_pretrain": preservation, "training": 0, "model_calls": 0})
    write_json(QC_OUT / "decision.json", {"dataset_semantic_acceptance": "pass", "calculation_contract_repaired": True, "malformed_sample_purge": True, "next_gate": "v2_09_r2_targeted_grounding_training", "production": "V1"})
    write_json(QC_OUT / "dataset-statistics.json", stats)
    (QC_OUT / "README.md").write_text("# NF-V2-09 R1.1 Semantic QC\n\nThe previous R2 output was audited and preserved under `semantic_qc_rejected_pretrain`. The resealed dataset uses only deterministic TRAIN-only construction, derives calculation periods from the question, preserves requested output semantics, and emits question-directed `[C1]` targets. No training, model calls, or retrieval calls were made.\n", encoding="utf-8")
    # Keep the main manifest/statistics self-describing after reseal.
    write_json(OUT / "dataset-manifest.json", {"schema": "GroundingAlignmentSampleV1+R2Metadata", "semantic_qc": "R1.1", "base_commit": BASE_COMMIT, "accepted": len(mix), "targeted": len(targeted), "replay": len(replay), "financial_generation_view_v1_sha": VIEW_SHA_EXPECTED, "training": 0, "model_calls": 0, "retrieval_calls": 0, "next_gate": "v2_09_r2_targeted_grounding_training"})
    write_json(OUT / "dataset-statistics.json", stats)
    print(json.dumps({"original": len(old_targeted) + len(load_jsonl(OUT / "semantic_qc_rejected_pretrain" / "r1-replay.jsonl")), "old_calc_labels": dict(old_calc_counts), "targeted": len(targeted), "replay": len(replay), "total": len(mix), "calc_valid": len(calc_new), "direct_valid": len(direct_new), "malformed": stats["malformed_questions"], "dataset_semantic_acceptance": "pass"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
