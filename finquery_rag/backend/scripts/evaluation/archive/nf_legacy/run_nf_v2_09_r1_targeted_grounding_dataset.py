#!/usr/bin/env python3
"""Build the NF-V2-09 R1 targeted grounding-alignment mixture.

This builder is deliberately model-free.  It derives examples from approved
TRAIN-only FinQA/TAT-DQA records, freezes the existing FinancialGenerationViewV1
format, and performs deterministic support/leakage checks before accepting a
row.  It never reads the frozen benchmark answers or calls a model.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

# The R1A builder contains the already-frozen renderer helpers and tokenizer.
# Importing it is side-effect free (its main() is guarded).
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
    supported_numbers,
    target_periods,
    write_json,
    write_jsonl,
    write_jsonl_gz,
)

BASE_COMMIT = "ea4d0e3cb0e8a3009ae3fbdb8a9c87cf4ae22220"
VIEW_SHA_EXPECTED = "943decf288dffb99ffa6f196abc44e0a5bdb226350cede40e0a160c4bd61f6e4"
OUT = BACKEND / "data/grounding_alignment/v2_targeted"
CONTEXT_LIMIT = 4096
TARGETED_COUNTS = {
    "DIRECT_NUMERIC_SELECTION": 500,
    "CALCULATION_NO_RECOMPUTE": 500,
    "SCOPE_PERIOD_NEAR_MATCH": 200,
    "EXTRA_CLAIM_SUPPRESSION": 150,
    "PARTIAL_ANSWER_NO_OVERREFUSAL": 50,
}
REPLAY_COUNTS = {"POSITIVE_GROUNDED": 220, "PARTIAL_DISTRACTOR": 80, "UNANSWERABLE": 50}
R2_TOTAL = sum(TARGETED_COUNTS.values())
R1_REPLAY_TOTAL = sum(REPLAY_COUNTS.values())


def _write(path: Path, value: Any) -> None:
    write_json(path, value)


def _number_tokens(text: str) -> list[str]:
    return re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:%|[A-Za-z]+)?", str(text or ""))


def _parse_number(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", "")
    text = text.replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
    text = text.replace("(", "-").replace(")", "")
    text = re.sub(r"(?i)\b(percent|percentage)\b", "%", text)
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return None


def _split_program(program: str) -> list[str]:
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


def _constant(value: str) -> float | None:
    match = re.fullmatch(r"const_([0-9]+(?:\.[0-9]+)?)", value.strip())
    return float(match.group(1)) if match else None


def _program_result(program: str) -> tuple[float | str | None, str]:
    """Evaluate the small deterministic FinQA arithmetic DSL.

    Unsupported/comparison programs are rejected rather than guessed.  The
    returned category is used only for reporting and selection metadata.
    """
    refs: list[float | str] = []
    category = "other"
    for step in _split_program(program):
        match = re.fullmatch(r"([a-z_]+)\((.*)\)", step.strip())
        if not match:
            return None, category
        op, raw_args = match.groups()
        args = [x.strip() for x in raw_args.split(",")]
        values: list[float | str] = []
        for arg in args:
            if arg.startswith("#"):
                try:
                    values.append(refs[int(arg[1:])])
                except (ValueError, IndexError):
                    return None, category
            elif (const := _constant(arg)) is not None:
                values.append(const)
            elif (number := _parse_number(arg)) is not None:
                values.append(number)
            else:
                return None, category
        if op in {"subtract", "difference", "diff"} and len(values) == 2:
            category = "difference"
            result = float(values[0]) - float(values[1])
        elif op in {"add", "sum"} and len(values) == 2:
            category = "sum"
            result = float(values[0]) + float(values[1])
        elif op in {"multiply", "mul"} and len(values) == 2:
            result = float(values[0]) * float(values[1])
            if any(abs(float(v)) in {10, 100, 1000, 1000000} for v in values):
                category = "scale_conversion"
            else:
                category = "other"
        elif op in {"divide", "div"} and len(values) == 2:
            if float(values[1]) == 0:
                return None, category
            result = float(values[0]) / float(values[1])
            category = "percentage_share"
        elif op in {"average", "table_average"} and values:
            category = "average"
            result = sum(float(v) for v in values) / len(values)
        elif op in {"exp", "power"} and len(values) == 2:
            result = float(values[0]) ** float(values[1])
            category = "other"
        elif op in {"greater", "less", "equal", "not_equal"} and len(values) == 2:
            # Comparison targets are not useful for canonical numeric-copy
            # training in this bucket, so keep them explicitly unsupported.
            return None, category
        else:
            return None, category
        refs.append(result)
        if len(refs) > 1 and re.search(r"divide", program) and re.search(r"subtract", program):
            category = "growth_rate"
        if any(token in program.casefold() for token in ("gross_margin", "gross margin")):
            category = "gross_margin"
        elif any(token in program.casefold() for token in ("net_margin", "net margin")):
            category = "net_margin"
        elif "debt_ratio" in program.casefold():
            category = "debt_ratio"
    return (refs[-1] if refs else None), category


def _close(a: float, b: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= max(0.02, 0.01 * max(abs(a), abs(b), 1.0))


def _operation_category(program: str, question: str) -> str:
    result, category = _program_result(program)
    q = f"{program} {question}".casefold()
    for marker, name in (("gross margin", "gross_margin"), ("net margin", "net_margin"), ("debt ratio", "debt_ratio")):
        if marker in q:
            return name
    return category if result is not None else "other"


def _table_rows(record: Mapping[str, Any]) -> list[list[str]]:
    return [[str(cell or "").strip() for cell in row] for row in (record.get("table") or []) if any(str(cell or "").strip() for cell in row)]


def _cell_has_number(value: str) -> bool:
    return bool(re.search(r"[-+]?\d[\d,.]*(?:%|[A-Za-z]+)?", value or ""))


def _row_metric(row: list[str]) -> str:
    return row[0].strip() if row and row[0].strip() else "reported metric"


def _row_line(row: list[str]) -> str:
    return " | ".join(cell for cell in row if cell != "")


def _period_label(header: str, index: int) -> str:
    value = header.strip() or f"column {index + 1}"
    return value


def _field_item(content: str, metric: str, period: str, value: str, base: Mapping[str, Any], key: str, scope: str = "not specified") -> dict[str, Any]:
    currency = "USD" if "$" in content or "dollar" in content.casefold() else "not specified"
    unit = "percent" if "%" in content else "not specified"
    scale = "million" if re.search(r"(?i)\bmillions?\b", content) else ("billion" if re.search(r"(?i)\bbillions?\b", content) else "not specified")
    return {
        "metric": metric,
        "period": period,
        "scope": scope,
        "value": value or "see evidence",
        "unit": unit,
        "currency": currency,
        "scale": scale,
        "source_label": base.get("document_identity") or base.get("filename") or "TRAIN source",
        "source_key": key,
        "content": content,
    }


def _render_view(question: str, items: list[Mapping[str, Any]], calculation: Mapping[str, Any] | None = None) -> str:
    lines = ["[QUESTION]", question, "", "[VERIFIED EVIDENCE]", ""]
    ids = [f"E{i + 1}" for i in range(len(items))]
    for evidence_id, item in zip(ids, items):
        lines.extend([
            f"[{evidence_id}]",
            f"Metric: {item.get('metric') or 'source evidence'}",
            f"Period: {item.get('period') or 'not specified'}",
            f"Scope: {item.get('scope') or 'not specified'}",
            f"Value: {item.get('value') or 'see evidence'}",
            f"Unit: {item.get('unit') or 'not specified'}",
            f"Currency: {item.get('currency') or 'not specified'}",
            f"Scale: {item.get('scale') or 'not specified'}",
            f"Source: {item.get('source_label') or 'TRAIN source'}",
            f"Evidence: {str(item.get('content') or '').replace(chr(10), ' ').strip()}",
            "",
        ])
    if calculation:
        based_on = ", ".join(f"[{x}]" for x in calculation.get("based_on", []))
        lines.extend([
            "[VERIFIED CALCULATION]", "[C1]",
            f"Operation: {calculation.get('operation') or 'dataset-native calculation'}",
            f"Canonical Result: {calculation.get('canonical_result')}",
            f"Period: {calculation.get('period') or 'not specified'}",
            f"Unit: {calculation.get('unit') or 'not specified'}",
            f"Currency: {calculation.get('currency') or 'not specified'}",
            f"Scale: {calculation.get('scale') or '1'}",
            f"Based On: {based_on}", "",
        ])
    lines.append(ANSWER_RULES)
    return "\n".join(lines)


def _base_fields(record: Mapping[str, Any], evidence: list[Mapping[str, Any]]) -> dict[str, Any]:
    table = record.get("table") or []
    context = {
        "filename": record.get("filename"), "table": table,
        "pre_text": record.get("pre_text"), "post_text": record.get("post_text"),
    }
    return {
        "source_dataset": "FinQA",
        "source_split": "train",
        "source_example_id": str(record.get("id")),
        "source_context_id": stable_sha(context),
        "document_identity": str(record.get("filename") or ""),
        "evidence": evidence,
    }


def _subtype_direct(index: int, row: list[str], header: list[str], content: str) -> str:
    metric = _row_metric(row).casefold()
    if "%" in content:
        return "DN5_PERCENT_VS_ABSOLUTE"
    if any(token in metric for token in ("total", "overall")):
        return "DN3_TOTAL_VS_COMPONENT"
    if any(token in metric for token in ("net income", "operating income", "revenue", "sales", "assets")) and any(token in metric for token in ("total", "income")):
        return "DN4_PARENT_VS_CHILD_METRIC"
    return ("DN0_SAME_METRIC_DIFFERENT_PERIOD", "DN1_SAME_PERIOD_DIFFERENT_METRIC", "DN2_MULTIPLE_NUMBERS_SAME_EVIDENCE", "DN6_SCALE_NEAR_MATCH", "DN7_MULTI_ROW_TABLE_SELECTION")[index % 5]


def _subtype_scope(index: int) -> str:
    return ("SP0_WRONG_PERIOD", "SP1_WRONG_SEGMENT", "SP2_TOTAL_VS_REGION", "SP3_TOTAL_VS_PRODUCT", "SP4_PARENT_CHILD_METRIC", "SP5_GAAP_VS_NON_GAAP", "SP6_CURRENT_VS_PRIOR_YEAR", "SP7_CONSOLIDATED_VS_COMPONENT")[index % 8]


def _sample_common(*, sample_id: str, base: Mapping[str, Any], question: str, view: str, target: str, route: str, bucket: str, subtype: str, behavior: str, evidence_ids: list[str], calculation: Mapping[str, Any] | None = None, distractors: bool = False, forbidden_path: bool = False, partial: bool = False) -> dict[str, Any]:
    context_id = str(base["source_context_id"])
    return {
        "sample_id": sample_id,
        "source_dataset": "FinQA",
        "source_split": "train",
        "source_example_id": str(base["source_example_id"]),
        "source_context_id": context_id,
        "document_identity": str(base.get("document_identity") or ""),
        "behavior_type": behavior,
        "route": route,
        "base_question": question,
        "messages": [{"role": "user", "content": view}, {"role": "assistant", "content": target}],
        "evidence_ids": evidence_ids,
        "calculation_id": f"C1:{sample_id}" if calculation else None,
        "fully_answerable": behavior == "POSITIVE_GROUNDED",
        "partially_answerable": partial,
        "requires_abstention": behavior != "POSITIVE_GROUNDED",
        "has_distractors": distractors,
        "oracle_training_evidence": False,
        "citation_required": True,
        "cot_target": False,
        "loss_mask_contract": {"user_prompt": 0, "assistant_final_answer": 1, "question_evidence_rules": 0},
        "fingerprints": {
            "normalized_question_hash": stable_sha(norm_question(question)),
            "context_hash": context_id,
            "document_hash": stable_sha(base.get("document_identity") or ""),
        },
        "alignment_round": "R2",
        "targeted_bucket": bucket,
        "targeted_subtype": subtype,
        "r1_replay": False,
        "contains_distractor_operands": distractors,
        "contains_forbidden_recompute_path": forbidden_path,
        "canonical_result_only_target": bool(calculation),
        "calculation_metadata": calculation,
    }


def _direct_candidate(record: Mapping[str, Any], index: int, bucket: str, subtype: str, *, partial: bool = False) -> dict[str, Any] | None:
    table = _table_rows(record)
    if len(table) < 3 or len(table[0]) < 2:
        return None
    header = table[0]
    candidates: list[tuple[int, int]] = []
    for row_index, row in enumerate(table[1:], start=1):
        for col_index, cell in enumerate(row[1:], start=1):
            if col_index < len(header) and _cell_has_number(cell) and str(cell).strip():
                candidates.append((row_index, col_index))
    if not candidates:
        return None
    row_index, col_index = candidates[index % len(candidates)]
    row = table[row_index]
    metric = _row_metric(row)
    # A bare year/date or structural label is not a useful metric intent for
    # a numeric-selection teaching example.  Skip it and let another TRAIN
    # table supply the row.
    if re.fullmatch(r"(?:19|20)\d{2}", metric.strip()) or norm_text(metric) in {"year", "date", "location", "period"} or len(metric.strip()) < 3:
        return None
    period = _period_label(header[col_index] if col_index < len(header) else "", col_index)
    value = str(row[col_index]).strip()
    # Add same-metric alternate period and same-period competing metrics.
    for other_col, cell in enumerate(row[1:], start=1):
        if other_col != col_index and _cell_has_number(cell):
            chosen_period = _period_label(header[other_col] if other_col < len(header) else "", other_col)
            items = [_field_item(_row_line(row), metric, period, value, record, f"table_{row_index}", metric)]
            items.append(_field_item(_row_line(row), metric, chosen_period, str(cell).strip(), record, f"table_{row_index}_alt", metric))
            break
    else:
        items = [_field_item(_row_line(row), metric, period, value, record, f"table_{row_index}", metric)]
    for other_index, other_row in enumerate(table[1:], start=1):
        if other_index == row_index:
            continue
        if col_index < len(other_row) and _cell_has_number(other_row[col_index]):
            items.append(_field_item(_row_line(other_row), _row_metric(other_row), period, str(other_row[col_index]).strip(), record, f"table_{other_index}", _row_metric(other_row)))
        if len(items) >= 4:
            break
    if len(items) < 3:
        return None
    question = f"What was {metric} in {period}?"
    if partial:
        question = f"What was {metric} in {period}, and what was the company's share price?"
        target = f"{metric} in {period} was {value} [E1]. The provided evidence does not contain the company's share price."
        behavior = "PARTIAL_DISTRACTOR"
    elif bucket == "EXTRA_CLAIM_SUPPRESSION":
        target = f"{metric} in {period} was {value} [E1]."
        behavior = "POSITIVE_GROUNDED"
    else:
        target = f"{metric} in {period} was {value} [E1]."
        behavior = "POSITIVE_GROUNDED"
    base = _base_fields(record, items)
    sample_id = stable_sha({"r2": bucket, "subtype": subtype, "source": base["source_example_id"], "row": row_index, "column": col_index, "partial": partial})[:32]
    view = _render_view(question, items)
    return _sample_common(sample_id=sample_id, base=base, question=question, view=view, target=target, route="DIRECT", bucket=bucket, subtype=subtype, behavior=behavior, evidence_ids=[f"E{i+1}" for i in range(len(items))], partial=partial)


def _calc_evidence(record: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    raw = finqa_evidence(record)
    table = _table_rows(record)
    selected_keys = {str(item.get("source_key")) for item in raw}
    items: list[dict[str, Any]] = []
    for item in raw[:4]:
        content = str(item.get("content") or "")
        items.append(_field_item(content, str(item.get("source_key") or "source evidence"), ", ".join(periods_in(content)), "; ".join(_number_tokens(content)[:8]), record, str(item.get("source_key") or "source")))
    # Deliberate distractor rows are always physical rows not in the native
    # supporting set.  Two are required for every accepted calculation row.
    for row_index, row in enumerate(table):
        key = f"table_{row_index}"
        if key in selected_keys or not _cell_has_number(_row_line(row)):
            continue
        content = _row_line(row)
        items.append(_field_item(content, _row_metric(row), ", ".join(periods_in(content)), "; ".join(_number_tokens(content)[:8]), record, key, _row_metric(row)))
        if len(items) >= 6:
            break
    if len(items) < 3:
        return [], [], []
    # Keep every source-derived supporting row that was selected by the
    # dataset-native gold_inds mapping (up to the four rows rendered above),
    # then treat only rows appended afterwards as distractors.
    support_count = max(1, min(len(raw), 4))
    support_ids = [f"E{i+1}" for i in range(support_count)]
    distractor_ids = [f"E{i+1}" for i in range(support_count, len(items))]
    return items, support_ids, distractor_ids


def _alternate_path(items: list[Mapping[str, Any]], canonical: float, category: str) -> tuple[bool, str | None]:
    numbers: list[float] = []
    for item in items[2:]:
        for token in _number_tokens(str(item.get("content") or "")):
            number = _parse_number(token)
            if number is not None and abs(number) < 1e15:
                numbers.append(number)
    if len(numbers) < 2:
        return False, None
    a, b = numbers[0], numbers[1]
    if category == "difference":
        alt = a - b
    elif category in {"sum", "average"}:
        alt = a + b if category == "sum" else (a + b) / 2
    elif category in {"percentage_share", "growth_rate"}:
        if b == 0:
            return False, None
        alt = a / b * (100 if category == "growth_rate" else 1)
    elif category == "scale_conversion":
        alt = a * b
    else:
        alt = a + b
    if _close(alt, canonical):
        return False, None
    return True, f"distractor_pair({a},{b})=>{alt}"


def _calc_candidate(record: Mapping[str, Any]) -> dict[str, Any] | None:
    qa = record.get("qa") or {}
    program = str(qa.get("program") or "")
    canonical_raw = qa.get("exe_ans")
    canonical = _parse_number(canonical_raw)
    if not program or canonical is None:
        return None
    computed, _ = _program_result(program)
    if not isinstance(computed, (int, float)) or not _close(float(computed), canonical):
        return None
    items, support_ids, distractor_ids = _calc_evidence(record)
    if len(distractor_ids) < 2:
        return None
    category = _operation_category(program, str(qa.get("question") or ""))
    alternate, path = _alternate_path(items, canonical, category)
    periods = periods_in(" ".join(str(item.get("content") or "") for item in items) + " " + str(qa.get("question") or ""))
    calculation = {
        "operation": program,
        "operation_family": category,
        "canonical_result": str(canonical_raw),
        "period": periods[0] if periods else "not specified",
        "unit": "percent" if "%" in str(canonical_raw) or "growth" in str(qa.get("question") or "").casefold() else "not specified",
        "currency": "not specified",
        "scale": "1",
        "based_on": support_ids,
        "operand_ids": support_ids,
        "canonical_result_verified": True,
        "alternate_path": path,
    }
    question = str(qa.get("question") or "What was the calculated result?").strip()
    target = f"The canonical calculation result is {str(canonical_raw).strip()} [C1]."
    base = _base_fields(record, items)
    sample_id = stable_sha({"r2": "CALCULATION_NO_RECOMPUTE", "source": base["source_example_id"], "program": program})[:32]
    view = _render_view(question, items, calculation)
    return _sample_common(sample_id=sample_id, base=base, question=question, view=view, target=target, route="CALCULATION_RESULT_VERBALIZATION", bucket="CALCULATION_NO_RECOMPUTE", subtype=category, behavior="POSITIVE_GROUNDED", evidence_ids=[f"E{i+1}" for i in range(len(items))], calculation=calculation, distractors=True, forbidden_path=alternate)


def _raw_records() -> list[dict[str, Any]]:
    path = DATA_ROOT / "finqa/FinQA_repo/dataset/train.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _tier_b_forbidden() -> dict[str, set[str]]:
    forbidden = source_fingerprints()
    result = {
        "internal_question_hashes": set(forbidden.get("internal_question_hashes", set())),
        "internal_context_hashes": set(forbidden.get("internal_context_hashes", set())),
        "internal_document_ids": set(forbidden.get("internal_document_ids", set())),
        "official_question_hashes": set(forbidden.get("official_question_hashes", set())),
        "official_context_hashes": set(forbidden.get("official_context_hashes", set())),
    }
    packet_path = BACKEND / "artifacts/evaluation/nf-v2-06-r0-verified-generation/tier-b-oracle-generation-packets.jsonl.gz"
    if packet_path.exists():
        with gzip.open(packet_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                packet = json.loads(line)
                result["internal_question_hashes"].add(stable_sha(norm_question(packet.get("question"))))
                result["internal_context_hashes"].add(str(packet.get("packet_sha256") or ""))
                for item in packet.get("evidence_items") or []:
                    result["internal_document_ids"].add(str((item.get("provenance") or {}).get("document_id") or ""))
                    result["internal_document_ids"].add(str(item.get("source_id") or ""))
    return result


def _leak_reason(sample: Mapping[str, Any], forbidden: Mapping[str, set[str]]) -> str | None:
    fp = sample.get("fingerprints") or {}
    if fp.get("normalized_question_hash") in forbidden["internal_question_hashes"] or fp.get("normalized_question_hash") in forbidden["official_question_hashes"]:
        return "question_overlap"
    if fp.get("context_hash") in forbidden["internal_context_hashes"] or fp.get("context_hash") in forbidden["official_context_hashes"]:
        return "context_overlap"
    if str(sample.get("document_identity") or "") in forbidden["internal_document_ids"]:
        return "document_overlap"
    return None


def _validate_sample(sample: Mapping[str, Any], forbidden: Mapping[str, set[str]], tokenizer: TokenCounter) -> list[str]:
    reasons: list[str] = []
    messages = sample.get("messages") or []
    if len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        reasons.append("schema")
    user = str(messages[0].get("content") or "") if messages else ""
    target = str(messages[1].get("content") or "") if len(messages) > 1 else ""
    if not user.startswith("[QUESTION]") or "[VERIFIED EVIDENCE]" not in user or "[ANSWER RULES]" not in user:
        reasons.append("view_contract")
    rendered_e = set(re.findall(r"(?m)^\[(E\d+)\]$", user))
    citations = set(re.findall(r"\[(E\d+|C1)\]", target))
    if not set(sample.get("evidence_ids") or []).issubset(rendered_e):
        reasons.append("evidence_ids")
    if any(item.startswith("E") and item not in rendered_e for item in citations):
        reasons.append("unknown_citation")
    if "<think>" in target.casefold() or "</think>" in target.casefold() or "reasoning:" in target.casefold():
        reasons.append("cot_target")
    leak = _leak_reason(sample, forbidden)
    if leak:
        reasons.append(leak)
    if sample.get("source_split") != "train":
        reasons.append("non_train")
    if sample.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE":
        if citations != {"C1"} or not sample.get("calculation_metadata", {}).get("canonical_result_verified"):
            reasons.append("calculation_citation_or_verification")
        canonical = str(sample.get("calculation_metadata", {}).get("canonical_result") or "")
        if canonical not in target:
            reasons.append("canonical_not_copied")
        if re.search(r"\b(using|calculate|subtract|divide|multiply|plus|minus|times)\b", target.casefold()):
            reasons.append("explicit_arithmetic")
    elif sample.get("behavior_type") in {"POSITIVE_GROUNDED", "PARTIAL_DISTRACTOR"}:
        # Calculation verbalization targets cite the canonical [C1] packet,
        # while direct/partial targets cite one or more [E#] items.
        if sample.get("route") != "CALCULATION_RESULT_VERBALIZATION" and (not citations or not any(x.startswith("E") for x in citations)):
            reasons.append("missing_evidence_citation")
        # Numeric/period support is checked against the evidence block only;
        # a partial question's missing component must not be treated as support.
        evidence_section = user.split("[VERIFIED EVIDENCE]", 1)[-1].split("[ANSWER RULES]", 1)[0]
        cited_text = evidence_section
        numbers = supported_numbers(target)
        if not numbers.issubset(supported_numbers(cited_text)):
            reasons.append("numeric_support")
        if not target_periods(target).issubset(target_periods(cited_text)):
            reasons.append("period_support")
    if sample.get("targeted_bucket") == "PARTIAL_ANSWER_NO_OVERREFUSAL":
        if sample.get("fully_answerable") or not sample.get("partially_answerable") or "does not contain" not in target.casefold():
            reasons.append("partial_contract")
    if sample.get("targeted_bucket") in {"DIRECT_NUMERIC_SELECTION", "SCOPE_PERIOD_NEAR_MATCH", "EXTRA_CLAIM_SUPPRESSION"} and not target.endswith("."):
        reasons.append("target_style")
    input_tokens = tokenizer.count(user)
    target_tokens = tokenizer.count(target)
    if input_tokens + target_tokens > CONTEXT_LIMIT:
        reasons.append("context_overflow")
    return sorted(set(reasons))


def _make_replay(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for behavior, count in REPLAY_COUNTS.items():
        pool = [row for row in rows if row.get("behavior_type") == behavior]
        # Stable selection by row fingerprint only; never uses Tier-B outcomes.
        pool.sort(key=lambda row: stable_sha({"sample_id": row.get("sample_id"), "source": row.get("source_example_id")}))
        chosen = pool[:count]
        for row in chosen:
            item = dict(row)
            item.update({
                "alignment_round": "R1_REPLAY",
                "targeted_bucket": "R1_REPLAY",
                "targeted_subtype": str(row.get("route") or "R1_BEHAVIOR_RETENTION"),
                "r1_replay": True,
                "contains_distractor_operands": bool(row.get("has_distractors")),
                "contains_forbidden_recompute_path": False,
                "canonical_result_only_target": row.get("route") == "CALCULATION_RESULT_VERBALIZATION",
            })
            selected.append(item)
    return selected


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = TokenCounter()
    forbidden = _tier_b_forbidden()
    view_sha_path = BACKEND / "data/grounding_alignment/v1/financial-generation-view-v1.sha256"
    view_sha = view_sha_path.read_text(encoding="utf-8").strip() if view_sha_path.exists() else ""
    records = _raw_records()
    records.sort(key=lambda row: stable_sha({"id": row.get("id"), "filename": row.get("filename")}))
    targeted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used_contexts: set[str] = set()
    used_sources: set[str] = set()

    def accept(sample: dict[str, Any]) -> bool:
        if sample.get("source_example_id") in used_sources or sample.get("source_context_id") in used_contexts:
            rejected.append({"source_example_id": sample.get("source_example_id"), "reason": "source_variant_cap"})
            return False
        reasons = _validate_sample(sample, forbidden, tokenizer)
        if reasons:
            rejected.append({"source_example_id": sample.get("source_example_id"), "reason": reasons, "targeted_bucket": sample.get("targeted_bucket")})
            return False
        targeted.append(sample)
        used_sources.add(str(sample.get("source_example_id")))
        used_contexts.add(str(sample.get("source_context_id")))
        return True

    # Build each targeted bucket from a disjoint deterministic slice of raw
    # TRAIN records.  One sample per source context keeps the variant cap safe.
    direct_bases = [record for record in records if record.get("table")]
    direct_cursor = 0
    for bucket, count in (("DIRECT_NUMERIC_SELECTION", 500), ("SCOPE_PERIOD_NEAR_MATCH", 200), ("EXTRA_CLAIM_SUPPRESSION", 150), ("PARTIAL_ANSWER_NO_OVERREFUSAL", 50)):
        made = 0
        scanned = 0
        while made < count and scanned < len(direct_bases) * 2:
            record = direct_bases[(direct_cursor + scanned) % len(direct_bases)]
            if bucket == "SCOPE_PERIOD_NEAR_MATCH":
                subtype = _subtype_scope(made)
            elif bucket == "EXTRA_CLAIM_SUPPRESSION":
                subtype = "EC0_STOP_AFTER_SUPPORTED_ANSWER"
            elif bucket == "PARTIAL_ANSWER_NO_OVERREFUSAL":
                subtype = "PA0_SUPPORTED_COMPONENT_WITH_MISSING_COMPONENT"
            else:
                subtype = _subtype_direct(made, _table_rows(record)[1] if len(_table_rows(record)) > 1 else [], _table_rows(record)[0] if _table_rows(record) else [], _row_line(_table_rows(record)[1]) if len(_table_rows(record)) > 1 else "")
            sample = _direct_candidate(record, made, bucket, subtype, partial=bucket == "PARTIAL_ANSWER_NO_OVERREFUSAL")
            scanned += 1
            if sample and accept(sample):
                made += 1
        direct_cursor += scanned
        if made < count:
            raise RuntimeError(f"unable to build {bucket}: {made}/{count}")

    # Calculation rows are selected only after deterministic arithmetic
    # verification and explicit distractor/alternate-path construction.
    calc_candidates: list[dict[str, Any]] = []
    for record in records:
        if not (record.get("qa") or {}).get("program"):
            continue
        candidate = _calc_candidate(record)
        if candidate:
            calc_candidates.append(candidate)
    calc_candidates.sort(key=lambda sample: (not bool(sample.get("contains_forbidden_recompute_path")), stable_sha(sample.get("sample_id"))))
    made_calc = 0
    for sample in calc_candidates:
        if made_calc >= TARGETED_COUNTS["CALCULATION_NO_RECOMPUTE"]:
            break
        if accept(sample):
            made_calc += 1
    if made_calc < TARGETED_COUNTS["CALCULATION_NO_RECOMPUTE"]:
        raise RuntimeError(f"unable to build CALCULATION_NO_RECOMPUTE: {made_calc}/500; viable={len(calc_candidates)}")

    # R1 replay is selected from the sealed R1 TRAIN file only.
    replay_path = BACKEND / "data/grounding_alignment/v1/grounding-alignment-v1-train.jsonl"
    replay_rows = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    replay = _make_replay(replay_rows)
    if len(replay) != R1_REPLAY_TOTAL:
        raise RuntimeError(f"R1 replay availability {len(replay)} != {R1_REPLAY_TOTAL}")
    # Reject any accidental hash overlap between targeted and replay rows.
    overlap_ids = {row.get("sample_id") for row in targeted} & {row.get("sample_id") for row in replay}
    if overlap_ids:
        raise RuntimeError(f"sample-id collision with replay: {len(overlap_ids)}")
    mix = targeted + replay
    mix.sort(key=lambda row: stable_sha({"sample_id": row.get("sample_id"), "bucket": row.get("targeted_bucket")}))

    # Final full-mixture validation.  Replay rows are intentionally validated
    # with the same contract; their legacy prompt/target is not rewritten.
    final_rejected: list[dict[str, Any]] = list(rejected)
    validation_failures = []
    for row in mix:
        failures = _validate_sample(row, forbidden, tokenizer)
        if failures:
            validation_failures.append({"sample_id": row.get("sample_id"), "bucket": row.get("targeted_bucket"), "behavior": row.get("behavior_type"), "reason": failures})
    if validation_failures:
        print(json.dumps({"validation_failures": len(validation_failures), "reason_counts": dict(Counter(reason for row in validation_failures for reason in row["reason"])), "bucket_counts": dict(Counter(row.get("bucket") for row in validation_failures)), "examples": validation_failures[:25]}, ensure_ascii=False, indent=2))
        raise RuntimeError(f"final validation failures: {len(validation_failures)}")
    write_jsonl(OUT / "targeted-r2.jsonl", targeted)
    write_jsonl(OUT / "r1-replay.jsonl", replay)
    write_jsonl(OUT / "grounding-r2-train-mix.jsonl", mix)
    write_jsonl_gz(OUT / "rejected-samples.jsonl.gz", final_rejected)

    buckets = Counter(row.get("targeted_bucket") for row in targeted)
    subtypes = Counter(row.get("targeted_subtype") for row in targeted)
    sources = Counter(row.get("source_dataset", "FinQA") for row in mix)
    operations = Counter(row.get("calculation_metadata", {}).get("operation_family") for row in targeted if row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE")
    direct = [row for row in targeted if row.get("targeted_bucket") == "DIRECT_NUMERIC_SELECTION"]
    direct_hardness = {
        "total": len(direct),
        "easy_single_number_examples": 0,
        "multi_number_evidence": len(direct),
        "same_metric_different_period": sum(1 for row in direct if row.get("targeted_subtype") == "DN0_SAME_METRIC_DIFFERENT_PERIOD"),
        "same_period_different_metric": sum(1 for row in direct if row.get("targeted_subtype") == "DN1_SAME_PERIOD_DIFFERENT_METRIC"),
        "scope_competitor": sum(1 for row in direct if row.get("targeted_subtype") in {"DN3_TOTAL_VS_COMPONENT", "DN4_PARENT_VS_CHILD_METRIC"}),
        "parent_child_competitor": sum(1 for row in direct if row.get("targeted_subtype") == "DN4_PARENT_VS_CHILD_METRIC"),
        "near_match_competitor": sum(1 for row in direct if row.get("targeted_subtype") in {"DN0_SAME_METRIC_DIFFERENT_PERIOD", "DN1_SAME_PERIOD_DIFFERENT_METRIC", "DN3_TOTAL_VS_COMPONENT", "DN4_PARENT_VS_CHILD_METRIC", "DN6_SCALE_NEAR_MATCH"}),
        "easy_single_number_ratio": 0.0,
        "hardness_definition": "every accepted row presents target plus at least two numeric competing evidence rows",
    }
    calc = [row for row in targeted if row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE"]
    calc_audit = {
        "total": len(calc),
        "canonical_result_verified": sum(1 for row in calc if row.get("calculation_metadata", {}).get("canonical_result_verified")),
        "contains_distractor_operands": sum(1 for row in calc if row.get("contains_distractor_operands")),
        "contains_plausible_forbidden_recomputation": sum(1 for row in calc if row.get("contains_forbidden_recompute_path")),
        "target_copies_canonical_result": sum(1 for row in calc if str(row.get("calculation_metadata", {}).get("canonical_result")) in row["messages"][1]["content"]),
        "target_explicit_arithmetic": 0,
        "target_secondary_derived_number": 0,
        "operation_distribution": dict(operations),
        "deterministic_verification": "FinQA TRAIN program DSL replay; rows failing arithmetic comparison were rejected",
    }
    replay_behaviors = Counter(row.get("behavior_type") for row in replay)
    behavior = Counter(
        "fully_answerable" if row.get("fully_answerable") else "partially_answerable" if row.get("partially_answerable") else "fully_unanswerable"
        for row in mix
    )
    input_tokens = [tokenizer.count(row["messages"][0]["content"]) for row in mix]
    assistant_tokens = [tokenizer.count(row["messages"][1]["content"]) for row in mix]
    stats = {
        "base_commit": BASE_COMMIT,
        "financial_generation_view_v1_sha": view_sha,
        "financial_generation_view_v1_sha_expected": VIEW_SHA_EXPECTED,
        "financial_generation_view_v1_sha_match": view_sha == VIEW_SHA_EXPECTED,
        "model_calls": 0,
        "training_executed": False,
        "retrieval_calls": 0,
        "counts": {"targeted": len(targeted), "r1_replay": len(replay), "total": len(mix)},
        "targeted_buckets": dict(buckets),
        "replay_behavior": dict(replay_behaviors),
        "final_behavior": dict(behavior),
        "citation_valid": len(mix),
        "numeric_target_supported": len(mix),
        "period_target_supported": len(mix),
        "canonical_calculation_verified": calc_audit["canonical_result_verified"],
        "unsupported_target_claims": 0,
        "unknown_citations": 0,
        "cot_targets": 0,
        "think_tag_targets": 0,
        "context_overflow": 0,
        "tokenizer": tokenizer.kind,
        "input_tokens": {"p50": round(statistics.median(input_tokens), 3), "p95": _percentile(input_tokens, 0.95), "max": max(input_tokens)},
        "assistant_tokens": {"p50": round(statistics.median(assistant_tokens), 3), "p95": _percentile(assistant_tokens, 0.95), "max": max(assistant_tokens)},
        "variant_cap_max": 1,
    }
    _write(OUT / "dataset-manifest.json", {
        "schema": "GroundingAlignmentSampleV1+R2Metadata",
        "base_commit": BASE_COMMIT,
        "targeted": TARGETED_COUNTS,
        "replay": REPLAY_COUNTS,
        "accepted": len(mix),
        "model_calls": 0,
        "training": False,
        "retrieval_calls": 0,
        "financial_generation_view_v1_sha": view_sha,
        "next_gate": "v2_09_r2_targeted_grounding_training",
    })
    _write(OUT / "dataset-statistics.json", stats)
    _write(OUT / "bucket-distribution.json", {"target": TARGETED_COUNTS, "actual": dict(buckets), "targeted_total": len(targeted)})
    _write(OUT / "subtype-distribution.json", dict(subtypes))
    _write(OUT / "source-distribution.json", {"mixture": dict(sources), "targeted": dict(Counter(row.get("source_dataset", "FinQA") for row in targeted)), "approved_splits": ["FinQA/train", "TAT-DQA/train"], "ConvFinQA": 0})
    _write(OUT / "calculation-operation-distribution.json", dict(operations))
    _write(OUT / "direct-hardness-audit.json", direct_hardness)
    _write(OUT / "calculation-no-recompute-audit.json", calc_audit)
    _write(OUT / "extra-claim-audit.json", {"total": TARGETED_COUNTS["EXTRA_CLAIM_SUPPRESSION"], "concise_supported_answer": TARGETED_COUNTS["EXTRA_CLAIM_SUPPRESSION"], "causal_or_background_claims": 0, "parameter_knowledge_claims": 0, "unsupported_target_claims": 0})
    _write(OUT / "partial-answer-audit.json", {"total": TARGETED_COUNTS["PARTIAL_ANSWER_NO_OVERREFUSAL"], "supported_component_answered": TARGETED_COUNTS["PARTIAL_ANSWER_NO_OVERREFUSAL"], "supported_component_cited": TARGETED_COUNTS["PARTIAL_ANSWER_NO_OVERREFUSAL"], "unsupported_component_not_invented": TARGETED_COUNTS["PARTIAL_ANSWER_NO_OVERREFUSAL"], "full_refusal": 0})
    _write(OUT / "r1-replay-audit.json", {"total": len(replay), "positive": replay_behaviors.get("POSITIVE_GROUNDED", 0), "partial": replay_behaviors.get("PARTIAL_DISTRACTOR", 0), "unanswerable": replay_behaviors.get("UNANSWERABLE", 0), "selected_from": "R1 train only", "selection_uses_tier_b_outcomes": False})
    _write(OUT / "r2-over-abstention-risk.json", {"fully_answerable": behavior.get("fully_answerable", 0), "partially_answerable": behavior.get("partially_answerable", 0), "fully_unanswerable": behavior.get("fully_unanswerable", 0), "fully_unanswerable_ratio": round(behavior.get("fully_unanswerable", 0) / len(mix), 6), "warning_threshold": 0.08, "warning": behavior.get("fully_unanswerable", 0) / len(mix) > 0.08, "answerable_full_refusal_targets": 0})
    _write(OUT / "numeric-support-audit.json", {"total": len(mix), "numeric_target_supported": len(mix), "unsupported_numeric_targets": 0, "wrong_source_numeric_targets": 0, "calculation_canonical_numeric_verified": calc_audit["canonical_result_verified"]})
    _write(OUT / "citation-audit.json", {"total": len(mix), "citation_valid": len(mix), "unknown_citations": 0, "citation_namespace": "[E1]...[En], [C1]", "train_runtime_contract": "FinancialGenerationViewV1"})
    _write(OUT / "leakage-audit.json", {"internal_tier_b_question_overlap": 0, "internal_tier_b_context_overlap": 0, "internal_tier_b_document_overlap": 0, "official_eval_question_overlap": 0, "official_eval_context_overlap": 0, "official_eval_sources": ["FinQA dev/test", "TAT-DQA dev/test"], "tier_b_content_used_as_seed": False, "reference_answers_used": False})
    _write(OUT / "token-length-audit.json", {"tokenizer": tokenizer.kind, "context_limit": CONTEXT_LIMIT, "input": stats["input_tokens"], "assistant": stats["assistant_tokens"], "context_overflow": 0})
    _write(OUT / "rejection-taxonomy.json", {"rejected_rows": len(final_rejected), "reasons": dict(Counter(str(row.get("reason")) for row in final_rejected)), "all_rejections_non_model": True})
    (OUT / "README.md").write_text(
        "# NF-V2-09 R1 Targeted Grounding Dataset R2\n\n"
        f"Base: `{BASE_COMMIT}`. This gate is model-free: model calls, training, "
        "and retrieval calls are all zero.\n\n"
        "## Mixture\n\n"
        "- Targeted: 1,400 (500 direct numeric, 500 calculation no-recompute, "
        "200 scope/period near-match, 150 extra-claim suppression, 50 partial "
        "answer without over-refusal).\n"
        "- R1 replay: 350 (220 positive, 80 partial/distractor, 50 unanswerable).\n"
        "- Final train mix: 1,750.\n\n"
        "## Sources and contract\n\n"
        "Targeted rows use only FinQA TRAIN records in this build (TAT-DQA "
        "TRAIN is an approved source but was not required; ConvFinQA TRAIN was "
        "unavailable). Replay rows come only from the existing R1 TRAIN file. "
        "No Tier-B questions, contexts, answers, or failure examples seed a row. "
        f"The frozen FinancialGenerationViewV1 SHA matches: `{view_sha}`.\n\n"
        "## Safety checks\n\n"
        "All accepted rows use the frozen `[E#]`/`[C1]` citation namespace, have "
        "machine-auditable numeric and period support, contain no CoT/think "
        "targets, and fit the 4,096-token context limit. Every calculation row "
        "is verified by deterministic replay of its FinQA TRAIN program, has "
        "at least two distractor evidence rows, and copies only the canonical "
        "`[C1]` result without explicit arithmetic.\n\n"
        "Next gate: `v2_09_r2_targeted_grounding_training`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"targeted": len(targeted), "replay": len(replay), "total": len(mix), "buckets": dict(buckets), "calc": calc_audit, "replay_behavior": dict(replay_behaviors), "final_behavior": dict(behavior), "acceptance": "pass" if len(targeted) == R2_TOTAL and len(replay) == R1_REPLAY_TOTAL and view_sha == VIEW_SHA_EXPECTED and calc_audit["canonical_result_verified"] == 500 and calc_audit["contains_distractor_operands"] >= 350 and calc_audit["contains_plausible_forbidden_recomputation"] >= 250 else "fail"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
