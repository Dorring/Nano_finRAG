"""Runtime-only deterministic validation for generated financial answers.

The validator intentionally does not attempt to be a semantic truth judge.  It
enforces the dimensions that can be proven from a verified packet: envelope,
citations, explicit numbers, periods, units/scales, and canonical calculation
results.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from .contracts import (AnswerEnvelopeV1, GenerationValidationFindingV1,
                        GenerationValidationReportV1, ValidationSeverity)

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
_PERIOD_RE = re.compile(r"\b(?:FY\s*\d{4}|Q[1-4]\s*FY?\s*\d{4}|\d{4}\s*Q[1-4]|20\d{2})\b", re.I)
_CURRENCY_RE = re.compile(r"(?:\$|€|£|¥|\b(?:USD|EUR|GBP|JPY|CNY)\b)", re.I)
_UNIT_RE = re.compile(r"\b(?:millions?|billions?|thousands?|percent|percentage|ratio|shares?|dollars?)\b|%", re.I)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _numbers(value: Any) -> list[Decimal]:
    result: list[Decimal] = []
    for token in _NUMBER_RE.findall(_text(value).replace("−", "-")):
        clean = token.replace(",", "").rstrip("%").strip()
        try:
            result.append(Decimal(clean))
        except InvalidOperation:
            continue
    return result


def _close(a: Decimal, b: Decimal) -> bool:
    try:
        return abs(a - b) <= max(Decimal("0.0002"), abs(b) * Decimal("0.0005"))
    except Exception:
        return False


def _periods(value: Any) -> set[str]:
    return {re.sub(r"\s+", "", item).upper() for item in _PERIOD_RE.findall(_text(value))}


def _iter_evidence(packet: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for item in packet.get("evidence_items", ()):
        if isinstance(item, Mapping):
            yield item


def _supported_numbers(packet: Mapping[str, Any]) -> list[Decimal]:
    values: list[Decimal] = []
    for item in _iter_evidence(packet):
        values.extend(_numbers(item.get("value")))
        values.extend(_numbers(item.get("source_text")))
    calculation = packet.get("calculation_result")
    if isinstance(calculation, Mapping):
        values.extend(_numbers(calculation.get("value")))
        for operand in calculation.get("operands", ()):
            if isinstance(operand, Mapping):
                values.extend(_numbers(operand.get("value")))
    return values


def _supported_periods(packet: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for item in _iter_evidence(packet):
        values |= _periods(item.get("period"))
        values |= _periods(item.get("source_text"))
        values |= _periods(item.get("column_header_path"))
    calculation = packet.get("calculation_result")
    if isinstance(calculation, Mapping):
        values |= _periods(calculation.get("period"))
        for operand in calculation.get("operands", ()):
            if isinstance(operand, Mapping):
                values |= _periods(operand.get("period"))
    return values


def _known_units(packet: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for item in _iter_evidence(packet):
        values.add(_text(item.get("unit")).strip().lower()) if item.get("unit") else None
        values.add(_text(item.get("currency")).strip().upper()) if item.get("currency") else None
        values.add(_text(item.get("scale")).strip().lower()) if item.get("scale") else None
        values |= {x.lower() for x in _UNIT_RE.findall(item.get("source_text", ""))}
        values |= {x.upper() for x in _CURRENCY_RE.findall(item.get("source_text", ""))}
    calculation = packet.get("calculation_result")
    if isinstance(calculation, Mapping):
        for key in ("unit", "currency", "scale"):
            if calculation.get(key):
                values.add(_text(calculation[key]).strip().lower())
    return {item for item in values if item}


class RuntimeGenerationValidatorV1:
    """Deterministic validator; no reference-answer or Gold access exists here."""

    def validate(self, packet: Mapping[str, Any], envelope: AnswerEnvelopeV1) -> GenerationValidationReportV1:
        findings: list[GenerationValidationFindingV1] = []
        checked = ["GV0_ENVELOPE_SCHEMA", "GV1_CITATION_ID_VALIDITY", "GV2_CITATION_REQUIREMENT",
                   "GV3_NUMERIC_FIDELITY", "GV4_PERIOD_FIDELITY", "GV5_UNIT_CURRENCY_SCALE_FIDELITY",
                   "GV6_CALCULATION_RESULT_PRESERVATION", "GV7_UNKNOWN_CITATION", "GV8_STRUCTURE"]

        def add(code: str, severity: ValidationSeverity, message: str) -> None:
            findings.append(GenerationValidationFindingV1(code, severity, message))

        if not isinstance(packet, Mapping) or packet.get("validation_status") != "VERIFIED":
            add("GV0_ENVELOPE_SCHEMA", ValidationSeverity.HARD_FAIL, "packet is not VERIFIED")
        if not isinstance(envelope, AnswerEnvelopeV1):
            add("GV0_ENVELOPE_SCHEMA", ValidationSeverity.HARD_FAIL, "malformed AnswerEnvelopeV1")
            return GenerationValidationReportV1(ValidationSeverity.HARD_FAIL, tuple(findings), tuple(checked))
        if envelope.query_id != packet.get("query_id") or envelope.route != packet.get("route"):
            add("GV0_ENVELOPE_SCHEMA", ValidationSeverity.HARD_FAIL, "query or route mismatch")
        if envelope.generation_status.lower() != "complete" or not envelope.answer_text.strip():
            add("GV0_ENVELOPE_SCHEMA", ValidationSeverity.HARD_FAIL, "generation is not complete")

        allowed = {str(item) for item in packet.get("allowed_citation_ids", ())}
        emitted = set(envelope.citation_ids)
        bracketed = {match.strip() for match in _CITATION_RE.findall(envelope.answer_text)}
        unknown = (emitted | bracketed) - allowed
        if unknown:
            add("GV7_UNKNOWN_CITATION", ValidationSeverity.HARD_FAIL,
                f"unknown citation IDs: {sorted(unknown)}")
        if bracketed - emitted:
            add("GV1_CITATION_ID_VALIDITY", ValidationSeverity.HARD_FAIL,
                "answer contains citation not declared in envelope")
        factual = bool(_NUMBER_RE.search(envelope.answer_text) or envelope.answer_text.strip())
        if factual and not emitted:
            add("GV2_CITATION_REQUIREMENT", ValidationSeverity.SOFT_FAIL,
                "factual answer has no supplied citation ID")

        # Do not count years in an explicit period as numeric claims.
        answer_without_periods = _CITATION_RE.sub(" ", _PERIOD_RE.sub(" ", envelope.answer_text))
        answer_nums = _numbers(answer_without_periods)
        supported = _supported_numbers(packet)
        calculation = packet.get("calculation_result")
        if calculation and isinstance(calculation, Mapping):
            canonical = _numbers(calculation.get("value"))
            # Ratios are commonly verbalized as percentages.
            supported += [item * Decimal("100") for item in canonical]
        unsupported = [item for item in answer_nums if not any(_close(item, known) for known in supported)]
        if unsupported:
            add("GV3_NUMERIC_FIDELITY", ValidationSeverity.HARD_FAIL,
                f"unsupported material number(s): {[str(item) for item in unsupported]}")

        answer_periods = _periods(envelope.answer_text)
        packet_periods = _supported_periods(packet)
        if answer_periods and packet_periods and not answer_periods <= packet_periods:
            add("GV4_PERIOD_FIDELITY", ValidationSeverity.HARD_FAIL,
                f"period(s) not supported by packet: {sorted(answer_periods - packet_periods)}")

        answer_currency = {item.upper() for item in _CURRENCY_RE.findall(envelope.answer_text)}
        known_currency = {item.upper() for item in _known_units(packet) if item.upper() in {"USD", "EUR", "GBP", "JPY", "CNY", "$", "€", "£", "¥"}}
        if answer_currency and known_currency and answer_currency.isdisjoint(known_currency):
            add("GV5_UNIT_CURRENCY_SCALE_FIDELITY", ValidationSeverity.HARD_FAIL,
                "answer currency conflicts with packet")
        answer_units = {item.lower() for item in _UNIT_RE.findall(envelope.answer_text)}
        known_unit_tokens = {item.lower() for item in _known_units(packet)}
        if answer_units and known_unit_tokens:
            incompatible = {"percent", "percentage", "%", "ratio"} & answer_units
            known_ratio = {"ratio", "percent", "percentage", "%"} & known_unit_tokens
            if incompatible and known_ratio and not incompatible & known_ratio:
                add("GV5_UNIT_CURRENCY_SCALE_FIDELITY", ValidationSeverity.HARD_FAIL,
                    "answer unit conflicts with packet")
        scale_words = {"thousand": 1, "thousands": 1, "million": 1000000, "millions": 1000000,
                       "billion": 1000000000, "billions": 1000000000}
        answer_scales = {scale_words[item.lower()] for item in _UNIT_RE.findall(envelope.answer_text)
                         if item.lower() in scale_words}
        packet_scales: set[int] = set()
        for item in _iter_evidence(packet):
            raw = _text(item.get("scale")).replace(",", "").strip()
            if raw:
                try:
                    packet_scales.add(int(Decimal(raw)))
                except (InvalidOperation, ValueError):
                    pass
        if isinstance(calculation, Mapping) and calculation.get("scale"):
            try:
                packet_scales.add(int(Decimal(_text(calculation["scale"]))))
            except (InvalidOperation, ValueError):
                pass
        if answer_scales and packet_scales and not any(scale in packet_scales for scale in answer_scales):
            add("GV5_UNIT_CURRENCY_SCALE_FIDELITY", ValidationSeverity.HARD_FAIL,
                "answer scale conflicts with packet")

        if isinstance(calculation, Mapping) and str(calculation.get("status", "")).lower() == "executed":
            canonical = _numbers(calculation.get("value"))
            if canonical and not any(_close(num, canonical[0]) or _close(num, canonical[0] * Decimal("100")) for num in answer_nums):
                add("GV6_CALCULATION_RESULT_PRESERVATION", ValidationSeverity.HARD_FAIL,
                    "canonical calculation result is not preserved")
            calc_period = _periods(calculation.get("period"))
            if calc_period and answer_periods and not answer_periods <= (packet_periods | calc_period):
                add("GV6_CALCULATION_RESULT_PRESERVATION", ValidationSeverity.HARD_FAIL,
                    "calculation period is contradicted")

        hard = any(item.severity is ValidationSeverity.HARD_FAIL for item in findings)
        soft = any(item.severity is ValidationSeverity.SOFT_FAIL for item in findings)
        status = ValidationSeverity.HARD_FAIL if hard else ValidationSeverity.SOFT_FAIL if soft else ValidationSeverity.PASS
        return GenerationValidationReportV1(status, tuple(findings), tuple(checked))
