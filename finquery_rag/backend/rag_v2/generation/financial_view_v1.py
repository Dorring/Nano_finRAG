"""Canonical FinancialGenerationViewV1 renderer.

The frozen Grounding Alignment contract is the source of truth for section
names, citation syntax, ordering, and answer rules.  This renderer is a pure
projection of a verified packet; it never reads Gold or evaluation labels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import GenerationInputV1

CONTRACT_SHA256 = "943decf288dffb99ffa6f196abc44e0a5bdb226350cede40e0a160c4bd61f6e4"
RENDERER_VERSION = "deterministic_text_v1"
ANSWER_RULES = (
    "1. Use only the verified evidence and calculation above.",
    "2. Do not introduce outside financial knowledge.",
    "3. Preserve supplied numbers, periods, units, currencies and scales exactly.",
    "4. Do not recalculate canonical calculation results.",
    "5. Cite factual claims using the supplied [E#] / [C#] IDs.",
    "6. If required evidence is missing, explicitly state that the provided evidence is insufficient.",
    "7. Answer concisely.",
)


def _value(value: Any) -> str:
    if value is None or value == "":
        return "not specified"
    if isinstance(value, (list, tuple)):
        return ", ".join(_value(item) for item in value)
    return str(value)


def _source(item: Mapping[str, Any]) -> str:
    provenance = item.get("provenance")
    if isinstance(provenance, Mapping):
        return _value(provenance.get("physical_source_id") or provenance.get("document_id") or item.get("source_id"))
    return _value(item.get("source_id"))


def _supporting_ids(raw: Any, evidence_ids: list[str]) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return evidence_ids[:]
    # Runtime packets may contain historical EV-* IDs.  The model-facing
    # namespace is always the frozen E1/E2/... sequence; this is a deterministic
    # packet-local projection, not a semantic alias or benchmark rule.
    mapping = {f"EV-{index + 1}": evidence_ids[index] for index in range(len(evidence_ids))}
    mapping.update({evidence_ids[index]: evidence_ids[index] for index in range(len(evidence_ids))})
    return [mapping.get(str(item), str(item)) for item in raw if mapping.get(str(item), str(item)) in evidence_ids]


@dataclass(frozen=True)
class FinancialGenerationViewV1:
    query_id: str
    route: str
    question: str
    rendered_text: str
    evidence_ids: tuple[str, ...]
    calculation_ids: tuple[str, ...]
    contract_sha256: str = CONTRACT_SHA256
    renderer_version: str = RENDERER_VERSION
    view_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.query_id or not self.route:
            raise ValueError("query_id and route are required")
        expected = hashlib.sha256(self.rendered_text.encode("utf-8")).hexdigest()
        if self.view_sha256 and self.view_sha256 != expected:
            raise ValueError("view_sha256 does not match rendered_text")
        object.__setattr__(self, "view_sha256", expected)

    @classmethod
    def from_verified_packet(cls, packet: Mapping[str, Any], *, question: str | None = None,
                             route: str | None = None) -> "FinancialGenerationViewV1":
        if packet.get("validation_status") != "VERIFIED":
            raise ValueError("FinancialGenerationViewV1 requires a VERIFIED packet")
        query_id = str(packet.get("query_id", ""))
        route_value = str(route or packet.get("route", ""))
        question_value = str(question if question is not None else packet.get("question", ""))
        items = packet.get("evidence_items")
        if not isinstance(items, (list, tuple)) or not items:
            raise ValueError("verified packet must contain evidence_items")
        evidence_ids = [f"E{index + 1}" for index in range(len(items))]
        lines = ["[QUESTION]", question_value, "", "[VERIFIED EVIDENCE]", ""]
        for evidence_id, item in zip(evidence_ids, items):
            if not isinstance(item, Mapping):
                raise ValueError("evidence item must be an object")
            lines.extend([f"[{evidence_id}]", f"Metric: {_value(item.get('metric') or item.get('normalized_metric'))}",
                          f"Period: {_value(item.get('period'))}", f"Scope: {_value(item.get('scope'))}",
                          f"Value: {_value(item.get('value'))}", f"Unit: {_value(item.get('unit'))}",
                          f"Currency: {_value(item.get('currency'))}", f"Scale: {_value(item.get('scale'))}",
                          f"Source: {_source(item)}", f"Evidence: {_value(item.get('source_text'))}", ""])
        calculation_ids: list[str] = []
        calculation = packet.get("calculation_result")
        if isinstance(calculation, Mapping):
            calculation_ids = ["C1"]
            calculation_value = calculation.get("value")
            lines.extend(["[VERIFIED CALCULATION]", "", "[C1]",
                          f"Operation: {_value(calculation.get('operation'))}",
                          f"Canonical Result: {_value(calculation_value)}",
                          f"Period: {_value(calculation.get('period'))}",
                          f"Unit: {_value(calculation.get('unit'))}",
                          f"Currency: {_value(calculation.get('currency'))}",
                          f"Scale: {_value(calculation.get('scale'))}",
                          f"Based On: {', '.join(f'[{item}]' for item in _supporting_ids(calculation.get('allowed_citation_ids'), evidence_ids))}", ""])
        lines.extend(["[ANSWER RULES]", *ANSWER_RULES])
        rendered = "\n".join(lines)
        return cls(query_id, route_value, question_value, rendered, tuple(evidence_ids), tuple(calculation_ids))

    def to_generation_input(self, trusted_packet: Mapping[str, Any]) -> GenerationInputV1:
        view_packet = {
            "query_id": self.query_id, "route": self.route, "validation_status": "VERIFIED",
            "allowed_citation_ids": [*self.evidence_ids, *self.calculation_ids],
            "evidence_items": [],
        }
        for index, evidence_id in enumerate(self.evidence_ids):
            item = trusted_packet["evidence_items"][index]
            sanitized = {key: item.get(key) for key in (
                "fact_id", "source_id", "metric", "normalized_metric", "period", "scope", "value",
                "unit", "currency", "scale", "source_text", "provenance") if key in item}
            sanitized["citation_id"] = evidence_id
            view_packet["evidence_items"].append(sanitized)
        if self.calculation_ids and isinstance(trusted_packet.get("calculation_result"), Mapping):
            calculation = dict(trusted_packet["calculation_result"])
            calculation["citation_ids"] = list(self.calculation_ids)
            calculation["allowed_citation_ids"] = list(self.evidence_ids)
            view_packet["calculation_result"] = calculation
        view_packet["question"] = self.question
        return GenerationInputV1(query_id=self.query_id, route=self.route, question=self.question,
                                 packet=view_packet, renderer_id="financial_generation_view_v1",
                                 rendered_text=self.rendered_text, view_sha256=self.view_sha256,
                                 trusted_packet=trusted_packet)


class FinancialGenerationViewRendererV1:
    """Adapter implementing the existing GenerationInputRendererV1 seam."""

    renderer_id = "financial_generation_view_v1"

    def render(self, packet: Mapping[str, Any]) -> GenerationInputV1:
        view = FinancialGenerationViewV1.from_verified_packet(packet)
        return view.to_generation_input(packet)
