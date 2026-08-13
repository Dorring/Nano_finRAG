"""Generic runtime trusted-evidence gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rag_v2.contracts.plan import SupervisorPlan


def _route_packet_name(plan: SupervisorPlan) -> str:
    return "DIRECT" if plan.intent.value == "DIRECT_FACT" else plan.intent.value


@dataclass(frozen=True)
class EvidenceGateResultV1:
    valid: bool
    reason: str
    source: str | None
    packet: Mapping[str, Any] | None


class TrustedEvidenceGateV1:
    """Validates only runtime packet structure and route readiness."""

    def validate(self, plan: SupervisorPlan, packet: Mapping[str, Any] | None,
                 query_id: str) -> EvidenceGateResultV1:
        if packet is None:
            return EvidenceGateResultV1(False, "NO_PACKET", None, None)
        if packet.get("query_id") != query_id:
            return EvidenceGateResultV1(False, "QUERY_ID_MISMATCH", None, packet)
        if packet.get("validation_status") != "VERIFIED":
            return EvidenceGateResultV1(False, "PACKET_NOT_VERIFIED", None, packet)
        expected_route = _route_packet_name(plan)
        if str(packet.get("route")) != expected_route:
            return EvidenceGateResultV1(False, "ROUTE_MISMATCH", None, packet)
        items = packet.get("evidence_items")
        if not isinstance(items, (list, tuple)) or not items:
            return EvidenceGateResultV1(False, "NO_EVIDENCE_ITEMS", None, packet)
        allowed = packet.get("allowed_citation_ids")
        if not isinstance(allowed, (list, tuple)):
            return EvidenceGateResultV1(False, "CITATION_ALLOWLIST_MISSING", None, packet)
        for item in items:
            if not isinstance(item, Mapping):
                return EvidenceGateResultV1(False, "MALFORMED_EVIDENCE_ITEM", None, packet)
            if not item.get("fact_id") or not item.get("citation_id"):
                return EvidenceGateResultV1(False, "FACT_ID_MISSING", None, packet)
            provenance = item.get("provenance")
            if not isinstance(provenance, Mapping):
                return EvidenceGateResultV1(False, "PROVENANCE_MISSING", None, packet)
            if not (provenance.get("physical_source_id") or item.get("source_id")):
                return EvidenceGateResultV1(False, "PHYSICAL_SOURCE_MISSING", None, packet)
        if plan.intent.value == "CALCULATION":
            calc = packet.get("calculation_result")
            if not isinstance(calc, Mapping) or not calc.get("runtime_calculation_ready", False):
                return EvidenceGateResultV1(False, "CALCULATION_NOT_READY", "calculation", packet)
        source = str(packet.get("evaluation_tier") or packet.get("evidence_source") or "runtime_packet")
        return EvidenceGateResultV1(True, "VERIFIED", source, packet)
