"""Deterministic temporal scope and consistency checks.

The resolver uses explicit structured fields only.  In particular, database
created_at/ingested_at values are never used as financial effective time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .adaptive_contracts import (
    ConsistencyDecision,
    EvidencePacketV1,
    PeriodSemantics,
    TemporalRelation,
)


def _norm(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    return text or None


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", text):
        return None
    try:
        return float(text.rstrip("%"))
    except ValueError:
        return None


@dataclass(frozen=True)
class TemporalResolutionV1:
    relation: TemporalRelation
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"relation": self.relation.value, "reason": self.reason}


class TemporalScopeResolverV1:
    """Resolve only relationships justified by explicit temporal metadata."""

    @staticmethod
    def _scope_key(packet: EvidencePacketV1) -> tuple[str | None, ...]:
        temporal = packet.temporal
        return (
            _norm(packet.entity or temporal.entity),
            _norm(packet.metric or temporal.metric),
            _norm(packet.scope or temporal.scope),
            _norm(packet.unit or temporal.unit),
            _norm(packet.currency or temporal.currency),
            _norm(packet.scale or temporal.scale),
        )

    @staticmethod
    def _period_key(packet: EvidencePacketV1) -> tuple[str | None, ...]:
        temporal = packet.temporal
        return (
            _norm(packet.period),
            _norm(temporal.fiscal_year),
            _norm(temporal.fiscal_quarter),
            _norm(temporal.period_start),
            _norm(temporal.period_end),
            temporal.period_semantics.value,
        )

    def resolve(self, left: EvidencePacketV1, right: EvidencePacketV1) -> TemporalResolutionV1:
        left_t, right_t = left.temporal, right.temporal
        if right_t.supersedes_document_id and right_t.supersedes_document_id == left.document_id:
            return TemporalResolutionV1(TemporalRelation.VERSION_SUPERSEDED, "explicit supersedes_document_id")
        if left_t.supersedes_document_id and left_t.supersedes_document_id == right.document_id:
            return TemporalResolutionV1(TemporalRelation.VERSION_SUPERSEDED, "explicit supersedes_document_id")

        left_scope = self._scope_key(left)
        right_scope = self._scope_key(right)
        if left_scope != right_scope:
            # Distinct metric slots in a multi-evidence request are not a
            # contradiction.  Compare entity/scope/unit dimensions first;
            # metric differences are independent evidence, not ambiguity.
            if left_scope[0] == right_scope[0] and left_scope[2:] == right_scope[2:]:
                return TemporalResolutionV1(TemporalRelation.DIFFERENT_SOURCE_SCOPE, "different explicit metric slot")
            if left_scope[:3] == right_scope[:3] and left.source != right.source:
                return TemporalResolutionV1(TemporalRelation.DIFFERENT_SOURCE_SCOPE, "different explicit source")
            return TemporalResolutionV1(TemporalRelation.AMBIGUOUS_SCOPE, "metric/entity/unit/scope differs")
        if left.source and right.source and left.source != right.source:
            return TemporalResolutionV1(TemporalRelation.DIFFERENT_SOURCE_SCOPE, "different explicit source")
        if self._period_key(left) != self._period_key(right):
            if left_t.period_semantics is PeriodSemantics.UNKNOWN or right_t.period_semantics is PeriodSemantics.UNKNOWN:
                return TemporalResolutionV1(TemporalRelation.AMBIGUOUS_SCOPE, "period semantics are unknown")
            return TemporalResolutionV1(TemporalRelation.TEMPORAL_SUCCESSION, "explicit periods differ")
        if left.document_id and right.document_id and left.document_id != right.document_id:
            return TemporalResolutionV1(TemporalRelation.SAME_FACT_SCOPE, "same explicit fact scope")
        return TemporalResolutionV1(TemporalRelation.SAME_FACT_SCOPE, "same explicit fact scope")


@dataclass(frozen=True)
class ConsistencyResultV1:
    decision: ConsistencyDecision
    relations: tuple[TemporalResolutionV1, ...] = ()
    conflict_pairs: tuple[tuple[str, str], ...] = ()
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "relations": [item.to_dict() for item in self.relations],
            "conflict_pairs": [list(item) for item in self.conflict_pairs],
            "reason": self.reason,
        }


class EvidenceConsistencyGateV1:
    """Detect unresolved same-scope contradictions without general NLI."""

    def __init__(self, resolver: TemporalScopeResolverV1 | None = None) -> None:
        self.resolver = resolver or TemporalScopeResolverV1()

    @staticmethod
    def _incompatible(left: EvidencePacketV1, right: EvidencePacketV1) -> bool:
        if left.value is None or right.value is None:
            return False
        # Exact strings are compatible; safe numeric equality is compatible.
        if str(left.value).strip() == str(right.value).strip():
            return False
        left_num, right_num = _number(left.value), _number(right.value)
        if left_num is not None and right_num is not None:
            return left_num != right_num
        # BUY/HOLD/SELL and other explicit categorical states are incompatible.
        return _norm(left.value) != _norm(right.value)

    def evaluate(self, packets: Iterable[EvidencePacketV1]) -> ConsistencyResultV1:
        packets = tuple(packet if isinstance(packet, EvidencePacketV1) else EvidencePacketV1.from_mapping(packet) for packet in packets)
        if len(packets) < 2:
            return ConsistencyResultV1(ConsistencyDecision.CONSISTENT, reason="one or zero evidence packets")
        relations: list[TemporalResolutionV1] = []
        conflicts: list[tuple[str, str]] = []
        saw_successor = False
        saw_source = False
        saw_superseded = False
        saw_ambiguous = False
        for index, left in enumerate(packets):
            for right in packets[index + 1:]:
                result = self.resolver.resolve(left, right)
                relations.append(result)
                if result.relation is TemporalRelation.VERSION_SUPERSEDED:
                    saw_superseded = True
                elif result.relation is TemporalRelation.TEMPORAL_SUCCESSION:
                    saw_successor = True
                elif result.relation is TemporalRelation.DIFFERENT_SOURCE_SCOPE:
                    saw_source = True
                elif result.relation is TemporalRelation.AMBIGUOUS_SCOPE:
                    saw_ambiguous = True
                elif result.relation is TemporalRelation.SAME_FACT_SCOPE and self._incompatible(left, right):
                    conflicts.append((left.evidence_id, right.evidence_id))
        if conflicts:
            return ConsistencyResultV1(
                ConsistencyDecision.UNRESOLVED_CONFLICT, tuple(relations), tuple(conflicts),
                "incompatible values in the same normalized metric/period/scope",
            )
        if saw_ambiguous:
            return ConsistencyResultV1(ConsistencyDecision.AMBIGUOUS, tuple(relations), reason="temporal/scope fields are incomplete")
        if saw_superseded:
            return ConsistencyResultV1(ConsistencyDecision.SUPERSEDED, tuple(relations), reason="explicit version succession")
        if saw_successor:
            return ConsistencyResultV1(ConsistencyDecision.TEMPORAL_SUCCESSION, tuple(relations), reason="different explicit financial periods")
        if saw_source:
            return ConsistencyResultV1(ConsistencyDecision.MULTI_SOURCE_COMPATIBLE, tuple(relations), reason="different sources, no same-scope conflict")
        return ConsistencyResultV1(ConsistencyDecision.CONSISTENT, tuple(relations), reason="no incompatible same-scope values")
