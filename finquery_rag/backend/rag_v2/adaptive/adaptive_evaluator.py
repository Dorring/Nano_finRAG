"""Evidence-state evaluation for the bounded adaptive loop."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .adaptive_contracts import (
    AdaptiveRAGStateV1,
    EvidenceDecision,
    EvidenceEvaluationV1,
    EvidencePacketV1,
    ReasonCode,
)
from .adaptive_temporal import EvidenceConsistencyGateV1


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    return text or None


class EvidenceStateEvaluatorV1:
    """Evaluate slots, calculation readiness, and temporal consistency.

    This evaluator is deterministic and never reads a reference answer.  A
    missing field is not guessed from raw text; it remains missing or
    ambiguous and can only be addressed by a concrete retriever action.
    """

    def __init__(self, consistency_gate: EvidenceConsistencyGateV1 | None = None) -> None:
        self.consistency_gate = consistency_gate or EvidenceConsistencyGateV1()

    @staticmethod
    def _slot_supports(packet: EvidencePacketV1, slot: Mapping[str, Any]) -> bool:
        slot_id = slot.get("slot_id")
        if slot_id and packet.slots and str(slot_id) not in packet.slots:
            return False
        for packet_key, slot_key in (
            ("metric", "metric"), ("period", "period"), ("entity", "entity"),
            ("scope", "scope"), ("unit", "unit"), ("currency", "currency"),
            ("scale", "scale"),
        ):
            expected = _norm(slot.get(slot_key))
            if expected is None:
                continue
            actual = _norm(getattr(packet, packet_key)) or _norm(getattr(packet.temporal, packet_key, None))
            if actual != expected:
                return False
        if slot.get("value_required", True) and packet.value is None and packet.calculation_result is None:
            return False
        return True

    @staticmethod
    def _period_ambiguous(packets: Sequence[EvidencePacketV1], slot: Mapping[str, Any]) -> bool:
        requested = _norm(slot.get("period"))
        if requested is None:
            return False
        candidates = [packet for packet in packets if _norm(packet.metric) == _norm(slot.get("metric"))]
        periods = {_norm(packet.period) for packet in candidates if packet.period}
        return len(periods) > 1 and requested not in periods

    def evaluate(self, state: AdaptiveRAGStateV1) -> EvidenceEvaluationV1:
        packets = tuple(EvidencePacketV1.from_mapping(item) for item in state.evidence_packets)
        slots = tuple(state.required_slots)
        requested_ids = tuple(str(slot.get("slot_id")) for slot in slots if slot.get("slot_id"))
        supported: dict[str, list[str]] = {}
        missing: list[str] = []
        reasons: list[ReasonCode] = []
        for slot in slots:
            slot_id = str(slot.get("slot_id", ""))
            matches = [packet.evidence_id for packet in packets if self._slot_supports(packet, slot)]
            if matches:
                supported[slot_id] = matches
            else:
                missing.append(slot_id)
                if self._period_ambiguous(packets, slot):
                    reasons.append(ReasonCode.AMBIGUOUS_PERIOD)
                elif any(_norm(packet.metric) == _norm(slot.get("metric")) for packet in packets):
                    reasons.append(ReasonCode.WRONG_PERIOD)
                elif any(_norm(packet.period) == _norm(slot.get("period")) and ((slot.get("entity") and _norm(packet.entity) != _norm(slot.get("entity"))) or (slot.get("scope") and _norm(packet.scope) != _norm(slot.get("scope")))) for packet in packets):
                    reasons.append(ReasonCode.WRONG_ENTITY_SCOPE)
                else:
                    reasons.append(ReasonCode.MISSING_SLOT)

        unique_reasons = list(dict.fromkeys(reasons))
        consistency = self.consistency_gate.evaluate(packets)
        conflicts = [
            {"left": left, "right": right}
            for left, right in consistency.conflict_pairs
        ]
        if consistency.decision.value == "UNRESOLVED_CONFLICT":
            return EvidenceEvaluationV1(
                EvidenceDecision.UNRESOLVED_CONFLICT,
                tuple(unique_reasons + [ReasonCode.EVIDENCE_CONFLICT]),
                requested_ids, tuple(sorted(supported)), tuple(missing),
                tuple(sorted({item for values in supported.values() for item in values})),
                consistency.decision.value, tuple(conflicts), False,
            )

        calc_requirements = state.calculation_requirements or {}
        operand_slots = tuple(str(item) for item in calc_requirements.get("operand_slots", ()))
        missing_operands = [item for item in operand_slots if item not in supported]
        calculation_ready = bool(calc_requirements.get("canonical_result") is not None) and not missing_operands
        if missing_operands:
            missing.extend(item for item in missing_operands if item not in missing)
            unique_reasons.append(ReasonCode.MISSING_OPERAND)

        evidence_ids = tuple(sorted({item for values in supported.values() for item in values}))
        if not missing and not missing_operands:
            return EvidenceEvaluationV1(
                EvidenceDecision.SUFFICIENT, tuple(unique_reasons), requested_ids,
                tuple(sorted(supported)), (), evidence_ids, consistency.decision.value,
                tuple(conflicts), calculation_ready,
            )

        # A missing/ambiguous slot is repairable only while the evaluator has
        # a concrete reason.  Unknown scope remains terminal to avoid guesses.
        if unique_reasons and all(reason not in {ReasonCode.WRONG_ENTITY_SCOPE} for reason in unique_reasons):
            decision = EvidenceDecision.REPAIRABLE
        else:
            decision = EvidenceDecision.TERMINAL_INSUFFICIENT
        return EvidenceEvaluationV1(
            decision, tuple(dict.fromkeys(unique_reasons or [ReasonCode.LOW_EVIDENCE_COVERAGE])),
            requested_ids, tuple(sorted(supported)), tuple(dict.fromkeys(missing)), evidence_ids,
            consistency.decision.value, tuple(conflicts), calculation_ready,
        )
