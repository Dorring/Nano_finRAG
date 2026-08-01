"""Read-only observation protocol for deterministic answer decisions."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProductionFactTrace:
    fact_id: str
    candidate_key: str | None
    candidate_rank: int | None
    document_id: str | None
    page: int | None
    extraction_stage: str
    source_span_hash: str
    raw_value: str | None
    canonical_value: str | None
    currency: str | None
    unit: str | None
    scale: str | None
    period: str | None
    # Evaluation-only local text used for matching; never artifact serialized.
    evaluation_text: str | None = field(default=None, repr=False, compare=False)


class DeterministicAnswerObserver(Protocol):
    def on_route_selected(self, *, route: str) -> None: ...
    def on_fact_candidate_extracted(self, *, candidate: ProductionFactTrace) -> None: ...
    def on_fact_selected(self, *, fact_id: str, reason_codes: tuple[str, ...]) -> None: ...
    def on_calculation_selected(self, *, operation: str, operand_fact_ids: tuple[str, ...]) -> None: ...
    def on_answer_rendered(self, *, source_fact_ids: tuple[str, ...], answer_hash: str) -> None: ...
    # NF42 R2 projection trace hooks (default no-op in Null/Recording).
    def on_numeric_candidate_projected(self, *, trace: dict[str, Any]) -> None: ...
    def on_fact_projection_excluded(self, *, fact_id: str, reason: str, source_span_hash: str) -> None: ...
    def on_pre_selector_ranked(self, *, ranked_candidate_ids: list[str]) -> None: ...
    def on_selector_input(self, *, input_candidate_ids: list[str]) -> None: ...
    def on_selector_output(self, *, output_candidate_ids: list[str]) -> None: ...
    def on_answer_values_selected(self, *, values: list[str], fact_ids: list[str]) -> None: ...


class NullDeterministicAnswerObserver:
    def on_route_selected(self, *, route: str) -> None: pass
    def on_fact_candidate_extracted(self, *, candidate: ProductionFactTrace) -> None: pass
    def on_fact_selected(self, *, fact_id: str, reason_codes: tuple[str, ...]) -> None: pass
    def on_calculation_selected(self, *, operation: str, operand_fact_ids: tuple[str, ...]) -> None: pass
    def on_answer_rendered(self, *, source_fact_ids: tuple[str, ...], answer_hash: str) -> None: pass
    def on_numeric_candidate_projected(self, *, trace: dict[str, Any]) -> None: pass
    def on_fact_projection_excluded(self, *, fact_id: str, reason: str, source_span_hash: str) -> None: pass
    def on_pre_selector_ranked(self, *, ranked_candidate_ids: list[str]) -> None: pass
    def on_selector_input(self, *, input_candidate_ids: list[str]) -> None: pass
    def on_selector_output(self, *, output_candidate_ids: list[str]) -> None: pass
    def on_answer_values_selected(self, *, values: list[str], fact_ids: list[str]) -> None: pass


@dataclass
class RecordingDeterministicAnswerObserver(NullDeterministicAnswerObserver):
    routes: list[str] = field(default_factory=list)
    facts: list[ProductionFactTrace] = field(default_factory=list)
    selected_fact_ids: list[str] = field(default_factory=list)
    selection_reason_codes: list[tuple[str, ...]] = field(default_factory=list)
    calculation_operation: str | None = None
    operand_fact_ids: tuple[str, ...] = ()
    rendered_fact_ids: tuple[str, ...] = ()
    # NF42 R2 projection trace storage.
    projected_candidates: list[dict[str, Any]] = field(default_factory=list)
    projection_exclusions: list[dict[str, Any]] = field(default_factory=list)
    pre_selector_ranking: list[str] = field(default_factory=list)
    selector_input_ids: list[str] = field(default_factory=list)
    selector_output_ids: list[str] = field(default_factory=list)
    selected_values: list[str] = field(default_factory=list)
    selected_value_fact_ids: list[str] = field(default_factory=list)

    def on_route_selected(self, *, route: str) -> None:
        self.routes.append(route)

    def on_fact_candidate_extracted(self, *, candidate: ProductionFactTrace) -> None:
        self.facts.append(candidate)

    def on_fact_selected(self, *, fact_id: str, reason_codes: tuple[str, ...]) -> None:
        self.selected_fact_ids.append(fact_id)
        self.selection_reason_codes.append(reason_codes)

    def on_calculation_selected(self, *, operation: str, operand_fact_ids: tuple[str, ...]) -> None:
        self.calculation_operation = operation
        self.operand_fact_ids = operand_fact_ids

    def on_answer_rendered(self, *, source_fact_ids: tuple[str, ...], answer_hash: str) -> None:
        self.rendered_fact_ids = source_fact_ids

    def on_numeric_candidate_projected(self, *, trace: dict[str, Any]) -> None:
        self.projected_candidates.append(trace)

    def on_fact_projection_excluded(self, *, fact_id: str, reason: str, source_span_hash: str) -> None:
        self.projection_exclusions.append({
            "fact_id": fact_id,
            "reason": reason,
            "source_span_hash": source_span_hash,
        })

    def on_pre_selector_ranked(self, *, ranked_candidate_ids: list[str]) -> None:
        self.pre_selector_ranking = list(ranked_candidate_ids)

    def on_selector_input(self, *, input_candidate_ids: list[str]) -> None:
        self.selector_input_ids = list(input_candidate_ids)

    def on_selector_output(self, *, output_candidate_ids: list[str]) -> None:
        self.selector_output_ids = list(output_candidate_ids)

    def on_answer_values_selected(self, *, values: list[str], fact_ids: list[str]) -> None:
        self.selected_values = list(values)
        self.selected_value_fact_ids = list(fact_ids)


def fact_id(*, candidate_key: str | None, stage: str, ordinal: int, raw_value: str) -> str:
    payload = f"{candidate_key or 'unresolved'}|{stage}|{ordinal}|{raw_value}".encode()
    return f"production-fact:v1:{hashlib.sha256(payload).hexdigest()}"


def _stable_json_hash(payload: Any) -> str:
    """SHA-256 of a canonical JSON representation."""
    import json
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def projected_candidate_id(
    *,
    provider: str,
    candidate_key: str | None,
    source_fact_ids: tuple[str, ...],
    source_span_hash: str,
    projected_value_hashes: tuple[str, ...],
) -> str:
    """Deterministic projected candidate ID independent of runtime addresses."""
    payload = {
        "schema": "numeric-projection/v1",
        "provider": provider,
        "candidate_key": candidate_key,
        "source_fact_ids": sorted(source_fact_ids),
        "source_span_hash": source_span_hash,
        "projected_value_hashes": sorted(projected_value_hashes),
    }
    return "projected:v1:" + _stable_json_hash(payload)
