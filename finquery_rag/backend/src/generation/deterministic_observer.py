"""Read-only observation protocol for deterministic answer decisions."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol


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


class DeterministicAnswerObserver(Protocol):
    def on_route_selected(self, *, route: str) -> None: ...
    def on_fact_candidate_extracted(self, *, candidate: ProductionFactTrace) -> None: ...
    def on_fact_selected(self, *, fact_id: str, reason_codes: tuple[str, ...]) -> None: ...
    def on_calculation_selected(self, *, operation: str, operand_fact_ids: tuple[str, ...]) -> None: ...
    def on_answer_rendered(self, *, source_fact_ids: tuple[str, ...], answer_hash: str) -> None: ...


class NullDeterministicAnswerObserver:
    def on_route_selected(self, *, route: str) -> None: pass
    def on_fact_candidate_extracted(self, *, candidate: ProductionFactTrace) -> None: pass
    def on_fact_selected(self, *, fact_id: str, reason_codes: tuple[str, ...]) -> None: pass
    def on_calculation_selected(self, *, operation: str, operand_fact_ids: tuple[str, ...]) -> None: pass
    def on_answer_rendered(self, *, source_fact_ids: tuple[str, ...], answer_hash: str) -> None: pass


@dataclass
class RecordingDeterministicAnswerObserver(NullDeterministicAnswerObserver):
    routes: list[str] = field(default_factory=list)
    facts: list[ProductionFactTrace] = field(default_factory=list)
    selected_fact_ids: list[str] = field(default_factory=list)
    selection_reason_codes: list[tuple[str, ...]] = field(default_factory=list)
    calculation_operation: str | None = None
    operand_fact_ids: tuple[str, ...] = ()
    rendered_fact_ids: tuple[str, ...] = ()

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


def fact_id(*, candidate_key: str | None, stage: str, ordinal: int, raw_value: str) -> str:
    payload = f"{candidate_key or 'unresolved'}|{stage}|{ordinal}|{raw_value}".encode("utf-8")
    return f"production-fact:v1:{hashlib.sha256(payload).hexdigest()}"
