"""Pure, evaluation-only classification for observed deterministic traces."""
from __future__ import annotations

from enum import StrEnum
from typing import Callable, Iterable


class ProductionFactFailure(StrEnum):
    PRODUCTION_FACT_NOT_EXTRACTED = "production_fact_not_extracted"
    PRODUCTION_FACT_AVAILABLE_NOT_SELECTED = "production_fact_available_not_selected"
    PRODUCTION_FACT_SELECTED_RENDER_WRONG = "production_fact_selected_render_wrong"
    PRODUCTION_CALCULATION_WRONG_OPERATION = "production_calculation_wrong_operation"
    PRODUCTION_CALCULATION_WRONG_OPERAND = "production_calculation_wrong_operand"
    PRODUCTION_TRACE_INSUFFICIENT = "production_trace_insufficient"
    CORRECT = "correct"


def classify_observed_fact_failure(
    *,
    facts: Iterable[object],
    selected_fact_ids: Iterable[str],
    raw_answer_correct: bool,
    fact_matches_gold: Callable[[object], bool],
) -> ProductionFactFailure:
    """Classify only what an actual observer has recorded.

    This function deliberately receives selected IDs from the observer.  It
    never derives a selection from the rendered answer.
    """
    observed = list(facts)
    if not observed:
        return ProductionFactFailure.PRODUCTION_TRACE_INSUFFICIENT
    correct = [fact for fact in observed if fact_matches_gold(fact)]
    if not correct:
        return ProductionFactFailure.PRODUCTION_FACT_NOT_EXTRACTED
    if not set(selected_fact_ids).intersection(fact.fact_id for fact in correct):
        return ProductionFactFailure.PRODUCTION_FACT_AVAILABLE_NOT_SELECTED
    if not raw_answer_correct:
        return ProductionFactFailure.PRODUCTION_FACT_SELECTED_RENDER_WRONG
    return ProductionFactFailure.CORRECT


def next_step_for_observed_failures(failures: dict[str, int]) -> tuple[str, str]:
    """Return a single next direction; incomplete traces always block NF42."""
    if failures.get(ProductionFactFailure.PRODUCTION_TRACE_INSUFFICIENT.value, 0):
        return "none", "expand_observer_coverage"
    if failures.get(ProductionFactFailure.PRODUCTION_FACT_NOT_EXTRACTED.value, 0) >= 3:
        return "extractor", "stop_for_review"
    if failures.get(ProductionFactFailure.PRODUCTION_FACT_AVAILABLE_NOT_SELECTED.value, 0) >= 3:
        return "selector", "stop_for_review"
    if failures.get(ProductionFactFailure.PRODUCTION_FACT_SELECTED_RENDER_WRONG.value, 0) >= 2:
        return "renderer", "stop_for_review"
    return "none", "stop_for_review"


def proxy_production_relation(*, proxy_failure: str | None, production_failure: str) -> str:
    if production_failure == ProductionFactFailure.PRODUCTION_TRACE_INSUFFICIENT.value:
        return "trace_insufficient"
    if proxy_failure is None:
        return "proxy_unavailable"
    return "agreement" if proxy_failure == production_failure else "conflict"
