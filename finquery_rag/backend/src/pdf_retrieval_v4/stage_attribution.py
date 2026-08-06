"""Stage-level failure attribution for Gate 08 R1 evaluation contract repair.

Distinguishes "Gold not in Structured Universe" from "Gold in Universe but
not retrieved" and "Gold retrieved but mapping failed".

The old Gate 08 scoring conflated these by checking ``all_stage_ids`` which
was the *retrieved pool*, not the *universe*.  This module fixes the
classification so each missing Gold gets exactly one failure category.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FirstFailureStage(str, Enum):
    """Unique failure stage for each missing Gold source."""

    NOT_IN_STRUCTURED_UNIVERSE = "not_in_structured_universe"
    STRUCTURED_VIEW_UNMAPPED = "structured_view_unmapped"
    STRUCTURED_MAPPING_AMBIGUOUS = "structured_mapping_ambiguous"
    GOLD_TABLE_NOT_RETRIEVED = "gold_table_not_retrieved"
    GOLD_ROW_NOT_RETRIEVED = "gold_row_not_retrieved"
    GOLD_FACT_NOT_RETRIEVED = "gold_fact_not_retrieved"
    FACT_RETRIEVED_MAPPING_FAILED = "fact_retrieved_mapping_failed"
    FACT_RETRIEVED_MAPPING_AMBIGUOUS = "fact_retrieved_mapping_ambiguous"
    STRUCTURED_BUDGET_TRUNCATED = "structured_budget_truncated"
    RECOVERED = "recovered"


@dataclass(frozen=True)
class StageAttributionInput:
    """All signals needed to classify one Gold source's first failure."""

    case_id: str
    gold_candidate_key: str

    in_structured_universe: bool
    universe_mapping_status: str

    gold_view_id: str | None

    retrieved_table_view_ids: set[str]
    retrieved_table_candidate_keys: set[str]
    retrieved_row_view_ids: set[str]
    retrieved_row_candidate_keys: set[str]
    retrieved_fact_view_ids: set[str]
    retrieved_fact_candidate_keys: set[str]

    structured_pool_candidate_keys: set[str]
    combined_pool_candidate_keys: set[str]

    structured_ambiguous_mapping_count: int


@dataclass(frozen=True)
class StageAttributionResult:
    """Result of attributing a single Gold source to a failure stage."""

    case_id: str
    gold_candidate_key: str
    first_failure_stage: FirstFailureStage
    recoverable_by_larger_k: bool = False
    in_structured_universe: bool = False
    in_retrieved_table: bool = False
    in_retrieved_row: bool = False
    in_retrieved_fact: bool = False
    in_combined_pool: bool = False
    mapping_status: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "candidate_identity": self.gold_candidate_key,
            "first_failure_stage": self.first_failure_stage.value,
            "recoverable_by_larger_k": self.recoverable_by_larger_k,
            "in_structured_universe": self.in_structured_universe,
            "in_retrieved_table": self.in_retrieved_table,
            "in_retrieved_row": self.in_retrieved_row,
            "in_retrieved_fact": self.in_retrieved_fact,
            "in_combined_pool": self.in_combined_pool,
            "mapping_status": self.mapping_status,
            "detail": self.detail,
        }


def classify_first_failure(data: StageAttributionInput) -> StageAttributionResult:
    """Classify a single Gold source's first failure stage.

    Classification order (each Gold gets exactly one category):

    A. Gold not in Structured Universe at all
       → ``not_in_structured_universe``

    B. Gold in Universe but view→candidate mapping failed
       → ``structured_view_unmapped`` or ``structured_mapping_ambiguous``

    C. Gold in Universe with unique mapping, but not retrieved at table stage
       → ``gold_table_not_retrieved``

    D. Table retrieved but row not retrieved
       → ``gold_row_not_retrieved``

    E. Row retrieved but fact not retrieved
       → ``gold_fact_not_retrieved``

    F. Fact retrieved but runtime mapping ambiguous
       → ``fact_retrieved_mapping_ambiguous``

    G. In structured pool but not in combined pool
       → ``structured_budget_truncated``

    H. Fact retrieved but candidate mapping failed
       → ``fact_retrieved_mapping_failed``

    If Gold is in combined pool → ``recovered``.
    """
    base_kwargs = {
        "case_id": data.case_id,
        "gold_candidate_key": data.gold_candidate_key,
        "in_structured_universe": data.in_structured_universe,
        "mapping_status": data.universe_mapping_status,
    }

    # 0. Recovered
    if data.gold_candidate_key in data.combined_pool_candidate_keys:
        return StageAttributionResult(
            **base_kwargs,
            first_failure_stage=FirstFailureStage.RECOVERED,
            in_combined_pool=True,
            in_retrieved_table=(
                data.gold_candidate_key
                in data.retrieved_table_candidate_keys
            ),
            in_retrieved_row=(
                data.gold_candidate_key
                in data.retrieved_row_candidate_keys
            ),
            in_retrieved_fact=(
                data.gold_candidate_key
                in data.retrieved_fact_candidate_keys
            ),
        )

    # A. Not in structured universe
    if not data.in_structured_universe:
        return StageAttributionResult(
            **base_kwargs,
            first_failure_stage=FirstFailureStage.NOT_IN_STRUCTURED_UNIVERSE,
            detail="Gold candidate_key not found in any structured view",
        )

    # B. In universe but mapping failed
    if data.universe_mapping_status == "unmapped":
        return StageAttributionResult(
            **base_kwargs,
            first_failure_stage=FirstFailureStage.STRUCTURED_VIEW_UNMAPPED,
            detail="View exists in universe but map_view returned unmapped",
        )

    if data.universe_mapping_status == "ambiguous":
        return StageAttributionResult(
            **base_kwargs,
            first_failure_stage=FirstFailureStage.STRUCTURED_MAPPING_AMBIGUOUS,
            detail="View exists but mapping is ambiguous",
        )

    # Check retrieval at each stage using both view_id and candidate_key
    in_table = (
        data.gold_view_id in data.retrieved_table_view_ids
        if data.gold_view_id
        else False
    ) or (
        data.gold_candidate_key in data.retrieved_table_candidate_keys
    )

    in_row = (
        data.gold_view_id in data.retrieved_row_view_ids
        if data.gold_view_id
        else False
    ) or (
        data.gold_candidate_key in data.retrieved_row_candidate_keys
    )

    in_fact = (
        data.gold_view_id in data.retrieved_fact_view_ids
        if data.gold_view_id
        else False
    ) or (
        data.gold_candidate_key in data.retrieved_fact_candidate_keys
    )

    # C. Not retrieved at table stage
    if not in_table:
        return StageAttributionResult(
            **base_kwargs,
            first_failure_stage=FirstFailureStage.GOLD_TABLE_NOT_RETRIEVED,
            recoverable_by_larger_k=True,
            in_retrieved_table=False,
        )

    # D. Table retrieved but row not retrieved
    if not in_row:
        return StageAttributionResult(
            **base_kwargs,
            first_failure_stage=FirstFailureStage.GOLD_ROW_NOT_RETRIEVED,
            recoverable_by_larger_k=True,
            in_retrieved_table=True,
            in_retrieved_row=False,
        )

    # E. Row retrieved but fact not retrieved
    if not in_fact:
        return StageAttributionResult(
            **base_kwargs,
            first_failure_stage=FirstFailureStage.GOLD_FACT_NOT_RETRIEVED,
            recoverable_by_larger_k=True,
            in_retrieved_table=True,
            in_retrieved_row=True,
            in_retrieved_fact=False,
        )

    # F. Fact retrieved but runtime mapping ambiguous
    if data.structured_ambiguous_mapping_count > 0:
        return StageAttributionResult(
            **base_kwargs,
            first_failure_stage=FirstFailureStage.FACT_RETRIEVED_MAPPING_AMBIGUOUS,
            in_retrieved_table=True,
            in_retrieved_row=True,
            in_retrieved_fact=True,
            detail="Fact retrieved but runtime mapping was ambiguous",
        )

    # G. In structured pool but not in combined pool — budget truncated
    if (
        data.gold_candidate_key
        in data.structured_pool_candidate_keys
    ):
        return StageAttributionResult(
            **base_kwargs,
            first_failure_stage=FirstFailureStage.STRUCTURED_BUDGET_TRUNCATED,
            recoverable_by_larger_k=True,
            in_retrieved_table=True,
            in_retrieved_row=True,
            in_retrieved_fact=True,
            detail="In structured pool but truncated from combined pool",
        )

    # H. Fact retrieved but candidate mapping failed
    return StageAttributionResult(
        **base_kwargs,
        first_failure_stage=FirstFailureStage.FACT_RETRIEVED_MAPPING_FAILED,
        in_retrieved_table=True,
        in_retrieved_row=True,
        in_retrieved_fact=True,
        detail="Fact retrieved but candidate mapping failed at runtime",
    )
