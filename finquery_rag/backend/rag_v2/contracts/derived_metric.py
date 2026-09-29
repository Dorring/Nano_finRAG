"""Audited, versioned definitions for multi-step composite financial metrics."""

from __future__ import annotations

from dataclasses import dataclass

from .plan import OperandRef, OperationStep


@dataclass(frozen=True)
class DerivedMetricDefinition:
    """An audited, versioned specification for a composite financial metric."""

    metric: str
    definition_version: str
    required_leaf_metrics: tuple[str, ...]
    steps: tuple[OperationStep, ...]
    output_step_id: str
    output_unit: str
    disclosure_conditions: tuple[str, ...] = ()
    description: str = ""


# Audited definitions:
GROSS_MARGIN_DAG_V1 = DerivedMetricDefinition(
    metric="gross_margin",
    definition_version="gross_margin.dag.v1",
    required_leaf_metrics=("revenue", "cogs"),
    steps=(
        OperationStep(
            step_id="gross_profit",
            operation="difference",
            operands=(
                OperandRef(ref_type="slot", ref_id="revenue", role="current"),
                OperandRef(ref_type="slot", ref_id="cogs", role="previous"),
            ),
            output_unit="base",
            target_metric="gross_profit",
        ),
        OperationStep(
            step_id="gross_margin",
            operation="percentage_share",
            operands=(
                OperandRef(ref_type="step", ref_id="gross_profit", role="part"),
                OperandRef(ref_type="slot", ref_id="revenue", role="total"),
            ),
            output_unit="ratio",
            target_metric="gross_margin",
        ),
    ),
    output_step_id="gross_margin",
    output_unit="ratio",
    disclosure_conditions=("same_period", "same_entity", "same_currency"),
    description="Multi-step gross margin derivation: gross_profit = revenue - cogs; gross_margin = gross_profit / revenue",
)

NET_DEBT_V1 = DerivedMetricDefinition(
    metric="net_debt",
    definition_version="net_debt.v1",
    required_leaf_metrics=("short_term_debt", "long_term_debt", "cash"),
    steps=(
        OperationStep(
            step_id="total_debt",
            operation="sum",
            operands=(
                OperandRef(ref_type="slot", ref_id="short_term_debt", role="operand"),
                OperandRef(ref_type="slot", ref_id="long_term_debt", role="operand"),
            ),
            output_unit="base",
            target_metric="total_debt",
        ),
        OperationStep(
            step_id="net_debt",
            operation="difference",
            operands=(
                OperandRef(ref_type="step", ref_id="total_debt", role="current"),
                OperandRef(ref_type="slot", ref_id="cash", role="previous"),
            ),
            output_unit="base",
            target_metric="net_debt",
        ),
    ),
    output_step_id="net_debt",
    output_unit="base",
    disclosure_conditions=("same_period", "same_entity", "same_currency", "unrestricted_cash_only"),
    description="Net debt derivation: total_debt = short_term_debt + long_term_debt; net_debt = total_debt - cash",
)

DERIVED_METRIC_DEFINITIONS: dict[str, DerivedMetricDefinition] = {
    "gross_margin": GROSS_MARGIN_DAG_V1,
    "gross_margin.dag.v1": GROSS_MARGIN_DAG_V1,
    "net_debt": NET_DEBT_V1,
    "net_debt.v1": NET_DEBT_V1,
}


def get_derived_metric_definition(identifier: str | None) -> DerivedMetricDefinition | None:
    if not identifier:
        return None
    normalized = str(identifier).strip().lower()
    return DERIVED_METRIC_DEFINITIONS.get(normalized)


def is_supported_derived_metric(metric: str | None) -> bool:
    if not metric:
        return False
    return str(metric).strip().lower() in DERIVED_METRIC_DEFINITIONS
