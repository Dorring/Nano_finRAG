"""Tests for Multi-Step Financial Calculation DAG execution, validation, and runtime integration."""

from __future__ import annotations

from decimal import Decimal
import pytest

from rag_v2.adaptive import AdaptiveRAGStateV1
from rag_v2.contracts import (
    Action,
    ContractError,
    ExecutionDAG,
    Intent,
    OperandRef,
    OperationStep,
    PlanValidationError,
    RequiredSlot,
    SupervisorPlan,
)
from rag_v2.supervisor import validate_plan_v2_01
from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
    CalculationStepRecord,
    IntermediateValue,
)
from src.finance.calculation_executor import execute_dag
from src.finance.calculation_renderer import render_calculation_result
from src.runtime.trusted_v2_calculation import DeterministicCalculationCapability
from src.runtime.trusted_v2_generation import (
    CandidateExecutionResult,
    TrustedV2GenerationCapability,
)
from src.runtime.trusted_v2_validation import TrustedReleaseValidationCapability


def _make_slot(slot_id: str, metric: str, period: str = "FY2025", role: str = "operand", entity: str = "AAPL") -> RequiredSlot:
    return RequiredSlot(
        slot_id=slot_id,
        metric=metric,
        period=period,
        role=role,
        value_type="numeric",
        unit="USD",
        entity=entity,
    )


def _make_operand(name: str, value: str, slot_id: str, chunk_id: str = "chk-1", unit: str = "USD") -> CalculationOperand:
    return CalculationOperand(
        name=name,
        value=Decimal(value),
        unit=unit,
        source_text=f"{name} was {value}",
        evidence_chunk_id=chunk_id,
        slot_id=slot_id,
    )


# ==============================================================================
# 1. 2-Step DAG Execution: Gross Margin
# ==============================================================================

def test_gross_margin_dag_execution() -> None:
    # Step 1: gross_profit = difference(revenue, cogs)
    # Step 2: gross_margin = percentage_share(gross_profit, revenue)
    step1 = OperationStep(
        step_id="step_gross_profit",
        operation="difference",
        operands=(
            OperandRef(ref_type="slot", ref_id="slot_rev", role="current"),
            OperandRef(ref_type="slot", ref_id="slot_cogs", role="previous"),
        ),
        output_unit="base",
        target_metric="gross_profit",
    )
    step2 = OperationStep(
        step_id="step_gross_margin",
        operation="percentage_share",
        operands=(
            OperandRef(ref_type="step", ref_id="step_gross_profit", role="part"),
            OperandRef(ref_type="slot", ref_id="slot_rev", role="total"),
        ),
        output_unit="ratio",
        target_metric="gross_margin",
    )
    dag = ExecutionDAG(
        steps=(step1, step2),
        output_step_id="step_gross_margin",
        target_metric="gross_margin",
        definition_version="gross_margin.dag.v1",
    )

    leaf_operands = {
        "slot_rev": _make_operand("revenue", "1000", "slot_rev", chunk_id="chunk_rev_01"),
        "slot_cogs": _make_operand("cogs", "600", "slot_cogs", chunk_id="chunk_cogs_02"),
    }

    result = execute_dag(dag, leaf_operands, precision=4)
    assert result.status is CalculationStatus.EXECUTED
    assert result.value == Decimal("0.4")
    assert result.unit == "ratio"
    assert result.target_metric == "gross_margin"
    assert len(result.steps) == 2

    # Step 1 record checks
    assert result.steps[0].step_id == "step_gross_profit"
    assert result.steps[0].value == Decimal("400")
    assert result.steps[0].unit == "USD"
    assert result.steps[0].input_refs == ("slot_rev", "slot_cogs")

    # Step 2 record checks
    assert result.steps[1].step_id == "step_gross_margin"
    assert result.steps[1].value == Decimal("0.4")
    assert result.steps[1].unit == "ratio"
    assert result.steps[1].input_refs == ("step_gross_profit", "slot_rev")

    # Provenance invariant: leaf operands only, no intermediate results in result.operands
    assert len(result.operands) == 2
    assert {op.evidence_chunk_id for op in result.operands} == {"chunk_rev_01", "chunk_cogs_02"}
    assert not any(isinstance(op, IntermediateValue) for op in result.operands)

    # Calculation ID is deterministic
    cid = result.calculation_id
    assert cid is not None
    assert cid.startswith("C1-")

    # Rendered output
    rendered = render_calculation_result(result)
    assert "Gross Margin: 40.00%" in rendered
    assert "Calculation Steps:" in rendered
    assert "step_gross_profit" in rendered
    assert "step_gross_margin" in rendered
    assert "Inputs:" in rendered
    assert "chunk_rev_01" in rendered
    assert "chunk_cogs_02" in rendered


# ==============================================================================
# 2. 2-Step DAG Execution: Net Debt
# ==============================================================================

def test_net_debt_dag_execution() -> None:
    # Step 1: total_debt = sum(short_term_debt, long_term_debt)
    # Step 2: net_debt = difference(total_debt, cash)
    step1 = OperationStep(
        step_id="step_total_debt",
        operation="sum",
        operands=(
            OperandRef(ref_type="slot", ref_id="slot_st_debt", role="operand"),
            OperandRef(ref_type="slot", ref_id="slot_lt_debt", role="operand"),
        ),
        output_unit="base",
        target_metric="total_debt",
    )
    step2 = OperationStep(
        step_id="step_net_debt",
        operation="difference",
        operands=(
            OperandRef(ref_type="step", ref_id="step_total_debt", role="current"),
            OperandRef(ref_type="slot", ref_id="slot_cash", role="previous"),
        ),
        output_unit="base",
        target_metric="net_debt",
    )
    dag = ExecutionDAG(
        steps=(step1, step2),
        output_step_id="step_net_debt",
        target_metric="net_debt",
        definition_version="net_debt.v1",
    )

    leaf_operands = {
        "slot_st_debt": _make_operand("short_term_debt", "250", "slot_st_debt", chunk_id="chunk_st"),
        "slot_lt_debt": _make_operand("long_term_debt", "750", "slot_lt_debt", chunk_id="chunk_lt"),
        "slot_cash": _make_operand("cash", "300", "slot_cash", chunk_id="chunk_cash"),
    }

    result = execute_dag(dag, leaf_operands, precision=2)
    assert result.status is CalculationStatus.EXECUTED
    assert result.value == Decimal("700.00")
    assert result.target_metric == "net_debt"
    assert len(result.steps) == 2
    assert result.steps[0].value == Decimal("1000.00")
    assert result.steps[1].value == Decimal("700.00")
    assert len(result.operands) == 3


# ==============================================================================
# 3. Graph Validation: Cycles, Dangling, Orphans, Duplicates
# ==============================================================================

def test_plan_validator_detects_dag_cycle() -> None:
    slot_a = _make_slot("s1", "revenue")
    slot_b = _make_slot("s2", "cogs")
    # Cycle: step1 -> step2 -> step1
    step1 = OperationStep(
        step_id="step1",
        operation="difference",
        operands=(
            OperandRef(ref_type="slot", ref_id="s1"),
            OperandRef(ref_type="step", ref_id="step2"),
        ),
    )
    step2 = OperationStep(
        step_id="step2",
        operation="percentage_share",
        operands=(
            OperandRef(ref_type="step", ref_id="step1"),
            OperandRef(ref_type="slot", ref_id="s2"),
        ),
    )
    dag = ExecutionDAG(steps=(step1, step2), output_step_id="step2")
    plan = SupervisorPlan(
        intent=Intent.CALCULATION,
        required_slots=(slot_a, slot_b),
        operation=None,
        next_action=Action.RETRIEVE,
        execution_dag=dag,
    )
    with pytest.raises(PlanValidationError, match="cycles"):
        validate_plan_v2_01(plan)


def test_plan_validator_detects_dangling_references() -> None:
    slot_a = _make_slot("s1", "revenue")
    # Dangling slot ref
    step_bad_slot = OperationStep(
        step_id="step1",
        operation="difference",
        operands=(
            OperandRef(ref_type="slot", ref_id="s1"),
            OperandRef(ref_type="slot", ref_id="non_existent_slot"),
        ),
    )
    dag = ExecutionDAG(steps=(step_bad_slot,), output_step_id="step1")
    plan = SupervisorPlan(
        intent=Intent.CALCULATION,
        required_slots=(slot_a,),
        operation=None,
        next_action=Action.RETRIEVE,
        execution_dag=dag,
    )
    with pytest.raises(PlanValidationError, match="dangling slot reference"):
        validate_plan_v2_01(plan)

    # Dangling step ref
    step_bad_step = OperationStep(
        step_id="step1",
        operation="difference",
        operands=(
            OperandRef(ref_type="slot", ref_id="s1"),
            OperandRef(ref_type="step", ref_id="non_existent_step"),
        ),
    )
    dag2 = ExecutionDAG(steps=(step_bad_step,), output_step_id="step1")
    plan2 = SupervisorPlan(
        intent=Intent.CALCULATION,
        required_slots=(slot_a,),
        operation=None,
        next_action=Action.RETRIEVE,
        execution_dag=dag2,
    )
    with pytest.raises(PlanValidationError, match="dangling step reference"):
        validate_plan_v2_01(plan2)


def test_plan_validator_detects_isolated_step() -> None:
    slot_a = _make_slot("s1", "revenue")
    slot_b = _make_slot("s2", "cogs")
    # step1 is computed but step2 produces output without using step1
    step1 = OperationStep(
        step_id="step1",
        operation="difference",
        operands=(OperandRef(ref_type="slot", ref_id="s1"), OperandRef(ref_type="slot", ref_id="s2")),
    )
    step2 = OperationStep(
        step_id="step2",
        operation="percentage_share",
        operands=(OperandRef(ref_type="slot", ref_id="s1"), OperandRef(ref_type="slot", ref_id="s2")),
    )
    dag = ExecutionDAG(steps=(step1, step2), output_step_id="step2")
    plan = SupervisorPlan(
        intent=Intent.CALCULATION,
        required_slots=(slot_a, slot_b),
        operation=None,
        next_action=Action.RETRIEVE,
        execution_dag=dag,
    )
    with pytest.raises(PlanValidationError, match="isolated or non-contributing"):
        validate_plan_v2_01(plan)


def test_plan_validator_detects_duplicate_step_ids() -> None:
    step1 = OperationStep(
        step_id="dup_id",
        operation="difference",
        operands=(OperandRef(ref_type="slot", ref_id="s1"), OperandRef(ref_type="slot", ref_id="s2")),
    )
    step2 = OperationStep(
        step_id="dup_id",
        operation="percentage_share",
        operands=(OperandRef(ref_type="slot", ref_id="s1"), OperandRef(ref_type="slot", ref_id="s2")),
    )
    with pytest.raises(ContractError, match="unique step_ids"):
        ExecutionDAG(steps=(step1, step2), output_step_id="dup_id")


def test_plan_validator_enforces_budget_limits() -> None:
    # Limit: max 10 steps
    slots = tuple(_make_slot(f"s{i}", f"metric_{i}") for i in range(12))
    steps = tuple(
        OperationStep(
            step_id=f"step_{i}",
            operation="difference",
            operands=(
                OperandRef(ref_type="slot", ref_id=f"s{i}"),
                OperandRef(ref_type="slot", ref_id=f"s{i+1}"),
            ),
        )
        for i in range(11)
    )
    dag = ExecutionDAG(steps=steps, output_step_id="step_10")
    plan = SupervisorPlan(
        intent=Intent.CALCULATION,
        required_slots=slots,
        operation=None,
        next_action=Action.RETRIEVE,
        execution_dag=dag,
    )
    with pytest.raises(PlanValidationError, match="steps exceed limit"):
        validate_plan_v2_01(plan)


# ==============================================================================
# 4. Financial Semantics & Disclosure Invariants
# ==============================================================================

def test_cross_entity_slots_in_dag_rejected() -> None:
    # Mixing AAPL revenue and TSLA cogs in a single-company derived metric is rejected
    slot_a = _make_slot("s1", "revenue", entity="AAPL")
    slot_b = _make_slot("s2", "cogs", entity="TSLA")
    step = OperationStep(
        step_id="step1",
        operation="difference",
        operands=(OperandRef(ref_type="slot", ref_id="s1"), OperandRef(ref_type="slot", ref_id="s2")),
    )
    dag = ExecutionDAG(steps=(step,), output_step_id="step1", target_metric="gross_profit")
    plan = SupervisorPlan(
        intent=Intent.CALCULATION,
        required_slots=(slot_a, slot_b),
        operation=None,
        next_action=Action.RETRIEVE,
        execution_dag=dag,
    )
    with pytest.raises(PlanValidationError, match="cannot mix distinct entities"):
        validate_plan_v2_01(plan)


def test_cross_period_slots_for_composite_ratio_rejected() -> None:
    # FY2024 cogs with FY2025 revenue for gross_margin is rejected
    slot_a = _make_slot("s1", "revenue", period="FY2025")
    slot_b = _make_slot("s2", "cogs", period="FY2024")
    step = OperationStep(
        step_id="step1",
        operation="difference",
        operands=(OperandRef(ref_type="slot", ref_id="s1"), OperandRef(ref_type="slot", ref_id="s2")),
    )
    dag = ExecutionDAG(steps=(step,), output_step_id="step1", target_metric="gross_margin")
    plan = SupervisorPlan(
        intent=Intent.CALCULATION,
        required_slots=(slot_a, slot_b),
        operation=None,
        next_action=Action.RETRIEVE,
        execution_dag=dag,
    )
    with pytest.raises(PlanValidationError, match="requires matching period"):
        validate_plan_v2_01(plan)


def test_restricted_cash_disclosure_ambiguity_rejected() -> None:
    slot_st = _make_slot("s1", "short_term_debt")
    slot_lt = _make_slot("s2", "long_term_debt")
    # Restricted cash in slot
    slot_cash = RequiredSlot(
        slot_id="s3",
        metric="cash and restricted cash",
        period="FY2025",
        role="operand",
        value_type="numeric",
        unit="USD",
        entity="AAPL",
    )
    step1 = OperationStep(
        step_id="step_total",
        operation="sum",
        operands=(OperandRef(ref_type="slot", ref_id="s1"), OperandRef(ref_type="slot", ref_id="s2")),
    )
    step2 = OperationStep(
        step_id="step_net",
        operation="difference",
        operands=(OperandRef(ref_type="step", ref_id="step_total"), OperandRef(ref_type="slot", ref_id="s3")),
    )
    dag = ExecutionDAG(steps=(step1, step2), output_step_id="step_net", target_metric="net_debt")
    plan = SupervisorPlan(
        intent=Intent.CALCULATION,
        required_slots=(slot_st, slot_lt, slot_cash),
        operation=None,
        next_action=Action.RETRIEVE,
        execution_dag=dag,
    )
    with pytest.raises(PlanValidationError, match="restricted_cash_ambiguity"):
        validate_plan_v2_01(plan)


# ==============================================================================
# 5. Arithmetic Edge Cases: Divide by Zero & Unit Mismatches
# ==============================================================================

def test_execute_dag_divide_by_zero_blocked() -> None:
    step1 = OperationStep(
        step_id="step1",
        operation="difference",
        operands=(OperandRef(ref_type="slot", ref_id="s1"), OperandRef(ref_type="slot", ref_id="s2")),
    )
    step2 = OperationStep(
        step_id="step2",
        operation="percentage_share",
        operands=(OperandRef(ref_type="step", ref_id="step1"), OperandRef(ref_type="slot", ref_id="s1")),
    )
    dag = ExecutionDAG(steps=(step1, step2), output_step_id="step2")
    # revenue = 0 -> division by zero in step 2
    leaf_operands = {
        "s1": _make_operand("revenue", "0", "s1"),
        "s2": _make_operand("cogs", "50", "s2"),
    }
    result = execute_dag(dag, leaf_operands)
    assert result.status is CalculationStatus.BLOCKED
    assert result.error_code == "PRIMITIVE_DECLINED"


def test_execute_dag_unit_mismatch_blocked() -> None:
    step1 = OperationStep(
        step_id="step1",
        operation="difference",
        operands=(OperandRef(ref_type="slot", ref_id="s1"), OperandRef(ref_type="slot", ref_id="s2")),
    )
    dag = ExecutionDAG(steps=(step1,), output_step_id="step1")
    # USD vs EUR
    leaf_operands = {
        "s1": _make_operand("revenue", "100", "s1", unit="USD"),
        "s2": _make_operand("cogs", "50", "s2", unit="EUR"),
    }
    result = execute_dag(dag, leaf_operands)
    assert result.status is CalculationStatus.BLOCKED
    assert result.error_code == "UNIT_MISMATCH"


# ==============================================================================
# 6. Determinism & Identity Stability
# ==============================================================================

def test_dag_calculation_id_stability() -> None:
    step1 = OperationStep(
        step_id="p",
        operation="difference",
        operands=(OperandRef(ref_type="slot", ref_id="s1"), OperandRef(ref_type="slot", ref_id="s2")),
    )
    dag = ExecutionDAG(steps=(step1,), output_step_id="p", target_metric="gross_profit")
    operands = {
        "s1": _make_operand("revenue", "100", "s1", chunk_id="chk1"),
        "s2": _make_operand("cogs", "40", "s2", chunk_id="chk2"),
    }
    res1 = execute_dag(dag, operands)
    res2 = execute_dag(dag, operands)
    assert res1.calculation_id == res2.calculation_id

    # Changing operand value changes calculation_id
    operands_diff = {
        "s1": _make_operand("revenue", "110", "s1", chunk_id="chk1"),
        "s2": _make_operand("cogs", "40", "s2", chunk_id="chk2"),
    }
    res3 = execute_dag(dag, operands_diff)
    assert res3.calculation_id != res1.calculation_id


# ==============================================================================
# 7. Capability Integration & State Validation
# ==============================================================================

def test_deterministic_calculation_capability_with_dag() -> None:
    step1 = OperationStep(
        step_id="step_gp",
        operation="difference",
        operands=(OperandRef(ref_type="slot", ref_id="slot_rev"), OperandRef(ref_type="slot", ref_id="slot_cogs")),
    )
    step2 = OperationStep(
        step_id="step_gm",
        operation="percentage_share",
        operands=(OperandRef(ref_type="step", ref_id="step_gp"), OperandRef(ref_type="slot", ref_id="slot_rev")),
        output_unit="ratio",
    )
    dag = ExecutionDAG(steps=(step1, step2), output_step_id="step_gm", target_metric="gross_margin")
    plan = SupervisorPlan(
        intent=Intent.CALCULATION,
        required_slots=(
            _make_slot("slot_rev", "revenue"),
            _make_slot("slot_cogs", "cogs"),
        ),
        operation=None,
        next_action=Action.RETRIEVE,
        execution_dag=dag,
    )

    state = AdaptiveRAGStateV1.new(
        request_id="req-dag-01",
        query="What was gross margin?",
        intent="CALCULATION",
        task_type="CALCULATION",
        plan={"supervisor_plan": plan.to_dict()},
    )
    state.bound_evidence_ids = ("ev_rev", "ev_cogs")
    state.bound_slot_bindings = {
        "slot_rev": ("ev_rev",),
        "slot_cogs": ("ev_cogs",),
    }
    state.evidence_packets = [
        {"evidence_id": "ev_rev", "value": "1000", "unit": "USD", "metric": "revenue"},
        {"evidence_id": "ev_cogs", "value": "600", "unit": "USD", "metric": "cogs"},
    ]
    state.calculation_requirements = {"precision": 4, "target_metric": "gross_margin"}

    capability = DeterministicCalculationCapability()
    result = capability.calculate(state)

    assert result.status is CalculationStatus.EXECUTED
    assert result.value == Decimal("0.4")
    assert capability.last_calculation_id is not None
    assert state.calculation_result_id == capability.last_calculation_id
    assert state._calculation_result_obj is result
    assert capability.last_operand_evidence_ids == ("ev_rev", "ev_cogs")


def test_missing_leaf_slot_blocks_calculation() -> None:
    step1 = OperationStep(
        step_id="step_gp",
        operation="difference",
        operands=(OperandRef(ref_type="slot", ref_id="slot_rev"), OperandRef(ref_type="slot", ref_id="slot_cogs")),
    )
    dag = ExecutionDAG(steps=(step1,), output_step_id="step_gp")
    plan = SupervisorPlan(
        intent=Intent.CALCULATION,
        required_slots=(
            _make_slot("slot_rev", "revenue"),
            _make_slot("slot_cogs", "cogs"),
        ),
        operation=None,
        next_action=Action.RETRIEVE,
        execution_dag=dag,
    )

    state = AdaptiveRAGStateV1.new(
        request_id="req-dag-missing",
        query="What was gross profit?",
        intent="CALCULATION",
        task_type="CALCULATION",
        plan={"supervisor_plan": plan.to_dict()},
    )
    # Only slot_rev is bound, slot_cogs is missing
    state.bound_evidence_ids = ("ev_rev",)
    state.bound_slot_bindings = {"slot_rev": ("ev_rev",)}
    state.evidence_packets = [
        {"evidence_id": "ev_rev", "value": "1000", "unit": "USD", "metric": "revenue"},
    ]

    capability = DeterministicCalculationCapability()
    result = capability.calculate(state)
    assert result.status is CalculationStatus.BLOCKED
    assert result.error_code == "INSUFFICIENT_OPERANDS"
    assert capability.last_calculation_id is None
    assert state.calculation_result_id is None


# ==============================================================================
# 8. Generation & Validation: Multi-step DAG Preservation and Release Invariants
# ==============================================================================

def test_generation_and_validation_with_dag_calculation() -> None:
    step1 = OperationStep(
        step_id="step_gp",
        operation="difference",
        operands=(OperandRef(ref_type="slot", ref_id="slot_rev"), OperandRef(ref_type="slot", ref_id="slot_cogs")),
    )
    step2 = OperationStep(
        step_id="step_gm",
        operation="percentage_share",
        operands=(OperandRef(ref_type="step", ref_id="step_gp"), OperandRef(ref_type="slot", ref_id="slot_rev")),
        output_unit="ratio",
    )
    dag = ExecutionDAG(steps=(step1, step2), output_step_id="step_gm", target_metric="gross_margin")
    plan = SupervisorPlan(
        intent=Intent.CALCULATION,
        required_slots=(
            _make_slot("slot_rev", "revenue"),
            _make_slot("slot_cogs", "cogs"),
        ),
        operation=None,
        next_action=Action.RETRIEVE,
        execution_dag=dag,
    )

    state = AdaptiveRAGStateV1.new(
        request_id="req-gen-val",
        query="What was Apple's gross margin?",
        intent="CALCULATION",
        task_type="CALCULATION",
        plan={"supervisor_plan": plan.to_dict()},
    )
    state.bound_evidence_ids = ("ev_rev", "ev_cogs")
    state.bound_slot_bindings = {
        "slot_rev": ("ev_rev",),
        "slot_cogs": ("ev_cogs",),
    }
    state.evidence_packets = [
        {"evidence_id": "ev_rev", "citation_id": "E1", "value": "1000", "unit": "USD", "metric": "revenue", "period": "FY2025"},
        {"evidence_id": "ev_cogs", "citation_id": "E2", "value": "600", "unit": "USD", "metric": "cogs", "period": "FY2025"},
    ]
    state.calculation_requirements = {"precision": 4, "target_metric": "gross_margin"}

    # 1. Run calculator
    calc_cap = DeterministicCalculationCapability()
    calc_res = calc_cap.calculate(state)
    assert calc_res.status is CalculationStatus.EXECUTED

    # 2. Run generator
    gen_cap = TrustedV2GenerationCapability()
    candidate = gen_cap.generate(state)
    assert candidate.route == "CALCULATION_SIMPLE"
    assert "Gross Margin: 40.00%" in candidate.candidate_answer
    assert candidate.calculation_ids == (state.calculation_result_id,)

    # 3. Run validator
    val_cap = TrustedReleaseValidationCapability()
    val_res = val_cap.validate(state, candidate)
    assert val_res.passed is True
    assert val_res.status == "PASS"


def test_validator_rejects_intermediate_step_as_evidence() -> None:
    # If a rogue candidate introduces an intermediate step id into bound_evidence_ids
    state = AdaptiveRAGStateV1.new(
        request_id="req-rogue",
        query="What was gross margin?",
        intent="CALCULATION",
        task_type="CALCULATION",
    )
    calc_rec = CalculationStepRecord(
        step_id="intermediate_step_id",
        operation=CalculationOperation.DIFFERENCE,
        value=Decimal("400"),
        unit="base",
        formula="rev - cogs",
        formula_version="diff.v1",
    )
    calc_res = CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.PERCENTAGE_SHARE,
        value=Decimal("0.4"),
        unit="ratio",
        steps=(calc_rec,),
    )
    state._calculation_result_obj = calc_res
    state.bound_evidence_ids = ("ev1", "intermediate_step_id")
    state.evidence_packets = [
        {"evidence_id": "ev1", "citation_id": "E1", "value": "1000", "unit": "USD"},
        {"evidence_id": "intermediate_step_id", "citation_id": "E2", "value": "400", "unit": "USD"},
    ]

    candidate = CandidateExecutionResult(
        candidate_answer="Gross margin was 40% [E1]",
        route="CALCULATION_SIMPLE",
        route_reason="test",
        bound_evidence_ids=("ev1", "intermediate_step_id"),
        citation_ids=("E1",),
        calculation_ids=(calc_res.calculation_id,) if calc_res.calculation_id else (),
    )

    val_cap = TrustedReleaseValidationCapability()
    val_res = val_cap.validate(state, candidate)
    assert val_res.passed is False
    assert "INTERMEDIATE_RESULT_AS_EVIDENCE" in val_res.reason_codes
