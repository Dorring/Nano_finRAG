from src.domain.calculation import CalculationOperation, CalculationStatus
from src.finance.calculation_intent import CalculationIntent
from src.finance.operation_router import RoutingDecision
from src.finance.structured_operand_binding import build_operand_specs


def _intent(operation, periods=("FY2024", "FY2025")):
    return CalculationIntent(True, operation, (), periods, (), 2, True, 1.0, (), None)


def test_growth_preserves_previous_current_roles():
    specs = build_operand_specs(
        question="What was the year-over-year growth rate of total revenue reported by Example from FY2024 to FY2025?",
        routing_decision=RoutingDecision(
            CalculationStatus.READY, operation=CalculationOperation.GROWTH_RATE
        ),
        calculation_intent=_intent(CalculationOperation.GROWTH_RATE),
    )
    assert [(spec.role, spec.period) for spec in specs] == [
        ("previous", "FY2024"),
        ("current", "FY2025"),
    ]


def test_difference_builds_two_operand_specs():
    specs = build_operand_specs(
        question="Which was higher in FY2025: Total automotive revenues or Energy generation revenue, and by how much?",
        routing_decision=RoutingDecision(
            CalculationStatus.READY, operation=CalculationOperation.DIFFERENCE
        ),
        calculation_intent=_intent(CalculationOperation.DIFFERENCE, ("FY2025",)),
    )
    assert [spec.role for spec in specs] == ["minuend", "subtrahend"]
