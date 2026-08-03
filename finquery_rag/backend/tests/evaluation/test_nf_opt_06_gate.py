from src.domain.calculation import CalculationStatus
from src.domain.evidence import EvidenceItem
from src.finance.calculation_pipeline import CalculationPipeline


def test_shadow_defaults_to_not_applicable():
    result = CalculationPipeline().try_structured_shadow(
        "calculate growth rate from FY2024 to FY2025",
        {"intent": "financial_calculation"},
        (EvidenceItem("chunk", "", "doc", 1, "text", 0.0, None, {}),),
    )
    assert result.status is CalculationStatus.NOT_APPLICABLE


def test_no_answer_does_not_execute():
    result = CalculationPipeline(
        enable_structured_operand_binding=True
    ).try_structured_shadow(
        "What is the private client contract amount?",
        {"intent": "document_qa"},
        (EvidenceItem("chunk", "", "doc", 1, "text", 0.0, None, {}),),
    )
    assert result.status is CalculationStatus.NOT_APPLICABLE
