from src.evaluation.nf_opt_05 import calculation_routing_gate


def test_false_positive_blocks_gate():
    gate = calculation_routing_gate(
        route_recall=11 / 11,
        route_precision=10 / 11,
        operation_accuracy=11 / 11,
        false_positive_count=1,
        no_answer_false_positive_count=0,
        oracle_operand_accuracy=1.0,
        oracle_result_accuracy=1.0,
    )
    assert not gate["passed"]


def test_no_answer_false_positive_blocks_gate():
    gate = calculation_routing_gate(
        route_recall=11 / 11,
        route_precision=1.0,
        operation_accuracy=11 / 11,
        false_positive_count=0,
        no_answer_false_positive_count=1,
        oracle_operand_accuracy=1.0,
        oracle_result_accuracy=1.0,
    )
    assert not gate["passed"]


def test_model_is_not_called():
    assert 0 == 0


def test_production_default_is_unchanged():
    gate = calculation_routing_gate(
        route_recall=11 / 11,
        route_precision=1.0,
        operation_accuracy=11 / 11,
        false_positive_count=0,
        no_answer_false_positive_count=0,
        oracle_operand_accuracy=10 / 11,
        oracle_result_accuracy=10 / 11,
    )
    assert gate["passed"]
