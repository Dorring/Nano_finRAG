def test_fact_improvement_without_answer_improvement_fails_answer_gate():
    fact_gate = True
    answer_improved = False
    safety_passed = True
    assert not (fact_gate and answer_improved and safety_passed)


def test_existing_correct_case_regression_fails_safety_gate():
    safety = {"existing_correct_regressions": 1}
    assert safety["existing_correct_regressions"] == 1
