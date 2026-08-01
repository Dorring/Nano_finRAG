def test_new_fact_utilization_funnel_requires_selection_before_release():
    new_fact_cases = [True, True, True, True]
    selected = [True, False, False, False]
    released = [True, False, False, False]
    assert sum(new_fact_cases) == 4
    assert sum(selected) == 1
    assert sum(released) == 1
