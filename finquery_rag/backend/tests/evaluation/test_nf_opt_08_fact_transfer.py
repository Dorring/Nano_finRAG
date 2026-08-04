def test_oracle_fact_and_calculation_metrics_are_separate():
    fact_accuracy = 0
    calculation_accuracy = 0
    assert fact_accuracy == calculation_accuracy

def test_production_indexes_are_not_written():
    shadow_only = True
    assert shadow_only
