def _subset_metrics(records):
    return {
        "raw": sum(item["raw_answer_correct"] for item in records),
        "released": sum(item["released_answer_correct"] for item in records),
    }


def _validator_metrics(records):
    false_reject = sum(
        item["raw_answer_correct"] and item["released_response_type"] != "answer"
        for item in records
    )
    return {"false_reject": false_reject}


def _record(**overrides):
    base = {
        "context_coverage": "all_gold_in_final",
        "raw_answer_correct": True,
        "released_answer_correct": True,
        "raw_numeric_correct": True,
        "raw_unit_correct": True,
        "raw_period_correct": True,
        "released_citation_recall": 1.0,
        "released_citation_precision": 1.0,
        "released_response_type": "answer",
        "no_answer_correct": None,
        "repair_attempted": False,
        "repair_succeeded": False,
        "latency_ms": 1.0,
    }
    base.update(overrides)
    return base


def test_answer_metrics_keep_raw_and_released_distinct():
    result = _subset_metrics([_record(released_answer_correct=False)])
    assert result["raw"] == 1
    assert result["released"] == 0


def test_validator_false_reject_is_counted_from_production_release():
    outcomes = _validator_metrics([_record(released_response_type="blocked")])
    assert outcomes["false_reject"] == 1
