from src.retrieval_v3.query_router import route_question


def test_unknown_fact_is_not_predicted_as_no_answer() -> None:
    profile = route_question("Does the FY2025 report disclose lunar mining royalties?")
    assert profile.task_type != "no_answer"
    assert profile.answerability_check_required is True
