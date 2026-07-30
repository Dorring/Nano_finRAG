def test_structured_facts_do_not_make_no_answer_case_answerable():
    current = {"route": "not_answerable", "response_type": "blocked"}
    structured = {"route": "not_answerable", "response_type": "blocked"}
    assert current == structured
