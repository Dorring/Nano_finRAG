from src.evaluation.nf_opt_12 import collapse_families, evidence_family_id, parse_query_slots


def test_table_row_family_is_gold_independent_and_keeps_rows_distinct():
    first = {
        "canonical_document_id": "doc",
        "page": 1,
        "evidence_id": "doc::page_1::table_1::row_2",
        "candidate_key": "a",
    }
    same = {**first, "candidate_key": "b"}
    other_row = {**first, "candidate_key": "c", "evidence_id": "doc::page_1::table_1::row_3"}
    assert evidence_family_id(first) == evidence_family_id(same)
    assert evidence_family_id(first) != evidence_family_id(other_row)
    assert collapse_families([first, same, other_row], 5) == [first, other_row]


def test_query_slots_only_use_query_fields():
    plan = parse_query_slots(
        {
            "case_id": "case",
            "question": "How much did revenue grow from FY2024 to FY2025?",
            "document_scope": ["doc"],
            "requires_calculation": True,
            "expected_answer": "must not be read",
        }
    )
    assert plan["expected_fields_read"] is False
    assert plan["operation"] == "growth_rate"
    assert [slot["period"] for slot in plan["slots"]] == ["FY2024", "FY2025"]
