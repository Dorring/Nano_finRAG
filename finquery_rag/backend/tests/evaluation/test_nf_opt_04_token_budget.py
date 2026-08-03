from src.evaluation.nf_opt_04 import select_token_budget

def candidate(key, tokens, gold=None):
    value = {"candidate_key": key, "tokens": tokens}
    if gold is not None:
        value["gold"] = gold
    return value

def count(items):
    return sum(item["tokens"] for item in items)

def test_token_budget_never_exceeds_max_evidence():
    result = select_token_budget([candidate("a", 3), candidate("b", 3), candidate("c", 3)], max_evidence=2, token_budget=10, count_context_tokens=count)
    assert len(result) == 2

def test_token_budget_skips_over_budget_candidate_without_reordering():
    result = select_token_budget([candidate("a", 7), candidate("b", 4), candidate("c", 3)], max_evidence=2, token_budget=7, count_context_tokens=count)
    assert [item["candidate_key"] for item in result] == ["a"]

def test_gold_labels_are_not_used_for_selection():
    result = select_token_budget([candidate("a", 2, False), candidate("b", 2, True)], max_evidence=2, token_budget=10, count_context_tokens=count)
    assert [item["candidate_key"] for item in result] == ["a", "b"]
