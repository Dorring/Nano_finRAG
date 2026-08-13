"""Focused model-free tests for the NF-V2-06 R1A data contract."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/evaluation/run_nf_v2_06_r1a_grounding_alignment.py"
SPEC = importlib.util.spec_from_file_location("nf_v2_06_r1a", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def base_record():
    return {
        "source_dataset": "FinQA",
        "source_example_id": "train-example-1",
        "source_context_id": "context-1",
        "document_identity": "training-doc-1",
        "question": "What was revenue in FY2021?",
        "answer": "$365.8 million",
        "evidence": [{"content": "Revenue for FY2021 was $365.8 million.", "source_label": "training-doc-1"}],
        "route": "DIRECT",
        "periods": ["FY2021"],
        "program": "",
        "canonical_result": None,
        "base_fingerprint": "base-1",
    }


def forbidden():
    return {
        "internal_question_hashes": set(),
        "internal_context_hashes": set(),
        "internal_document_ids": set(),
        "official_question_hashes": set(),
        "official_context_hashes": set(),
    }


def test_financial_generation_view_and_plain_citations():
    view, ids = MODULE.evidence_view(base_record(), MODULE.make_evidence_items(base_record()))
    assert ids == ["E1"]
    assert "[QUESTION]" in view
    assert "[VERIFIED EVIDENCE]" in view
    assert "[E1]" in view
    assert "[ANSWER RULES]" in view
    sample = MODULE.make_sample(base_record(), "POSITIVE_GROUNDED")
    assert MODULE.validate_sample(sample, forbidden(), MODULE.TokenCounter()) == []
    assert "[E1]" in sample["messages"][1]["content"]


def test_unanswerable_and_partial_contracts():
    base = base_record()
    negative = MODULE.make_sample(base, "UNANSWERABLE", "DIRECT", 1)
    partial_base = base | {"answer": "7", "evidence": [{"content": "Revenue was 7.", "source_label": "training-doc-1"}]}
    partial_base["question"] = "What was revenue?"
    partial_base["periods"] = []
    partial = MODULE.make_sample(partial_base, "PARTIAL_DISTRACTOR", "DIRECT", 2)
    assert MODULE.validate_sample(negative, forbidden(), MODULE.TokenCounter()) == []
    assert MODULE.validate_sample(partial, forbidden(), MODULE.TokenCounter()) == []
    assert negative["requires_abstention"] is True
    assert partial["partially_answerable"] is True


def test_calculation_result_is_verbalized_without_recalculation():
    base = base_record() | {
        "route": "CALCULATION_RESULT_VERBALIZATION",
        "program": "subtract(2021,2020)",
        "canonical_result": "12.4%",
    }
    sample = MODULE.make_sample(base, "POSITIVE_GROUNDED", base["route"])
    assert "The canonical calculation result is 12.4% [C1]." == sample["messages"][1]["content"]
    assert MODULE.validate_sample(sample, forbidden(), MODULE.TokenCounter()) == []


def test_forbidden_fingerprint_and_no_think_rules():
    sample = MODULE.make_sample(base_record(), "POSITIVE_GROUNDED")
    blocked = forbidden()
    blocked["internal_question_hashes"].add(sample["fingerprints"]["normalized_question_hash"])
    assert "QV8_question_overlap" in MODULE.validate_sample(sample, blocked, MODULE.TokenCounter())
    sample["messages"][1]["content"] = "<think>private</think> [E1]"
    assert "QV10_think_target" in MODULE.validate_sample(sample, forbidden(), MODULE.TokenCounter())


def test_split_group_isolation():
    old_targets = MODULE.SPLIT_TARGET
    try:
        MODULE.SPLIT_TARGET = {"train": 4, "alignment_dev": 2, "alignment_holdout": 2}
        samples = []
        for group_index, size in enumerate((2, 2, 2, 2)):
            for row_index in range(size):
                row = MODULE.make_sample(base_record() | {
                    "source_example_id": f"example-{group_index}-{row_index}",
                    "source_context_id": f"context-{group_index}",
                }, "POSITIVE_GROUNDED")
                samples.append(row)
        splits = MODULE.assign_splits(samples)
        assert {len(value) for value in splits.values()} == {2, 4}
        memberships = {}
        for split, rows in splits.items():
            for row in rows:
                memberships.setdefault(row["source_context_id"], set()).add(split)
        assert all(len(value) == 1 for value in memberships.values())
    finally:
        MODULE.SPLIT_TARGET = old_targets
