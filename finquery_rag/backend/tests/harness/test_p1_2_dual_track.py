"""P1.2: the plan fixtures and the dual-track scorer.

Two things are checked here that a green run could otherwise hide.

**The fixtures carry plan fields and nothing else.**  A benchmark that injects
gold evidence, a gold calculation or a gold answer is not a RAG benchmark, and
the leak would be invisible in the results -- everything would simply look
better.  So the test asserts the *absence* of the gold's answer material from
every fixture, by key, over the whole corpus.

**The scorer separates candidate from release.**  The existing canonical scorer
promotes a value match to a binding hit, calls any release a correct comparison,
and treats any non-release as a correct reason attribution.  Each of those makes
a metric true for most rows and therefore worth nothing; each is pinned here by
the case that would expose it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_v2.contracts import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.supervisor import validate_plan_v2_01
from scripts.evaluation.build_p1_2_plan_fixtures import (
    author_plan,
    build_fixtures,
    render_fixtures,
)
from src.evaluation.p1_2_dual_track import (
    CITATION_PLACEHOLDER,
    citation_diagnostics,
    numeric_match,
    score_downstream,
    score_reachability,
)

EVAL_SET = (
    Path(__file__).resolve().parents[2]
    / "artifacts/evaluation/tv2-final-01-canonical-eval-set/canonical-eval-v1.jsonl"
)
GOLD = (
    Path(__file__).resolve().parents[2]
    / "artifacts/evaluation/tv2-final-01-canonical-eval-set/gold-evidence-v1.jsonl"
)

corpus_required = pytest.mark.skipif(
    not (EVAL_SET.exists() and GOLD.exists()),
    reason="the canonical eval set is not present in this checkout",
)

#: Keys that never belong in a fixture, because they are the answer the runtime
#: is supposed to find.  Asserted absent rather than merely unused.
_ANSWER_MATERIAL = ("fact_ids", "expected_value", "operands", "values", "expected_higher")


# --- the fixtures -------------------------------------------------------------------------------


def _synthetic(tmp_path: Path) -> tuple[Path, Path]:
    eval_path = tmp_path / "eval.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "What was Apple's Total revenue in FY2025?",
                "stratum": "factual_lookup",
                "expected_intent": "financial_lookup",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gold_path.write_text(
        json.dumps(
            {
                "id": "q1",
                "metric": "Total revenue",
                "period": "FY2025",
                "expected_outcome": "ANSWER",
                "expected_value": "391",
                "fact_ids": ["v2fact:deadbeef"],
                "operands": {"current": "391"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return eval_path, gold_path


def test_a_gold_derived_plan_is_a_plan_the_runtime_would_accept(tmp_path: Path) -> None:
    """Authored plans must be valid `SupervisorPlan`s, not shapes that look right.

    Checked with the production validator rather than by inspecting fields: the
    validator is what the coordinator runs, and a fixture that failed it would
    produce a `INVALID_PLAN` failure that the benchmark would misread as a
    retrieval outcome.
    """

    eval_path, gold_path = _synthetic(tmp_path)
    fixtures = build_fixtures(eval_path, gold_path)

    assert len(fixtures) == 1
    plan_blob = fixtures[0]["plan"]
    plan = SupervisorPlan(
        Intent(plan_blob["intent"]),
        tuple(
            RequiredSlot(
                slot["slot_id"],
                slot["metric"],
                slot["period"],
                slot["role"],
                slot["value_type"],
                slot["unit"],
            )
            for slot in plan_blob["required_slots"]
        ),
        plan_blob["operation"],
        Action(plan_blob["next_action"]),
    )
    assert validate_plan_v2_01(plan) is plan


def test_no_fixture_carries_answer_material(tmp_path: Path) -> None:
    """The leak that would make every downstream number meaningless."""

    eval_path, gold_path = _synthetic(tmp_path)
    rendered = render_fixtures(build_fixtures(eval_path, gold_path))

    for key in _ANSWER_MATERIAL:
        assert key not in rendered, f"{key} leaked into a fixture"


@corpus_required
def test_the_corpus_fixtures_are_deterministic() -> None:
    """Same inputs, same bytes -- so the frozen digest means something."""

    first = render_fixtures(build_fixtures(EVAL_SET, GOLD))
    second = render_fixtures(build_fixtures(EVAL_SET, GOLD))

    assert first == second


@corpus_required
def test_editing_a_fixture_changes_its_digest() -> None:
    """The injection check for the freeze: a mutated fixture cannot pass unseen."""

    import hashlib

    fixtures = build_fixtures(EVAL_SET, GOLD)
    before = hashlib.sha256(render_fixtures(fixtures).encode("utf-8")).hexdigest()

    fixtures[0]["plan"]["required_slots"][0]["metric"] = "mutated"
    after = hashlib.sha256(render_fixtures(fixtures).encode("utf-8")).hexdigest()

    assert before != after


@corpus_required
def test_the_corpus_fixtures_never_carry_answer_material() -> None:
    rendered = render_fixtures(build_fixtures(EVAL_SET, GOLD))

    for key in _ANSWER_MATERIAL:
        assert key not in rendered, f"{key} leaked into a fixture"


@corpus_required
def test_every_corpus_fixture_is_a_valid_plan() -> None:
    for row in build_fixtures(EVAL_SET, GOLD):
        blob = row["plan"]
        plan = SupervisorPlan(
            Intent(blob["intent"]),
            tuple(
                RequiredSlot(
                    slot["slot_id"],
                    slot["metric"],
                    slot["period"],
                    slot["role"],
                    slot["value_type"],
                    slot["unit"],
                )
                for slot in blob["required_slots"]
            ),
            blob["operation"],
            Action(blob["next_action"]),
        )
        validate_plan_v2_01(plan)


def test_an_abstention_question_does_not_get_an_abstain_plan() -> None:
    """The plan must not assert the outcome the retrieval is supposed to reach.

    A fixture that said ABSTAIN would produce `SUPERVISOR_ABSTAIN`, which is a
    different failure from the gold's `expected_failure_reason` -- and it would
    make that reason unreachable by construction, so the benchmark could never
    score it.
    """

    row = author_plan(
        "What was Amazon's Total revenue in FY2025?",
        {"expected_outcome": "ABSTENTION", "expected_failure_reason": "ENTITY_NOT_FOUND"},
        {"id": "q", "stratum": "adversarial_abstention", "expected_intent": "financial_lookup"},
    )

    assert row["plan"]["next_action"] == "RETRIEVE"
    assert row["plan"]["intent"] == "DIRECT_FACT"


# --- numeric matching ---------------------------------------------------------------------------


def test_a_percentage_answer_matches_a_fractional_gold() -> None:
    """``31.05%`` and ``0.3105`` are one quantity.

    The existing scorer strips the percent sign and compares the bare number, so
    these two never match -- which is a silent, systematic miss on every growth
    rate the model states the way an analyst writes it.
    """

    assert numeric_match("growth was 31.05%", "0.3105", tolerance="0.0001")
    assert numeric_match("growth was 0.3105", "0.3105", tolerance="0.0001")


def test_accounting_negatives_and_grouping_are_one_quantity() -> None:
    assert numeric_match("cost of revenue (8,077)", "-8077")
    assert numeric_match("revenue $ 47,627", "47627")
    assert not numeric_match("revenue $ 47,628", "47627")


def test_tolerance_is_actually_applied() -> None:
    assert numeric_match("0.3106", "0.3105", tolerance="0.0001")
    assert not numeric_match("0.3115", "0.3105", tolerance="0.0001")


def test_a_structured_calculation_value_is_read_before_the_text() -> None:
    assert numeric_match(
        "see the calculation",
        "1234",
        structured=["1,234"],
    )
    assert not numeric_match("see the calculation", "1234", structured=["1,235"])


# --- candidate quality apart from released quality -----------------------------------------------


def _row(**overrides) -> dict:
    base = {
        "id": "q1",
        "stratum": "factual_lookup",
        "question": "What was revenue?",
        "release_status": "NOT_RELEASED",
        "status": "FAIL_CLOSED",
        "answer": "",
        "reason_codes": [],
        "calculations": [],
        "reached": {},
    }
    base.update(overrides)
    return base


def _gold(outcome: str = "ANSWER", value: str = "391", **extra) -> dict:
    return {"expected_outcome": outcome, "expected_value": value, **extra}


def test_a_wrong_candidate_that_was_blocked_is_not_a_false_release() -> None:
    """The central separation: the Harness did its job and the model did not.

    Counting this as either "the model answered" or "the runtime failed" would
    be wrong in opposite directions, so it is counted as neither.
    """

    rows = [_row(answer="revenue was 999")]
    scored = score_downstream(rows, {"q1": _gold()})

    assert scored["counts"]["blocked_and_wrong"] == 1
    assert scored["counts"]["false_release"] == 0
    assert scored["counts"]["over_conservative_block"] == 0
    assert scored["counts"]["candidate_correct"] == 0


def test_a_correct_candidate_that_was_blocked_is_over_conservative() -> None:
    rows = [_row(answer="revenue was 391")]
    scored = score_downstream(rows, {"q1": _gold()})

    assert scored["counts"]["over_conservative_block"] == 1
    assert scored["counts"]["released_correct"] == 0


def test_a_wrong_released_answer_is_a_false_release() -> None:
    rows = [_row(answer="revenue was 999", release_status="RELEASED", status="ANSWER")]
    scored = score_downstream(rows, {"q1": _gold()})

    assert scored["counts"]["false_release"] == 1
    assert scored["counts"]["released_correct"] == 0


def test_a_correct_released_answer_is_the_success_case() -> None:
    rows = [_row(answer="revenue was 391", release_status="RELEASED", status="ANSWER")]
    scored = score_downstream(rows, {"q1": _gold()})

    assert scored["counts"]["released_correct"] == 1
    assert scored["answerable_correctness"] == 1.0


def test_a_blanket_fail_closed_does_not_attribute_the_gold_reason() -> None:
    """The defect the existing scorer has, pinned as a failing case here.

    ``reason_codes=["FAIL_CLOSED"]`` is what the sealed artifact carries for
    every one of its 114 fail-closed rows, and the existing scorer counts each
    as a correct reason attribution because *any* non-release is accepted.
    """

    rows = [
        _row(
            stratum="adversarial_abstention",
            reason_codes=["FAIL_CLOSED"],
        )
    ]
    scored = score_downstream(
        rows,
        {"q1": {"expected_outcome": "ABSTENTION", "expected_failure_reason": "ENTITY_NOT_FOUND"}},
    )

    assert scored["counts"]["no_answer_correct"] == 1
    assert scored["no_answer_reason_attribution"] == 0.0

    attributed = [_row(stratum="adversarial_abstention", reason_codes=["ENTITY_NOT_FOUND"])]
    scored = score_downstream(
        attributed,
        {"q1": {"expected_outcome": "ABSTENTION", "expected_failure_reason": "ENTITY_NOT_FOUND"}},
    )
    assert scored["no_answer_reason_attribution"] == 1.0


def test_releasing_on_an_abstention_question_is_a_false_release() -> None:
    rows = [
        _row(
            stratum="adversarial_abstention",
            answer="Amazon's total revenue was 600,000",
            release_status="RELEASED",
        )
    ]
    scored = score_downstream(
        rows, {"q1": {"expected_outcome": "ABSTENTION", "expected_failure_reason": "ENTITY_NOT_FOUND"}}
    )

    assert scored["counts"]["no_answer_false_release"] == 1
    assert scored["no_answer_correctness"] == 0.0


# --- reachability -------------------------------------------------------------------------------


def test_reachability_counts_stages_and_keeps_the_real_gate_verdict() -> None:
    rows = [
        _row(computed_align_status="MISMATCH", align_overridden=True, reached={}),
        _row(
            computed_align_status="ALIGNED",
            reached={"retrieval": True, "binding": True, "generation": True, "validation": True},
        ),
    ]
    scored = score_reachability(rows)

    assert scored["align_pass_rate"] == 0.5
    assert scored["align_override_count"] == 1
    assert scored["reached_generation"] == 1


def test_the_reject_distribution_is_kept_by_reason() -> None:
    rows = [
        _row(reason_codes=["QUERY_PLAN_SEMANTIC_MISMATCH"]),
        _row(reason_codes=["QUERY_PLAN_SEMANTIC_MISMATCH"]),
        _row(reason_codes=["MISSING_SLOT"]),
    ]
    scored = score_reachability(rows)

    assert scored["reject_reason_distribution"] == {
        "MISSING_SLOT": 1,
        "QUERY_PLAN_SEMANTIC_MISMATCH": 2,
    }


# --- citation diagnostics -----------------------------------------------------------------------


def test_the_placeholder_is_measured_rather_than_repaired() -> None:
    rows = [
        _row(answer=f"risk factors: [E1] {CITATION_PLACEHOLDER}, supply chain [E1]"),
        _row(answer="risk factors: [E1], supply chain [E2]"),
    ]
    diagnostics = citation_diagnostics(rows)

    assert diagnostics["literal_citation_placeholder_rate"] == 0.5
    assert diagnostics["malformed_citation_rate"] == 0.0


def test_a_handle_beyond_the_pack_is_counted() -> None:
    rows = [
        _row(answer="revenue [E7]", pack_handle_count=3),
        _row(answer="revenue [E2]", pack_handle_count=3),
    ]
    diagnostics = citation_diagnostics(rows)

    assert diagnostics["cited_handle_not_in_pack_rate"] == 0.5
    assert diagnostics["cited_handle_not_in_pack_denominator"] == 2


def test_token_columns_stay_separate() -> None:
    """Three counts, three questions -- and they must not be averaged together.

    ``context_payload_tokens`` is what the compiler measured and the binding's
    exact counter reported.  ``actual_model_input_tokens`` is what the provider
    fed the model, which is that payload plus the renderer's static text plus
    the four frame tokens.  They are close, they are never equal, and a report
    that used one for the other would be wrong by an amount nobody could see.
    """

    rows = [
        _row(tokens={"context_payload_tokens": 240, "actual_model_input_tokens": 263, "actual_output_tokens": 22}),
        _row(tokens={"context_payload_tokens": 210, "actual_model_input_tokens": 231, "actual_output_tokens": 33}),
    ]
    scored = score_downstream(rows, {"q1": _gold()})

    assert scored["tokens"]["context_payload_tokens"] == {
        "count": 2,
        "mean": 225.0,
        "p50": 240,
        "max": 240,
    }
    assert scored["tokens"]["actual_model_input_tokens"]["mean"] == 247.0
    assert scored["tokens"]["actual_output_tokens"]["max"] == 33
    assert scored["tokens"]["provider_frame_tokens"] == 4
