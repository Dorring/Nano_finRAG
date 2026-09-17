"""The TV2-07 readiness benchmark, as a test.

This is H2A-0: every one of the repository's 22 labelled production-readiness
cases is executable, scored against its own label, and the cases that do not yet
meet their label are named rather than skipped.

It is deliberately not a pass/fail gate on the runtime.  Six cases do not meet
their label today and three of those are *false releases* -- the runtime
releases where the readiness contract says it should abstain.  That is a finding
about the runtime, and a benchmark that skipped it would be reporting on itself
rather than on the system.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.benchmark.tv2_readiness_cases import FIXTURE_SPECS, KNOWN_MISMATCHES, cases
from tests.benchmark.tv2_readiness_scoring import run_benchmark


@pytest.fixture(scope="module")
def benchmark() -> dict[str, Any]:
    return run_benchmark()


def test_every_labelled_case_has_an_executable_fixture() -> None:
    """H1.1 could only report 10 of 22 reachable, because there was nothing to run.

    ``cases()`` raises if a case has no spec, so this failing means the sealed
    set and the labels have drifted apart.
    """

    assert len(cases()) == 22
    assert len(FIXTURE_SPECS) == 22


def test_no_case_fails_for_a_reason_nobody_has_looked_at(benchmark: dict[str, Any]) -> None:
    """A mismatch must be a recorded finding, not a surprise.

    The first version of this registry was written by inspecting the labels and
    five of its entries were wrong.  It is now derived from the run, and this
    assertion is what keeps it that way.
    """

    assert benchmark["summary"]["unexplained_mismatch"] == 0
    mismatched = {
        case["fixture_key"] for case in benchmark["results"] if not case["matches_label"]
    }
    assert mismatched == set(KNOWN_MISMATCHES)


def test_the_recorded_mismatches_are_still_the_recorded_ones(
    benchmark: dict[str, Any],
) -> None:
    """Pin what the runtime does, so a change in either direction is visible."""

    assert benchmark["summary"]["task_success"] == 17
    # The H2A-1E conflict gate took this from 3 to 0.  It is the headline of
    # that change and the one number that must not creep back up.
    assert benchmark["summary"]["false_release"] == 0
    # One case has become over-conservative: multi_evidence asks for two
    # corroborating facts for one slot, and the fixture supplies two that
    # disagree, which the gate now refuses.  Counting it here rather than
    # adjusting the fixture keeps the disagreement visible.
    assert benchmark["summary"]["over_conservative"] == 1


def test_no_case_releases_where_the_readiness_contract_says_abstain(
    benchmark: dict[str, Any],
) -> None:
    """The defect this phase fixed, stated as an invariant.

    Three cases used to release here -- conflict, multi_evidence_two and
    qualitative -- all the same shape: several candidates for one slot bound to
    one of them and the disagreement was never examined.  This is the assertion
    that must not go back to listing them.
    """

    false_releases = [
        case["fixture_key"]
        for case in benchmark["results"]
        if case["released"] and not case["expected_release"]
    ]

    assert false_releases == []


def test_the_two_kinds_of_mismatch_are_not_conflated(benchmark: dict[str, Any]) -> None:
    """A vocabulary difference is not a runtime failure.

    Three of the remaining mismatches are cases where the runtime now behaves
    correctly and the label says it in different words. Counting them against
    the runtime would hide the two that are real capability gaps.
    """

    from tests.benchmark.tv2_readiness_cases import (
        BEHAVIOUR_MATCHES_LABEL,
        CAPABILITY_GAPS,
    )

    assert set(BEHAVIOUR_MATCHES_LABEL) & set(CAPABILITY_GAPS) == set()
    assert len(BEHAVIOUR_MATCHES_LABEL) == 3
    assert len(CAPABILITY_GAPS) == 2


def test_token_counts_are_never_faked(benchmark: dict[str, Any]) -> None:
    """No tokenizer means null, not zero.

    The H2A ablation compares token figures across two configurations.  A
    character count or a word-count approximation wearing the word "tokens"
    would make that comparison meaningless, and the repo already contains one
    such number (``ContextBudgetManager.count_tokens`` is a 1.15x word count
    reported as "context tokens").
    """

    if not benchmark["summary"]["token_counts_available"]:
        assert benchmark["summary"]["evidence_tokens"] is None
        assert benchmark["summary"]["answer_tokens"] is None
        assert benchmark["tokenizer"] == "unavailable"
    else:
        assert isinstance(benchmark["summary"]["evidence_tokens"], int)
