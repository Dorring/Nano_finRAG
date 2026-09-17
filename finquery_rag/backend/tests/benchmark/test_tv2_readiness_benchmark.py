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

    assert benchmark["summary"]["task_success"] == 16
    # Rising means the runtime got closer to the readiness contract; falling
    # means it regressed.  Either way it should not move silently.
    assert benchmark["summary"]["false_release"] == 3
    assert benchmark["summary"]["over_conservative"] == 0


def test_false_releases_are_the_absence_of_conflict_detection(
    benchmark: dict[str, Any],
) -> None:
    """Name the finding rather than only counting it.

    All three are the same shape: several candidates for one slot resolve to one
    admitted fact, and the runtime releases without noticing that the other
    candidate disagreed.  That is the ``conflict`` case in the readiness contract.
    """

    false_releases = [
        case["fixture_key"]
        for case in benchmark["results"]
        if case["released"] and not case["expected_release"]
    ]

    assert false_releases == ["conflict", "multi_evidence_two", "qualitative"]


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
