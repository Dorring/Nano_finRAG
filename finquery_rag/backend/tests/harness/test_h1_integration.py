"""NF-V3 H1.1: the sealed-fixture ablation, as a test.

``scripts/runtime/run_nf_v3_h1_harness_integration.py`` produces the report
artifact; these tests are the same assertions in the suite, so a divergence
fails a normal test run rather than only a manual script run.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.harness.h1_integration import (
    FIXTURES,
    fixture_report,
    run_all,
    sealed_digest,
)

FIXTURE_IDS = sorted(fixture.fixture_id for fixture in FIXTURES)


@pytest.fixture(scope="module")
def run() -> dict[str, Any]:
    return run_all()


def _report(run: dict[str, Any], fixture_id: str) -> dict[str, Any]:
    return next(
        item for item in run["fixtures"] if item["fixture_id"] == fixture_id
    )


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_modes_reach_the_same_decision(run: dict[str, Any], fixture_id: str) -> None:
    report = _report(run, fixture_id)
    assert report["decision_equivalent"], report["differences"]


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_fixture_meets_its_own_expectation(
    run: dict[str, Any], fixture_id: str
) -> None:
    report = _report(run, fixture_id)
    assert report["expectation_failures"] == []


def test_no_fixture_releases_where_the_other_mode_does_not(run: dict[str, Any]) -> None:
    assert run["summary"]["release_bypass"] == 0


def test_a_blocked_or_failed_calculation_never_releases(run: dict[str, Any]) -> None:
    assert run["summary"]["false_calculation_release"] == 0


def test_every_fixture_ran(run: dict[str, Any]) -> None:
    assert run["summary"]["decision_equivalence"] == len(FIXTURES)
    assert run["fixture_count"] == len(FIXTURES)


def test_the_successful_calculation_traces_the_full_harness_path() -> None:
    """The end-to-end trace claim, on real wiring rather than a composed loop.

    A successful calculation query must be *seen* to reach CALCULATE and then
    RELEASE.  Anything less and the closed loop is a claim about a code path
    that no fixture walks.
    """

    report = fixture_report(next(f for f in FIXTURES if f.fixture_id == "calculation_growth_rate"))

    assert report["harness_v3"]["transitions"] == [
        "ACT",
        "OBSERVE",
        "EVALUATE",
        "CALCULATE",
        "READY_TO_GENERATE",
        "GENERATE",
        "VERIFY",
        "RELEASE",
    ]
    assert report["harness_v3"]["released"] is True
    # ``legacy`` never enters CALCULATE: the calculator stays outside its loop.
    assert "CALCULATE" not in report["legacy"]["turns"]


def test_the_sealed_digest_is_stable_and_recomputed(run: dict[str, Any]) -> None:
    """A changed fixture set must change the digest the report claims."""

    assert run["sealed_digest"] == sealed_digest()
    assert len(run["sealed_digest"]) == 64


def test_rejected_validator_cannot_release_in_either_mode(run: dict[str, Any]) -> None:
    report = _report(run, "validator_rejection")

    assert report["legacy"]["released"] is False
    assert report["harness_v3"]["released"] is False
    assert "UNBOUND_CITATION_METADATA" in report["harness_v3"]["reason_codes"]
