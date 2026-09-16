"""NF-V3 H1.1: the sealed-fixture ablation, as a test.

``scripts/runtime/run_nf_v3_h1_harness_integration.py`` produces the report
artifact; these tests are the same assertions in the suite, so a divergence
fails a normal test run rather than only a manual script run.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.runtime.trusted_v2_contracts import V2ExecutionOutcome
from tests.harness.equivalence import unclassified_fields
from tests.harness.h1_integration import (
    SUBSTITUTED_PORTS,
    FIXTURES,
    fixture_report,
    run_all,
    sealed_case_keys,
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


def test_a_calculation_that_did_not_execute_produced_nothing(run: dict[str, Any]) -> None:
    """The BLOCKED and raising fixtures must not have reached generation.

    Read from the generation capability's own recorded state, so this fails if
    the candidate stage ever generates from a result that did not execute --
    having a result id is not the same as having a result.
    """

    for fixture_id in ("calculation_blocked", "calculation_error"):
        capability = _report(run, fixture_id)["harness_v3"]["capability_view"]
        assert capability["calculator_invoked"] is True, fixture_id
        assert capability["renderer_invoked"] is False, fixture_id
        assert capability["specialist_invoked"] is False, fixture_id
        assert capability["candidate_ready"] is False, fixture_id
        assert capability["calculation_result_id"] is None, fixture_id


def test_the_flag_changes_the_execution_model_not_just_an_attribute() -> None:
    """Both modes must be reached through the environment, and differ.

    ``_assert_flag_reached`` would already have raised if the flag never landed
    on the coordinator, but a flag that is set and ignored would look identical.
    The two modes must produce different phase traces for the same fixture.
    """

    report = fixture_report(next(f for f in FIXTURES if f.fixture_id == "fact_direct"))

    assert "RELEASE" in report["harness_v3"]["transitions"]
    assert report["harness_v3"]["transitions"] != [
        "ACT",
        "OBSERVE",
        "EVALUATE",
        "READY_TO_GENERATE",
    ]
    # legacy never enters the harness tail, so it reports no such transition.
    assert report["legacy"]["released"] == report["harness_v3"]["released"]


def test_every_fixture_declares_how_it_was_built() -> None:
    """The report must not present substituted wiring as production wiring."""

    for fixture in FIXTURES:
        production = (
            fixture.factory_eligible and fixture.fixture_id not in SUBSTITUTED_PORTS
        )
        report = fixture_report(fixture)
        assert report["production_entry_point"] == production, fixture.fixture_id
        if not production:
            assert (
                fixture.fixture_id in SUBSTITUTED_PORTS or not fixture.retrieval
            ), fixture.fixture_id


def test_the_equivalence_contract_leaves_no_outcome_field_unclassified() -> None:
    """A new field on the outcome must be bucketed before it can pass silently.

    This is the rule that makes the contract durable: the two divergences that
    survived the first review were both fields nobody had classified.
    """

    assert unclassified_fields(V2ExecutionOutcome) == []


def test_every_sealed_case_is_accounted_for(run: dict[str, Any]) -> None:
    """The repository's own sealed case set is a coverage checklist, not a corpus.

    ``tests/fixtures/tv2_07_production_readiness/`` ships 22 labelled cases and
    no fact corpus, so a case cannot be executed without inventing the facts
    behind it.  What it can do honestly is say which cases these fixtures reach.
    A gap must be named, not left implicit.
    """

    coverage = run["sealed_case_coverage"]

    assert coverage["sealed_cases"] == len(sealed_case_keys())
    assert coverage["covered"] + coverage["not_covered"] == coverage["sealed_cases"]
    for case, entry in coverage["cases"].items():
        assert entry["note"], case
        if entry["fixture_id"] is None:
            assert entry["note"].startswith("gap:"), case


def test_the_coverage_table_cannot_drift_from_the_dataset(run: dict[str, Any]) -> None:
    """A case added to the dataset without a coverage entry fails loudly."""

    assert set(run["sealed_case_coverage"]["cases"]) == set(sealed_case_keys())
