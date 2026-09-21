"""NF-V3 H1.1: the sealed-fixture ablation, as a test.

``scripts/runtime/run_nf_v3_h1_harness_integration.py`` produces the report
artifact; these tests are the same assertions in the suite, so a divergence
fails a normal test run rather than only a manual script run.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from src.runtime.trusted_v2_contracts import V2ExecutionOutcome
from tests.harness.equivalence import (
    CONTRACT_FIELDS,
    unclassified_fields,
    unknown_contract_fields,
)
from tests.harness.h1_integration import (
    substituted_ports,
    FIXTURES,
    H1Fixture,
    run_all,
    run_fixture,
    sealed_case_keys,
    sealed_digest,
)
from src.runtime.harness_runtime_mode import AgentRuntimeMode

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


def test_the_successful_calculation_traces_the_full_harness_path(
    run: dict[str, Any],
) -> None:
    """The end-to-end trace claim, on real wiring rather than a composed loop.

    A successful calculation query must be *seen* to reach CALCULATE and then
    RELEASE.  Anything less and the closed loop is a claim about a code path
    that no fixture walks.
    """

    report = _report(run, "calculation_growth_rate")

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


def test_the_flag_changes_the_execution_model_not_just_an_attribute(
    run: dict[str, Any],
) -> None:
    """Both modes must be reached through the environment, and genuinely differ.

    ``_assert_flag_reached`` raises if the flag never landed on the coordinator,
    but a flag that is set and then ignored would look identical.  The same
    fixture must produce *different recorded work* in the two modes while
    reaching the same verdict -- that is what makes this an ablation rather than
    a rename.
    """

    report = _report(run, "fact_direct")

    assert report["legacy"]["turns"] == ["SEMANTIC_RETRIEVAL"]
    assert report["harness_v3"]["turns"] == [
        "SEMANTIC_RETRIEVAL",
        "GENERATE",
        "VERIFY",
    ]
    assert "RELEASE" in report["harness_v3"]["transitions"]
    assert report["harness_v3"]["status"] == "READY_FOR_RELEASE"
    assert report["legacy"]["released"] == report["harness_v3"]["released"] is True


def test_every_fixture_declares_how_it_was_built(run: dict[str, Any]) -> None:
    """The report must not present substituted wiring as production wiring."""

    by_id = {fixture.fixture_id: fixture for fixture in FIXTURES}
    assert {report["fixture_id"] for report in run["fixtures"]} == set(by_id)
    for report in run["fixtures"]:
        fixture = by_id[report["fixture_id"]]
        production = (
            fixture.factory_eligible and substituted_ports(fixture) is None
        )
        assert report["production_entry_point"] == production, fixture.fixture_id
        if not production:
            assert (
                substituted_ports(fixture) is not None or not fixture.retrieval
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


def test_every_declared_fixture_field_is_sealed() -> None:
    """A fixture field that is not in ``spec()`` is not covered by the digest.

    The first version of ``spec()`` omitted every ``expect_*`` field, so the
    sealed digest did not move when a fixture's expectation did: it sealed the
    inputs while the report described it as sealing the fixture.
    """

    declared = {field.name for field in dataclasses.fields(H1Fixture)}
    for fixture in FIXTURES:
        assert set(fixture.spec()) == declared, fixture.fixture_id


def test_the_sealed_digest_moves_when_a_fixture_moves() -> None:
    """Keeping the digest stable is not the same as the digest working."""

    base = sealed_digest()
    assert base == sealed_digest()

    for field, value in (
        ("query", "A different question?"),
        ("expect_released", not FIXTURES[0].expect_released),
        ("expect_status", "EXECUTION_ERROR"),
        ("factory_eligible", False),
    ):
        mutated = (dataclasses.replace(FIXTURES[0], **{field: value}), *FIXTURES[1:])
        assert sealed_digest(mutated) != base, field


def test_the_calculation_marker_is_set_only_by_the_harness_mode() -> None:
    """The contract strips this key, so nothing else would notice it vanish.

    ``calculation_in_harness`` is classified harness-only, which is correct: it
    records that calculation ran as a loop phase, and ``legacy`` has no such
    phase.  Stripping it also means the fixture suite would stay green if the
    coordinator stopped writing it -- so it is asserted here, directly, the same
    way the composed suite asserts it.
    """

    fixture = next(f for f in FIXTURES if f.fixture_id == "calculation_growth_rate")

    harness = run_fixture(fixture, AgentRuntimeMode.HARNESS_V3)
    legacy = run_fixture(fixture, AgentRuntimeMode.LEGACY)

    assert harness.runtime_metadata.get("calculation_in_harness") is True
    assert "calculation_in_harness" not in legacy.runtime_metadata


def test_no_fixture_runs_past_the_bounded_turn_limit(run: dict[str, Any]) -> None:
    """The script gates on this, so the suite must assert it too."""

    assert run["summary"]["infinite_loop"] == 0


def test_the_contract_is_not_vacuous(run: dict[str, Any]) -> None:
    """'The modes agree' and 'the comparison collapsed' must be distinguishable.

    Replacing ``canonicalize_decision_result`` with a function returning ``{}``
    leaves every fixture comparison green, because two empty payloads are equal.
    These assertions pin the shape the comparison depends on.
    """

    from tests.harness.equivalence import canonicalize_decision_result

    outcome = run_fixture(
        next(f for f in FIXTURES if f.fixture_id == "fact_direct"),
        AgentRuntimeMode.HARNESS_V3,
    )
    payload = canonicalize_decision_result(outcome)

    assert set(CONTRACT_FIELDS) <= set(payload)
    assert payload["status"] == "READY_FOR_RELEASE"
    assert isinstance(payload["runtime_metadata"], dict) and payload["runtime_metadata"]
    assert payload["reason_codes"]


def test_the_contract_names_only_real_outcome_fields() -> None:
    """A typo in the contract used to compare ``None == None`` and pass."""

    assert unknown_contract_fields(V2ExecutionOutcome) == []


def test_a_coverage_entry_for_a_case_that_no_longer_exists_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reverse direction of the drift check."""

    from tests.harness import h1_integration

    monkeypatch.setitem(
        h1_integration.SEALED_CASE_COVERAGE, "a-case-not-in-the-dataset", (None, "gap: stale")
    )
    with pytest.raises(AssertionError, match="unknown sealed cases"):
        h1_integration.sealed_coverage_report()


def test_every_contract_name_reaches_the_payload(
    run: dict[str, Any],
) -> None:
    """Every declared name must actually be compared, not merely declared.

    ``runtime_metadata`` is named by ``DECISION_BEARING_METADATA`` rather than by
    ``DECISION_BEARING_FIELDS``, and for a while the payload was built from the
    second alone -- so the field carrying ``release_decision``,
    ``validation_status``, ``failed_checks`` and the terminal state was declared
    decision-bearing and never compared.  Every differential test stayed green.
    """

    from tests.harness.equivalence import canonicalize_decision_result

    payload = canonicalize_decision_result(
        run_fixture(FIXTURES[0], AgentRuntimeMode.HARNESS_V3)
    )

    for name in CONTRACT_FIELDS:
        assert name in payload, name
    assert payload["runtime_metadata"].get("release_decision")


def test_a_collapsed_contract_is_refused_rather_than_reported_as_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'The modes agree' and 'nothing was compared' must not look the same.

    Letting the payload be empty makes every fixture equivalent -- the two
    outcomes have no differing fields because they have no fields.  The refusal
    lives on the comparison path, so every caller gets it.

    Note what the first check could *not* do: comparing the payload against
    ``CONTRACT_FIELDS`` is circular, because the builder iterates that same list,
    so removing a name removes it from both sides.  That check agreed with
    itself for as long as it existed.  The two assertions below are the pair
    that works -- the builder-collapsed case is caught by coverage of the
    outcome's own fields, and a truncated contract by the type-level check.
    """

    from tests.harness import equivalence

    outcome = run_fixture(FIXTURES[0], AgentRuntimeMode.HARNESS_V3)
    monkeypatch.setattr(equivalence, "canonicalize_decision_result", lambda item: {})

    with pytest.raises(AssertionError, match="missing fields the outcome carries"):
        equivalence.decision_differences(outcome, outcome)
