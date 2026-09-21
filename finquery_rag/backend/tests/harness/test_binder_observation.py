"""The binder's own vocabulary, captured instead of inferred.

``EVIDENCE_CONFLICT`` is one reason code over three unrelated mechanisms --
no-consensus among admissible candidates, a provider ``AMBIGUOUS`` verdict, and
a provider ``INVALID`` verdict, which is a schema failure rather than a conflict.
A count that merges them cannot say why binding failed, so before anything is
said about binding, the split has to be recorded.

These tests are written against the binding statuses the capability actually
produces, and they pin the two ways the capture could quietly lie: reporting an
earlier round than the one the outcome reflects, and losing the row entirely
when the capability cannot be read.
"""

from __future__ import annotations

from typing import Any

from scripts.evaluation.run_p1_2_dual_track_benchmark import _binder_observation


class _Capability:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self._snapshot = snapshot

    def trace_snapshot(self) -> dict[str, Any]:
        return self._snapshot


class _Coordinator:
    def __init__(self, capability: Any) -> None:
        self.capabilities = type("Caps", (), {"evidence_evaluator": capability})()


class _Runtime:
    def __init__(self, capability: Any) -> None:
        self.coordinator = _Coordinator(capability)


def _round(
    status: str,
    *,
    bound: list[str] | None = None,
    missing: list[str] | None = None,
    ambiguous: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "bound_slot_ids": bound or [],
        "missing_slot_ids": missing or [],
        "ambiguous_slot_ids": ambiguous or [],
    }


def _snapshot(rounds: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"binder_rounds": rounds, **extra}


def test_the_three_conflict_mechanisms_are_distinguishable() -> None:
    """Each mechanism the reason code merges reports its own status.

    If these collapsed to one value the split would be cosmetic, and a repair
    aimed at "conflicts" would still be aimed at all three.
    """

    no_consensus = _binder_observation(
        _Runtime(_Capability(_snapshot([_round("OK", ambiguous=["s1"])])))
    )
    ambiguous = _binder_observation(
        _Runtime(_Capability(_snapshot([_round("AMBIGUOUS", ambiguous=["s1"])])))
    )
    invalid = _binder_observation(
        _Runtime(_Capability(_snapshot([_round("INVALID", missing=["s1"])])))
    )

    statuses = {
        no_consensus["binder_final_status"],
        ambiguous["binder_final_status"],
        invalid["binder_final_status"],
    }
    assert len(statuses) == 3
    assert invalid["binder_final_status"] == "INVALID"


def test_the_reported_round_is_the_last_one() -> None:
    """The outcome reflects the final round, so that is the one to record.

    A repair that recovers a slot changes the status between rounds; reading
    round one would report the failure the run already moved past.
    """

    observation = _binder_observation(
        _Runtime(
            _Capability(
                _snapshot(
                    [
                        _round("MISSING", missing=["s1", "s2"]),
                        _round("OK", bound=["s1", "s2"]),
                    ]
                )
            )
        )
    )

    assert observation["binder_final_status"] == "OK"
    assert observation["binder_final_bound_slot_ids"] == ["s1", "s2"]
    assert observation["binder_final_missing_slot_ids"] == []
    assert observation["binder_round_count"] == 2
    # Every round is still reported, so a run that needed repair is visible as
    # such rather than looking like a first-round success.
    assert observation["binder_round_statuses"] == ["MISSING", "OK"]


def test_a_slot_that_never_bound_survives_into_the_observation() -> None:
    """``missing_slot_ids`` is what a targeted repair would act on."""

    observation = _binder_observation(
        _Runtime(
            _Capability(_snapshot([_round("MISSING", bound=["s1"], missing=["s2"])]))
        )
    )

    assert observation["binder_final_bound_slot_ids"] == ["s1"]
    assert observation["binder_final_missing_slot_ids"] == ["s2"]


def test_a_capability_that_cannot_be_read_does_not_cost_the_row() -> None:
    """Observation is a lens, not a dependency.

    A benchmark row is one run of an expensive pipeline; losing it because the
    diagnostic could not be read would trade the measurement for the metric.
    """

    class _Broken:
        def trace_snapshot(self) -> dict[str, Any]:
            raise RuntimeError("capability did not expose a snapshot")

    assert _binder_observation(_Runtime(_Broken())) == {}


def test_no_runtime_observes_nothing() -> None:
    """The error path builds a row with no runtime; it still must be emitted."""

    assert _binder_observation(None) == {}


def test_an_empty_trace_reports_no_round() -> None:
    """A run that never reached the binder has no status, not a default one."""

    observation = _binder_observation(_Runtime(_Capability(_snapshot([]))))

    assert observation["binder_final_status"] is None
    assert observation["binder_round_count"] == 0
