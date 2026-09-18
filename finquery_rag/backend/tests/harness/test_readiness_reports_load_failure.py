"""Readiness must move when the service stops being able to serve.

``/readyz`` validated *configuration* -- paths, env, the checkpoint's digest --
and reported ready on a host where the specialist could not be loaded at all,
because another user's job had filled the GPU.  Every query returned 500 for
forty minutes and the probe never moved.  A readiness signal that does not move
when the service stops working is not a readiness signal.

What this does not yet do is *proactively* load the model at startup: the check
is reactive, and reports the last build attempt's outcome.  A service that has
not been asked to load its model is not claiming that it can -- which is why a
cold service still reports ready, and why the proactive variant (build the
resources during readiness, at the cost of a slower and heavier start) is a
deployment decision rather than an obvious win.
"""

from __future__ import annotations

import pytest

from src.runtime import trusted_v2_production as production


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch):
    """Do not disturb the module's real cache or failure record."""

    monkeypatch.setattr(production, "_RESOURCE_CACHE", {})
    monkeypatch.setattr(production, "_RESOURCE_LOAD_FAILURE", None)
    monkeypatch.setattr(production, "_configuration_fingerprint", lambda environ: "k")
    yield


def test_a_service_that_has_never_loaded_reports_no_failure() -> None:
    """Cold is not the same as broken, and the report says which it is."""

    assert production.last_resource_load_failure() is None


def test_a_failed_build_is_recorded_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(environ: object) -> None:
        raise RuntimeError("CUDA out of memory: the configured device is full")

    monkeypatch.setattr(production, "_load_resources", boom)

    with pytest.raises(RuntimeError):
        production._cached_resources({"ANY": "1"})

    recorded = production.last_resource_load_failure()
    assert recorded is not None
    assert "RuntimeError" in recorded
    assert "the configured device is full" in recorded


def test_a_later_success_clears_the_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """A recovered service must stop reporting the old failure.

    A record that only ever goes one way is a sticky alarm, and operators learn
    to ignore those.
    """

    def boom(environ: object) -> None:
        raise RuntimeError("temporarily full")

    monkeypatch.setattr(production, "_load_resources", boom)
    with pytest.raises(RuntimeError):
        production._cached_resources({"ANY": "1"})
    assert production.last_resource_load_failure() is not None

    monkeypatch.setattr(production, "_load_resources", lambda environ: "resources")
    assert production._cached_resources({"ANY": "1"}) == "resources"
    assert production.last_resource_load_failure() is None


def test_a_cached_service_is_not_reloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def load(environ: object) -> str:
        calls.append(environ)
        return "resources"

    monkeypatch.setattr(production, "_load_resources", load)

    assert production._cached_resources({"ANY": "1"}) == "resources"
    assert production._cached_resources({"ANY": "1"}) == "resources"
    assert len(calls) == 1
