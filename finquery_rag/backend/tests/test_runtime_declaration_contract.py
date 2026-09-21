"""NF-V3 H1.2: runtime-sensitive tests must declare the runtime they exercise.

Thirty-nine tests across five modules broke together when an ambient default
moved. They patched `src.main.get_rag_engine` -- which only the V1 lifecycle
calls -- and asserted V1 response shapes, while inheriting whichever runtime mode
and conversation mode happened to be the default. Nothing about the code under
test had changed, so the failure said nothing about the code under test, and 34
of them passed again the moment the mode was supplied.

The defaults themselves are correct and are pinned deliberately elsewhere:

- ``FINANCIAL_RUNTIME_MODE`` defaults to ``v2``
  (``test_runtime_router_shadow.py::test_mode_defaults_to_v2_and_accepts_explicit_runtime_modes``).
  ``v2`` is the official Trusted V2 path and fails closed without a configured
  builder; ``v1`` is the explicit rollback/compatibility path. That is a product
  contract, not a test convenience:
  ``docs/showcase/trusted-runtime-v2-production-integration.md``.
- ``MULTITURN_CONTEXT_MODE`` defaults to ``on``
  (``tests/conversation/test_shadow_service.py``).

What was missing was the other half of the contract: a test that drives the real
application must say which runtime it is driving. These tests are the tripwire
that keeps that true.
"""

from __future__ import annotations

from pathlib import Path

TESTS_ROOT = Path(__file__).parent

#: Declaring the ambient runtime, by either supported route: the shared fixture,
#: or an explicit environment reference the module sets up itself.
_DECLARATIONS = (
    "legacy_single_turn_endpoint",
    "FINANCIAL_RUNTIME_MODE",
    "MULTITURN_CONTEXT_MODE",
)

#: Driving the real HTTP surface -- the only path where the ambient runtime
#: decides which lifecycle runs.
_ENDPOINT_CALLS = ('"/query"', '"/query/stream"', "'/query'", "'/query/stream'")

_ENGINE_PATCH = "get_rag_engine"


def _test_modules() -> list[Path]:
    return sorted(
        path
        for path in TESTS_ROOT.rglob("test_*.py")
        if "__pycache__" not in path.parts
    )


def test_every_endpoint_test_module_declares_its_runtime() -> None:
    """Patching the V1 engine and calling /query means you are testing V1.

    Say so. Otherwise the module is testing the current default, and the day the
    default changes it fails for a reason that has nothing to do with it.
    """

    offenders = []
    for path in _test_modules():
        text = path.read_text(encoding="utf-8")
        if _ENGINE_PATCH not in text:
            continue
        if not any(call in text for call in _ENDPOINT_CALLS):
            continue
        if not any(token in text for token in _DECLARATIONS):
            offenders.append(path.relative_to(TESTS_ROOT).as_posix())

    assert not offenders, (
        "these modules drive a real endpoint with a patched V1 engine but never "
        f"declared a runtime mode: {offenders}. Use the "
        "`legacy_single_turn_endpoint` fixture from conftest.py, or set the mode "
        "explicitly."
    )


def test_the_declaration_fixture_matches_the_documented_defaults() -> None:
    """The fixture's value is only meaningful next to the default it overrides.

    If either default moves, this fails and someone has to decide whether the
    fixture still describes what those tests exercise -- which is the decision
    we want to be forced into, rather than thirty-nine silent failures.
    """

    from src.conversation.config import resolve_multiturn_context_mode
    from src.runtime.shadow_contracts import resolve_financial_runtime_mode

    assert resolve_financial_runtime_mode(environ={}) == "v2"
    assert resolve_multiturn_context_mode(environ={}) == "on"

    conftest = (TESTS_ROOT.parent / "conftest.py").read_text(encoding="utf-8")
    assert "def legacy_single_turn_endpoint" in conftest, (
        "the declaration fixture was removed"
    )
    body = conftest.split("def legacy_single_turn_endpoint", 1)[1]
    assert 'setenv("FINANCIAL_RUNTIME_MODE", "v1")' in body
    assert 'setenv("MULTITURN_CONTEXT_MODE", "off")' in body
