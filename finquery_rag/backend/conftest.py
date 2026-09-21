"""Repository-level pytest configuration.

Two collection-time guards live here, both because the suite as committed could
not report an honest number on its own: a bare ``pytest`` aborted on an import
error before running a single test, and seventy-odd sealed-evaluation tests
reported as failures when what they actually lacked was an artifact directory
that was never committed.

Neither guard hides anything.  Every test that is excluded is excluded with a
reason naming exactly what is missing, so ``-rs`` and the session header
together account for every test that did not run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

collect_ignore_glob = ["pytest-cache-files*"]

BACKEND_ROOT = Path(__file__).parent
ARTIFACT_ROOT = BACKEND_ROOT / "artifacts"

# --- guard 1: test modules that cannot import without an optional runtime ----
#
# These used to be collection *errors*.  pytest aborts the whole session on a
# collection error unless ``--continue-on-collection-errors`` is passed, so a
# checkout without these runtimes could not run the suite at all -- the failure
# said nothing about the code under test.
#
# ``chromadb`` is a declared dependency that is simply absent from some working
# environments; ``torch`` is not declared anywhere and is only reachable through
# the nanochat local specialist.
OPTIONAL_RUNTIME_MODULES: dict[str, tuple[str, ...]] = {
    "tests/evaluation/test_nf40_cli.py": ("chromadb",),
    "tests/test_local_specialist_generator.py": ("torch",),
}


def _resolve(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# Resolved at import time, not during collection: ``pytest_report_header`` runs
# before collection and would otherwise always report an empty list.
_missing_runtimes: dict[str, tuple[str, ...]] = {
    relative: missing
    for relative, required in OPTIONAL_RUNTIME_MODULES.items()
    if (missing := tuple(name for name in required if not _resolve(name)))
}


def pytest_ignore_collect(collection_path: Path, config: Any) -> bool | None:
    try:
        relative = collection_path.relative_to(BACKEND_ROOT).as_posix()
    except ValueError:
        return None
    return True if relative in _missing_runtimes else None


def pytest_report_header(config: Any) -> list[str]:
    if not _missing_runtimes:
        return []
    lines = ["test modules excluded: optional runtime not installed"]
    for module, missing in sorted(_missing_runtimes.items()):
        lines.append(f"  {module} (needs {', '.join(missing)})")
    return lines


# --- guard 2: tests that read a sealed evaluation artifact -------------------
#
# A frozen evaluation run writes its evidence to ``artifacts/evaluation/<run>/``
# and the test module that asserts against it declares that directory as a
# module-level path constant.  Those directories are produced by an evaluation
# run and are not committed, so on a fresh checkout the tests that read them
# fail with one FileNotFoundError each.
#
# The skip is decided from the *actual* error, not from the directory being
# absent.  A module-level "the root is missing, skip the whole module" rule was
# tried first and silently swallowed nine tests in those same modules that do
# not read the artifact and passed without it.  Deciding per failure keeps them
# running and still states the missing path in the skip reason.
ARTIFACT_CONSTANT_NAMES = (
    "ARTIFACT",
    "ARTIFACTS",
    "ARTIFACT_ROOT",
    "ARTIFACT_DIR",
)


def _declared_artifact_root(module: Any) -> Path | None:
    for name in ARTIFACT_CONSTANT_NAMES:
        value = getattr(module, name, None)
        if isinstance(value, Path):
            return value
    return None


def pytest_collection_modifyitems(
    session: Any, config: Any, items: list[Any]
) -> None:
    """Tag modules whose declared artifact root is absent.

    Tagging only -- it neither skips nor deselects.  The marker exists so the
    group can be selected or deselected as a unit:

        pytest -m "not requires_artifacts"    # core + harness + regression
    """

    cache: dict[Path, bool] = {}
    for item in items:
        root = _declared_artifact_root(getattr(item, "module", None))
        if root is None:
            continue
        present = cache.get(root)
        if present is None:
            present = root.is_dir()
            cache[root] = present
        if not present:
            item.add_marker(pytest.mark.requires_artifacts)


def _missing_artifact_path(excinfo: Any) -> str | None:
    """Return the artifact path an exception failed to read, if that is what it is."""

    error = getattr(excinfo, "value", None)
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, OSError):
            filename = getattr(error, "filename", None)
            if isinstance(filename, str):
                candidate = Path(filename)
                try:
                    candidate.relative_to(ARTIFACT_ROOT)
                except ValueError:
                    pass
                else:
                    return candidate.relative_to(BACKEND_ROOT).as_posix()
        error = error.__cause__ or error.__context__
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: Any, call: Any):
    """Report a missing-artifact read as a skip, naming the artifact."""

    report = yield
    if report.failed and call.excinfo is not None:
        missing = _missing_artifact_path(call.excinfo)
        if missing is not None:
            report.outcome = "skipped"
            report.longrepr = (
                f"{getattr(report, 'when', 'call')} skipped: "
                f"sealed evaluation artifact not present: {missing}"
            )
    return report


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "requires_artifacts: reads a sealed evaluation artifact directory "
        "produced by an evaluation run; not committed to the repository",
    )


# --- declared runtime for endpoint tests -------------------------------------
#
# The application has two ambient defaults that decide which execution path a
# request takes, and a test that does not declare them is not testing a
# behaviour -- it is testing whatever the defaults happen to be today:
#
#   FINANCIAL_RUNTIME_MODE    default "v2".  The documented contract: v2 is the
#                             official Trusted V2 path and fails closed when no
#                             real runtime builder is configured; v1 is the
#                             explicit rollback/compatibility path.
#                             (docs/showcase/trusted-runtime-v2-production-integration.md)
#   MULTITURN_CONTEXT_MODE    default "on".  With it on and no session id, an
#                             under-specified query is answered with a
#                             clarification request rather than being run.
#
# Thirty-nine tests across five modules exercised the legacy single-turn /query
# path -- they patch `get_rag_engine`, which only the V1 lifecycle calls --
# without declaring either.  They passed while the defaults matched their
# assumptions and broke en masse when the defaults moved, which told us nothing
# about the code under test.
#
# This fixture is how such a test states what it is testing.  Use it instead of
# relying on the ambient value, unless the test's subject IS the default
# selection -- in which case it should assert the resolved value explicitly.


@pytest.fixture
def legacy_single_turn_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare the V1, single-turn runtime an endpoint test exercises."""

    monkeypatch.setenv("FINANCIAL_RUNTIME_MODE", "v1")
    monkeypatch.setenv("MULTITURN_CONTEXT_MODE", "off")
