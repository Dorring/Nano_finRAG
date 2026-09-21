"""The layering invariant the shadow architecture rests on.

``rag_v2`` is the V2 shadow architecture and ``src`` is the application layer
that consumes it.  The dependency runs one way: ``src/runtime`` imports
``rag_v2``, and ``rag_v2`` imports ``src`` nowhere.  That is what lets the V2
contracts be reasoned about, and tested, without the application layer present.

It was an invariant with no test until H2A-3B1 added a package that could
plausibly have broken it.  ``rag_v2/context`` needs an authoritative calculation
payload, and the obvious way to be sure you have one is
``isinstance(value, CalculationResult)`` -- but ``CalculationResult`` lives in
``src/domain``.  The compiler duck-types instead.  That decision is invisible
without this file, and a future contributor reaching for the import would get no
signal at all.

The check is on import statements, not on runtime behaviour: a conditional
import inside a function is the same dependency and is caught too.
"""

from __future__ import annotations

import ast
import pathlib

RAG_V2 = pathlib.Path("rag_v2")


def _imported_roots(path: pathlib.Path) -> set[str]:
    """Every top-level module name this file imports, at any nesting depth."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # ``level`` is non-zero for a relative import, which cannot name
            # another package and is therefore never a layering violation.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_rag_v2_never_imports_the_application_layer() -> None:
    """The one-way dependency, asserted over every module in the package."""

    offenders = {
        path.as_posix(): sorted(_imported_roots(path) & {"src"})
        for path in sorted(RAG_V2.rglob("*.py"))
        if "src" in _imported_roots(path)
    }

    assert offenders == {}, (
        "rag_v2 must not import src; the dependency runs src -> rag_v2 only.\n"
        f"{offenders}"
    )


def test_the_check_can_actually_fail(tmp_path: pathlib.Path) -> None:
    """A guard that cannot fail is not a guard.

    Exercised against the real helper with a synthetic module, so a refactor
    that quietly stopped finding imports fails here rather than reporting a
    clean package forever.
    """

    offending = tmp_path / "offending.py"
    offending.write_text(
        "import json\n"
        "from src.domain.calculation import CalculationResult\n",
        encoding="utf-8",
    )
    assert "src" in _imported_roots(offending)

    clean = tmp_path / "clean.py"
    clean.write_text(
        "from .contracts import ContextBudgetV1\n"
        "from rag_v2.evidence import disclosure\n",
        encoding="utf-8",
    )
    # A relative import cannot name another package, so it is never a violation.
    assert _imported_roots(clean) == {"rag_v2"}
