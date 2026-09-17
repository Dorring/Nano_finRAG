"""H2A-3E: the Harness's negative contracts, as executable guards.

Every one of these asserts that something **cannot** happen.  They are worth
more than another feature: the whole point of H2A-3 was to put a governed
boundary around context and around model output, and a boundary is only real for
as long as nobody can reach past it.  A future contributor adding a convenient
import would break the guarantee silently, and no behavioural test would notice
-- the code would still work, it would just stop being trustworthy.

So the rule for this file: **each guard names the thing it prevents and why the
thing would matter.**  Where an equivalent assertion already lives beside the
code it protects (the pack refusing nested mappings, the request's exact field
set), this file does not repeat it -- it holds the module-level facts, which is
what nothing else was checking.

Absence is asserted on the syntax tree, never on the source text.  A comment
naming what was removed must not be able to satisfy a guard -- that mistake has
already been made once in this project, and the failure it produced was a check
that its own docstring passed.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from rag_v2.context import AgentContextPackV1
from rag_v2.invocation import ModelProviderV1, ModelRequestV1

#: One row per boundary, as (subject, forbidden import roots, why).
#:
#: The subject is a file or a directory.  The forbidden roots are matched
#: against the first one and two components of every import in it, so
#: ``rag_v2.context`` catches ``from rag_v2.context import ...`` and ``rag_v2``
#: catches anything at all in that package.
BOUNDARIES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "rag_v2/invocation/provider.py",
        ("rag_v2.context", "rag_v2.adaptive", "rag_v2.derived", "src"),
        "a provider is handed a ModelRequestV1 and nothing else.  The pack, the "
        "RunState, the evidence contract and the admission contracts are all "
        "reachable from this package otherwise, and a provider that could read "
        "them would be a second route to the model that Disclosure Authority "
        "does not govern",
    ),
    (
        "rag_v2/context",
        ("rag_v2.invocation", "rag_v2.derived", "src"),
        "the compiler selects and discloses context; it does not execute models "
        "and it does not verify model output.  Importing either contract family "
        "would let it acquire an opinion it must not have",
    ),
    (
        "rag_v2/invocation",
        ("rag_v2.derived", "src"),
        "a provider executes and a runtime wires; neither admits artifacts.  "
        "Admission is a separate authority over model output, and an execution "
        "boundary that could reach it would be judging its own work",
    ),
    (
        "rag_v2/derived",
        ("rag_v2.context", "rag_v2.invocation", "src"),
        "admission answers to the authoritative source artifact alone.  A "
        "verifier that could reach the compiler or a provider would be "
        "checking model output against model output",
    ),
    (
        "src/generation/specialist_prompt.py",
        ("rag_v2.adaptive", "rag_v2.evidence.disclosure", "src.runtime"),
        "the renderer consumes an AgentContextPack and no runtime authority: "
        "not the RunState it came from, not the Disclosure Authority it already "
        "passed through, and not the capability that called it",
    ),
    (
        "src/services",
        ("rag_v2.invocation",),
        "ingestion writes trusted fields.  If it cannot see a ModelResponseV1 it "
        "cannot write one into a chunk, however a future edit is phrased -- "
        "which is the F6 defect stated as an import rule",
    ),
    (
        "src/retrieval",
        ("rag_v2.invocation",),
        "retrieval reads what ingestion admitted.  Reaching the invocation "
        "contracts from here would be a second path from a model to the "
        "evidence a reader is shown",
    ),
)


def _imported(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module)
    return roots


def _modules(subject: str) -> list[pathlib.Path]:
    path = pathlib.Path(subject)
    if path.is_dir():
        return sorted(path.rglob("*.py"))
    return [path]


def _matches(imported: str, forbidden: str) -> bool:
    return imported == forbidden or imported.startswith(forbidden + ".")


@pytest.mark.parametrize(
    ("subject", "forbidden", "why"), BOUNDARIES, ids=[row[0] for row in BOUNDARIES]
)
def test_a_boundary_cannot_be_reached_past(
    subject: str, forbidden: tuple[str, ...], why: str
) -> None:
    offenders = {
        path.as_posix(): sorted(
            name
            for name in _imported(path)
            if any(_matches(name, root) for root in forbidden)
        )
        for path in _modules(subject)
    }
    offenders = {path: names for path, names in offenders.items() if names}

    assert offenders == {}, f"{subject} must not import {forbidden}: {why}\n{offenders}"


def test_the_guard_can_actually_fail(tmp_path: pathlib.Path) -> None:
    """A guard that cannot fail is not a guard.

    Exercised against the real helpers with a synthetic module, so a refactor
    that quietly stopped finding imports fails here rather than reporting a
    clean repository forever.
    """

    offending = tmp_path / "offending.py"
    offending.write_text(
        "from rag_v2.context import AgentContextPackV1\n"
        "import rag_v2.derived\n"
        "from . import sibling\n",
        encoding="utf-8",
    )
    imported = _imported(offending)

    assert any(_matches(name, "rag_v2.context") for name in imported)
    assert any(_matches(name, "rag_v2.derived") for name in imported)
    # A relative import cannot name another package, so it is never a violation.
    assert not any(_matches(name, "sibling") for name in imported)


# --- the same boundaries, at the level of what crosses them --------------------------------


def test_a_provider_is_asked_for_one_request_and_nothing_else() -> None:
    """The interface itself, not a caller's manners."""

    parameters = list(inspect.signature(ModelProviderV1.invoke).parameters.values())

    assert [parameter.name for parameter in parameters] == ["self", "request"]
    assert parameters[1].annotation in ("ModelRequestV1", ModelRequestV1)


def test_the_request_has_nowhere_to_put_an_authority_object() -> None:
    """The F6-shaped hole, closed at the contract level.

    A provider cannot receive what the request has no field for, so this is the
    boundary made structural rather than a rule callers are trusted to keep.
    """

    fields = set(ModelRequestV1.__dataclass_fields__)

    assert fields == {
        "invocation_id",
        "role",
        "prompt",
        "provider_id",
        "model_id",
    }
    for absent in (
        "pack",
        "state",
        "evidence",
        "evidence_items",
        "calculation",
        "metadata",
        "artifacts",
    ):
        assert absent not in fields, absent


def test_the_renderer_is_handed_the_pack_and_nothing_else() -> None:
    from src.generation.specialist_prompt import render_specialist_prompt

    parameters = list(inspect.signature(render_specialist_prompt).parameters.values())

    assert [parameter.name for parameter in parameters] == ["pack"]


def test_a_pack_has_nowhere_to_put_a_runtime_authority_either() -> None:
    """The other half of the same rule.

    A pack is a per-invocation projection, not a narrowed RunState, and not an
    artifact store and not memory.  Its field set is asserted exactly elsewhere
    too; what is added here is that none of the *names* describe something the
    pack is not.
    """

    fields = set(AgentContextPackV1.__dataclass_fields__)

    assert fields == {
        "role",
        "invocation_id",
        "query",
        "evidence",
        "calculation",
        "references",
        "budget",
        "selection",
    }
    for absent in ("state", "run_state", "evidence_packets", "store", "memory", "binding"):
        assert absent not in fields, absent


def test_the_compiler_holds_no_provider_and_no_verifier() -> None:
    """The compiler's collaborators, read off its own construction."""

    from rag_v2.context import ContextCompilerV1

    parameters = set(inspect.signature(ContextCompilerV1.__init__).parameters)

    assert parameters == {"self", "policy", "budget", "token_counter"}
