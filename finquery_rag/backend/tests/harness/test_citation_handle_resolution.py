"""The two citation vocabularies, and the bridge between them.

The renderer shows the model short handles -- ``[E1]``, ``[E2]`` -- and asks it
to cite with them, because a prompt that repeated a sixty-four character digest
beside every evidence item would spend its context budget on identifiers.  The
validator resolves citations against canonical fact-store ids.

Nothing reconciled the two.  Every specialist answer that followed the prompt's
own instruction was therefore rejected as ``GV7_UNKNOWN_CITATION``: the model was
told to write one vocabulary and graded in another, and across the whole P1.2
replay run **not one model answer was released** -- all six of the invocations
that happened failed on exactly this.

The expected values here are read off the frozen ``BASELINE_V3`` fixture and the
pack the compiler actually produced, not off the resolver.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.runtime.trusted_v2_generation import (
    TrustedV2GenerationCapability,
    _resolve_cited_handles,
)
from tests.harness.b3_legacy_context_baseline import SCENARIOS, build_state


class _AnsweringBackend:
    """A backend that answers with whatever text the test authored."""

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, prompt: str) -> str:
        return self.text


def _capability_with(answer: str) -> TrustedV2GenerationCapability:
    return TrustedV2GenerationCapability(model_backend=_AnsweringBackend(answer))


def _authored_citation_ids() -> tuple[str, ...]:
    """The citation ids the ``multi_fact`` fixture authors, in evidence order.

    Read off the *authored scenario* -- ``e1 -> citation:a``, ``e2 ->
    citation:b`` -- and not off the frozen baseline.  The baseline is captured
    from the implementation, so asserting against it here would make this file
    check the code against itself, which is the one thing a test of a
    translation cannot afford: it would pass just as happily if the translation
    mapped handles to the wrong ids.
    """

    return tuple(item["citation_id"] for item in SCENARIOS["multi_fact"]["evidence"])


def _multi_fact_pack() -> Any:
    capability = _capability_with("placeholder")
    capability.generate(build_state("multi_fact"))
    return capability.last_context_pack, capability


# --- the bridge itself --------------------------------------------------------------------------


def test_a_cited_handle_becomes_the_identity_it_names() -> None:
    """``[E1]`` and ``citation:a`` name one evidence item; the pack says which.

    The fixture is the frozen ``multi_fact`` scenario, whose two evidence items
    carry ``citation:a`` and ``citation:b`` -- so the expectation is read off the
    baseline rather than off the resolver.
    """

    pack, _ = _multi_fact_pack()
    first, second = _authored_citation_ids()

    rewritten, cited, record = _resolve_cited_handles(
        "revenue was 391 [E1] against 383 [E2]", pack
    )

    assert cited == (first, second)
    assert rewritten == f"revenue was 391 [{first}] against 383 [{second}]"
    assert record["unresolved_cited_handles"] == ()


def test_a_handle_the_pack_does_not_carry_is_left_as_written() -> None:
    """A hallucinated handle must stay visible, not be dropped or invented.

    Dropping it would hide a fabrication from the validator; assigning it an id
    would manufacture one.  Leaving it means the answer still fails validation,
    which is the outcome that should happen.
    """

    pack, _ = _multi_fact_pack()

    rewritten, cited, record = _resolve_cited_handles("revenue was 391 [E9]", pack)

    assert rewritten == "revenue was 391 [E9]"
    assert cited == ()
    assert record["unresolved_cited_handles"] == ("E9",)


def test_a_calculation_handle_is_left_alone() -> None:
    """``C1`` names the calculation, which is not an evidence citation."""

    _, capability = _multi_fact_pack()
    pack = capability.last_context_pack
    capability.generate(build_state("calculation_with_explanation"))
    pack = capability.last_context_pack

    rewritten, cited, record = _resolve_cited_handles("the difference was 8 USD [C1]", pack)

    assert rewritten == "the difference was 8 USD [C1]"
    assert cited == ()


def test_an_answer_with_no_handles_is_unchanged() -> None:
    pack, _ = _multi_fact_pack()

    rewritten, cited, record = _resolve_cited_handles("revenue was 391", pack)

    assert rewritten == "revenue was 391"
    assert cited == ()
    assert record["cited_handles_resolved"] == 0


# --- end to end through the capability ----------------------------------------------------------


def test_the_capability_takes_its_citations_from_the_answers_own_handles() -> None:
    """The tier that did not exist: what the model actually cited.

    Before it, a model that cited correctly resolved to nothing and fell through
    to the bound-evidence fallback, which attributes *every* citation to a
    candidate that named none -- so a careful answer and a hallucinating one
    produced the same citation set.
    """

    capability = _capability_with("revenue was 391 [E1] against 383 [E2]")

    result = capability.generate(build_state("multi_fact"))

    first, second = _authored_citation_ids()
    assert result.citation_ids == (first, second)
    assert result.generation_metadata["citation_source"] == "answer_handles"
    assert result.candidate_answer == (
        f"revenue was 391 [{first}] against 383 [{second}]"
    )


def test_only_the_item_the_answer_cited_is_attributed() -> None:
    """The discriminating case, and the reason this tier had to exist.

    Two items are bound; the answer cites one.  The old fallback attributed
    *both*, because a candidate that declared no ids was handed the whole
    allow-list -- so a careful answer and a hallucinating one produced the same
    citation set, and nothing downstream could tell them apart.
    """

    capability = _capability_with("revenue was 391 [E1]")

    result = capability.generate(build_state("multi_fact"))

    assert result.citation_ids == (_authored_citation_ids()[0],)
    assert result.citation_ids != _authored_citation_ids()
    assert result.generation_metadata["citation_source"] == "answer_handles"


def test_a_model_that_cites_nothing_still_falls_back() -> None:
    """The fallback is unchanged, but it is now the last tier rather than the only one."""

    capability = _capability_with("revenue was 391 against 383")

    result = capability.generate(build_state("multi_fact"))

    assert result.citation_ids == _authored_citation_ids()
    assert result.generation_metadata["citation_source"] == "bound_evidence_fallback"


def test_a_fabricated_handle_is_reported_and_does_not_become_a_citation() -> None:
    capability = _capability_with("revenue was 391 [E9]")

    result = capability.generate(build_state("multi_fact"))

    assert result.generation_metadata["unresolved_cited_handles"] == ("E9",)
    assert _authored_citation_ids()[0] not in result.candidate_answer


@pytest.mark.parametrize("handle", ["E1", "E2"])
def test_every_handle_the_baseline_pack_issues_resolves(handle: str) -> None:
    pack, _ = _multi_fact_pack()

    _, cited, record = _resolve_cited_handles(f"see [{handle}]", pack)

    assert cited
    assert record["unresolved_cited_handles"] == ()
