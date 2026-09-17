"""H2A-3B1-A: the Context Compiler kernel's own contracts.

These tests are role-agnostic where they can be.  The kernel's guarantees --
immutability, disclosed-projections-only, deterministic selection, budget
enforcement, disclosure ordering, exact-token discipline -- must hold for any
role, so most of them are asserted against small purpose-built policies rather
than against B3.  The B3 boundary is proven separately, against real fixtures,
in ``test_b3_shadow_parity.py``.

The one thing none of this proves is what a model *saw*.  Nothing here touches a
model call: H2A-3B1 builds the framework and proves it beside the production
path, and the production path keeps its own context assembly until 3B2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rag_v2.context import (
    AgentContextPackV1,
    ContextBudgetUnsupported,
    ContextBudgetV1,
    ContextCompilerV1,
    ContextRequestV1,
    ContextRoleV1,
    ContextSelectionEntryV1,
    ContextSelectionReasonV1,
    PackIntegrityError,
    SelectionResultV1,
    UnknownContextRole,
)
from rag_v2.evidence.disclosure import (
    EvidenceDisclosureProfile,
    allowed_fields,
)

RAW_TEXT = "RAW SOURCE TEXT THAT MUST NOT CROSS"


# --- a purpose-built role, so the kernel is tested on its own terms ---------------


@dataclass(frozen=True)
class _Policy:
    """A minimal role policy: everything admitted, in order, under the cap."""

    role: ContextRoleV1 = ContextRoleV1.SPECIALIST

    @property
    def evidence_profile(self) -> EvidenceDisclosureProfile:
        return EvidenceDisclosureProfile.SPECIALIST

    @property
    def calculation_profile(self) -> EvidenceDisclosureProfile:
        return EvidenceDisclosureProfile.SPECIALIST

    def select(
        self, admitted: Any, budget: ContextBudgetV1
    ) -> SelectionResultV1:
        limit = budget.max_evidence_items
        selected: list[Any] = []
        entries: list[ContextSelectionEntryV1] = []
        for rank, item in enumerate(admitted):
            kept = limit is None or len(selected) < limit
            if kept:
                selected.append(item)
            entries.append(
                ContextSelectionEntryV1(
                    evidence_id=str(item.get("evidence_id", "")),
                    reason=(
                        ContextSelectionReasonV1.ADMITTED
                        if kept
                        else ContextSelectionReasonV1.DROPPED_BY_EVIDENCE_BUDGET
                    ),
                    rank=rank,
                )
            )
        return SelectionResultV1(tuple(selected), tuple(entries))


def _artifact(evidence_id: str = "e1", **extra: Any) -> dict[str, Any]:
    """An authoritative evidence artifact, as the runtime carries it.

    Deliberately shaped like a real packet: the forbidden content is *present*
    on the authoritative object, so a test asserting its absence from the pack
    is asserting something the projection actually had to remove.
    """

    return {
        "evidence_id": evidence_id,
        "fact_id": evidence_id,
        "metric": "Revenue",
        "value": "391",
        "period": "FY2024",
        "document_id": "doc-1",
        "page": 7,
        "citation_id": "citation:a",
        "source_text": RAW_TEXT,
        "content": RAW_TEXT,
        "metadata": {"source_text": RAW_TEXT},
        **extra,
    }


def _compile(*artifacts: dict[str, Any], **kwargs: Any) -> AgentContextPackV1:
    compiler = ContextCompilerV1(_Policy(), **kwargs)
    return compiler.compile(
        ContextRequestV1(
            role=ContextRoleV1.SPECIALIST,
            invocation_id="inv-1",
            query="What was revenue?",
            admitted_evidence=tuple(artifacts),
        )
    )


# --- immutability and lifetime -------------------------------------------------------


def test_a_pack_is_frozen() -> None:
    """Lifetime is one invocation; mutating it would make it state."""

    pack = _compile(_artifact())

    with pytest.raises(Exception):
        pack.query = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        pack.evidence[0]["metric"] = "changed"  # type: ignore[index]


def test_a_pack_carries_no_raw_authority_object() -> None:
    """The structural guarantee, not a field-by-field promise.

    A raw evidence packet is *identified* by carrying containers -- ``metadata``
    above all.  Refusing nested mappings is what makes "no authoritative object
    was smuggled in" enforceable rather than merely intended.
    """

    from rag_v2.adaptive.adaptive_contracts import EvidencePacketV1

    raw = EvidencePacketV1.from_mapping(
        {**_artifact(), "temporal": {"period": "FY2024"}}
    ).to_dict()
    assert isinstance(raw["metadata"], dict), "the fixture must really be raw"

    with pytest.raises(PackIntegrityError):
        AgentContextPackV1(
            role=ContextRoleV1.SPECIALIST,
            invocation_id="inv-raw",
            query="q",
            evidence=(raw,),
            calculation=None,
            references=(),
            budget=_compile(_artifact()).budget,
            selection=_compile(_artifact()).selection,
        )


def test_an_invalid_role_or_identity_is_refused() -> None:
    pack = _compile(_artifact())
    with pytest.raises(PackIntegrityError):
        AgentContextPackV1(
            role="SPECIALIST",  # type: ignore[arg-type]
            invocation_id="inv",
            query="q",
            evidence=(),
            calculation=None,
            references=(),
            budget=pack.budget,
            selection=pack.selection,
        )
    with pytest.raises(PackIntegrityError):
        AgentContextPackV1(
            role=ContextRoleV1.SPECIALIST,
            invocation_id="   ",
            query="q",
            evidence=(),
            calculation=None,
            references=(),
            budget=pack.budget,
            selection=pack.selection,
        )


# --- disclosure is the only way in ----------------------------------------------------


def test_the_pack_holds_disclosed_projections_only() -> None:
    """The authoritative object carried the text; the pack must not."""

    pack = _compile(_artifact())

    assert RAW_TEXT not in repr(pack.evidence)
    for forbidden in ("source_text", "content", "metadata"):
        assert all(forbidden not in item for item in pack.evidence), forbidden


def test_selection_may_read_authority_but_the_pack_may_not_hold_it() -> None:
    """The ordering's whole point, stated as a test.

    A policy selects on an authoritative field the disclosure profile does not
    admit.  Selection works, and the field is absent from the pack -- so
    "select on authority, disclose what crosses" is a property of the kernel,
    not a convention its callers are trusted to follow.
    """

    @dataclass(frozen=True)
    class _RawMetricPolicy(_Policy):
        def select(self, admitted: Any, budget: ContextBudgetV1) -> SelectionResultV1:
            kept = [item for item in admitted if item.get("raw_metric") == "Total"]
            entries = tuple(
                ContextSelectionEntryV1(
                    evidence_id=str(item.get("evidence_id", "")),
                    reason=(
                        ContextSelectionReasonV1.ADMITTED
                        if item.get("raw_metric") == "Total"
                        else ContextSelectionReasonV1.DROPPED_BY_EVIDENCE_BUDGET
                    ),
                    rank=rank,
                )
                for rank, item in enumerate(admitted)
            )
            return SelectionResultV1(tuple(kept), entries)

    compiler = ContextCompilerV1(_RawMetricPolicy())
    pack = compiler.compile(
        ContextRequestV1(
            role=ContextRoleV1.SPECIALIST,
            invocation_id="inv-2",
            query="q",
            admitted_evidence=(
                _artifact("e1", raw_metric="Segment"),
                _artifact("e2", raw_metric="Total"),
            ),
        )
    )

    # Selection read the authoritative field.
    assert pack.evidence_ids == ("e2",)
    # And it did not cross.
    assert "raw_metric" not in set(allowed_fields(EvidenceDisclosureProfile.SPECIALIST))
    assert all("raw_metric" not in item for item in pack.evidence)


def test_an_unlisted_field_never_reaches_the_pack() -> None:
    """Deny by default, at the boundary the pack is built from."""

    pack = _compile(_artifact(brand_new_field="leaked?", internal_score=0.9))

    for item in pack.evidence:
        assert set(item) <= set(allowed_fields(EvidenceDisclosureProfile.SPECIALIST))


def test_the_calculation_projection_is_the_three_permitted_fields() -> None:
    """The internal diagnostics form must not cross."""

    compiler = ContextCompilerV1(_Policy())
    pack = compiler.compile(
        ContextRequestV1(
            role=ContextRoleV1.SPECIALIST,
            invocation_id="inv-3",
            query="q",
            admitted_evidence=(_artifact(),),
            calculation={
                "operation": "difference",
                "unit": "USD",
                "value": "8",
                "operands": [{"source_text": RAW_TEXT}],
                "error_message": "INTERNAL /var/lib/secret.py",
            },
        )
    )

    assert dict(pack.calculation or {}) == {
        "operation": "difference",
        "unit": "USD",
        "value": "8",
    }
    assert RAW_TEXT not in repr(pack)


# --- determinism -------------------------------------------------------------------------


def test_the_same_input_produces_the_same_pack() -> None:
    """Not merely equal outcomes -- an equal pack, field by field."""

    artifacts = (_artifact("e1", page=7), _artifact("e2", page=0))

    first = _compile(*artifacts)
    second = _compile(*artifacts)

    assert first == second
    assert first.evidence == second.evidence
    assert first.references == second.references
    assert first.selection == second.selection


def test_selection_order_is_the_admitted_order() -> None:
    """Stable, and derived from the input rather than from an internal ordering."""

    pack = _compile(_artifact("c"), _artifact("a"), _artifact("b"))

    assert pack.evidence_ids == ("c", "a", "b")


# --- evidence-count budget ------------------------------------------------------------------


def test_the_evidence_budget_keeps_the_head_and_records_the_rest() -> None:
    pack = _compile(
        _artifact("e1"), _artifact("e2"), _artifact("e3"),
        budget=ContextBudgetV1(max_evidence_items=2),
    )

    assert pack.evidence_ids == ("e1", "e2")
    assert pack.budget.evidence_considered == 3
    assert pack.budget.evidence_selected == 2
    assert pack.budget.evidence_dropped == 1
    assert pack.selection.dropped_evidence_ids == ("e3",)
    # The trace is complete: every considered artifact appears exactly once.
    assert len(pack.selection.entries) == 3


def test_an_evidence_budget_of_zero_is_refused() -> None:
    with pytest.raises(ValueError):
        ContextBudgetV1(max_evidence_items=0)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_a_malformed_budget_is_refused(value: Any) -> None:
    with pytest.raises(ValueError):
        ContextBudgetV1(max_evidence_items=value)


# --- the exact-token requirement -------------------------------------------------------------


class _WordCounter:
    """TEST-ONLY deterministic counter.  Not a production measurement.

    It exists to exercise budgeting *mechanics*, and it must never be quoted as
    a token count for any real boundary.  It is deliberately not a tokenizer: it
    is named ``word-count`` and its ``counter_id`` says so, which is how a reader
    of a pack's accounting can tell it apart from a real one.
    """

    counter_id = "test-only-word-count"

    def count(self, text: str) -> int:
        return len(text.split())


def test_a_token_bound_without_an_exact_counter_is_refused() -> None:
    """The rule, at construction rather than at the first call.

    A character count, a word count, or a tokenizer for a different model are
    all forbidden substitutes.  The failure must be a configuration failure.
    """

    with pytest.raises(ContextBudgetUnsupported):
        ContextCompilerV1(_Policy(), budget=ContextBudgetV1(max_input_tokens=100))


def test_a_token_bound_with_a_counter_is_accepted() -> None:
    compiler = ContextCompilerV1(
        _Policy(),
        budget=ContextBudgetV1(max_input_tokens=10_000),
        token_counter=_WordCounter(),
    )
    assert compiler.budget.max_input_tokens == 10_000


def test_the_budget_sheds_evidence_until_the_payload_fits() -> None:
    """Mechanics only.  The numbers below are word counts, not tokens.

    The bound is derived from a measured full payload rather than guessed, so
    the test asserts a relationship -- the payload shrank to fit -- instead of a
    magic number that would have to be re-guessed whenever the serializer's
    shape changed.
    """

    artifacts = tuple(
        _artifact(f"e{index}", metric=f"Metric number {index} with several words")
        for index in range(1, 6)
    )
    full = _compile(*artifacts, token_counter=_WordCounter()).budget
    assert full.selected_context_tokens is not None
    limit = full.selected_context_tokens // 2

    pack = _compile(
        *artifacts,
        budget=ContextBudgetV1(max_input_tokens=limit),
        token_counter=_WordCounter(),
    )

    assert len(pack.evidence) < len(artifacts)
    assert pack.budget.selected_context_tokens is not None
    assert pack.budget.selected_context_tokens <= limit
    # Shedding is visible as a reason, not as an absence: the trace still
    # accounts for every artifact that was considered.
    assert len(pack.selection.entries) == len(artifacts)
    assert ContextSelectionReasonV1.DROPPED_BY_TOKEN_BUDGET in {
        entry.reason for entry in pack.selection.entries
    }
    # Shedding happens at the tail, so the survivors keep the admitted prefix.
    assert pack.evidence_ids == tuple(
        f"e{index}" for index in range(1, len(pack.evidence) + 1)
    )


def test_the_accounting_names_the_counter_that_produced_the_number() -> None:
    """A number without its counter's identity can be quoted as the wrong thing."""

    pack = _compile(_artifact(), token_counter=_WordCounter())

    assert pack.budget.token_counter_id == "test-only-word-count"
    assert pack.budget.selected_context_tokens is not None


def test_no_counter_means_no_token_number_at_all() -> None:
    """Absence is recorded as absence, never as zero and never as an estimate."""

    pack = _compile(_artifact())

    assert pack.budget.selected_context_tokens is None
    assert pack.budget.token_counter_id is None
    assert pack.budget.token_bound_was_measured is False


# --- the kernel refuses to misbehave ------------------------------------------------------------


def test_a_policy_that_invents_evidence_is_refused() -> None:
    """Selection may subset; it may not manufacture."""

    @dataclass(frozen=True)
    class _Inventing(_Policy):
        def select(self, admitted: Any, budget: ContextBudgetV1) -> SelectionResultV1:
            return SelectionResultV1(
                tuple(list(admitted) + [_artifact("invented")]), ()
            )

    compiler = ContextCompilerV1(_Inventing())
    with pytest.raises(PackIntegrityError):
        compiler.compile(
            ContextRequestV1(
                role=ContextRoleV1.SPECIALIST,
                invocation_id="inv",
                query="q",
                admitted_evidence=(_artifact(),),
            )
        )


def test_an_unrecognised_role_policy_is_refused() -> None:
    class _OtherRole:
        role = "NOT_A_ROLE"
        evidence_profile = EvidenceDisclosureProfile.SPECIALIST
        calculation_profile = EvidenceDisclosureProfile.SPECIALIST

        def select(self, admitted: Any, budget: ContextBudgetV1) -> SelectionResultV1:
            return SelectionResultV1((), ())

    with pytest.raises(UnknownContextRole):
        ContextCompilerV1(_OtherRole())  # type: ignore[arg-type]


def test_a_request_for_another_role_is_refused() -> None:
    """A pack is role-specific; compiling for the wrong one is not a detail.

    The role decides which disclosure profile applies, so accepting a
    mismatched request would silently project one boundary's artifacts through
    another boundary's allowlist.
    """

    compiler = ContextCompilerV1(_Policy())

    with pytest.raises(UnknownContextRole):
        compiler.compile(
            ContextRequestV1(
                role="SPECIALIST",  # type: ignore[arg-type]
                invocation_id="inv",
                query="q",
            )
        )


# --- ContextBudget is not RunBudget ----------------------------------------------------------------


def test_context_budget_and_run_budget_are_separate_types() -> None:
    """They answer different questions and must not be merged.

    RunBudget bounds how much *work* the runtime performs; ContextBudget bounds
    how much *information* one invocation sees.  A shared type would mean one
    knob changing both, which is how a context change becomes a latency change.
    """

    from rag_v2.adaptive.adaptive_budget import AdaptiveRAGBudgetV1

    run_fields = set(AdaptiveRAGBudgetV1().__dataclass_fields__)
    context_fields = set(ContextBudgetV1().__dataclass_fields__)

    assert run_fields.isdisjoint(context_fields)
    assert {"max_replan_rounds", "max_total_tool_calls"} <= run_fields
    assert {"max_input_tokens", "max_evidence_items"} <= context_fields
    # No retry, tool or timeout field belongs to a context budget.
    for field in ("retry", "tool", "timeout", "replan"):
        assert not any(field in name for name in context_fields), field
