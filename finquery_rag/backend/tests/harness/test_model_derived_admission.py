"""H2A-3D: a model's output is a candidate until a deterministic check admits it.

F6 was not "a model might write something poor".  It was that the ingestion path
treated *generation succeeding* as *admission succeeding*, so a table-cleaning
model's answer became trusted retrieval content with no authority transition in
between -- and from that point on nothing downstream could tell a number a
language model typed from a number the extractor read.

The fixtures below are authored, and the expected outcomes are read off the
fixtures rather than off the verifier.  That distinction is the whole reason
this file exists: a fidelity check whose two sides both came from the same
component would prove only that the component agrees with itself, and a test
whose expectations came from the code under test would do the same.

The adversarial set is the one the phase specifies -- preservation, presentation
normalisation, changed, invented, dropped, reassigned, duplicated, unverifiable
-- and each case is one transformation of *one* source table, so a failure names
which rule stopped working.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from rag_v2.derived import (
    AdmissionOutcomeV1,
    AdmissionReasonV1,
    AdmittedDerivedArtifactV1,
    ArtifactAdmissionError,
    ArtifactAdmissionResultV1,
    ModelDerivedArtifactV1,
    TransformationKindV1,
    numeric_facts_of,
    verify_table_fidelity,
)

#: The authoritative extraction.  Four figures, two labels, two columns.
SOURCE = """| Metric | FY2024 | FY2023 |
| --- | --- | --- |
| Revenue | 100 | 90 |
| Cost | 80 | 70 |
"""

SOURCE_REF = "report.pdf::page_7::table_1"


def _candidate(content: str) -> ModelDerivedArtifactV1:
    return ModelDerivedArtifactV1(
        artifact_id=f"{SOURCE_REF}::table-cleaning",
        content=content,
        transformation=TransformationKindV1.TABLE_CLEANING,
        source_reference=SOURCE_REF,
        provider_id="nvidia",
        model_id="meta/llama-3.1-8b-instruct",
    )


def _verdict(content: str, source: str = SOURCE) -> ArtifactAdmissionResultV1:
    return verify_table_fidelity(_candidate(content), source)


# --- the adversarial set ------------------------------------------------------------------


def test_an_exact_preservation_is_admitted() -> None:
    assert _verdict(SOURCE) == ArtifactAdmissionResultV1(
        artifact_id=f"{SOURCE_REF}::table-cleaning",
        outcome=AdmissionOutcomeV1.ADMITTED,
        reason=AdmissionReasonV1.FIDELITY_PRESERVED,
        source_reference=SOURCE_REF,
    )


PRESENTATION_SOURCE = """| Metric | FY2024 |
| --- | --- |
| Revenue | 1000 |
"""


def test_presentation_only_normalisation_is_admitted() -> None:
    """``1,000`` and ``1000`` are one quantity, and the contract allows the change.

    The shared financial semantics fold the digit-grouping separator, so this is
    the same number written two ways -- not a formatting loophole, and not a
    licence to require string equality either.
    """

    also_usable = """| Metric | FY2024 |
| --- | --- |
| Revenue | 1,000 |
"""

    assert _verdict(also_usable, PRESENTATION_SOURCE).outcome is AdmissionOutcomeV1.ADMITTED


def test_a_changed_value_is_rejected() -> None:
    """One digit, and the table now says something the source never said."""

    changed = SOURCE.replace("| Revenue | 100 |", "| Revenue | 101 |")

    result = _verdict(changed)

    assert result.outcome is AdmissionOutcomeV1.REJECTED
    assert result.reason is AdmissionReasonV1.NUMERIC_VALUE_CHANGED
    assert result.location == "revenue / fy2024"


def test_an_invented_value_is_rejected() -> None:
    """A figure the source never stated, however plausible."""

    invented = SOURCE + "| Profit | 120 | 110 |\n"

    result = _verdict(invented)

    assert result.outcome is AdmissionOutcomeV1.REJECTED
    assert result.reason is AdmissionReasonV1.NUMERIC_VALUE_INVENTED
    assert result.location == "profit / fy2024"


def test_a_dropped_required_value_is_rejected() -> None:
    dropped = SOURCE.replace("| Cost | 80 | 70 |\n", "")

    result = _verdict(dropped)

    assert result.outcome is AdmissionOutcomeV1.REJECTED
    assert result.reason is AdmissionReasonV1.NUMERIC_VALUE_REMOVED
    assert result.location == "cost / fy2024"


def test_a_reassignment_is_rejected_even_though_every_number_survived() -> None:
    """The case a bag-of-numbers check calls perfect.

    The multiset is identical -- 100, 90, 80, 70 all present, same counts -- and
    the table now says Revenue was 80 and Cost was 100.  Anything that compared
    sorted values would admit this, which is why association is checked.
    """

    reassigned = SOURCE.replace(
        "| Revenue | 100 | 90 |", "| Revenue | 80 | 90 |"
    ).replace("| Cost | 80 | 70 |", "| Cost | 100 | 70 |")

    result = _verdict(reassigned)

    assert result.outcome is AdmissionOutcomeV1.REJECTED
    assert result.reason is AdmissionReasonV1.NUMERIC_VALUE_REASSIGNED
    assert result.location == "revenue / fy2024"

    # The claim the rejection rests on: the two tables have the same numbers.
    source = numeric_facts_of(SOURCE)
    derived = numeric_facts_of(reassigned)
    assert source is not None and derived is not None
    assert sorted(fact.value for fact in source) == sorted(fact.value for fact in derived)


def test_a_duplicated_figure_is_rejected() -> None:
    """One source fact stated twice is an extra fact, not a familiar value."""

    duplicated = SOURCE.replace(
        "| Cost | 80 | 70 |",
        "| Cost | 80 | 70 |\n| Cost | 80 | 70 |",
    )

    result = _verdict(duplicated)

    assert result.outcome is AdmissionOutcomeV1.REJECTED
    assert result.reason is AdmissionReasonV1.NUMERIC_VALUE_INVENTED
    assert result.location == "cost / fy2024"


def test_an_unverifiable_transformation_fails_closed() -> None:
    """A prose answer, and a ragged one, are both failures to verify."""

    for content in (
        "I have cleaned the table. It shows revenue of 100.",
        "| Metric | FY2024 |\n| --- | --- |\n| Revenue | 100 | 90 |\n",
        "| Metric |\n| --- |\n",
    ):
        result = _verdict(content)
        assert result.outcome is AdmissionOutcomeV1.REJECTED, content
        assert result.reason is AdmissionReasonV1.STRUCTURE_UNVERIFIABLE, content


def test_an_unverifiable_source_fails_closed_too() -> None:
    """The oracle being unusable is not the model's fault and still refuses.

    Admitting here would mean "we could not check", which is the one answer an
    admission contract must never give.
    """

    result = _verdict(SOURCE, "not a table at all")

    assert result.outcome is AdmissionOutcomeV1.REJECTED
    assert result.reason is AdmissionReasonV1.STRUCTURE_UNVERIFIABLE


def test_the_oracle_is_the_source_and_not_another_model_output() -> None:
    """The same candidate against two sources gets two answers.

    A check whose two sides were both model-produced would agree with itself
    whatever the model did; this is the property that rules that out.
    """

    candidate = _candidate(SOURCE.replace("| Revenue | 100 |", "| Revenue | 101 |"))

    assert verify_table_fidelity(candidate, SOURCE).outcome is AdmissionOutcomeV1.REJECTED
    assert (
        verify_table_fidelity(candidate, candidate.content).outcome
        is AdmissionOutcomeV1.ADMITTED
    )


# --- the contract: unverified is a type, not a flag ------------------------------------------


def test_a_candidate_carries_no_trust_to_flip() -> None:
    """One object changing ``trusted=False`` to ``True`` in place is the thing
    this contract exists to prevent: a reader holding a reference could not say
    which state it was in."""

    candidate = _candidate(SOURCE)

    for absent in ("trusted", "admitted", "is_admitted", "verified"):
        assert not hasattr(candidate, absent), absent
    assert set(ModelDerivedArtifactV1.__dataclass_fields__) == {
        "artifact_id",
        "content",
        "transformation",
        "source_reference",
        "provider_id",
        "model_id",
        "invocation_id",
    }


def test_an_admitted_artifact_cannot_be_built_from_a_rejection() -> None:
    candidate = _candidate(SOURCE.replace("| Revenue | 100 |", "| Revenue | 101 |"))
    rejected = _verdict(candidate.content)

    assert rejected.outcome is AdmissionOutcomeV1.REJECTED
    with pytest.raises(ArtifactAdmissionError):
        AdmittedDerivedArtifactV1(candidate, rejected)


def test_an_admitted_artifact_cannot_be_built_from_a_mismatched_result() -> None:
    """A result for one artifact does not admit another."""

    admitted = _verdict(SOURCE)
    other = ModelDerivedArtifactV1(
        artifact_id="some-other-table",
        content=SOURCE,
        transformation=TransformationKindV1.TABLE_CLEANING,
        source_reference=SOURCE_REF,
    )

    with pytest.raises(ArtifactAdmissionError):
        AdmittedDerivedArtifactV1(other, admitted)


def test_an_admitted_artifact_cannot_be_built_from_a_bare_result() -> None:
    """Not even a passing result admits content on its own -- the candidate is
    part of the construction, so there is no way to assert admission without
    having verified something."""

    admitted = _verdict(SOURCE)

    with pytest.raises(ArtifactAdmissionError):
        AdmittedDerivedArtifactV1(SOURCE, admitted)  # type: ignore[arg-type]


def test_a_result_cannot_claim_admission_under_a_refusal_reason() -> None:
    """The reason is what was verified, and there is one thing to verify."""

    with pytest.raises(ValueError):
        ArtifactAdmissionResultV1(
            artifact_id="a",
            outcome=AdmissionOutcomeV1.ADMITTED,
            reason=AdmissionReasonV1.NUMERIC_VALUE_CHANGED,
            source_reference=SOURCE_REF,
        )
    with pytest.raises(ValueError):
        ArtifactAdmissionResultV1(
            artifact_id="a",
            outcome=AdmissionOutcomeV1.REJECTED,
            reason=AdmissionReasonV1.FIDELITY_PRESERVED,
            source_reference=SOURCE_REF,
        )


def test_source_lineage_survives_admission() -> None:
    """The admitted artifact says which source it came from.

    Without this, model-derived content becomes a piece of text with no origin
    -- which is the state the whole contract exists to leave behind.
    """

    candidate = _candidate(SOURCE)
    admitted = AdmittedDerivedArtifactV1(candidate, _verdict(SOURCE))

    assert admitted.source_reference == SOURCE_REF
    assert admitted.content == SOURCE
    assert admitted.admission_reason is AdmissionReasonV1.FIDELITY_PRESERVED
    assert admitted.model_id == "meta/llama-3.1-8b-instruct"
    # A derived representation, not a new evidence identity.
    assert admitted.artifact_id == candidate.artifact_id


def test_the_admitted_artifact_does_not_hold_the_candidate() -> None:
    """The candidate is not admitted; an admitted object that referenced one
    would keep unverified content reachable through it."""

    candidate = _candidate(SOURCE)
    admitted = AdmittedDerivedArtifactV1(candidate, _verdict(SOURCE))

    assert admitted.content is not candidate.content or admitted.content == candidate.content
    assert not any(
        value is candidate for value in vars(admitted).values()
    )


# --- structure parsing ------------------------------------------------------------------------


def test_a_table_with_no_numbers_is_parseable_and_empty() -> None:
    """``()`` says "a well-formed table with no figures"; ``None`` says "not a
    table".  The two are different answers and the verifier treats them
    differently."""

    facts = numeric_facts_of("| Metric | Note |\n| --- | --- |\n| Revenue | see above |\n")

    assert facts == ()


def test_the_structural_parser_uses_the_shared_financial_semantics() -> None:
    """``(25)`` is minus twenty-five and ``1,000`` is one thousand, by the same
    rules the rest of the project uses -- not by rules invented here."""

    facts = numeric_facts_of("| Metric | FY2024 |\n| --- | --- |\n| Cost | (25) |\n| Revenue | 1,000 |\n")

    assert facts is not None
    assert {fact.row_label: fact.value for fact in facts} == {
        "cost": -25,
        "revenue": 1000,
    }


def test_prose_is_never_pushed_through_the_numeric_parser() -> None:
    """A label is a label.  A cell that will not canonicalise is not silently
    coerced into a number."""

    facts = numeric_facts_of(
        "| Metric | FY2024 |\n| --- | --- |\n| Revenue growth of 12 percent | 100 |\n"
    )

    assert facts is not None
    assert [fact.value for fact in facts] == [100]
    assert facts[0].row_label == "revenue growth of 12 percent"


# --- whose job admission is not ------------------------------------------------------------------


def _imported_roots(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module)
        elif isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
    return roots


@pytest.mark.parametrize("package", ["rag_v2/context", "rag_v2/invocation"])
def test_admission_is_not_the_compiler_s_or_the_provider_s_job(package: str) -> None:
    """Two boundaries that must not grow a verifier.

    A ``ContextCompilerV1`` that re-checked raw model output would be a second
    admission authority; a ``ModelProviderV1`` that decided its own output was
    trustworthy would be the original defect with better manners.  Neither may
    even import the contracts, so neither can quietly acquire an opinion.
    """

    offenders = [
        path.as_posix()
        for path in pathlib.Path(package).rglob("*.py")
        if "rag_v2.derived" in _imported_roots(path)
    ]

    assert offenders == [], offenders


def test_the_derived_package_imports_no_application_layer() -> None:
    """The admission contract is low; only the table-specific caller is above it."""

    for path in pathlib.Path("rag_v2/derived").rglob("*.py"):
        roots = _imported_roots(path)
        assert not {name for name in roots if name.split(".")[0] == "src"}, path
