"""Model-derived artifacts: what a model produced, before anything trusts it.

H2A-3D closes F6.  The ingestion path ran a table-cleaning model over an
extracted financial table and wrote its answer straight into the chunk's
``content`` and ``parent_excerpt`` -- the same fields authoritative document text
occupies.  Nothing stood between the model's answer and the retrieval index, so
the graph was:

    raw table -> model -> trusted retrieval content

The defect is not that the model might write badly.  It is that *generation
succeeding was treated as admission succeeding*, and a candidate truth became an
evidence truth with no authority transition in between.  A retrieved number that
a language model typed is not the same kind of thing as a number the extractor
read, and after that write nothing downstream could tell them apart.

This package puts the transition back:

    authoritative source artifact
      -> model invocation
      -> ModelDerivedArtifactV1          (produced, unverified)
      -> deterministic verifier
      -> ArtifactAdmissionResultV1
      -> AdmittedDerivedArtifactV1       (only from a successful admission)
      -> content / parent_excerpt / indexing

Three properties are structural rather than promised:

* **Unverified is a type, not a flag.**  ``ModelDerivedArtifactV1`` has no
  ``trusted`` field to flip, and ``AdmittedDerivedArtifactV1`` cannot be
  constructed from anything except a candidate *and* the admission result that
  admitted it.  One mutable object changing ``trusted=False`` to
  ``trusted=True`` in place would leave the lifecycle unreadable -- a reader
  holding a reference could not say which state it was in.
* **The verifier's oracle is the source, never another model.**  A check whose
  two sides both came from a model would prove only that the model agrees with
  itself.
* **Admission belongs here, not to the compiler and not to the provider.**  A
  provider returns a ``ModelResponseV1`` and has no opinion about whether its
  output is trustworthy; ``ContextCompilerV1`` receives only artifacts that were
  already admitted upstream.  Both are asserted by tests rather than assumed.

There is deliberately no ``ArtifactStore``.  What this phase needed was an
artifact state, a source relation and an admission result, and a store would
have been a much larger thing built for a requirement nobody has stated.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum

__all__ = [
    "AdmissionOutcomeV1",
    "AdmissionReasonV1",
    "AdmittedDerivedArtifactV1",
    "ArtifactAdmissionError",
    "ArtifactAdmissionResultV1",
    "ModelDerivedArtifactV1",
    "TransformationKindV1",
]


class ArtifactAdmissionError(RuntimeError):
    """Raised when an admitted representation would be built without an admission."""


class TransformationKindV1(str, Enum):
    """What a model was asked to do to the source artifact.

    One member, because one production path invokes a model over a source
    artifact today.  A kind invented for symmetry would be a policy nobody
    validated against a real call -- the mistake ``EvidenceDisclosureProfile``
    and ``ContextRoleV1`` have each already recorded avoiding once.
    """

    TABLE_CLEANING = "TABLE_CLEANING"


class AdmissionOutcomeV1(str, Enum):
    ADMITTED = "ADMITTED"
    REJECTED = "REJECTED"


class AdmissionReasonV1(str, Enum):
    """Why a transformation was admitted or refused.

    Five refusal reasons and one acceptance, which is the whole taxonomy.  F6's
    documented risk is numeric fidelity in a cleaned financial table, so the
    reasons are the ways *that* can fail -- not a general classification of
    model error, and not a severity scale.

    The four numeric reasons are kept apart from each other because they are
    different repairs for whoever reads the log.  A changed value says the model
    edited a number; an invented one says it added a figure the source never
    stated; a removed one says it dropped a row; a reassigned one says the
    numbers all survived and the *assignment* moved -- which a bag-of-numbers
    comparison would call perfect.
    """

    FIDELITY_PRESERVED = "FIDELITY_PRESERVED"
    NUMERIC_VALUE_CHANGED = "NUMERIC_VALUE_CHANGED"
    NUMERIC_VALUE_INVENTED = "NUMERIC_VALUE_INVENTED"
    NUMERIC_VALUE_REMOVED = "NUMERIC_VALUE_REMOVED"
    NUMERIC_VALUE_REASSIGNED = "NUMERIC_VALUE_REASSIGNED"
    STRUCTURE_UNVERIFIABLE = "STRUCTURE_UNVERIFIABLE"


@dataclass(frozen=True)
class ModelDerivedArtifactV1:
    """Content a model produced from a source artifact, before admission.

    This object is a *candidate*.  Nothing in the Harness treats it as evidence,
    it carries no trusted flag to be flipped, and the only way past it is an
    admission result.

    ``source_reference`` is the identity of the artifact this was derived from,
    and it is carried rather than reconstructed: the point of the whole contract
    is that model-generated content must never become indistinguishable from
    source content, and lineage is what keeps them apart after admission.

    The provider and model identifiers are optional because ingestion reaches
    its model through a plain HTTP call today, not through ``ModelProviderV1`` --
    so what is available is the model name, and the field says so rather than
    being filled with a synthetic invocation id.  The invocation contracts from
    H2A-3C are deliberately unchanged by this phase.
    """

    artifact_id: str
    content: str
    transformation: TransformationKindV1
    source_reference: str
    provider_id: str | None = None
    model_id: str | None = None
    invocation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        if not isinstance(self.transformation, TransformationKindV1):
            raise ValueError("transformation must be a TransformationKindV1")
        if not isinstance(self.source_reference, str) or not self.source_reference.strip():
            raise ValueError("source_reference must be a non-empty string")


@dataclass(frozen=True)
class ArtifactAdmissionResultV1:
    """The decision, and the smallest record of why.

    ``location`` names the structured position the decision turned on -- a row
    label and column header, taken from the *source* table.  It is bounded,
    source-derived, and answers "where" without carrying the model's content
    into a log.  A reason without a location is still actionable; a content dump
    is not something an admission record should be.
    """

    artifact_id: str
    outcome: AdmissionOutcomeV1
    reason: AdmissionReasonV1
    source_reference: str
    location: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        if not isinstance(self.outcome, AdmissionOutcomeV1):
            raise ValueError("outcome must be an AdmissionOutcomeV1")
        if not isinstance(self.reason, AdmissionReasonV1):
            raise ValueError("reason must be an AdmissionReasonV1")
        if self.outcome is AdmissionOutcomeV1.ADMITTED and (
            self.reason is not AdmissionReasonV1.FIDELITY_PRESERVED
        ):
            raise ValueError(
                "an admitted result must carry FIDELITY_PRESERVED; the reason is "
                "what was verified, and there is only one thing this verifier "
                "establishes"
            )
        if self.outcome is AdmissionOutcomeV1.REJECTED and (
            self.reason is AdmissionReasonV1.FIDELITY_PRESERVED
        ):
            raise ValueError("a rejected result must carry a refusal reason")

    @property
    def admitted(self) -> bool:
        return self.outcome is AdmissionOutcomeV1.ADMITTED


@dataclass(frozen=True, init=False)
class AdmittedDerivedArtifactV1:
    """A model-derived artifact that a deterministic verifier admitted.

    **Only constructible from a candidate and the result that admitted it**, and
    that is the phase's central invariant expressed in the type system rather
    than in a convention.  There is no public field to set and no ``trusted``
    attribute to assign: reaching this class *is* the authority transition, and
    it cannot happen because generation succeeded.

    Admission does not make the model the evidence authority.  The source
    document remains the factual grounding; what this represents is a *verified
    derived representation* of it, and ``source_reference`` travels with the
    content so a consumer can always tell which of the two it is holding.
    """

    artifact_id: str
    content: str
    transformation: TransformationKindV1
    source_reference: str
    admission_reason: AdmissionReasonV1
    provider_id: str | None = None
    model_id: str | None = None
    invocation_id: str | None = None

    def __init__(
        self,
        candidate: ModelDerivedArtifactV1,
        result: ArtifactAdmissionResultV1,
    ) -> None:
        if not isinstance(candidate, ModelDerivedArtifactV1):
            raise ArtifactAdmissionError(
                f"an admitted artifact is built from a ModelDerivedArtifactV1, got "
                f"{type(candidate).__name__}"
            )
        if not isinstance(result, ArtifactAdmissionResultV1):
            raise ArtifactAdmissionError(
                f"an admitted artifact is built from an ArtifactAdmissionResultV1, "
                f"got {type(result).__name__}"
            )
        if result.artifact_id != candidate.artifact_id:
            raise ArtifactAdmissionError(
                f"admission result is for {result.artifact_id!r}, candidate is "
                f"{candidate.artifact_id!r}"
            )
        if not result.admitted:
            raise ArtifactAdmissionError(
                f"artifact {candidate.artifact_id!r} was not admitted "
                f"({result.reason.value}); the admitted representation is only "
                f"constructible through a successful admission"
            )

        # Copied field by field rather than held as a reference to the candidate:
        # the candidate is not admitted, and an admitted object that held one
        # would keep unverified content reachable through it.
        for field in fields(candidate):
            object.__setattr__(self, field.name, getattr(candidate, field.name))
        object.__setattr__(self, "admission_reason", result.reason)
