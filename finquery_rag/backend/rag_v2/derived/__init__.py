"""Model-derived artifacts and their deterministic admission.

    authoritative source artifact
      -> model invocation
      -> ModelDerivedArtifactV1      (produced, unverified)
      -> deterministic verifier
      -> ArtifactAdmissionResultV1
      -> AdmittedDerivedArtifactV1   (only from a successful admission)
      -> trusted retrieval / evidence

The invariant the package exists to hold:

    model output != admitted artifact

until a deterministic contract succeeds.  The failure it closes was not that a
model might write something poor; it was that *generation succeeding* was
treated as admission succeeding, so a candidate truth became an evidence truth
with no authority transition in between.  After the transition, nothing
downstream could tell that a number had been typed by a language model rather
than read by the extractor.

What is not here, deliberately: an ``ArtifactStore``.  The phase needed an
artifact state, a source relation and an admission result; a store would have
been a much larger thing built for a requirement nobody has stated.

Nothing in this package imports ``src``.  ``contracts.py`` is pure; ``tables.py``
depends only on the shared financial semantics, which live in ``rag_v2`` beside
it.
"""

from .contracts import (
    AdmissionOutcomeV1,
    AdmissionReasonV1,
    AdmittedDerivedArtifactV1,
    ArtifactAdmissionError,
    ArtifactAdmissionResultV1,
    ModelDerivedArtifactV1,
    TransformationKindV1,
)
from .tables import NumericFactV1, numeric_facts_of, verify_table_fidelity

__all__ = [
    "AdmissionOutcomeV1",
    "AdmissionReasonV1",
    "AdmittedDerivedArtifactV1",
    "ArtifactAdmissionError",
    "ArtifactAdmissionResultV1",
    "ModelDerivedArtifactV1",
    "NumericFactV1",
    "TransformationKindV1",
    "numeric_facts_of",
    "verify_table_fidelity",
]
