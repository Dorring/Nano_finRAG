"""The Context Compiler: governed, per-invocation, model-visible context.

The compiler's responsibility, frozen:

    already-admitted authoritative artifacts
        -> deterministic model-visible selection
        -> Disclosure Authority projection
        -> ContextBudget enforcement
        -> temporary AgentContextPackV1

It does **not** own retrieval, evidence admission, consensus or conflict
resolution, slot binding, calculation, claim verification, replanning, memory,
artifact storage, prompt repair, or durable runtime state.  It is not a second
authority, and a pack is not state: it exists for one invocation and is
discarded with that invocation's trace.

Only the SPECIALIST role exists so far -- B3 is the first boundary and is the
only one the framework is proven against.  B5, B6 and V1 add themselves when
their requirements are known.  A role invented ahead of a real prompt would be a
field list nobody validates, which is the mistake ``disclosure.py`` already
records avoiding once.
"""

from .compiler import (
    ContextCompilerV1,
    ContextRequestV1,
    EvidenceSelectionPolicyV1,
    SelectionResultV1,
)
from .contracts import (
    AgentContextPackV1,
    ContextBudgetAccountingV1,
    ContextBudgetUnsupported,
    ContextBudgetV1,
    ContextReferenceV1,
    ContextReferencesV1,
    ContextRoleV1,
    ContextSelectionEntryV1,
    ContextSelectionReasonV1,
    ContextSelectionTraceV1,
    ContextSupportGroupV1,
    ExactTokenCounterV1,
    PackIntegrityError,
    UnknownContextRole,
)
from .specialist import (
    SPECIALIST_INVOCATION,
    SpecialistContextPolicyV1,
    admitted_specialist_evidence,
    evidence_identity,
    specialist_context_request,
)

__all__ = [
    "AgentContextPackV1",
    "ContextBudgetAccountingV1",
    "ContextBudgetUnsupported",
    "ContextBudgetV1",
    "ContextCompilerV1",
    "ContextReferenceV1",
    "ContextReferencesV1",
    "ContextRequestV1",
    "ContextRoleV1",
    "ContextSelectionEntryV1",
    "ContextSelectionReasonV1",
    "ContextSelectionTraceV1",
    "ContextSupportGroupV1",
    "EvidenceSelectionPolicyV1",
    "ExactTokenCounterV1",
    "PackIntegrityError",
    "SPECIALIST_INVOCATION",
    "SelectionResultV1",
    "SpecialistContextPolicyV1",
    "UnknownContextRole",
    "admitted_specialist_evidence",
    "evidence_identity",
    "specialist_context_request",
]
