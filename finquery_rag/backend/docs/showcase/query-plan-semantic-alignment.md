# Query-to-Plan Semantic Alignment

`SupervisorPlan` answers whether a plan is structurally valid.  It does not
prove that the plan means the same thing as the user's question.  The runtime
therefore uses two deterministic gates:

```text
user query
    ↓
explicit semantic frame
    ↓
SupervisorPlan
    ↓
query ↔ plan alignment
    ├── ALIGNED      → retrieval may start
    ├── MISMATCH     → fail closed before retrieval
    ├── AMBIGUOUS    → fail closed; do not guess
    └── UNKNOWN      → compatibility policy or strict direct-fact policy
    ↓
R4 retrieval → candidate evidence → Semantic Binder
    ↓
bound evidence ↔ query/plan cross-check
    ├── ALIGNED      → downstream capability may continue
    └── MISMATCH     → discard bound evidence and fail closed
```

This covers the important case where a plan is legal but answers a different
metric, such as a query for `operating income` paired with a plan for `net
income`.  A structurally valid wrong-metric fact is also rejected after the
Binder, before calculation or generation.

## What is checked

The explicit frame uses a finite, deterministic production vocabulary.  Alias
normalization maps equivalent phrases such as `sales` and `revenue` to one
metric ID, while `operating income` and `net income` remain distinct.  Annual
and quarterly expressions normalize to IDs such as `FY2024` and `FY2024-Q1`.
The frame also records recognized operation, entity, and reporting-scope
mentions.  Entity and scope expectations are only enforced when supplied by
an already-authorized caller; the gate never invents a company or scope.

The pre-retrieval gate rejects explicit metric, period, operation, or optional
entity/scope contradictions.  Multiple explicit targets that cannot be
represented by the plan are `AMBIGUOUS`, not silently reduced to one target.
Derived comparison periods remain valid: a plan may add the prior period when
the query names only the current period.

The post-Binder gate independently checks admitted facts against their bound
slots and the explicit query.  Retrieval candidates are not treated as
verified evidence; only Binder-admitted IDs are retained as V2 provenance.
Evidence IDs, citations, and calculation IDs are always taken from structured
runtime objects, never parsed from answer text.

## UNKNOWN policy

The default `compatibility` policy preserves generic operation-only queries
such as `Compare the two years`.  Production construction uses
`strict_direct_fact`: a direct-fact plan with no recognized metric is blocked
instead of allowing the Supervisor to guess one.  This policy does not turn
calculation queries into a second conversational resolver.  Even when the
metric is unknown, explicit periods and operations are still checked.

The policy and both gate decisions are emitted in runtime metadata and the
structured trace, making a fail-closed decision auditable without answer-text
parsing, retrieval heuristics, or another model call.

## Contract boundary

`RequiredSlot` and `SupervisorPlan` remain frozen internal dataclasses.  They
are the canonical domain contract and are validated before execution.  A
wholesale Pydantic migration would improve serialization at an external
boundary, but it cannot identify that `operating income` and `net income` are
different meanings.  Semantic alignment therefore remains an explicit layer
instead of being hidden inside schema validation.

This is a deterministic safety gate, not a complete natural-language
understanding system.  Its finite vocabulary cannot prove every business
qualifier, entity alias, or domain-specific synonym.  Unknown or unsupported
meaning remains visible as `UNKNOWN`, and trusted evidence checks provide the
second firewall rather than guessing.
