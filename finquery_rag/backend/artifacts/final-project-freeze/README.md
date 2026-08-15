# Final project freeze artifacts

This directory is the authoritative NF-V2-11 closeout package.

Start with:

1. final-metrics-registry.json - scoped metrics separated into public, internal retrieval, component, validator, and strict E2E sections.
2. claim-registry.json - numerical claims with provenance, scope, and allowed wording.
3. training-lineage.json - R1 -> R2 -> R2.2 -> R2.3.1 lineage and disclosed training deviations.
4. architecture-freeze.json and production-decision.json - frozen roles and V1 production decision.
5. known-limitations.md, resume-evidence.md, and interview-evidence.md - safe communication boundaries.

The final decision is PROJECT_FREEZE_V1_PRODUCTION. NF-V2-15 adds the validated post-generation SemanticClaimVerifierV1 to the frozen V2 execution boundary; it does not authorize a production switch, additional grounded training, retrieval optimization, or LoRA/DPO runtime behavior.

NF-V2-15 comparison: the historical NF-V2-10 strict replay released 4 answers,
of which 3 were correct and 1 was post-hoc semantically unsafe. With
SemanticClaimVerifierV1, 3 answers were released, all 3 were correct, and the
unsafe unit claim was blocked. The 72-question result remains a frozen replay,
not fresh-blind E2E accuracy.
