# Final project freeze artifacts

This directory is the authoritative NF-V2-11 closeout package.

Start with:

1. final-metrics-registry.json - scoped metrics separated into public, internal retrieval, component, validator, and strict E2E sections.
2. claim-registry.json - numerical claims with provenance, scope, and allowed wording.
3. training-lineage.json - R1 -> R2 -> R2.2 -> R2.3.1 lineage and disclosed training deviations.
4. architecture-freeze.json and production-decision.json - frozen roles and V1 production decision.
5. known-limitations.md, resume-evidence.md, and interview-evidence.md - safe communication boundaries.

The final decision is PROJECT_FREEZE_V1_PRODUCTION. No additional grounded training, retrieval optimization, model checkpoint selection, or production switch is authorized by this freeze.
