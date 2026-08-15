# NF-V2-11 final project summary

- Base/freeze commit: eab5d2669e08b6f68b1dce1442d676860da68554
- Production: V1
- V2 switch: false
- Final Financial model: finquery-finance-grounded-v3-r231
- Role: financial_selective_generator
- Grounding training: closed
- Retrieval optimization: closed
- Architecture: frozen

The final architecture is a single Supervisor followed by retrieval/query rewrite, route-specific trusted evidence, deterministic calculation, grounded Financial Specialist generation, deterministic validation, and fail-closed recovery. The Specialist is never the answerability judge or final safety authority.

Component evaluation shows meaningful grounding/numeric/calculation gains, but strict E2E coverage is constrained by trusted evidence readiness. The final E2E result was 3/64 answerable final-correct, 4/64 released, 8/8 no-answer refusals, 0 false execution, 0 false binding, and 68/72 fail-closed. One post-hoc semantic unsafe release remains disclosed. The decision is PROJECT_FREEZE_V1_PRODUCTION.

See final-metrics-registry.json and claim-registry.json for provenance and scope. Checkpoints and failed-run binaries are intentionally not committed.
