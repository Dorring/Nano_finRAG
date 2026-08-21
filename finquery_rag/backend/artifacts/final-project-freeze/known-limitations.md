# Known limitations

This is the final NF-V2-11 project freeze.

- The internal 72-question benchmark is development-heavy; it is not a representative production prevalence estimate.
- NF-V2-06/NF-V2-09 Tier-B numbers are component-only evaluations using oracle/trusted evidence (oracle_evidence=true, fresh_blind=false). They are not end-to-end RAG accuracy.
- Runtime retrieval and trusted-evidence coverage remain limited; the final strict run had only 4 trusted DIRECT packets.
- Calculation E2E readiness remains insufficient: 0/11 trusted operand packets and 0/11 strict calculation successes.
- Multi-evidence E2E readiness remains insufficient: 0/16 complete dependency packets and 0/16 complete final answers.
- The Financial Specialist is not a reliable answerability judge. Evidence sufficiency and release policy stay outside the model.
- The historical NF-V2-10 deterministic-only run released one post-hoc semantic unsafe unit claim. NF-V2-15 adopts SemanticClaimVerifierV1 after generation; the re-evaluated frozen run detected and blocked that claim, yielding 0 semantic-unsafe final releases while preserving 3 correct releases.
- NF-V2-15 is a model-free replay of the sealed 72-question run: four sealed generation envelopes were replayed, with no new model or retrieval calls. It is not a fresh-blind quality estimate.
- V2 therefore remains non-production. Production stays V1 with fail-closed behavior.
- Latency figures have different scopes: V1 is a full 72-query run, whereas V2 measures four trusted DIRECT generations with retrieval timing reused from a sealed trace; it is not a matched hardware-only comparison.
