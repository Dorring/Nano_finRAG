# Resume evidence (NF-V2-11 freeze)

Use only the qualified claims in claim-registry.json.

- Designed and froze a single-Supervisor Financial RAG architecture: query/rewrite and route planning -> trusted evidence -> deterministic calculation -> grounded Financial Specialist -> deterministic validator -> fallback/fail-closed.
- Kept the Financial Specialist narrowly scoped to grounded generation; it is not the Supervisor, an answerability judge, a calculator, or final safety authority.
- Built a trusted-evidence and fail-closed runtime. In the strict final 72-question run, no-answer refusal was 8/8, false execution was 0, false binding was 0, and fail-closed was 68/72.
- Grounded Financial SFT materially improved component behavior: 47/64 Grounded, 52/64 Numeric, 7/11 canonical Calculation preservation, and 5/5 explicit Multi grounded. These are oracle-evidence component results, not E2E accuracy.
- Final strict E2E result was 3/64 grounded/final-correct with 4/64 released; this is why production remained V1. Do not present 47/64 as production or E2E accuracy.
- The public retrieval statement is limited to the pre-frozen calibration: Qwen reranker Recall@5 = 88.5639% on the specified public T2-RAGBench contract. The internal frozen runtime identity trace recorded 13/80 source Recall@5.
- Timing evidence is qualified by scope: V1 average 2460.234 ms versus the V2 trusted-generation subset average 293.341 ms; these are not matched full-pipeline benchmarks.

## NF-V2-15 semantic claim verifier (qualified)

- Adopted `SemanticClaimVerifierV1` as a post-generation, fail-closed claim/evidence check; the pre-generation sufficiency gate remains non-mandatory.
- On the same sealed 72-question replay, the verifier retained 3/3 previously correct released answers and blocked the one historical unsupported-unit release: 3 released, 3 correct, 0 semantic-unsafe final releases, 69/72 fail-closed.
- This is frozen replay/component evidence with zero new model or retrieval calls, not fresh-blind E2E accuracy and not a Production V2 acceptance claim. Production remains V1.
- R1 + LoRA/DPO is recorded as `LORA_DPO_INEFFECTIVE` and is not integrated.
