# NF-V2-16 Bounded Adaptive RAG

Base 74910f27d9675a6537914581f9ff74ddd7d57f51; branch xp/nf-v2-16-bounded-adaptive-rag.

This is a deterministic component experiment. No model calls, retrieval calls, fine-tuning, benchmark reads, or production changes were made.

## Results

- Bounded loop: passed; maximum observed tool calls 2, no infinite loop, no invalid transition, no budget violation.
- Recoverable/replan: A, C, and N recovered only after a concrete second observation.
- No-progress: B and M fail closed with NO_PROGRESS.
- Temporal cases: 9/9 expected decisions, including annual/quarter succession, YTD scope distinction, FY succession, explicit supersession, same-scope conflicts, cross-source compatibility, same-time rating conflict, and ingestion-time trap.
- Safety: false binding 0, false execution 0, unsafe release 0; NF-V2-15 sealed four-case regression recorded as safe retained 3/3 and unsafe blocked 1/1.

The frozen 72-question replay is optional under the task contract and was not run: this experiment does not tune or reinterpret that benchmark. Production remains V1 and the switch remains false.
