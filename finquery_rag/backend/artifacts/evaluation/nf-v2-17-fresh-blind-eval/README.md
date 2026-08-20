# NF-V2-17B1/B2 Fresh-Blind Trusted Agentic RAG Pack

This pack is a frozen, two-pass deterministic annotation of 120 questions from the sealed primary holdout (GOOGL and AMZN, 12 filings). Three A5 version sidecars are reserved only for version-temporal/ambiguity cases and remain outside the primary 60-file corpus count.

- Corpus freeze: `63620b2183c4635f1ecff974935bc81a4d8ce678c72e72e94155d8f0a96e6929`
- Searchable corpus: `3ef3d8e772dfb2d4e2594d18efe3c101c4a4a3bb108e0faa0d75d11c667421a3`
- Reservation: `8708ecf5b0f5ee056cf003238a510345c96cce720a41709d5eeb0c5d47e1dc23`
- Final system execution: **not performed**
- Model calls/training/tuning: **0**
- Human double-review claim: none; this is two-pass annotation/QC plus a 20-item manual review packet.

The runtime projection contains only `question_id`, `question`, and the authorized blind-corpus handle. Gold evidence, required slots, expected replans, conflict labels, and reference answers are evaluation-side only. B3 is the single final execution gate; failures must not trigger benchmark-specific fixes.

Dense documentation: the frozen dense index has 8,554 vectors over TEXT/TABLE coarse objects, while 95,154 searchable units remain available through the tiered lexical/structured path; TABLE_ROW units are not accidental missing dense vectors.
