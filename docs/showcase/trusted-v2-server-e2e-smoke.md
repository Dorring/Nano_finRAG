# Trusted Financial Runtime V2 — Server E2E Smoke Record

Date: 2026-09-12  
Scope: isolated server worktree; this is an integration smoke record, **not** a held-out readiness benchmark or a model-quality claim.

## What ran

The server exercise used the real `build_trusted_v2_runtime_for_request()` factory with:

- the production V2 coordinator, Supervisor, bounded runtime, R4-compatible fact index, Semantic Binder, deterministic calculator, validator and release gate;
- the configured DeepSeek-compatible Supervisor and Binder providers;
- the sealed local Financial Specialist checkpoint, loaded successfully on the isolated CUDA device;
- an isolated worktree and temporary test configuration. No deployed service, production router, session database or `online.env` was changed.

Network egress for the provider was limited to a temporary, loopback-only SSH relay authorized for this test. Evidence was never copied into this repository or emitted by the result summaries.

## Verified controls

| Control | Result |
| --- | --- |
| Focused Trusted V2 regression + compile | 94 passed |
| Focused Ruff check | passed |
| Real Specialist checkpoint load | passed; CUDA device initialized |
| Structured single-fact route | `ANSWER` / `RELEASED`; one admitted evidence and citation |
| Deterministic calculation route | `ANSWER` / `RELEASED`; two admitted evidence/citations and one calculation lineage ID |
| Unsupported-evidence request | `FAIL_CLOSED` / `NOT_RELEASED`; `MISSING_SLOT`, then bounded budget termination |
| Multi-evidence conflict request | `FAIL_CLOSED` / `NOT_RELEASED`; `EVIDENCE_CONFLICT` |
| Qualitative multi-evidence request in this corpus | `FAIL_CLOSED` / `NOT_RELEASED`; `MISSING_SLOT`, then bounded budget termination |

The negative outcomes are expected safety behavior: no candidate answer, raw retrieval candidate, or unbound evidence was released as a financial answer.

## What this record does and does not establish

It establishes that the real V2 factory can execute in the server environment across released fact/calculation paths and fail-closed paths, and that the Specialist artifact itself is loadable in the isolated runtime.

It does **not** establish aggregate accuracy, Recall@K, generation correctness, false-binding rate, P95 latency, held-out generalization, or a production-readiness decision. In particular, the qualitative/multi-evidence examples above did not release, so they must not be presented as successful end-to-end Specialist generations.

## Interview-safe wording

> I validated the real V2 runtime on an isolated GPU server with its configured provider, retrieval assets, Binder, calculator, validator and local Specialist checkpoint. The smoke run released a grounded fact and a deterministic calculation with structured provenance, while unsupported, missing-slot and conflicting-evidence cases failed closed. I treat this as an integration result rather than a benchmark; I do not claim aggregate quality or latency without a frozen evaluation run.