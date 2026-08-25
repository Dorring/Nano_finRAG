# TV2-07R1+ßuÁ‚ùÁT Canonical Trusted V2 Readiness Evaluation

Status: TV2-07R1_CANONICAL_READINESS = PENDING

This phase is the formal, frozen production-readiness run after the TV2-07
harness verification. It does not change the V2 runtime, the router,
Conversation layer, or the production mode.

## Current decision

The 22-case TV2-07 wiring fixture remains a harness test only. It is not part
of a readiness headline. The existing nf-v2-17-fresh-blind-eval output is a
consumed/sealed evaluation and the R1 loader rejects it. No canonical,
eligible readiness query/label set is currently checked into this worktree, so
the formal run has not executed and no Canary decision is claimed.

The safe current state is:

~~~text
TV2-07 Framework             = IMPLEMENTED
TV2-07 Wiring Fixture         = VERIFIED (22 cases)
TV2-07R1 Canonical Readiness  = PENDING
Production Runtime            = V1
V2 Production Authority      = OFF
V2 Canary                    = NOT STARTED
~~~

## Frozen runtime inputs already available

The R1 preflight can verify the existing corpus/index freeze without rebuilding
it:

- corpus freeze: artifacts/evaluation/nf-v2-17-financial-corpus-v2/financial-corpus-v2-freeze.json
- searchable corpus SHA: 3ef3d8e772dfb2d4e2594d18efe3c101c4a4a3bb108e0faa0d75d11c667421a3
- searchable manifest SHA: 8b180c8a19f62ff358880878aa9dc78798b1d65ada69336e9b5953da0508d24c
- primary documents: 60
- index build and integrity reports: matching index-*.json files in the same directory
- Specialist model manifest: artifacts/runtime/nf-v2-21-local-specialist-integration/runtime-model-config.json
- Specialist checkpoint identity: d24_grounded_specialist_v3_lr5e6/model_000156.pt
- Specialist checkpoint SHA-256: 3bda9f032d7bfb29a3bdf7e0eeeee930a57a05e899e11e67e108483ca920894a

The checkpoint path and hash are read from the manifest and verified by the
preflight; they are not hard-coded into the evaluation runner.

## R1 dataset contract

The reserved directory is
tests/fixtures/tv2_07_canonical_readiness/. It contains a schema README but
no fabricated cases. A formal set must be a distinct, frozen JSONL pair with
at least 100 cases (preferably 120∫w^~)ﬁw200), and each query/label row must declare
one of:

~~~text
fresh_company_held_out
untouched_frozen_eval
~~~

Required strata include direct fact, multi-evidence, calculation, qualitative
synthesis, no-answer/insufficient evidence, wrong-period and unit/scale traps,
recovery/repair, and a real multi-turn subset. Multi-turn rows preserve the
user-visible input turns; the resolved standalone query is recorded only after
Conversation execution.

Gold labels are loaded after blind runtime execution. They are never put in
FinancialQueryRequest, and the R1 runner rejects raw conversation history and
Gold/reference fields in runtime requests.

## Preflight and execution

Use:

~~~text
PYTHONPATH=/mnt/disk/mxf/anaconda3/lib/python3.12/site-packages   .venv/bin/python scripts/evaluation/run_tv2_07_r1_readiness.py
~~~

The command performs a preflight first and writes a separate
canonical-r1/ artifact directory. Without --execute, or when the canonical
set is absent/ineligible, it writes preflight.json, manifest.json, and
decision.json with HOLD_FOR_QUALITY; it never invokes V1/V2.

A formal run requires injected V1 and real TV2-05 V2 factories:

~~~text
--execute
--v1-factory module:factory
--v2-factory module:factory
~~~

If any case contains input_turns, an explicit request factory is also
required:

~~~text
--request-factory module:build_request
~~~

The request factory must return the same blind FinancialQueryRequest for V1
and V2, with standalone_query produced by the Conversation layer. The Gold
scorer remains outside the runtime call.

## Hard safety gates

The formal run is blocked by any non-zero count for unsafe release, incorrect
release, false binding, false calculation execution, incorrect calculation,
unknown citation, unsupported claim, wrong period, wrong unit/scale,
assistant-history fact leak, unvalidated release, repeated repair,
internal V1 fallback, unexpected error/timeout, or Gold/runtime leakage.

A safe but over-conservative refusal is recorded separately from an unsafe
release. The final decision remains three-state:

~~~text
READY_FOR_CANARY
HOLD_FOR_QUALITY
BLOCKED_FOR_SAFETY
~~~

No numeric quality threshold is invented by the harness; coverage, latency, and
qualitative review are explicit review inputs before READY_FOR_CANARY.

## Artifacts

Formal output is kept separate from the wiring fixture:

~~~text
artifacts/evaluation/tv2-07-production-readiness/
  wiring-fixture/
  canonical-r1/
~~~

canonical-r1/ contains the frozen manifest, corpus/index/model preflight,
blind case results, safety gates, route/binding/calculation/recovery/repair
breakdowns, V1/V2 comparison, latency/error summaries, qualitative review, and
the three-state decision.

If a formal run changes code, runtime configuration, corpus/index, or model
identity, that run is invalidated and must restart from a new frozen SHA. No
consumed benchmark is repackaged as fresh, and TV2-08 remains disallowed until
the canonical R1 decision is established.
