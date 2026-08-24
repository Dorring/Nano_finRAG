# I5-R1 API Integration Verification Closure

Status: **I5_IMPLEMENTED_API_INTEGRATION_VERIFICATION_PENDING**

This verification-only sprint did not change Conversation behavior, Resolver behavior,
V1 execution, legacy rewrite, /query/stream, or any benchmark artifact.

## Scope and baseline

- Base: c3561c2c5f85de8aeadc279c62b0e4657dc81989
- Branch: codex/conversation-shadow-i5-r1
- Production runtime: V1
- MULTITURN_CONTEXT_MODE: off and shadow verified; on rejected
- Benchmark rerun: false
- /query/stream: unchanged legacy V1 path
- V2: not integrated

## Dependency audit

The five previously skipped endpoint tests were skipped by the test helper because
the base interpreter did not have the API imports. This was an incomplete test
environment, not an undeclared production dependency.

| Import | Declared project dependency | Verification |
| --- | --- | --- |
| jose | python-jose[cryptography]>=3.5.0 | present in backend pyproject.toml and uv.lock |
| bcrypt | bcrypt==4.0.1 | present in backend pyproject.toml and uv.lock |
| passlib | passlib[bcrypt]>=1.7.4 | present in backend pyproject.toml and uv.lock |
| multipart | python-multipart>=0.0.21 | present in backend pyproject.toml and uv.lock |
| fitz | pymupdf>=1.24.0 | present in backend pyproject.toml and uv.lock |
| camelot | camelot-py[cv]>=1.0.9 | present in backend pyproject.toml and uv.lock |

An isolated backend .venv was created with uv sync --locked. The production
Conda environment was not modified. The backend lock does not declare pytest as a
runtime dependency, so the test runner used the existing local pytest installation
as a test-only harness; no production package contract was changed.

## API test execution

All five formerly skipped tests in
tests/test_query_runtime_integration.py were executed (zero skips):

- 4 passed:
  - calculation/blocked public-shape preservation
  - session message write count
  - unresolved request identity/scope
  - engine error HTTP semantics
- 1 failed:
  - test_adapter_and_direct_paths_have_identical_public_payload

The failure is an existing I3 adapter parity defect, isolated to the engine-call
input mapping. The public JSON payload is identical, but the direct path passes
conversation_history=None while the adapter path normalizes the same explicit
metadata value to conversation_history=[]. No R1 production code changed this
behavior. Because the test compares the complete engine call contract, the I5-R1
API gate cannot be sealed until that I3 parity issue is fixed and reverified.

Combined result for the original API suite plus the new R1 endpoint suite:

1 failed, 9 passed, 0 skipped

## New real endpoint verification

tests/conversation/test_shadow_api_integration.py contains four endpoint-level
tests, all passing:

- test_off_shadow_response_and_raw_history_parity
  - off and shadow return the same HTTP status and public payload;
  - raw SessionManager messages remain equivalent;
  - shadow observation is internal only.
- test_shadow_provider_and_sqlite_failures_do_not_change_v1
  - timeout, invalid-provider response metadata, and SQLite shadow-write failure
    leave the official V1 response successful.
- test_restart_current_turn_once_clear_and_user_isolation
  - SQLite state survives store recreation;
  - the current user turn reaches the resolver exactly once;
  - clear removes structured state;
  - (user_id, session_id) isolation holds.
- test_active_mode_is_rejected_without_calling_v1
  - MULTITURN_CONTEXT_MODE=on is rejected and V1 is not invoked.

The endpoint tests use a deterministic offline resolver/client and a fake injected
RAG engine; they exercise the real FastAPI/auth/session/shadow wiring without network
model calls or benchmark questions.

## Focused regressions and static checks

- I1/I2/I4 conversation, SQLite, adapter, contract and shadow tests: 68 passed
- New R1 endpoint tests: 4 passed
- Targeted Ruff for R1 tests and runtime regression files: passed
- Python compilation: passed
- A broader Ruff check still reports two pre-existing findings in
  src/conversation/resolver.py (F401 and F841); they are outside the R1
  diff and were not changed.
- Only the new verification test file is changed by R1; no production runtime,
  stream, rewrite, or validator file is modified.

## Gate result

| Gate | Result |
| --- | --- |
| Correct locked API dependency environment | PASS |
| Five former skips actually executed | PASS |
| off endpoint behavior | PASS |
| shadow endpoint behavior | PASS |
| Shadow failure isolation | PASS |
| Restart, clear, and user/session isolation | PASS |
| Current turn exactly once | PASS |
| on configuration rejection | PASS |
| Production response difference caused by shadow in R1 scenarios | 0 |
| Direct-vs-adapter engine-call parity | **FAIL** |
| I5 production shadow seal | **BLOCKED** |

Therefore:

I5 implementation                         = COMPLETE
API integration verification               = PENDING
CONVERSATION_SHADOW_INTEGRATED             = NOT SEALED
Production runtime                         = V1

The remaining action is an I3 adapter-contract fix for the None versus [] history
mapping, followed by rerunning the five API tests. That fix is deliberately outside
this verification-only sprint.
