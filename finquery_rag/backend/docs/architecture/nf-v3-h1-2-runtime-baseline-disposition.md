# NF-V3 H1.2 — Runtime / Test Baseline Disposition

Status: **CLOSED — 0 production correctness defects, 39/39 classified**

Scope was deliberately narrow: classify the failing tests, decide what the
runtime defaults should be, and make runtime-sensitive tests declare what they
exercise. No harness capability was added and no production default was changed.

## The result

```
before   4068 collected = 3886 passed + 39 failed + 143 skipped
after    4070 collected = 3927 passed +  0 failed + 143 skipped
                          0 xfailed, 0 xpassed, 0 deselected, 0 errors
```

Both rows are raw `pytest` summary lines, and each identity holds:
`3886 + 39 + 143 = 4068` and `3927 + 143 = 4070`.

The two-test difference is the declaration-contract module added by this change.
The 39 failures became passes in place: `3886 + 39 + 2 = 3927`.

An earlier revision of this section quoted `4047 = 3865 + 39 + 143` as the
"before" row. That row is real but belongs to `7d39b34`, two commits earlier --
it was the H1.1 state, not the state this change started from. Quoting a
self-consistent number from the wrong commit is the same failure mode this phase
exists to remove, so it is recorded rather than quietly corrected.

## What each failure actually was

Every failure was attributed by running it and reading its own failure line, not
by heredity from the summary.

| # | Class | Count | Cause |
| --- | --- | --- | --- |
| A | Implicit `FINANCIAL_RUNTIME_MODE` | 34 | The module patches `src.main.get_rag_engine` — which only the V1 lifecycle calls — and asserts V1 response shapes, but never declared a mode. It inherited the `v2` default and hit the fail-closed builder check. |
| B | Implicit `MULTITURN_CONTEXT_MODE` | 3 | Posts an under-specified question with no session id and never declared a mode. It inherited `on` and got the clarification gate instead of the ordinary response it asserts. |
| C | Stubbed `session_manager` with no usable `db_path` | 2 | `getattr(MagicMock(), "db_path", None)` returns a Mock, not `None`, so the store raised `db_path must be a non-empty string`. Only reachable because A and B let the request get that far. |
| D | **Production correctness defect** | **0** | — |
| E | External artifact / environment | 0 | Already handled by H1.1's artifact guard. |
| F | Deprecated legacy behaviour | 0 | — |

C is worth naming because the failure *looked* like a production bug and is a
test-setup one: the code reads `getattr(session_manager, "db_path", None)` and a
bare `MagicMock` answers every attribute. The store then fails closed on the
non-string, which is the correct response to a non-string path.

## The product-contract question

**The default must remain `v2`. It was not changed.**

The contract is documented, not incidental
(`docs/showcase/trusted-runtime-v2-production-integration.md:11-23`, reproduced
from the code that implements it):

| Mode | Official result | V2 behaviour |
| --- | --- | --- |
| `v1` | Legacy V1 | Explicit rollback/compatibility path |
| `shadow` | V1 | V2 as bounded observation only |
| `v2` | Trusted V2 | Official V2 result; no V1 call and no fallback |

> "The default is v2." — and `v2` is validated at
> `src/main.py:_validate_financial_runtime_configuration`, which fails closed
> when no real runtime builder is configured.

Supporting evidence: `src/runtime/shadow_contracts.py:39` resolves an absent
value to `v2`; `src/services/health.py:221` reports `v2`; `.env.example:49`
sets `v2` and `:57` sets the builder. It changed deliberately in `b84fa3a`
("feat(runtime): activate trusted v2 production integration", 2026-08-26).

**Which tests were testing default selection?** None of the 39. That behaviour is
covered by `test_runtime_router_shadow.py::test_mode_defaults_to_v2_and_accepts_explicit_runtime_modes`
and, for conversation mode, `tests/conversation/test_shadow_service.py:309`.
Both pass, and both were already correct.

**Would changing the default change production semantics?** Yes. Defaulting to
`v1` would route all production `/query` traffic to the legacy engine and
silently undo the V2 activation. That is why it was not done.

## The fix

The five modules now declare both defaults through one fixture,
`legacy_single_turn_endpoint` in the root `conftest.py`, whose docstring states
why a test that patches the V1 engine must say so rather than inherit.

Two tests keep it true:

- **`test_every_endpoint_test_module_declares_its_runtime`** fails if a module
  patches `src.main.get_rag_engine` and drives `/query` or `/query/stream`
  without declaring a mode — the invariant that was silently violated. It holds
  across all 10 engine-patching modules today; the two that do not declare are
  the ones that never drive an endpoint.
- **`test_the_declaration_fixture_matches_the_documented_defaults`** fails if
  either ambient default moves, forcing the next person to decide whether the
  fixture still describes what those tests exercise — rather than rediscovering
  it as 39 unrelated failures.

## One documentation defect found and not yet fixed

Three showcase documents state the default is `off`. It is `on`:

| Document | Says | Line |
| --- | --- | --- |
| `docs/showcase/i6-active-conversation-integration.md` | "The default remains off." | 20 |
| `docs/showcase/query-stream-conversation-parity.md` | "The default remains MULTITURN_CONTEXT_MODE=off." | 41 |
| `docs/showcase/trusted-runtime-v2-integration-audit.md` | "Default `MULTITURN_CONTEXT_MODE`: `off`" | 9, 229 |

All three were written on 2026-08-25; the default changed to `on` in `b84fa3a`
on 2026-08-26 (`src/conversation/config.py:35`). They are accurate as records of
the phase they describe and wrong as statements about the system today. For a
financial runtime, a reader who believes active conversation is opt-in when it
is on has the wrong safety model, so this is recorded rather than left.

Left unfixed here because correcting historical seal documents is a decision
about how this project treats its own records, not a test-hygiene fix. The
recommendation is a dated note at the top of each rather than an edit to the
body.
