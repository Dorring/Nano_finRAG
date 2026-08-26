# I5-R1 Conversation Shadow API Integration Seal

Status: **CONVERSATION_SHADOW_INTEGRATED = PASS**

This is the post-I3-R1 closure of the earlier verification report. The first
run correctly remained pending because the adapter passed conversation_history=[]
where the direct V1 path passed None. I3-R1 fixed that invocation-contract drift
without changing the direct path.

## Commits

- I5-R1 verification tests/report: f32e6574a5bda7b1db8e307ccd4067e0b9d5390e
- I3-R1 parity fix: 2a69c6e951184ef5108f4bbb4606a5511f9cbd58

## Final verification

- Formerly skipped API tests: 5/5 passed, 0 skipped
- New Shadow endpoint tests: 4/4 passed
- I1 contract + I2 adapter + I3 API parity + I5 Shadow API suite: 36 passed
- Conversation, SQLite state, lifecycle and Shadow regression suite: 44 passed
- Direct versus adapter invocation parity: PASS
- Direct versus adapter public response parity: PASS
- Production response difference caused by Shadow: 0
- Targeted Ruff: PASS
- Python compilation: PASS

The backend dependency environment was created from uv.lock with uv sync --locked
in an isolated worktree .venv. The production Conda environment was not modified.
No benchmark, model generation, or external resolver call was used.

## Verified gates

- MULTITURN_CONTEXT_MODE=off preserves the V1 endpoint path and does not invoke
  Shadow.
- MULTITURN_CONTEXT_MODE=shadow executes the real FastAPI/session/shadow wiring,
  but the official request remains original_query == standalone_query and
  query_as_resolved == false.
- Shadow provider timeout, invalid-provider metadata and SQLite shadow-write
  failure do not change the official V1 response.
- Shadow state survives store recreation, current user input reaches the resolver
  exactly once, clear removes state, and user_id + session_id isolation holds.
- Shadow ambiguity and internal observations remain out of the public response.
- Assistant text is not promoted to evidence or a calculator operand.
- MULTITURN_CONTEXT_MODE=on is rejected rather than silently treated as shadow.
- The legacy rewrite and V1 validation/release behavior were not changed.
- /query/stream remains the legacy direct V1 path in this phase.
- V2 remains not integrated.

## Production status

The Conversation Shadow capability is now verified in the production /query
lifecycle, with SQLite structured state and failure isolation. The default mode
remains off, so official users continue to receive V1 behavior without active
context rewriting.

The next gate is I6 Active Conversation Resolution + Legacy Rewrite Bypass. That
gate must separately prove query_as_resolved=true and active standalone-query
routing; it is not enabled by this seal.
