# I3-R1 Legacy Adapter Invocation Parity Fix

Status: **I3-R1 PASS**

## Scope

This fix is limited to the LegacyFinancialRuntimeAdapter invocation contract.
No RAGEngine, RAGOrchestrator, SessionManager, legacy rewrite, public response,
/query/stream, ConversationContextManager, or V2 code was changed.

The source of truth is the existing direct V1 path:

- omitted/None conversation history: None
- explicitly supplied empty history: []

## Root cause

LegacyFinancialRuntimeAdapter._request_options used:

- metadata.get("conversation_history", [])
- _history_list(None) -> []

The direct endpoint path supplied None for a stateless request, so the adapter
changed the internal RAGEngine invocation even though the public response was
unchanged.

## Fix

The adapter now:

1. defaults an omitted conversation_history to None;
2. preserves an explicit None as None;
3. preserves an explicit [] as an empty list;
4. retains the existing validation and normalization for non-empty history.

This preserves Optional semantics instead of collapsing None and [].

## Contract tests

Added:

- test_adapter_preserves_none_conversation_history
- test_adapter_preserves_explicit_empty_conversation_history

Updated the adapter default expectation to match the canonical direct V1 path.

## Verification

- I1 contract, I2 adapter and I3 API parity tests: 19 passed
- Direct versus adapter invocation parity: PASS
- Public response parity: PASS
- Targeted Ruff: PASS
- Python compilation: PASS
- No benchmark or model calls: confirmed

This commit is the small I3-R1 repair required before sealing the I5 Shadow
API integration gate.
