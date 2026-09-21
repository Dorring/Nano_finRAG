# NF-V3 H2A-1D — Content Identity: Consumer Audit

Status: **AUDIT ONLY. No hash semantics changed.**

Baseline: `5aff975`. Companion to `nf-v3-h2a-context-artifact-runtime-plan.md`.

## The defect being scoped

`EvidencePacketV1.content_hash` hashes `to_dict()`, which includes
`evidence_id`. So two packets with byte-identical content under different
evidence ids hash differently — it is a *serialized-record* hash wearing a
content-hash name.

```python
EvidencePacketV1.from_mapping({..., "evidence_id": "E-A"}).content_hash
EvidencePacketV1.from_mapping({..., "evidence_id": "E-B"}).content_hash
# identical content, different hashes
```

## The question that decides the fix

Before changing it: has this value reached anything persisted, sealed, cached
or publicly serialized? If it has, correcting the semantics silently invalidates
those records.

**It has not.**

## Consumer audit

| Consumer | Where | Needs | Evidence |
| --- | --- | --- | --- |
| `ProgressDetector.signature` | `adaptive_progress.py:26` | **semantic content identity** | `"evidence": sorted(packet.content_hash for packet in packets)` — the only functional reader |
| `AdaptiveRAGStateV1.evidence_hashes` | `adaptive_contracts.py:305`, populated `:410` | **nothing — no reader** | grep finds only the field and its assignment |
| `AdaptiveRAGStateV1.to_dict` | `adaptive_contracts.py` → `AdaptiveRunResultV1.to_dict` (`adaptive_state_machine.py:47`) | — | the only serializer; used by the harness's own return value |
| V2 execution trace | `trusted_v2_coordinator.py` | — | grep for `content_hash` / `evidence_hashes` / `progress_signatures`: **no hits** |
| public response (`V2ExecutionOutcome`) | `trusted_v2_contracts.py` | — | not among its fields |
| fixtures / sealed artifacts / cache keys | `tests/`, `artifacts/`, `scripts/evaluation/` | — | the `content_hash` there is a **different field** (candidate-level, from frozen artifacts, e.g. `frozen_candidate_integrity.py:55`) |
| `document_registry.content_hash` | `src/services/document_registry.py:24` | — | **different concept**: a document-level dedup hash persisted in SQLite, computed by `DocumentRegistry.content_hash(chunks)`, never from a packet |
| persistence | `sqlite_store.py:157`, `context_budget.py:71` | — | those serialize `DialogueState`, a different class in `src/conversation/` |

## Verdict: case A — runtime-internal

`EvidencePacketV1.content_hash` is read by exactly one functional consumer,
inside the adaptive loop, during a single run. It is not persisted, not sealed,
not cached, not published. Correcting its semantics breaks no stored record and
no external contract.

**The recommendation is therefore to correct it directly rather than add a
parallel `content_fingerprint`.** The "legacy plus new field" pattern is the
right call when a value has escaped into a contract; this one has not, and
adding a second identity beside a wrong one would leave the wrong one available
to be reached for again.

That recommendation carries one condition, below.

## The condition: ProgressDetector's semantics must be settled first

`signature()` is what "no progress" means, so changing what it hashes changes
*when the loop decides it is stuck*. That is not a refactor; it is a behaviour
change to the control plane, and it is the whole reason this audit exists.

Today the signature includes `sorted(packet.content_hash ...)`, which is
per-*instance*. Two rounds that retrieve the same row under a different
evidence id therefore look like different evidence, and progress is reported
where there is none — the loop keeps going on evidence it already has.

So the change is a real improvement to the loop's stop condition, and it must be
made with:

- a test that a repeated round returning the same content under a new id is
  detected as **no progress**, and that the run terminates rather than burning
  budget;
- a test that genuinely different content is still progress;
- a check that no existing fixture depends on the current per-instance
  behaviour — the readiness suite should be unaffected, since its fixtures
  return new keys only on genuine recovery.

## Three identities, which D must keep separate

C established that provenance and content are different questions. D must not
undo that by folding source/page into the new identity:

```
evidence_id          instance identity      "which runtime object is this"
content_fingerprint  semantic identity      "is this the same evidence"
source/page          provenance identity    "where did it come from"
```

Two documents reporting the same figure on different pages are the **same
content** and **different provenance**. Merging them would break the
corroboration semantics H2A-1E's consensus relies on: the conflict gate counts
`100`s from distinct physical sources as agreement, and it does so by comparing
canonical values while keeping sources apart. A content identity that swallowed
the page would make distinct sources look like one.

So the fingerprint is built from the canonical semantic fields — metric,
period, entity, scope, canonicalised value, unit, currency — and **not** from
`page`, `document_id`, `evidence_id`, or any trace-level identifier.

What exactly those fields are is a code question this audit does not settle,
because it belongs with the value canonicalisation C just touched
(`_fact_value_key`, `resolve_scale_keyword`) and should reuse it rather than
reimplement it.

## What this audit did not do

No code changed. No hash changed. The next step is the ProgressDetector
semantics test, written first and expected to fail, before anything is
recomputed.
