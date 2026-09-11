# Trusted V2 Binder Context Hardening

## Status

Implemented and unit/integration-regression tested. This document records an
execution-boundary improvement; it is **not** an end-to-end quality benchmark or
a production-readiness claim.

## Problem

The R4 candidate packet is intentionally broad for retrieval audit and recovery.
Passing its raw `content`, `raw_content`, `source_text`, or any accidental
conversation-like field directly to an external Binder provider makes the model
context larger and weakens the boundary between:

```text
retrieved candidate context
!=
Binder-admitted financial evidence
```

It also made it harder to distinguish an actual missing structural metadata field
from an unbounded prompt payload.

## Implemented boundary

`BinderRequest.to_dict()` now emits a strict allowlisted fact view:

```text
actual fact_id / evidence_id / citation_id
+ source identity
+ entity / metric / period / value / unit / currency / scale
+ table / row / scope / statement fields when source-derived
```

It excludes by construction:

```text
content
raw_content
source_text
evidence_text
assistant_text
conversation history
model-generated summary
private reasoning / chain-of-thought
opaque diagnostic objects
```

The full candidate fact remains in-process for deterministic binding validation,
provenance, calculation, and release validation. The external Binder can select
only real packet fact IDs; it cannot receive an answer, Gold label, or a
pre-assigned RequiredSlot hint.

## Optional structural enrichment artifact

Older fact artifacts may keep row/table metadata in a separate
`structured-views.jsonl`. The new offline builder joins only existing source
identities in this priority order:

```text
Evidence ID match
→ Row ID match
→ Table-fragment match
```

A table-fragment-only match may add table-level provenance but **never** adds
row-level metric or period metadata. Existing fact-store fields are never
overwritten. Raw view text and nested extraction blobs are never copied.

Example command:

```bash
python scripts/evaluation/enrich_trusted_v2_fact_store.py \
  --facts /path/native-facts.jsonl \
  --structured-views /path/structured-views.jsonl \
  --out /path/native-facts-enriched.jsonl
```

The output manifest records input/output SHA-256 values, linkage counts, and
asserts that the builder does not read questions, Gold, answer text, or models.
Using the output requires an explicit `TRUSTED_V2_FACT_STORE_PATH` deployment
configuration change; this implementation does not silently switch production.

## What this does not solve

It does not license the system to choose among genuinely conflicting facts.
For example, several Microsoft `FY2024 Operating Income` rows may remain
source-valid but lack a source-derived consolidated/segment discriminator. In
that case Binder ambiguity and Fail-Closed remain the correct outcome. The
improvement is context quality and provenance observability, not a numerical
value-selection heuristic.

## Verification

- Binder input trust-boundary regression: raw/assistant/history/reasoning fields
  are absent; source-derived fact IDs and structural context remain available.
- Trusted V2 routing, Binder, lifecycle, and production-integration regression
  suite: 158 tests passed locally after the change.
- The offline builder was smoke-tested against the current server facts and
  structured views in `/tmp` only. It linked 20,394/20,394 records by existing
  evidence IDs. This is **lineage coverage**, not Recall, answer accuracy,
  Binder correctness, or release precision.

## Interview-safe description

> We separate a broad retrieval packet from the model-facing Binder view. The
> Binder sees only source-derived typed fact metadata and real IDs; assistant
> text, raw context, and model summaries cannot become financial evidence. A
> deterministic artifact join restores available table/row provenance, while
> unresolved source-valid conflicts remain Fail-Closed rather than being ranked
> into an answer.
