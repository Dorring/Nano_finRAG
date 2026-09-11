# Trusted V2 R4 Asset Compatibility

## Why this contract exists

The production V2 retrieval graph has two separate, immutable assets:

```text
R4 candidate index
        candidate_key
             ↓
StructuredFactStore.materialize(candidate_key)
             ↓
Binder-admitted financial evidence
```

They must be built from the same candidate-key namespace.  An R4 index can be
healthy on its own and a Fact Store can be healthy on its own while the pair is
unsafe together: a retrieval hit with no Fact Store materialization has no
structured provenance and must never enter the Binder or calculator.

`validate_trusted_v2_production_configuration()` now checks that every R4
candidate key is materializable before the V2 production graph is constructed.
It fails fast rather than allowing the first user request to reach a
`candidate_materialization_failed` execution error.

## Build a compatible R4 asset

Use the existing four-lane `CandidateViewIndexBuilder` against the same
source-derived Fact Store configured as `TRUSTED_V2_FACT_STORE_PATH`:

```bash
cd finquery_rag/backend
uv run python scripts/evaluation/build_trusted_v2_fact_candidate_indexes.py \
  --facts /absolute/path/native-facts.jsonl \
  --out-dir /absolute/path/trusted-v2-r4-fact-index \
  --encoder-model /absolute/path/to/all-MiniLM-L6-v2/snapshot \
  --encoder-device cuda:0
```

`--encoder-device` affects only the offline dense-index build.  The serving
reader selects its model through `EMBEDDING_MODEL_NAME` and retains its
separate runtime-device policy.  Use `cpu` when a reproducible CPU-only build
is preferred.

The builder creates the existing R4 layout:

```text
candidate_raw_bm25/
candidate_raw_dense/
candidate_structured_bm25/
candidate_structured_dense/
candidate-metadata.sqlite
```

Each raw/structured view pair retains the exact Fact Store `candidate_key`.
The structured view uses source-derived entity, metric, period, value,
unit/scale, row/table and document fields.  The raw view may contain only the
physically extracted source text already present in the Fact Store.  Neither
view consumes questions, Gold evidence, answers, assistant history, model
summaries, or private reasoning.

## Metadata readiness diagnostic

The generated `candidate_contract.explicit_metadata_coverage` records the
count and ratio of candidates with non-placeholder source fields for entity,
metric, period, scope, unit, currency, scale and statement type. It is an
asset-quality signal, not a release override: a populated field does not make
a candidate Binder-admitted, and a missing field never permits a generated
answer to bypass the Binder.

Review this section before interpreting a smoke result. For example, sparse
period/unit/scope coverage predicts conservative multi-slot and qualitative
outcomes; the appropriate remedy is to enrich or correct source-derived fact
metadata and rebuild the reviewed index, not to loosen evidence admission or
silently infer missing qualifiers.

## Deployment sequence

1. Preserve the existing Fact Store; build a new index directory instead of
   overwriting an historical evaluation index.
2. Review `trusted-v2-fact-index-manifest.json`, including its source SHA,
   candidate count and four-lane integrity report.
3. Configure `TRUSTED_V2_R4_INDEX_DIR` to that reviewed directory and
   `TRUSTED_V2_FACT_STORE_PATH` to the exact source Fact Store used to build
   it.
4. Run the V2 production preflight.  A candidate-key mismatch is a deployment
   error, not a reason to fall back to V1 or to derive facts from retrieval
   text.
5. Only then run isolated V2 smoke/integration tests before changing a live
   service configuration.

## Scope

This closes an asset-contract and deployment-safety gap.  It does not change
Supervisor planning, R4 ranking, Binder policy, calculator logic, generation,
validation, or release thresholds.  It is also not an end-to-end quality
benchmark: index/fact compatibility must not be represented as recall,
accuracy, evidence-binding precision, or production-readiness evidence.
