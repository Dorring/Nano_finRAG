# TV2 canonical eval set and plan fixtures

The frozen ground truth the trusted-V2 benchmark runs against. It is tracked
here, rather than under `artifacts/`, because `artifacts/` is gitignored and
these files are not results — they are *inputs*, and a delta measured against
them is only meaningful to someone who has the same bytes.

| File | What it is |
| --- | --- |
| `canonical-eval-v1.jsonl` | 120 questions: `id`, `question`, `stratum`, `expected_intent` |
| `gold-evidence-v1.jsonl` | one gold row per question: `metric`, `operation`, `fact_ids`, `operands`, `expected_value`, `expected_outcome` |
| `plan-fixtures-v1.jsonl` | ALIGNED-REPLAY plans, as first authored — **known defective, kept as a change record** |
| `plan-fixtures-v2.jsonl` | the same plans with the operand-period defect repaired |

The four strata are `factual_lookup` (40), `arithmetic_calculation` (35),
`cross_entity_comparison` (20) and `adversarial_abstention` (25).

Every fixture carries `sourced_from`, recording which tier produced each field,
so how much of a plan is the gold's and how much is question text is readable
rather than assumed.

## Provenance

Both the eval set and the gold were frozen by the TV2-FINAL-01 stage and moved
here unchanged. Their digests are unchanged by the move and are re-recorded in
every fixture manifest (`eval_set_sha256`, `gold_sha256`), so a manifest from an
older run still identifies the same inputs.

`dataset-manifest.json` is the TV2-FINAL-01 record and its `output_files` paths
still name the location that stage wrote to. They are left as written: the
manifest is a statement about a past run, and the digests it records are the
part that still verifies. Editing the paths to match the move would falsify a
record to make a file tidier.

## v1 → v2, and why both are kept

v1's `sum` and `average` plans were authored by a builder that, finding no
period in the gold, took **the first period the question named and gave it to
every operand**. Both slots then required the same quantity. The binder refused
the duplicate — it has to, or a sum would count one number as twice itself — and
ten questions that were sound scored as binder failures. The slot *count* was
right, which is why nothing caught it.

v2 reads each operand's period from the gold operand fact itself, joining on the
same id the gold uses. Ten cases changed, and only those ten; `metric` was
already correct everywhere. `sourced_from.period` records the new
`gold_operand_facts` tier.

The rule this produced, which outlives the bug:

> The fixture builder may **serialise** ground truth. It may not **create** it.

v1 is kept rather than corrected because the detector written for this defect is
tested against it: `tests/harness/test_p1_2_plan_fixture_ground_truth.py` asserts
the check still *fails* on v1. A test the defective and the corrected artifact
both pass would be measuring nothing.

## Reproducing a fixture

The builder needs the deployment's fact store, which does not live in the
checkout:

```bash
cd finquery_rag/backend
.venv/bin/python scripts/evaluation/build_p1_2_plan_fixtures.py \
    --fact-store /disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl \
    --name plan-fixtures-v2
```

Without the store the build **fails**, naming the case it could not author. That
is deliberate: a period invented to cover a missing fact would measure the
builder's guess instead of the binder.

## Auditing a fixture

```bash
.venv/bin/python scripts/evaluation/audit_p1_2_plan_fixture_ground_truth.py \
    --out-dir artifacts/evaluation/p1-3-ground-truth-audit
```

It judges the fixtures against the question and the gold — never against what
the binder did — and reports per stratum. Current verdict: **100 of 120 fixtures
carry their own ground truth**; the 20 `cross_entity_comparison` ones do not, and
cannot until `RequiredSlot` (`rag_v2/contracts/plan.py`) gains an entity
coordinate, because two companies at one period currently have nowhere to differ.
