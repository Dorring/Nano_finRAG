# Nano_finRAG — final sealed state

**This is the entry point.** Everything else under `docs/evaluation/` is the
record of how this state was reached; where a historical document disagrees with
this one, this one is current.

Branch `feat/nf-v3-interview-final`. Sealed at P1.8 (benchmark) and P1.9
(hygiene). Behaviour is frozen: no capability work, no metric tuning, and no
change to the runtime is permitted after this point.

---

## 1. The final architecture

**There are two runtimes in the tree and only one is production.**

```
FINANCIAL_RUNTIME_MODE=v2   (the default, set in .env.example)
  src/main.py:265            _build_financial_runtime
  src/main.py:154            _configured_trusted_v2_runtime_builder
  src/runtime/trusted_v2_production.py:1271
                             build_trusted_v2_runtime_for_request
        |-- trusted_v2_coordinator      orchestration
        |-- trusted_v2_binder           slot -> evidence binding
        |-- trusted_v2_calculation      arithmetic + operand guard
        |-- finance/operand_ambiguity   "does this coordinate identify one value"
        `-- finance/source_label_grounding
```

The V1 engine (`src/services/rag_engine.py` → `finance/calculation_pipeline.py`)
is **never constructed** in the default deployment:
`src/main.py:245-247 _financial_runtime_requires_legacy_engine()` returns `False`
for `v2`, and `src/runtime/query_lifecycle.py:267-272` passes `engine = None`.

Its dependency `finance/structured_operand_binding.py` is therefore **imported on
every V2 request and executed never** — the only call site is the shadow entry
`CalculationPipeline.try_structured_shadow`, which returns early under
`ENABLE_STRUCTURED_OPERAND_BINDING = False`, a flag nothing sets.
**It is not a production capability and must not be described as one.**

**No experimental flag is on by default.** Verified end to end in the P1.9
audit (`docs/evaluation/p1-9-repository-audit.md` §5).

---

## 2. The final benchmark

Benchmark **V2** = `tv2-canonical-v1-post-p1.8-d1-c3`.

```
gold            gold-evidence-v1.jsonl    a3d17211bbb42c83c07d1077cb1ed7ef81c2af7e0d27159cac2b58c591f8f6e4
eval set        canonical-eval-v1.jsonl   227f0341d94ab8d4b9e7ee033feaa9b40e86ec32c3137a74d2931657e9281ece
fixtures        plan-fixtures-v9.jsonl    c20afaec24bcab1360bbcbf9e880d9c8661c24551a1bcef2c59dc0248f3147da
120 questions   77 answerable, 43 abstention
```

The repository stores V1 (gold `3d2a0c5b`); **V2 is reproduced, not stored as the
primary artefact** — the migration is the authority:

```
python scripts/evaluation/build_p1_8c_benchmark_v2.py \
    --base benchmarks/tv2_canonical_v1 --out <dir>          # 3d2a0c5b -> a3d17211
python scripts/evaluation/build_p1_8_d1_fixture_v9.py --apply   # v8 -> v9, by contract
```

Both are verified byte-exact in the P1.9 freeze guard. This is what makes the
final benchmark re-runnable from a clean checkout: no measured file is carried
outside the repository.

---

## 3. Final metrics

### Trusted end-to-end (Benchmark V2, fixtures v9, 120 questions)

| metric | value |
|---|---|
| Answerable | 77 |
| Abstention | 43 |
| Released | 52 |
| **Release coverage** | **52/77 = 67.53%** |
| **Released accuracy** | **52/52 = 100%** |
| **Incorrect release** | **0** |
| **Correct refusal** | **43/43 = 100%** |

### Citation (over the 52 released cases)

| | canonical identity | strict ids |
|---|---|---|
| precision | **95/99 = 96.0%** | 68/99 = 68.7% |
| recall | **82/86 = 95.3%** | 68/86 = 79.1% |

**Use the identity column.** The strict column is not a grounding measurement:
gold for the cross-entity stratum names rebuilt-iXBRL keys while every citation
names a legacy candidate id, so the strict comparison reports a miss for a
correct citation. Measured: iXBRL-keyed gold misses **16/16**, legacy-keyed gold
misses **2/70**.

### Retrieval

`run_nf_v3_retrieval_benchmark.py` reports six separate measurements that are
never merged into one number. **75 of 120 cases are scorable**; the 20-case
cross-entity stratum is `STRUCTURALLY_UNMEASURABLE` because all 47 of its gold
ids are `ixbrl:` keys and the R4 index is keyed on `v2fact:` — that is an
id-space finding, not a retrieval failure, and is reported blank rather than 0.

**Production retrieval — the shipped path, re-measured on fixture v9:**

| retriever | R@5 | R@10 | R@20 |
|---|---|---|---|
| **RRF hybrid (shipped)** | **50.7%** | **58.0%** | **65.3%** |
| RRF hybrid + heuristic rerank | 46.0% | 58.7% | 69.3% |

The shipped row **reproduces the P1.6 measurement exactly**, on v9 fixtures, from
the current tree. The fixture change did not move retrieval.

**Structured-rerank arm — evaluation only, default disabled:**

```
R@5 85.333%   R@10 93.333%   R@20 95.333%      (P1.6 measurement, n=75)
```

These are the numbers usually quoted as "the retrieval benchmark". They are the
**structured-rerank experiment**, which is off by default, and they were **not
re-run in P1.9** — the default invocation of the benchmark reports
`reranker: heuristic`, and reproducing the reranked arm needs that arm's own run.
They are carried here as a historical measurement with their arm named. See §6:
quoting them as production retrieval is a do-not-claim.

---

## 4. How to reproduce

```
python scripts/evaluation/final_regression_guard.py --expect --write baseline.json
python scripts/evaluation/final_regression_guard.py --check baseline.json
```

The guard re-derives rather than trusts: it rebuilds V2 from this commit's V1,
hashes the store and the fixtures, runs the C5 fixture guard, and re-scores the
sealed predictions to recover the E2E and citation metrics above. Exit code 0
means `BEHAVIORAL_DRIFT = 0`.

---

## 5. Important negative findings

These are results, not caveats. Each was measured and each changes what the
system may be claimed to do.

**The structured reranker improved offline ranking and produced no demonstrated
end-to-end gain.** It is benchmark-positive on retrieval and was measured over
six runs (16,16,16 versus 16,16,16 on the arm that mattered). Its production
gain was not demonstrated, so it is **off by default** — enforced structurally,
by the `pool_reranker` seam defaulting to `None` with no production caller, not
by a flag someone has to remember to set.

**The cross-entity sign defect was a stale pinned fixture, not a runtime bug.**
`crossdiff-001` and `crossdiff-002` carried the operand order of the pre-P1.6-0H
question. The runtime executed them *faithfully* — `current - previous`, in the
order the plan gave — and released both with the wrong sign. The root cause was a
partial migration that rewrote the slots' metric but never re-derived their
entities or their order. Fixed by regenerating the fixture through the authoring
contract; **the runtime was not modified**, no operand-order patch is live, and
the narrow cross-entity patch is not promoted. A fixture integrity guard now
fails verification when a `difference … between A and B` plan names the companies
in an order the question does not.

**The remaining coverage gap is dominated by representation and ambiguity
limitations, not by retrieval policy.** Of the 25 unreleased answerable cases,
14 have at least one slot whose gold never reached the Binder's pool on the first
pass, and 8 have their gold in the pool at a coordinate holding several values.
`compare-010` is the clean statement: the pool holds Microsoft's three FY2025
segment operating incomes (14,166 + 69,773 + 44,589) and the gold 128,528 is
exactly their sum — **the pool carries the row's parts and not the row's total**.
Each record's `content` is its own row, so the cells survive; what is missing is
which row it is, and the header that would say so is flattened onto every row
equally.

**Row / column / logical-table semantics remain Future Work**, together with the
`tsla-035` unit-emission defect and two source-ambiguous golds.

**A measurement limit, stated rather than hidden.** The pool trace models one
retrieval pass; the Binder may run more (`binder_round_count = 2` on several
released cases). `TRUE_RETRIEVAL_MISS` is therefore an upper bound. And the
layered identity ladder's `MATCH_PROVENANCE` rung is **unrunnable** on this store
pair: the iXBRL records carry `concept`/`context_ref` and the legacy records
carry `row_id`/`table_fragment_id`, with no field in common.

---

## 6. Do-not-claim

These statements are false or unsupported, and must not appear in a résumé,
an interview, or a summary of this work:

* **"The reranker produced the 67.5% E2E coverage."** It produced no demonstrated
  end-to-end gain and is default-disabled. The coverage is what the runtime does
  without it.
* **"Production retrieval is R@5 = 85.3%."** That figure is the structured-rerank
  evaluation arm, which is off by default. Production retrieval is the hybrid RRF
  row. The two must be quoted with their arm named or not at all.
* **"All remaining failures are retrieval failures."** 14 of 25 are retrieval
  *at the first pass*; 8 are binding at a coordinate that does not identify one
  value; 2 are validation. The bucket is not a single cause.
* **"Coverage could safely reach 75% with a simple Binder rule."** No single
  failure bucket contains ≥6 deterministic-recoverable cases
  (`DETERMINISTIC_RECOVERABLE = 0`). Reaching 75% would require touching several
  frozen safety layers at once, and the phase stopped rather than do that.
* **"Strict citation ID precision/recall reflects grounding quality."** It
  measures a namespace boundary: 16 of 16 iXBRL-keyed golds are "missed" by
  construction. The canonical-identity numbers are the grounding measurement.

---

## 7. Known debt, deliberately not fixed

Registered rather than applied, because each fails the "provably
behaviour-neutral" bar for a hygiene pass. Full detail in
`docs/evaluation/p1-9-repository-audit.md` §12.

1. Host-absolute default paths on the final-path scripts
   (`run_tv2_canonical_benchmark.py:720,725`;
   `run_nf_v3_retrieval_benchmark.py:693,698`). Making a default env-overridable
   is neutral only if the effective default is preserved exactly, and these sit
   on the measured path.
2. `docs/evaluation/p1-7-release-coverage.md:220-224` describes
   `TransportRetryPolicy` as a live invariant; it is dead code with no `src/`
   reference.
3. `src/conversation/shadow_service.py` is live and load-bearing under a name
   that says otherwise.
4. 127 private `write_json` re-implementations against the shared authority in
   `benchmark_foundation.py`.
5. ~45 `src/pdf_retrieval_v4/` modules reachable only from `scripts/`.

---

## 8. Where things live

```
docs/evaluation/p1-8-final-seal.md       the benchmark seal and its evidence
docs/evaluation/p1-9-repository-audit.md the P1.9 classification, inventory and debt
docs/evaluation/p1-8-coverage-c0.md      the gate-on / gate-bypass attribution
scripts/evaluation/archive/              superseded eras (nf_legacy, pdf_retrieval_v4)
benchmarks/tv2_canonical_v1/             V1 benchmark + the v9 fixture + migrations
```
