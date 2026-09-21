# P1.9 Phase 1 — repository audit (read-only, nothing modified)

Scope: `E:\nanochat\finquery_rag\backend`. 492 files in `scripts/evaluation/`,
103 docs in `docs/evaluation/`, 568 `tests/test_*.py`, **2,940 tracked files
under `artifacts/`**. Two independent sweeps were run — one over
`scripts/evaluation/`, one over `src/` + `rag_v2/` — and their claims are
spot-checked here rather than taken on faith.

**No file was deleted or moved to produce this document.**

---

## 0. The yardstick: what the final path actually is

Everything below is classified against one question — *is this on the path a
clean checkout runs to reproduce the sealed benchmark?* That path is:

**Production runtime (V2, the default).**
`FINANCIAL_RUNTIME_MODE=v2` → `src/main.py:265 _build_financial_runtime` →
`src/runtime/trusted_v2_production.py:1271 build_trusted_v2_runtime_for_request`
→ `trusted_v2_coordinator` → `trusted_v2_binder` / `trusted_v2_calculation` /
`operand_ambiguity` / `source_label_grounding`.

**Benchmark and evaluation.**
```
build_p1_8c_benchmark_v2.py          V1 -> V2 benchmark (gold a3d17211)
build_p1_8_d1_fixture_v9.py          v8 -> v9 fixtures (contract regeneration)
verify_fixture_integrity.py          the C5 operand-order guard
run_p1_2_dual_track_benchmark.py     the replay track; seals with provenance
run_tv2_canonical_benchmark.py       the raw HTTP track
run_nf_v3_retrieval_benchmark.py     Recall@K
score_nf_v3_final.py                 the final scorer
score_p1_8_d1_canonical.py           the D1 canonical scorer
measure_p1_8_identity_closure.py     layered gold/pool identity
measure_p1_8_citation.py             citation P/R, both readings
attribute_p1_8_coverage.py           first-failure attribution
final_regression_guard.py            the P1.9 freeze guard
```

---

## 1. modified / untracked files

41 paths: **2 modified, 39 untracked**, all of them P1.8 work.

```
M  benchmarks/tv2_canonical_v1/benchmark-version.json     the p1.8-d1-c3 migration entry
M  scripts/evaluation/build_p1_2_plan_fixtures.py         the C5 generation-side guard

?? benchmarks/tv2_canonical_v1/{plan-fixtures-v9.jsonl,.sha256,.manifest.json}
?? benchmarks/tv2_canonical_v1/migrations/p1.8-d1-c3.json
?? docs/evaluation/p1-8-*.md                              16 documents
?? scripts/evaluation/*.py                                15 P1.8 scripts
?? scripts/evaluation/run_p1_8_d1_c3.sh
```

`KEEP_PRODUCTION` for the benchmark files and the guard; `KEEP_EVALUATION` for
the scripts; `KEEP_HISTORICAL` for most of the documents (§10).

**Nothing under `src/` or `rag_v2/` is modified** — the runtime is untouched,
which is what D1 claimed and what §4 re-verifies.

---

## 2. The headline finding: 295 of 492 eval scripts are referenced nowhere

| | count |
|---|---:|
| total scripts in `scripts/evaluation/` | 492 |
| referenced anywhere outside themselves | 198 |
| **never referenced anywhere** | **295 (60%)** |
| added on or before 2025-08-25 | 218 of those 295 |

By naming pattern the unreferenced set is `run_*` 130/221, `score_*` 40/47,
`audit_*` 34/48, `build_*` 15/28, `probe_*` 13/15.

**Caveat, and it is not a small one:** "never referenced" ≠ "stale". A one-shot
probe whose output is quoted in a markdown doc is legitimately unreferenced in
code. The classification below therefore uses era + docstring + artifact
directory, not the reference count alone.

---

## 3. Duplicate / overlapping utilities

### 3.1 Four builders of "the" fact store

| script | disposition |
|---|---|
| `build_store_v2.py` | **KEEP_PRODUCTION** — on the documented final path, test-referenced |
| `build_ixbrl_fact_store.py` | ARCHIVE — superseded |
| `rebuild_ixbrl_facts.py` | ARCHIVE — same job as the above, added later |
| `build_trusted_v2_canonical_fact_store.py` | ARCHIVE — zero references |
| `account_store_migration.py`, `enrich_trusted_v2_fact_store.py` | ARCHIVE — one-shot migrations, zero references |

### 3.2 Ten scorers for one gate round

`score_pdf_v4_gate_08_r8_{r1, r1_2, r2a, r2a_2, r3, r3_1, r3_2, r3_3, se1,
candidate_depth}.py` — `_r1` and `_r1_2` differ only in output directory and
prediction path. ARCHIVE (whole family).

### 3.3 Scorers that shadow the final one

`score_p1_8_d1_canonical.py` is **not** a duplicate: it exists because the final
scorer cannot parse `(1,694)`-style parenthesised negatives, which is the D1
reconciliation finding. KEEP_EVALUATION, and the distinction is stated in §6.
`score_financial_calculation_final_showcase.py` — ARCHIVE, zero references.

### 3.4 Runners over the same canonical set

`run_p1_2_dual_track_benchmark.py` is the **live replay track** (it produced
every sealed number, and two live tests import it). `run_tv2_07_readiness.py` is
superseded by `run_tv2_07_r1_readiness.py`. ARCHIVE the first.

### 3.5 Measured near-duplicates (line-set containment)

| pair | containment |
|---|---|
| `run_nf_e2e_04_r0` ⊂ `run_nf_e2e_05_r0` ⊂ `run_nf_e2e_06_r0` | 0.986 / 0.998 |
| `run_nf_opt_23_r1` / `run_nf_opt_25_r0` | 0.94 |
| `run_nf_v2_09_r0` / `run_nf_v2_09_r2_1` | 0.88 |

Each round is a copy-paste superset of the last. ARCHIVE the superseded ones.

### 3.6 Utility duplication

`def write_json` is re-implemented privately in **127** scripts, `sha256` in 102,
`read_jsonl` in 68. The shared authority already exists and is used:
`benchmark_foundation.py` has 21 importers. **NEEDS_REVIEW** — collapsing 127
private copies is a large mechanical change with no behavioural payoff, and
`benchmark_foundation.py` imports would have to survive the archive move.

---

## 4. Dead / shadow-only runtime code

**There is no live cross-entity operand-order patch in the tree.** Verified
independently of the D1 document:

* `_cross_entity_difference_specs`, `_ENTITY_ALIASES` and `_match_entity` occur
  **only** inside `scripts/evaluation/apply_p1_8_d1_operand_order.py`, an
  applier that was never run against `src/`.
* `src/finance/structured_operand_binding.py:309-321` still carries the
  pre-patch branch shape.
* `git status -- src rag_v2` is empty.

What *is* dead or shadow-only:

| what | evidence | disposition |
|---|---|---|
| `ENABLE_STRUCTURED_OPERAND_BINDING` | `False` at `src/finance/calculation_pipeline.py:49`; no `src/` caller sets it | **KEEP_HISTORICAL** — flag stays, documented as shadow-only |
| `src/finance/structured_operand_binding.py` | imported on every V2 request via `finance/__init__.py` → `calculation_pipeline`, **invoked never** | **NEEDS_REVIEW** — see §6 |
| `rag_v2/evidence/{constrained_binder_provider, pairwise_binder, slotwise_binder, shortlist_comparative_binder, selective_admission*, transport_retry}.py` | not exported from `rag_v2/evidence/__init__.py`; zero `src/` references | ARCHIVE |
| `src/retrieval_v3/` | its own docstring calls it "shadow experiments"; only `scripts/`/`tests/` consume it | ARCHIVE |
| `src/pdf_retrieval_v4/shadow_index_reader.py` and ~45 sibling modules | reachable only from `scripts/`/`tests/` | NEEDS_REVIEW — large, and some are index-build tooling |
| `structured reranker` | seam defaults to `None` at `trusted_v2_production.py:1278` and `trusted_v2_r4.py:368, 383-388`; no `src/` caller | **KEEP_EVALUATION**, default-disabled (§7) |
| structured context sidecar (`scripts/evaluation/build_structural_sidecar.py`) | **no `src/` module mentions "sidecar"**; its only consumer is `audit_benchmark_source_truth.py` | ARCHIVE |

### One doc/code mismatch found

`docs/evaluation/p1-7-release-coverage.md:220-224` states as a live property that
"`TransportRetryPolicy` freezes one semantic response and one transport retry".
`TransportRetryPolicy` (`rag_v2/evidence/transport_retry.py:73`) is dead code with
no `src/` reference. **The documented invariant is not enforced by the code the
document names.** Logged as NEEDS_REVIEW — the document is a historical record of
a measurement, so the honest fix is a note, not a rewrite.

### One live module with a misleading name

`src/conversation/shadow_service.py` is **live and load-bearing** — wired at
`src/main.py:1247`, used by `src/runtime/query_lifecycle.py:391,438`, default
`on`. The name is a legacy of when it was an observer. **Rename is
behavior-neutral but touches a production path; registered as debt (§11).**

---

## 5. Deprecated feature flags

| switch | default | enabled by anything? | disposition |
|---|---|---|---|
| `ENABLE_STRUCTURED_OPERAND_BINDING` | `False` | no | KEEP_HISTORICAL |
| `DETERMINISTIC_FACT_EXTRACTOR` | `"current"` | no | KEEP_HISTORICAL |
| `entity_key_lookup` | `None` | no — "None on every production call path" | KEEP_HISTORICAL |
| `pool_reranker` | `None` | no | KEEP_EVALUATION |
| `retriever_factory` | `None` | no | KEEP_EVALUATION |
| `alignment_override` | `None` | no | KEEP_EVALUATION |
| `NF_AGENT_RUNTIME_MODE` | `legacy` | no | KEEP_HISTORICAL |
| `MULTITURN_CONTEXT_ENABLED` | — | superseded by `MULTITURN_CONTEXT_MODE` | KEEP_HISTORICAL |
| `ENABLE_TABLE_FACT_EXTRACTION` | `False` | dead symbol, no reader | **DELETE_SAFE** (symbol only) |
| `PRODUCTION_SWITCH_ALLOWED`, `V2_ARCH_VERSION` | — | defined at `rag_v2/__init__.py:3-4`, **no reader anywhere** | **DELETE_SAFE** |

**No experimental flag is on by default.** That is the acceptance criterion, and
it holds.

---

## 6. `structured_operand_binding.py` — the honest status

It is **not** a production capability and must not be described as one. It is
imported on every V2 request (through `finance/__init__.py`) and *executed never*:
the only call site is `CalculationPipeline.try_structured_shadow`
(`calculation_pipeline.py:109`), which returns `NOT_APPLICABLE_RESULT` at `:116`
under a flag that is `False` and that nothing sets.

Its only callers anywhere are `scripts/evaluation/run_nf_opt_06.py` and one test.

**Recommendation: KEEP_HISTORICAL, explicitly labelled shadow-only.** Its 3
call sites are not zero, and a module imported on the production request path
should not be deleted on a hygiene pass. Deleting it would also require touching
`finance/__init__.py`, which is a production import chain — that fails the
"provably behavior-neutral" bar for this phase.

---

## 7. Docs referring to obsolete metrics

16 documents cite at least one superseded headline (`17.9%`, `48.4%`, `52.6%`,
`5.3%`); 9 cite the V1 gold `3d2a0c5b`; 5 cite `plan-fixtures-v8`.

These are **not** errors to correct. Each records what was measured at its time,
and the P1.8 documents in particular are the evidence chain for the seal.
Disposition: **KEEP_HISTORICAL**, with the single entry point in §10 stating
which numbers are current, so a reader cannot mistake a historical figure for the
sealed one.

---

## 8. Host-specific and GPU-specific paths

**172 of 492 scripts (35%) contain a host-absolute path**, and all 11 `.sh` files
do. Roots: `/disk/qh/nano-finrag/**`, `/mnt/disk/mxf/**`, `/home/mxf/.cache/**`.

**The final-path scripts are affected, and this is the finding that matters:**

```
run_tv2_canonical_benchmark.py:720   default fact store  /disk/qh/...
run_tv2_canonical_benchmark.py:725   default sessions.db /disk/qh/...
run_nf_v3_retrieval_benchmark.py:693,698   same fact-store default
```

GPU pins (`cuda:N` literals, `CUDA_VISIBLE_DEVICES` guards) appear in 29 scripts,
all in the `nf-opt-2x` / `nf-v2-09` / `nf-v2-18a` eras. **None of the final-path
scripts pins a GPU** — they are CPU-only.

Disposition: **NEEDS_REVIEW**, and deliberately *not* fixed in this phase. Making
a hardcoded default env-overridable is behavior-neutral only if the effective
default is preserved exactly, and these sit on the measured path. Registered in
§11 rather than applied.

---

## 9. Generated artifacts tracked in git

**2,940 tracked files under `artifacts/`**, including 106 `.jsonl` and single
files up to ~44 MB.

Disposition: **KEEP_HISTORICAL**. They are the evidence for historical claims,
they are already in history, and removing them changes no behaviour while
destroying the audit trail. The interview-facing cost is that they are *findable*
— addressed by the entry document (§10) naming the final artifacts, not by
deletion.

---

## 10. Stale comments, TODOs, and bytecode

**`TODO` / `FIXME` / `XXX` / `HACK` across all 492 scripts: zero hits.** The
staleness signal in this codebase is prose-in-docstrings (each script states which
round superseded it), not inline markers.

`scripts/evaluation/__pycache__/` holds `.pyc` for scripts whose source has since
changed, and those binaries carry `/disk/qh/...` strings — inside a directory
imported as `scripts.evaluation`. **DELETE_SAFE.**

---

## 11. Classification summary

```
KEEP_PRODUCTION    ~12 scripts + the benchmark tree + the guard
KEEP_EVALUATION    the P1.7/P1.8 measurement chain (~15) + the 10 library modules
KEEP_HISTORICAL    2,940 artifacts, 103 docs, the shadow flags, structured_operand_binding
ARCHIVE            ~295 unreferenced scripts + the duplicate families in §3
DELETE_SAFE        __pycache__ trees; two dead symbols in rag_v2/__init__.py
NEEDS_REVIEW       the 127 private write_json copies; ~45 pdf_retrieval_v4 modules;
                   the final-path host-path defaults; the p1-7 TransportRetryPolicy note;
                   shadow_service.py's name
```

## 12. Registered as debt, not applied

1. Host-absolute defaults on the final-path scripts — behavior-adjacent, so
   documented rather than rewritten.
2. `TransportRetryPolicy` documented as live but dead.
3. `shadow_service.py` — live under a name that says otherwise.
4. 127 private `write_json` re-implementations against a shared authority.
5. ~45 `src/pdf_retrieval_v4/` modules reachable only from `scripts/`.
