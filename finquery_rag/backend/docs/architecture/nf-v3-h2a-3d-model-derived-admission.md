# NF-V3 H2A-3D — Model-Derived Artifact Admission

Status: **CLOSED.** Predecessors: the H2A-3 audits and notes, through
`nf-v3-h2a-3c-model-invocation.md`.
Sealed H2A-2 remains at `9497943`; this phase starts from `b374069`.

F6, the last real Harness trust debt. A table-cleaning model's answer was written
into a chunk's `content` and `parent_excerpt` — the same fields authoritative
document text occupies — with nothing between the model and the retrieval index.
The defect was not that the model might write poorly. It was that **generation
succeeding was treated as admission succeeding**, so a candidate truth became an
evidence truth with no authority transition in between, and every consumer
downstream read a number a language model typed as though the extractor had read
it.

No financial model, no DeepSeek, no F1, no H2A-4, no provider widening.

---

## 1 · The exact old authority transition

`src/services/process_tables.py:221`, in `enhance_table_with_context`:

```python
cleaned_table = table_md["md"]                      # authoritative default
table_match = re.search(r"CLEANED TABLE:\s*(.*)", enhanced, re.DOTALL)
if table_match:
    parsed_table = table_match.group(1).strip()
    if parsed_table:
        cleaned_table = parsed_table             # raw model text replaces it
return {"summary": summary, "content": cleaned_table}
```

One regex and one truthiness check, and that is the whole of it. From line 221
onward the return shape is **identical** to the three failure fallbacks
(`process_tables.py:142`, `:225`), so no caller could distinguish model text
from parser text.

Where it went:

| Surface | Site |
| --- | --- |
| chunk `content` | `ingest.py:690` |
| chunk `parent_excerpt` | `ingest.py:699`, from the same string |
| the model's prose summary | folded into **both**, `ingest.py:685-686` |
| dense embedding | `main.py:1147` → `vector_store.py:104,133` |
| BM25 / FTS index | `main.py:1151` → `retrieval.py:144,152` |
| row/cell child evidence | `ingest.py:737` → `mineru_parser.py:201,223,349` |
| the answer model's context | `context_builder.py:100-140` |
| deterministic numeric answers | `deterministic_answers.py:226` |

**The first irreversible write is `process_tables.py:221`.** At `:219` the only
variable holding the authoritative table is overwritten, and both strings are
still in scope there — which is why the fix is placed in that function rather
than at any of the downstream consumers. Patching the answer prompt, or the
context builder, would have left the model text in the index.

Two things already existed and were not being used: `is_usable_table_markdown`
(`process_tables.py:40`) validated *parser* output and was never applied to the
model's, and `layout_table_rows` is a coordinate-derived evidence stream
documented as unable to manufacture a value. Neither was an admission contract.
Nothing in the path recorded that any content was model-produced.

## 2 · The contracts

New package `rag_v2/derived/` — low-level, imports no `src`, depends only on the
shared financial semantics beside it.

```python
ModelDerivedArtifactV1            # produced, unverified
├── artifact_id: str
├── content: str
├── transformation: TransformationKindV1     # TABLE_CLEANING
├── source_reference: str
├── provider_id: str | None = None
├── model_id: str | None = None
└── invocation_id: str | None = None

ArtifactAdmissionResultV1
├── artifact_id: str
├── outcome: AdmissionOutcomeV1              # ADMITTED | REJECTED
├── reason: AdmissionReasonV1
├── source_reference: str
└── location: str = ""                       # source-derived row / column
```

`AdmittedDerivedArtifactV1` is `@dataclass(frozen=True, init=False)` with exactly
one constructor:

```python
AdmittedDerivedArtifactV1(candidate: ModelDerivedArtifactV1,
                          result: ArtifactAdmissionResultV1)
```

**Unverified is a type, not a flag.** There is no `trusted` attribute to flip and
no public field to set; the constructor refuses a rejection, refuses a result
belonging to a different artifact, refuses a bare string, and copies the fields
rather than holding the candidate — an admitted object that referenced one would
keep unverified content reachable through it. Reaching this class *is* the
authority transition, and it cannot happen because generation succeeded.

One reason enum, six members — the smallest taxonomy that keeps the four failure
shapes apart, because they are different repairs for whoever reads the log:

| Reason | Meaning |
| --- | --- |
| `FIDELITY_PRESERVED` | admitted |
| `NUMERIC_VALUE_CHANGED` | a figure was edited |
| `NUMERIC_VALUE_INVENTED` | a figure the source never stated |
| `NUMERIC_VALUE_REMOVED` | a source figure was dropped |
| `NUMERIC_VALUE_REASSIGNED` | every value survived; the assignment moved |
| `STRUCTURE_UNVERIFIABLE` | fidelity could not be established |

`ArtifactAdmissionResultV1` refuses an admitted result carrying a refusal reason
and a rejected one carrying `FIDELITY_PRESERVED`: the reason says what was
verified, and there is one thing to verify.

No `ArtifactStore`. The phase needed an artifact state, a source relation and an
admission result.

## 3 · Verifier semantics

`verify_table_fidelity(candidate, authoritative_markdown)`, in
`rag_v2/derived/tables.py`.

**The oracle is the source.** The authoritative pre-model markdown is the other
side of every comparison, so the check cannot be satisfied by a model agreeing
with itself. The prompt's "Do NOT change numeric values" is a request; this is
the check.

**Association is checked, not just the multiset.** A row is keyed by its label
and a figure by that label plus the column header above it. This is refused:

```
source            derived
Revenue  100      Revenue   80
Cost      80      Cost     100
```

Every number survives and every count matches — a sorted-value comparison calls
that perfect. It is `NUMERIC_VALUE_REASSIGNED` rather than `CHANGED` precisely
because the value exists *elsewhere* in the source, which is what distinguishes a
move from an edit.

**Structure, not string equality.** Cells go through `canonical_decimal` from the
H2A-2B shared semantics, so `1,000` and `1000` are one quantity, `(25)` is minus
twenty-five, and a footnote marker or currency edge is not part of the number.
Prose is never pushed through the parser: a cell that will not canonicalise is a
label, and labels are compared as text.

**Bag-of-numbers is not enough, and neither is a bag of keys.** The parse is
pipe-table structure only — the strongest authoritative structure the pipeline
possesses at that point, because it keeps no cell coordinates through to here.
Inventing a row/column model the extraction never had would be inventing
semantics rather than using them.

The source is walked in row order and the model's table only afterwards, so the
reported reason is deterministic. One classification reads oddly and is recorded
rather than hidden: **renaming a column header changes every key beneath it**, so
it surfaces as a removal and an invention. That is the honest reading from
structure alone, the contract asks the model to preserve headers, and the
consequence of over-refusing is the safe direction.

## 4 · Fallback behaviour

Chosen after auditing what already existed; option A was already the code's own
default and needed no new machinery.

| Situation | Result |
| --- | --- |
| **A** — verification refused | the authoritative table is used |
| **B** — no model produced a candidate (no key, failed call, no `CLEANED TABLE:`) | the authoritative table is used; no artifact |
| **C** — fail ingestion | **not used**, and deliberately |

Ingestion does not fail for a bad model answer. These fallbacks predate the
phase, and the change is that content reaching the trusted fields is now
*always* either the authoritative table or an admitted derivative — never raw
model output, and never the model's prose.

## 5 · Source lineage

The admitted artifact carries `source_reference`, built before the model is
called (`{pdf_path}::page_N::table_M`) because the answer is only ever a derived
candidate *for that table* and lineage has to exist before there is anything to
attach it to. `artifact_id` is `{source_reference}::table-cleaning`, so the
derived artifact is identified by its source rather than by a new unrelated
identity.

Admission does not make the model the evidence authority. The source document
remains the factual grounding; what is admitted is a *verified derived
representation*, and the chunk now records which of the two it holds:

```python
"derived_admission": {
    "transformation", "admitted", "reason", "location",
    "source_reference", "artifact_id", "model_id",
}
```

`None` means the content is not model-derived at all. Names, a reason and
lineage — the record can say *where* a refusal turned on (a source row label and
column header) without quoting what it refused.

## 6 · Production path after migration

```
camelot table.df
  -> format_table() -> authoritative markdown        (the oracle, never rebound)
  -> model call
  -> CLEANED TABLE: section -> ModelDerivedArtifactV1
  -> verify_table_fidelity(candidate, authoritative markdown)
       ADMITTED -> AdmittedDerivedArtifactV1(candidate, result) -> content
       REJECTED / absent -> authoritative markdown             -> content
  -> chunk.content, chunk.metadata.parent_excerpt, chunk.metadata.derived_admission
  -> embedding / BM25 / row-and-cell children / answer context
```

`content` is now *always* safe to write, and the record says which of the two it
is. The `parent_excerpt` is derived from that same string — pinned by an AST test,
because context expansion replaces a child's content with it and it is what the
answer model actually reads.

## 7 · Direct writes removed

| Write | Before | After |
| --- | --- | --- |
| `content` | raw model text or the fallback, indistinguishable | admitted content or the authoritative table |
| `parent_excerpt` | from the same string, including the summary | from the same string, summary excluded |
| the model's prose summary | folded into **both** | **not written anywhere** |
| a provider-declared bag | — | already removed in H2A-3C |

The summary's removal is a behaviour change and is flagged (§9.1). It is the same
defect one step removed: unverifiable model prose occupying a trusted field, and
a summary carrying a figure the table never stated is exactly what a fidelity
check on the table would not catch. The prompt still asks for it; the answer is
not used.

**One import was moved.** `camelot` was imported at module scope in
`process_tables.py`, which made that module — and `ingest.py`, which imports it —
unimportable without a PDF stack. The consequence was concrete: no test could
exercise `enhance_table_with_context` at all, which is part of why F6 survived.
The import is now inside `_read_tables`, the one function that needs it. Same
reasoning as H2A-3B0 moving the renderer out of the torch-importing module.

## 8 · Adversarial before/after

The "before" is a record, not a re-run: `_pre_fix_content` reproduces the removed
decision in five lines so the claim stays checkable after the code is gone — the
same reason `b3_legacy_context_baseline_v1` is kept.

| Case | Source | Candidate | Verdict |
| --- | --- | --- | --- |
| exact preservation | 100/90/80/70 | identical | ADMITTED |
| presentation only | 1000 | `1,000` | ADMITTED |
| changed value | 100 | 101 | REJECTED · `CHANGED` · `revenue / fy2024` |
| invented value | — | adds Profit 120 | REJECTED · `INVENTED` |
| dropped value | 80 | Cost row removed | REJECTED · `REMOVED` |
| reassignment | Rev 100, Cost 80 | Rev 80, Cost 100 | REJECTED · `REASSIGNED` |
| duplicate | 80 once | 80 twice | REJECTED · `INVENTED` |
| unverifiable | — | prose, ragged, header-only | REJECTED · `STRUCTURE_UNVERIFIABLE` |

The reassignment case is asserted **with** the property that makes it interesting:
the two tables' sorted values are equal, so a bag comparison would admit it.

Before/after on the live path, same adversarial answer, stubbed HTTP:

```
before:  content = "| Revenue | 101 | 90 | ..."     (from _pre_fix_content)
after:   content = the authoritative table
         admission_record = {admitted: False, reason: NUMERIC_VALUE_CHANGED,
                             location: "revenue / fy2024"}
```

## 9 · Decisions flagged for review

1. **The model's prose summary is no longer written anywhere.** It was verified
   by nothing and reached both `content` and `parent_excerpt`. The alternative —
   a numeric check on prose — would be passing arbitrary text through the
   financial parser, which H2A-2B's semantics are not for.
2. **A renamed column header refuses.** It changes every key beneath it, so it
   reads as removal plus invention. Safe direction, honest classification, and
   worth knowing about if enhancement ever appears to stop working.
3. **The ingestion model call does not go through `ModelProviderV1`.** It is a
   direct `requests.post` with its own auth and payload, and this phase's §20
   forbade widening the invocation contracts without a concrete missing field.
   So `invocation_id` is `None` on the artifact and `model_id` carries the model
   name — stated rather than filled with a synthetic id. Migrating ingestion onto
   the provider framework is a separate question from admission and is not an
   admission blocker.
4. **`camelot`'s import moved into the function that uses it** (§7). A module-level
   import of a heavy library that one function needs is what made this path
   untestable.

## 10 · Verification

* **Three injections, all caught**, sources restored byte-identically: writing
  the model's answer directly on a successful call (the F6 defect itself);
  comparing numbers as a bag instead of by structured position; letting an
  admitted artifact be built without a successful admission.
* **Full suite: 4471 passed, 143 skipped, 0 failed** (+37 over H2A-3C's 4434).
* **Readiness unchanged:** `task_success 19, semantic_mismatch 0, label_alias 3,
  unexplained_mismatch 0, false_release 0, over_conservative 0`.
* **The 4 `test_phase108_pdf_table_fallback` failures** (`pymupdf` absent) are
  pre-existing: confirmed by stashing the changes and reproducing them on the
  untouched tree. They pass in a full-suite run because an earlier test installs
  a stub, which is exactly the ordering dependence this phase's tests avoid — the
  F6 tests pass standalone and in the suite.
* **Layering:** `rag_v2/derived` imports no `src`; `rag_v2/context` and
  `rag_v2/invocation` import no `rag_v2.derived`, asserted by test. Admission is
  neither the compiler's job nor the provider's.

## 11 · Remaining debt before the H2A-3 final seal

1. **B5/B6/V1 have no role.** `ContextRoleV1` keeps one member; the framework has
   to support their arrival without `AgentContextPackV1` becoming a universal
   RunState. Whether they are migrated at all is a modernization question, not a
   seal blocker.
2. **The ingestion model call is outside `ModelProviderV1`** (§9.3).
3. **Provider capabilities are still one method.** Streaming, health and
   cancellation await a provider that needs them; `ModelRequestV1` carries text
   only.
4. **`specialist=` should be renamed**; `generation` has no producer; the
   support-topology opt-in path is specified and unexercised.
5. **`is_usable_table_markdown` remains a parser-output check.** The verifier's
   own structural parse subsumes much of it for the model path, but the two were
   not unified — deliberately, since one guards extraction and the other guards a
   transformation.

## 12 · What H2A-3D did not do

* no financial model, no DeepSeek, no provider widening
* no F1 tokenizer work; no token bound configured or approximated
* no V1, B5/B6 or query-rewrite migration
* no `ArtifactStore`
* no change to `ContextBudgetV1`, `RunBudget`, or the H2A-3C invocation contracts
* no new seal; H2A-2 remains at `9497943`
* no H2A-4 and no token-efficiency number of any kind
