# NF-V3 H2A-3B0 — Context Correctness Preconditions

Status: **CLOSED.** Predecessor: `nf-v3-h2a-3-context-surface-audit.md`.
Baseline: `HEAD 3b09de0` (H2A-3A), seal `nf-v3-h2a2-artifact-authority` at `9497943`.

H2A-3B0 fixes the two things that would otherwise have contaminated the compiler
baseline, and freezes that baseline. It introduces **no** Context Compiler, **no**
ContextBudget, **no** `AgentContextPack`. F1 and F6 are **recorded, not fixed**.

---

## 1 · F5 — the specialist was told every source was on page 1

### What was wrong

`rag_v2/evidence/disclosure.py` did not admit `page` to the SPECIALIST profile,
so `project()` never emitted it, so the prompt's

```python
page = ev.get("page") or 1
```

took its fallback on **every call** and rendered `Source: {doc}:1` for all
evidence. The model was given, as provenance, a page nobody had established.

Two errors were stacked in that expression, and they fail differently — which is
why the tests keep them apart:

| Error | Effect | Kind of wrong |
| --- | --- | --- |
| `or` treats a real page `0` as absent | a recorded page is dropped | a **lost** fact |
| the fallback invents `1` rather than stating the absence | a page is asserted that was never recorded | an **asserted false** fact |

`EvidencePacketV1.page` documents `0` as valid with absence preserved as
absence, so the two states are contractually distinct and a truthiness test
conflates them.

The deeper point is the one this module was written for. `disclosure.py`'s
comment had recorded `page` as *deliberately deferred*: "it is added here when
it survives that far — not before, or the profile would name a field that is
always absent." The deferral condition was met by H2A-2D-2B, which made
`EvidencePacketV1.page` the single authority for page provenance. Until it was
admitted, the field being *missing from the profile* was what produced the false
statement — a leak was not the only failure mode an allowlist could cause.

### What changed

| File | Change |
| --- | --- |
| `rag_v2/evidence/disclosure.py` | `page` added to `_SPECIALIST_FIELDS`; deferral comment replaced with the decision and the reason the deferral expired |
| `src/generation/specialist_prompt.py` | **new** — the renderer, moved; the `or 1` fallback removed |
| `src/generation/local_specialist_generator.py` | `render_prompt` delegates; `import re` dropped with the moved code |

The source is the canonical field and only the canonical field. `project()`'s
lookup reads the top level, and the SPECIALIST profile governs no container, so a
`metadata` bag carrying `page`/`pdf_page` cannot supply it and cannot override it.
`pdf_page` is deliberately **not** admitted: it is normalised to `page` once at
ingestion (`EvidencePacketV1.from_mapping`), and admitting both would put a
second page field back on the model-facing surface — the duplication H2A-2
removed.

Absence renders in the idiom the neighbouring fields already use:

```
page recorded   ->  Source: doc-1:7
page is 0       ->  Source: doc-1:0
page absent     ->  Source: doc-1:not specified
```

`not specified` matches `unit` and `currency` two lines above and is
distinguishable from a real value, which `1` was not.

### Why the renderer moved

`local_specialist_generator` imports `torch` at module scope, and `conftest.py`
excludes a test module from collection entirely when an optional runtime is
missing (`OPTIONAL_RUNTIME_MODULES`). A prompt assertion written next to
`import torch` would therefore **silently never run** on any checkout without
torch — reporting green while testing nothing. Since the phase's whole point is
that the model-visible prompt is now asserted on, the renderer moved to a
torch-free module and `render_prompt` delegates. The public API is unchanged.

### Evidence

Ten tests in `tests/harness/test_specialist_page_disclosure.py`, covering page 7,
page 0, absent, explicit `None`, a `metadata` bag that disagrees with the
canonical field, `pdf_page` normalisation, per-source pages across three items,
and — through the real `TrustedV2GenerationCapability`, not the helper — the
prompt the production boundary hands a model.

**The tests are load-bearing.** Restoring the exact pre-3B0 expression makes
**6 of the 10 fail**; the file was byte-identical afterwards (SHA256 verified
before and after).

---

## 2 · The B3 legacy context baseline

Frozen in `tests/harness/b3_legacy_context_baseline.py`, asserted in
`tests/harness/test_b3_legacy_context_baseline.py` (24 tests). Five authored
scenarios, chosen to cover every shape that reaches B3:

| Scenario | Route | Target |
| --- | --- | --- |
| `multi_fact` | MULTI | LOCAL_SPECIALIST |
| `temporal` | TEMPORAL_SYNTHESIS | LOCAL_SPECIALIST |
| `qualitative_single` | QUALITATIVE | LOCAL_SPECIALIST |
| `calculation_with_explanation` | CALCULATION_WITH_EXPLANATION | LOCAL_SPECIALIST |
| `page_variants` | MULTI | LOCAL_SPECIALIST |

Each freezes: route + reason + target, selected evidence ids, citation ids,
calculation ids, post-disclosure field names, the projected evidence payload, the
projected calculation payload, the exact model input **and its SHA256**, the
candidate answer/status/generation id, and the token count.

### What it is, and what it is not

**A change detector, not an oracle.** Every expected value was captured from the
implementation, so it cannot testify that the implementation is *correct* — only
that it has not *moved*. That is exactly the question H2A-3B2 asks when it
replaces `select → project → assemble` with `ContextCompiler.compile(...)`.

The inputs are **authored**; the outputs are **captured**. A scenario that
stopped reaching the specialist fails rather than silently recording a new shape.

This file alone would be the tautology the anti-circularity rule forbids, so the
module carries a second kind of assertion. Alongside equivalence, the **frozen
record itself** is checked against expectations derived from the *inputs* — every
`Source:` line's page must equal the page its authored evidence recorded. That
assertion holds even if the baseline were regenerated after a regression, which
is the one failure mode a snapshot comparison structurally cannot catch.

Both directions were verified by injection:

| Injected fault | Caught by |
| --- | --- |
| live code regressed to `or 1` | equivalence tests, on exactly the three page-bearing scenarios |
| baseline re-captured after that regression | the input-derived assertion, on the same three |

### Token counts are NOT DETERMINED

B3's tokenizer is the checkpoint's, behind `torch` and a SHA256-pinned checkpoint
that the suite cannot reach. `input_tokens` is recorded as `NOT_DETERMINED` — not
approximated with a character or word count. A test pins that value so filling it
in later must be a deliberate edit. This follows H2A-3A's tokenizer matrix: a
boundary without a reliable mapping is recorded as undetermined rather than
invented.

---

## 3 · F1 and F6 — recorded, not fixed

**F1 (V1 tokenizer mismatch).** Untouched. `ContextBuilder` still budgets with
`tiktoken cl100k_base` for a nanochat model, with `len/3` as the silent fallback.
Scheduled for H2A-3C, where the real tokenizer is resolved or declared NOT
DETERMINED and V1 is barred from any token-efficiency number either way.

**F6 (model-derived table text).** Untouched. The ingest model's `CLEANED TABLE:`
block is still written into the index verbatim and later permitted into the V1
answer prompt. Scheduled for H2A-3C as a deterministic numeric-fidelity gate.

Neither is partially mitigated here. A half-fix in a preconditions phase would
have made the 3C gate unverifiable.

---

## 4 · F10 — new finding, same class as F5

Found while fixing F5 and recorded rather than fixed, because it was not
authorized and because it would change the baseline just frozen.

The specialist renderer still fabricates **four more** values, and two of them are
financially material:

| Line | Expression | Assessment |
| --- | --- | --- |
| `metric` | `... or "Metric"` | fabricates a metric name |
| `period` | `... or "Period"` | fabricates a period |
| `scale` | `... or "1"` | **fabricates a magnitude multiplier** — a wrong scale is a wrong number |
| `document_id` | `... or "filing"` | fabricates a document name |
| `scope` | `... or metric` | asserts scope equals metric |
| `unit`, `currency` | `... or "not specified"` | **honest** — an absence marker distinguishable from a value |

`scale or "1"` is the sharpest: the answer rules tell the model to preserve
supplied scales exactly, and `1` is indistinguishable from a scale the extractor
recorded. It is the same defect as F5 with a worse consequence.

**Recommendation:** treat "not specified" as the only admissible fallback in this
renderer, and make the other five honest absences. This is a small change but it
alters the model input, so it must either land **before** H2A-3B2 builds on the
baseline or be handled as its own re-freeze — not folded into the migration,
where a prompt diff would be indistinguishable from a compiler regression.

---

## 5 · What 3B0 did not do

- no `ContextCompiler` and no `AgentContextPack`
- no `ContextBudget`, and `RunBudget` untouched
- no change to disclosure ordering, Binder projection, or the B1/B2 boundaries
- no V1 change of any kind (F1, F6)
- no re-capture of any sealed artifact; no seal created or moved

**Zero production behaviour changed except the specialist's `Source:` line.** The
full suite is re-run at closure; the H2A-2 seal is untouched at `9497943`.
