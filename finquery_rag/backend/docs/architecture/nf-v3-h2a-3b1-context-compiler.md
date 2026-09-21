# NF-V3 H2A-3B1 — The Context Compiler

Status: **CLOSED.** Predecessors: `nf-v3-h2a-3-context-surface-audit.md`,
`nf-v3-h2a-3b0-context-preconditions.md`.
Baseline: `HEAD fc1e82d` (H2A-3B0), seal `nf-v3-h2a2-artifact-authority` at `9497943`.

H2A-3B1 builds the complete minimum Context Compiler framework and proves it
reproduces B3's intended dynamic context **in shadow**, without touching the
production model path. The production B3 prompt still receives the legacy
context. F10 is not fixed. B3 is not migrated.

---

## 1 · The compiler's responsibility, frozen

```
already-admitted authoritative runtime artifacts
      ↓  deterministic model-visible selection     (reads authoritative fields)
      ↓  Disclosure Authority projection           (deny by default)
      ↓  ContextBudget enforcement                 (on the disclosed payload)
   AgentContextPackV1                              (disclosed projections only)
```

It does **not** own retrieval, evidence admission, consensus or conflict
resolution, slot binding, calculation execution, claim verification, replanning,
memory, artifact storage, prompt repair, or durable runtime state. It is not a
second authority. A compiler that could admit evidence would be a second Binder;
one that could decide which value is canonical would be a second binding
authority.

## 2 · New modules

| Module | What it is |
| --- | --- |
| `rag_v2/context/contracts.py` | `AgentContextPackV1`, `ContextBudgetV1`, `ExactTokenCounterV1`, selection trace, accounting |
| `rag_v2/context/compiler.py` | `ContextCompilerV1` — the kernel, and `ContextRequestV1` |
| `rag_v2/context/specialist.py` | B3's role policy and the RunState → request adapter |
| `tests/harness/test_context_compiler.py` | 23 kernel tests, role-agnostic |
| `tests/harness/test_b3_shadow_parity.py` | 39 B3 shadow and parity tests |
| `tests/architecture/test_rag_v2_layering.py` | 2 tests pinning `rag_v2` ⇏ `src` |

The compiler lives in `rag_v2` because the Disclosure Authority does, and
`src/runtime` imports `rag_v2` — never the reverse. The layering test is new
because this package is the first that could plausibly have broken it: the
natural way to be sure a value *is* a calculation is
`isinstance(value, CalculationResult)`, and that type lives in `src/domain`. The
adapter **duck-types on `to_dict`** instead. That decision is invisible without
the test.

## 3 · `AgentContextPackV1` — exact shape

Eight fields, and the set is asserted exactly by test, so adding a ninth
requires editing a test and saying so.

| Field | Meaning |
| --- | --- |
| `role: ContextRoleV1` | the boundary; currently `SPECIALIST` only |
| `invocation_id: str` | one model call, e.g. `{request_id}:candidate-generation` |
| `query: str` | the dynamically model-visible query |
| `evidence: tuple[Mapping, ...]` | **disclosed projections**, frozen views |
| `calculation: Mapping \| None` | the disclosed calculation projection |
| `references: tuple[ContextReferenceV1, ...]` | model-visible handles (`E1..En`, `C1`) bound to authoritative identities |
| `budget: ContextBudgetAccountingV1` | what governed the invocation and what it did |
| `selection: ContextSelectionTraceV1` | content-free record of every considered artifact |

**Lifetime:** built immediately before an invocation, discarded with it and its
trace. It is not state, not a store, not a cache, not a source of truth. Nothing
may read a pack to learn what the runtime believes — only what one model was
shown.

**Negative requirements, enforced rather than promised.** `__post_init__`
refuses any nested mapping anywhere in the evidence. That is not defensive
style: a raw evidence packet is *identified* by carrying containers
(`metadata`, `temporal`), so refusing containers is the structural form of "no
authoritative object was smuggled in". A disclosed projection on this profile is
flat by construction, so nothing legitimate is lost — the kernel test that
constructs a real `EvidencePacketV1.to_dict()` and asserts refusal is the proof.

**Deliberately absent:** static instructions. The renderer owns system text,
formatting and output-schema wording. Putting instructions in the pack would
turn a context compiler into a universal prompt compiler, which is a larger and
different thing. Also absent: conversation, memory, plan, tools, trace,
`RunState`, raw `EvidencePacketV1`, raw `CalculationResult`, `ClaimProvenance`,
`ArtifactStore`.

## 4 · `ContextBudgetV1` — exact shape, and why it is not `RunBudget`

```python
max_input_tokens:      int | None = None   # enforced only with an exact counter
max_evidence_items:    int | None = None   # enforced by the compiler
reserved_output_tokens:int | None = None   # carried, not enforced
```

Three fields; a test asserts the field sets of `ContextBudgetV1` and
`AdaptiveRAGBudgetV1` are **disjoint**, and that no context field is named for a
retry, tool, timeout or replan.

The two answer different questions. `RunBudget` bounds how much **work** the
runtime may perform; `ContextBudget` bounds how much **information** one
invocation sees. A shared type would mean one knob changing both, which is how a
context change becomes a latency change.

`reserved_output_tokens` is carried and not subtracted. It is the caller's
declaration that the window must also hold the answer; silently narrowing
`max_input_tokens` by it would reinterpret a number the caller already chose.

`max_evidence_items=0` is refused. A boundary whose model-visible evidence cap is
zero is a misconfiguration, not a budget; genuinely-absent evidence is the
runtime's fail-closed path, and the two are kept apart.

## 5 · Exact-token discipline

`ExactTokenCounterV1` is a protocol with `counter_id` and `count`.

**The rule:** if `max_input_tokens` is configured, an exact counter for that
model must be supplied. If it is not, construction raises
`ContextBudgetUnsupported` — at configuration time, not at first use. Word
counts, character counts, `tiktoken` on a model that is not a `tiktoken` model,
and every other substitution are forbidden. An estimate presented as a
measurement is worse than no bound, because it silently truncates context the
boundary believed it had room for.

`counter_id` is not decoration: it is recorded into every pack's accounting, so
a reader can tell which counter produced a number. The kernel tests use a
`test-only-word-count` double to exercise budgeting mechanics — and the counter's
name is the warning. **Those numbers are not token counts for any real boundary
and no H2A-4 metric may be produced from them.**

`selected_context_tokens` measures **the compiler's own serialized payload**, not
the model's prompt: the prompt also carries static instructions the renderer
owns. The field is named for exactly that reason. When no counter is supplied the
value is `None` — absence recorded as absence, never zero, never an estimate.

**B3 runs with no token bound.** Evidence-count bounding is available and
exercised; token admission is not enabled, because the exact tokenizer is the
checkpoint's and this environment cannot reach it. Not enabling it is the honest
state, not an omission.

## 6 · Deterministic selection

`SpecialistContextPolicyV1` is deliberately thin, because B3's current semantics
*are* "show the model every item the Binder admitted, in the state's order".
Narrowing that here would be a behaviour change smuggled into a framework phase,
and 3B2's differential would then be comparing two changes at once.

What the policy owns is the ordering contract and the evidence budget. Relevance
ranking is **explicitly absent**; if it is ever added, the contract is that it
must be deterministic and inspectable. An LLM context selector would put a second
model between the evidence and the model that answers from it, and no such thing
is contemplated.

Selection reasons are content-free: `ADMITTED`, `DROPPED_BY_EVIDENCE_BUDGET`,
`DROPPED_BY_TOKEN_BUDGET`. The trace is **complete** — every considered artifact
appears exactly once in admitted order — so shedding is visible as a *reason*
rather than an absence. Token shedding removes from the tail (the role policy's
least-preferred position), never the middle, so the surviving order does not
depend on how many were dropped. This is the minimum trace; there is no second
observability system.

**Selection happens before projection, deliberately.** Selection must read
authoritative semantic fields — which metric, which period — to decide anything
at all, and for V1 would otherwise need a container admitted that the profile
governs key-by-key. So selection inspects authority and the pack never holds it.
A kernel test proves this is a property rather than a convention: a policy that
selects on `raw_metric` works, and the pack has no `raw_metric`.

## 7 · B3 shadow parity

Compiled for all five authored baseline scenarios and compared field by field
against the frozen legacy projection.

| Legacy B3 context field | Compiled behaviour | Class |
| --- | --- | --- |
| projected evidence (values, order, count) | identical | **A** |
| disclosed field set (namespaced) | identical | **A** |
| authoritative page per item — `7`, `0` | identical | **A** |
| authoritative page — absent | absent in both; no default | **A** |
| calculation projection (`operation`/`unit`/`value`) | identical | **A** |
| model-visible handles `E1..En`, `C1` | identical (positional) | **A** |
| static instructions / `[ANSWER RULES]` | not in the pack; still renderer-owned | **B** |
| formatted prompt text | not produced by the compiler at all | **B** |
| claim/support cardinality | absent from both | see §9 |

**There is no class D divergence.** Recorded plainly rather than manufactured:
the compiled pack and the legacy projection agree on every field both produce.

## 8 · Known F10 debt, and where it lives

F10 is **not fixed**, as instructed. The parity matrix shows why it can wait:

The fabricated fallbacks — `scale or "1"`, `document_id or "filing"`,
`metric or "Metric"`, `period or "Period"`, `scope or metric` — are in the
**renderer**. Neither the legacy *projection* nor the compiled pack contains
them: both carry absence as absence, because `project()` emits only what exists.
The over-reach happens at render time, downstream of both.

So F10 is not a legacy-vs-compiled divergence at all. It is a shared prerequisite,
and it is why H2A-3B1C fixes it once in the renderer. **A compiled prompt rendered
by the current renderer would carry the same F10 defects as the legacy prompt** —
which is the correct and expected consequence of not having fixed them yet.

The current baseline remains a **pre-F10** baseline. 3B2's differential must run
against the **post-F10** baseline (v2); the pre-F10 digests are not the final
parity target.

## 9 · Known gap recorded for 3B2 — claim cardinality

> **Closed by H2A-3B2.** The pack now carries the Binder's claim/support
> relation as `references.support_groups`, projected rather than re-derived, and
> the pack's field set is unchanged. See
> `nf-v3-h2a-3b2-context-semantics.md` §2–§6. Everything below is the record of
> what was open at this commit, kept as written.

The pack has **no representation of "one claim with N supports"**. If a
multi-support state were compiled, it would carry both supports as two ordinary
evidence items, indistinguishable in shape from two genuinely distinct facts, and
a renderer reading it could not avoid stating the value twice.

This does not happen in production: the routing policy sends that shape to the
deterministic renderer. Verified against the real capability, not read from the
policy's source — `renderer_invoked=True`, `specialist_invoked=False`.

It is recorded rather than fixed because the correct representation is a *claim
with N supports*, which is precisely the structure H2A-2C spent a phase
establishing. Inventing a second version of it inside a context compiler would be
exactly the "compiler becomes an authority" failure this phase is built to avoid.

## 10 · Migration preconditions for 3B2

1. **F10 fixed and a post-F10 baseline (v2) frozen.** The pre-F10 digests are not
   a valid parity target, and a prompt diff during migration would otherwise be
   unattributable.
2. **A decision on §9.** Either the pack gains a claim/support representation, or
   B3's migration is explicitly scoped to shapes the router sends it and the gap
   is documented as out of scope.
3. **The duplicated selection is removed, not left.** `_bound_items` in
   `src/runtime/trusted_v2_generation.py` and `admitted_specialist_evidence` in
   `rag_v2/context/specialist.py` implement the same selection. The duplication is
   the price of proving the framework without moving production; 3B2 is where it
   is paid off.
4. **The call site is switched in one commit** — pack constructed, renderer fed
   from it, `_bound_items` deleted — so that the differential covers a single
   change.
5. **`test_the_production_generation_path_does_not_use_the_compiler` is inverted
   deliberately.** That test exists to fail the moment B3 is opted in early; 3B2
   replaces it with a test that the production path uses the *compiler*, and the
   replacement is the visible act of migration.

## 11 · What 3B1 did not do

- no F10 fix; no change to any renderer fallback
- no migration of the B3 model path; production still builds context the legacy way
- no `ArtifactStore`, no change to `RunBudget`
- no V1, B5 or B6 integration — each gets its own role when its requirements are known
- no LLM context selector, and no embedding-based relevance ranking
- no new seal; H2A-2 remains at `9497943`

Readiness is unchanged at `19 / 0 / 0 / 3 / 0 / 0` with `SEMANTIC_MISMATCHES = {}`.
