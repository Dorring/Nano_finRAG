# NF-V3 H2A-3B3 — Production B3 ContextCompiler Migration

Status: **CLOSED.** Predecessors: `nf-v3-h2a-3-context-surface-audit.md`,
`nf-v3-h2a-3b0-context-preconditions.md`, `nf-v3-h2a-3b1-context-compiler.md`,
`nf-v3-h2a-3b2-context-semantics.md`.
Baseline: the sealed H2A-2 is at `9497943`; this phase starts from `d5b7f86`.

The Context Runtime stopped being a shadow. Production B3 no longer builds its
own context: the compiler is the only path from the runtime's authoritative
state to what the model is shown, and the legacy assembly that worked beside it
is gone rather than dormant.

This is an **architecture migration, not a prompt redesign**. The model input
after it is byte-identical to the post-F10 baseline v2, on every authored
scenario.

---

## 1 · The production call graph, before and after

**Before** (hand-written assembly, inside `trusted_v2_generation.py`):

```
state.evidence_packets ─ _bound_items ─┬─ routing policy
                                       └─ project(SPECIALIST)  ─┐
state._calculation_result_obj ─ project_calculation(SPECIALIST) ─┤
                                                                 ↓
                     specialist.generate(question, evidence_items, calculation_result)
                                                                 ↓
                                     LocalSpecialistGenerator renders + generates
```

**After**:

```
state (authoritative)                       src/runtime/trusted_v2_generation.py
  ├─ bound_evidence_ids   ─┐
  ├─ bound_slot_bindings  ─┤   specialist_context_request        (rag_v2 adapter)
  ├─ normalized_query     ─┼──────────────────────────────────────────┐
  └─ calculation object   ─┘                                          ↓
                                                          ContextRequestV1
                                                                      ↓
                                                     ContextCompilerV1.compile
                                                      (selection → disclosure →
                                                       budget → AgentContextPackV1)
                                                                      ↓
                                              render_specialist_prompt(pack)
                                                        (src/generation)
                                                                      ↓
                                                     specialist.generate(prompt)
                                                                      ↓
                                              LocalSpecialistGenerator: tokenize,
                                              engine, decode
```

`_bound_items` survives, and only as the **routing policy's** input: the router
decides whether a generator is needed at all, before any context exists, and it
reads `entity`/`company`/`ticker` -- which the SPECIALIST disclosure profile does
not admit, so it cannot consume a pack. It no longer selects; it delegates to the
adapter's `admitted_specialist_evidence`, so "which evidence is admitted" has one
implementation here instead of two.

## 2 · The renderer boundary, and the interface it required

`render_specialist_prompt` now takes an `AgentContextPackV1` and nothing else:

```python
def render_specialist_prompt(pack: AgentContextPackV1) -> str
```

It was previously handed `(question, evidence_items, calculation_result)` -- the
pack's three dynamic fields travelling loose, which meant nothing stopped a
caller assembling them another way and nothing stopped the renderer reaching past
them. It has no reference to a RunState, an `EvidencePacketV1`, a
`CalculationResult` or a binding from which to reach around the compiler.

**This forced the specialist backend protocol to change**, from
`generate(question, evidence_items, calculation_result)` to
`generate(prompt: str)`, and that is the one decision in this phase that reaches
into the provider interface. It is flagged for review as §9.1. The reasoning:

* §4 requires the renderer's input to be the pack, and §15 requires "one
  renderer invocation" to be *pinnable*. Both hold only if the renderer is
  invoked where the count is observable. A renderer called from inside
  `local_specialist_generator.py` is not observable from any test this
  environment can run -- that module is excluded from collection without torch,
  which is exactly the trap H2A-3B0 extracted the renderer out of.
* The narrowing is a **disclosure guarantee, not a tidy-up**. After it, the
  specialist boundary is a string: there is no argument through which evidence
  could reach a model without passing Disclosure Authority. The migration's
  §11 requirement ("must not add another projection that bypasses the
  established role profiles") becomes structural rather than a claim about
  discipline.

The cost is enumerated in §8. The alternative considered and rejected was
passing the pack to the backend instead of the text: identical blast radius, and
it leaves the renderer invocation inside a module the suite cannot reach.

## 3 · Legacy construction removed

| Superseded | Where | Disposition |
| --- | --- | --- |
| evidence selection loop | `_bound_items`, `trusted_v2_generation.py` | **delegates** to `admitted_specialist_evidence`; the loop is gone |
| evidence projection | `_call_specialist` | **deleted** -- the compiler's |
| calculation projection | `_call_specialist` | **deleted** -- the compiler's |
| `render_prompt` | `LocalSpecialistGenerator` | **deleted**; the renderer is `render_specialist_prompt(pack)` |
| the adapter's 3-argument forwarding | `LocalSpecialistGenerationAdapter` | narrowed to one argument |

No feature flag, no dual path, no shadow compile. `_bound_items` is the one
survivor, and §1 says why it is not a survivor of *context construction*.

The audit that preceded this found a fifth duplicate the brief did not name:
`scripts/runtime/run_nf_v2_21_runtime_integration.py` drives
`LocalSpecialistGenerator` directly, bypassing the Harness entirely, and had
re-derived the context by hand -- complete with its own `"Metric"` / `"Period"` /
`"1"` defaults and its own decision to pass `source_text` through. It now renders
through the adapter and compiler. **Its model input changes**: `source_text` is
not on the SPECIALIST profile, so re-running it produces prompts without the
narrative evidence text. That difference is Disclosure Authority doing its job,
and its committed artifacts are historical (§9.2).

## 4 · What the pack gained, and what it deliberately did not

No new pack field, no new budget mechanism, no model-adaptation capability. The
pack is byte-for-byte the one H2A-3B2 built.

One structural guard was added, in `AgentContextPackV1._check_references`
(renamed from `_check_support_groups`): **one evidence handle per evidence
item**. The renderer now reads its citation handles from `pack.references`
rather than minting `E{i}` positionally, so the correspondence between handles
and evidence became load-bearing where it used to be coincidental -- and a pack
whose handle list did not correspond would render a citation that resolves to
the wrong item. It is a guard, not a field.

## 5 · The rendering decisions, frozen

Per the phase's §1 and §2:

* `pack.references.support_groups` is **Harness semantic metadata**. The current
  B3 renderer does not render the support topology, the `G1`/`G2` handles, or
  `canonical_slot`.
* `canonical_slot` is an internal semantic reference copied from the binding
  authority. It is not an evidence disclosure field, not a citation handle, not
  a model-facing value, and not a slot authority. It is not run through the
  SPECIALIST evidence profile, is not rendered, and is not rebuilt anywhere.
* Pack capability may exceed renderer use. A provider-agnostic semantic contract
  does not oblige every renderer to expose every field, and whether a future
  provider renders the topology is a provider/renderer policy decision that can
  be evaluated on its own rather than folded into a migration.

This is pinned as its own regression: **changing only the support groups, with
evidence and calculation identical, does not move the prompt.** The two packs in
that test differ in slot keys, so a renderer that leaked group identity would
fail rather than pass by accident.

## 6 · Support topology, through the production path

All four shapes the phase names, compiled through the production adapter. The
table records which of them the *router* actually delivers to the specialist,
because that is a separate decision from whether the context layer can represent
them:

| Shape | Groups | Driven by |
| --- | --- | --- |
| one slot, one support | `G1 → E1` | the capability (`QUALITATIVE` hint) |
| one slot, two supports | `G1 → E1, E2` | the capability (temporal query) |
| two slots | `G1 → E1`, `G2 → E2` | the capability (`MULTI`, distinct facts) |
| mixed | `G1 → E1, E2`, `G2 → E3` | the capability (`MULTI`, distinct keys) |

The router sends a state whose items collapse to *one* canonical fact to the
deterministic renderer -- that is H2A-2C's decision and this phase does not
reopen it -- so the one-slot-two-supports shape reaches the specialist only
through the temporal branch, which is checked first. The test asserts both: the
capability path, and the same state through the adapter alone, with identical
topology either way.

**The topology is not inferred, and the test for that is comparative.** The same
evidence, bound two ways (apart, together), produces two different topologies
while the evidence crossing is identical -- so if any part of the production path
derived groups from metric/period/value, the two would come out the same.

No synthetic artifact: a group carries a handle, a slot key and a list of
handles. It has no value, no metric and no citation, and two witnesses stay two
items in the authority's order. The prompt states the value twice because there
are two witnesses; the topology adds no third copy.

## 7 · Differential and correctness

**Change detector.** The migrated path reproduces `BASELINE_V2` exactly on all
five authored scenarios -- every recorded field, plus the prompt and its SHA256.
Verified twice: once through `observe()`, once through a test that drives the
capability directly.

The baseline's `projected_evidence` / `projected_calculation` now come from
`capability.last_context_pack` rather than from what the backend was handed,
because the backend is handed a string. The *expected* values are unchanged:
they were captured from the legacy path before the compiler took over, so
reproducing them is the differential rather than a restatement of it.

**Independent semantic oracle.** Unchanged and still green:
`test_specialist_truthfulness` (35, fixture-derived, total over every field),
`test_b3_support_topology` (20), the fixture-derived half of
`test_b3_legacy_context_baseline`. Neither half substitutes for the other, and
the differential alone would pass if the capture and the implementation were
regenerated from each other.

## 8 · Blast radius of the interface narrowing

| Touched | Count |
| --- | --- |
| backend implementations migrated to `generate(prompt)` | 11 (2 production, 9 test) |
| `render_specialist_prompt` call sites updated | 7 (2 production, 5 test) |
| modules whose *model input* changed | 1 (`run_nf_v2_21_runtime_integration.py`) |

`tests/test_local_specialist_generator.py` needed **no** change -- it exercises
`_resolve_nanochat_repo` and never calls `generate`. That is luck rather than
design, and worth recording: it is the only module that imports
`LocalSpecialistGenerator` directly, and it is excluded from collection wherever
torch is absent, so a change *to that class* could not have been verified here.
The class's own behaviour after this phase is: tokenize the prompt it is given,
run the engine, decode. No rendering.

## 9 · Decisions flagged for review

1. **The specialist backend protocol was narrowed to `generate(prompt: str)`.**
   §2 gives the reasoning and this is the phase's largest interface change. It
   was necessary to make §4's "renderer input = AgentContextPackV1" literal and
   §15's renderer count observable. If the intent was to leave the provider
   interface entirely to H2A-3C, the alternative is to have the provider receive
   the pack and render it -- the same blast radius, a weaker guarantee, and a
   renderer invocation the suite cannot observe.
2. **`run_nf_v2_21_runtime_integration.py` now produces different prompts.** Its
   hand-rolled context passed `source_text` to the model and invented `Metric` /
   `Period` / `1` defaults. It now goes through the compiler, so the narrative
   evidence text does not cross. The script is an archived NF-V2-21 instrument
   whose outputs are committed; those artifacts stay historical.

## 10 · Verification

* **Three injections, all caught**, sources restored byte-identically
  (SHA256-verified): rendering the support topology into the prompt; rendering
  twice per invocation; reimplementing selection inside `_bound_items`.
* **Full suite: 4387 passed, 143 skipped, 0 failed** (+25 over H2A-3B2's 4362).
* **Readiness unchanged:** `task_success 19, semantic_mismatch 0, label_alias 3,
  unexplained_mismatch 0, false_release 0, over_conservative 0`,
  `token_counts_available: False`.
* **Layering green:** `rag_v2` still imports `src` zero times. The adapter is the
  only place that knows what a RunState is, and it stays duck-typed.
* **Disclosure unchanged:** the production generation module no longer imports
  the Disclosure Authority at all, asserted on its syntax tree (the first
  version of that check grepped the source and failed on its own comments --
  the right failure, since a check a comment can pass is not checking the code).

## 11 · Remaining Harness debt before H2A-3C

1. **B5/B6/V1 have no role.** `ContextRoleV1` still has one member. Each gets
   its own policy, profile and renderer when its requirements are known.
2. **`canonical_slot` disclosure is decided but not yet reviewed.** It travels in
   the pack and is not rendered here; H2A-3C should confirm that is the permanent
   answer rather than a phase-local one.
3. **The renderer is chosen by the capability, not by the provider.** A future
   provider wanting its own rendering of the pack has no seam to plug into. That
   is deliberate for this phase -- adding one would have been model-adaptation
   capability -- and it is the first thing H2A-3C's provider interface has to
   settle.
4. **`_bound_items` still exists.** It is routing's input, not context's, but the
   name invites a future contributor to treat it as the selection again. A rename
   belongs with whatever settles the router's own interface.
5. **The traced counts are new and unexercised in production.** They are asserted
   in tests; no runtime consumer reads them yet.
6. **F1 and F6 are untouched**, as scoped. The tokenizer is a provider
   capability; model-derived artifact admission is its own trust contract.

## 12 · What H2A-3B3 did not do

* no change to `ContextBudgetV1`; no token bound is configured anywhere, and B3
  runs with the token limit unset
* no F1, no F6, no training-dataset renderer alignment
* no DeepSeek or financial-model provider, no provider-specific behaviour
* no V1, B5 or B6 integration
* no new seal; H2A-2 remains at `9497943`
* no H2A-4, and no token-efficiency number of any kind
