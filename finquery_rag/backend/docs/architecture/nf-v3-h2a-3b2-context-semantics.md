# NF-V3 H2A-3B2 — Context Semantic Completeness & Specialist Truthfulness

Status: **CLOSED.** Predecessors: `nf-v3-h2a-3-context-surface-audit.md`,
`nf-v3-h2a-3b0-context-preconditions.md`, `nf-v3-h2a-3b1-context-compiler.md`.
Baseline: `HEAD f7e1b96` (H2A-3B1), seal `nf-v3-h2a2-artifact-authority` at `9497943`.

Two independent pieces of work, deliberately in one phase because both are
prerequisites for the same migration:

* **A.** the pack gains the one thing H2A-3B1 recorded as a gap -- a canonical
  claim and its N supports, as a **projection of the binding authority**;
* **B.** F10 is fixed: five fabricated defaults are removed from the specialist
  renderer, and the legacy baseline is re-frozen as **v2**.

Production B3 is still on the legacy path. No model is reached differently. No
token tuning, no V1, no H2A-4.

---

## Part A · One canonical claim, N supports

### 1 · Where the relation comes from, and where it must not

The relation is `slot_bindings`: a mapping from a Binder slot key to the
evidence ids admitted as that slot's supports. It is established at
`rag_v2/contracts/evidence.py:33` (`EvidenceBinding.slot_bindings`,
`Mapping[str, tuple[str, ...]]`), projected onto the runtime state by
`src/runtime/trusted_v2_coordinator.py:658`, and read by the compiler from
`AdaptiveRAGStateV1.bound_slot_bindings` (`rag_v2/adaptive/adaptive_contracts.py:403`).

```
EvidenceBinding.slot_bindings        (the authority, upstream)
      -> state.bound_slot_bindings   (the runtime's own projection of it)
      -> ContextRequestV1.slot_bindings
      -> ContextCompilerV1           (a rename and a filter)
      -> pack.references.support_groups
```

The compiler does not decide when two evidence items are one claim, and it has
no code that could. Three independent reasons, in decreasing strength:

1. **It cannot reach the code that does.** Canonicalisation lives in
   `src/runtime/trusted_v2_binder.py` and `src/generation`, and `rag_v2` imports
   `src` zero times -- pinned by `tests/architecture/test_rag_v2_layering.py`.
2. **It would be a second binding authority.** `_consensus_fact_for_slot` groups
   by canonical quantity, deduplicates by physical source, and requires a strict
   majority before returning a winner. A compiler re-deriving any of that would
   disagree with the Binder the first time either changed.
3. **There is no canonical slot-key function to call.** Claim identity is the
   supervisor's `slot_id`; nothing in the repository composes one from
   metric+period+scope. `GeneratorRoutingPolicy._semantic_slot_identity` comes
   closest and is a *routing* judgment in the application layer.

### 2 · The pack shape

`references` is still **one** top-level field -- the pack's eight fields are
unchanged, and a test asserts the exact set. It changed type, from a bare tuple
of handles to:

```python
ContextReferencesV1
├── handles:        tuple[ContextReferenceV1, ...]   # E1..En, C1 -- unchanged
└── support_groups: tuple[ContextSupportGroupV1, ...] # G1..Gm

ContextSupportGroupV1
├── handle:         "G1"
├── canonical_slot: "revenue/FY2024"   # the authority's slot key, verbatim
└── supports:       ("E1", "E2")       # handles, not identities
```

Two questions kept as two fields rather than mixed into one sequence: `handles`
is the citation namespace a model writes in and a validator resolves against,
and a consumer resolving a citation should not have to filter group entries out
to do it. `ContextReferencesV1.evidence_handles` is the derived accessor.

### 3 · The projection rules

`_support_groups` is a rename and a filter. Per slot, in the authority's order:

* ids are translated to the handles the model will actually cite;
* **an id the pack does not carry is left out** -- the boundary did not release
  it, so a group naming it would assert a corroboration the model cannot check
  against evidence it can see;
* **a slot whose every support was left out is not emitted at all** -- an empty
  group is that same claim with nothing behind it;
* **a repeated id collapses.** The authority deduplicates where bindings are
  built, so this does not arise in production. The guard is there because if it
  ever did, reporting the id twice would state a corroboration that does not
  exist -- the failure this phase removes from the renderer, one layer up.

The pack enforces the first rule **structurally** rather than trusting the
compiler: `AgentContextPackV1._check_support_groups` raises `PackIntegrityError`
for a group naming a handle it does not carry, for an empty group, and for a
duplicated support. That is the same move as refusing nested mappings -- it makes
the property hold against a future caller, not only against today's one.

Group handles are `G1..Gm` over *emitted* groups, so they are contiguous and
deterministic.

### 4 · The six scenarios

| Shape | Pack |
| --- | --- |
| single evidence | `G1 = [E1]` |
| one canonical fact, two independent supports | `G1 = [E1, E2]` |
| two distinct facts | `G1 = [E1]`, `G2 = [E2]` |
| multi-support claim beside a single-support claim | `G1 = [E1, E2]`, `G2 = [E3]` |
| conflicting candidate | in no group, and in no pack |
| physical-source duplicate | one support, as the Binder counted it |

Singletons are emitted deliberately. If only multi-support slots produced a
group, a reader could not distinguish "these two are one claim" from "the
topology was not established" -- and "these are two separate claims" would be
the one thing the pack could not say.

**Conflict** is refused one layer up: `_unresolved_conflict_slots` clears both
the admitted set and the slot bindings, so a conflicting candidate never reaches
the compiler. The test asserts the compiler's own half as well -- an id named in
a binding but not admitted must not enter a group -- because either half alone
would leave the property resting on the other.

**Physical-source duplicates** are the case where the compiler's *silence* is the
correct behaviour, and it is tested in both directions: given the Binder's
verdict (one id for the slot) the second packet is admitted evidence that does
**not** join the group; given a binding that names both, both are reported,
because collapsing them would be the compiler overruling a binding it did not
make.

### 5 · The behavioural proof that nothing is recomputed

`test_the_compiler_carries_a_grouping_it_would_not_have_chosen_itself` binds two
items with *different values in different periods* into one slot and asserts the
pack reports them as one claim. Any canonicaliser would call them two facts. That
is the point: a compiler that "corrected" this would be recomputing the relation,
and this test is what makes such a change visible instead of silent.

### 6 · When the authority is silent

No binding produces **no groups** -- not one group per item. "The authority did
not place this item" and "this item is its own claim" are different statements,
and only the first is one a projection may make. The consequence is that an older
or partial state compiles to a pack with no topology, which is honest and is
visibly different from a pack whose authority said "these are three separate
claims".

**Recorded limitation:** the pack does not carry a flag distinguishing "no
topology established" from "topology established, and this evidence is in no
slot". Both render as an absence of groups. The second does not arise today --
every admitted id on the live path comes from a slot -- and a boolean for it
would have been a ninth field bought for a case nobody has reached.

### 7 · The measured payload now includes the topology

`selected_context_tokens` measures `_payload_text`, which now serializes
`support_groups` as well as the evidence and calculation. A support group is
context, so a bound measured without it would under-count the thing it exists to
bound. Within the token-shedding loop the groups are **rebuilt** after each pop,
because a shed support is no longer a witness this invocation can see and the
pack refuses a group naming a handle it does not carry.

---

## Part B · F10 — five fabricated defaults, removed

### 8 · What was there

```python
metric   = ev.get("metric") or ev.get("normalized_metric") or "Metric"
period   = ev.get("period") or "Period"
scale    = ev.get("scale") or "1"
scope    = ev.get("scope") or metric
source   = ev.get("document_id") or "filing"
```

Frozen in the v1 baseline, verbatim: `page_variants` renders `Scope: Revenue`,
`Scope: Cost of revenue` and `Scope: Operating income` -- the scope of each
figure asserted to be its metric -- and every scenario with no scale renders
`Scale: 1`.

They are worse than an omission, and the difference is why they are treated as a
truthfulness defect rather than untidiness. `Currency: not specified` sits two
lines below and is a marker no one can mistake for a currency; there is no metric
named `Metric`, no filing named `filing`, and no page `1` nobody recorded. Those
are **values** where the record had none. `scope or metric` is the sharpest case:
it answers an unknown with the assertion that scope *equals* metric, which is a
new fact about the world rather than a placeholder.

### 9 · The audit: marker or omission

The brief's rule was to handle fields by semantics, prefer omitting where absence
does not break the structure, and use an explicit marker where the schema needs a
placeholder. **The audit found the schema needs the placeholder**, on three pieces
of evidence:

1. The frozen contract (`data/grounding_alignment/v1/financial-generation-view-v1.md`)
   is a **fixed set of lines** per evidence block, not a variable one.
2. The canonical renderer for that schema -- `rag_v2/generation/financial_view_v1.py`,
   whose `CONTRACT_SHA256` pins it -- spells absence as `not specified` in its
   `_value` helper, **uniformly, for every field including metric, period and
   scope**.
3. Answer rule 6 asks the model to notice missing evidence. A stated absence
   serves that; a vanished line does not.

So every field goes through one `_render`, and absence is always the same
marker. **This is a deliberate divergence from the brief's stated lean** (which
preferred omitting document/metric/period/scope), and it is the one decision in
this phase a reviewer should check first. Omitting those lines remains available
if the reviewer prefers it; it is a change to `_render`'s call sites and the
expectations beside them, and nothing else in the phase depends on the answer.

The alternative is not free in either direction, and the asymmetry is why the
marker won: an omitted line makes the block's shape depend on the data, so a
reader can no longer tell a field that was absent from a renderer version that
stopped emitting it.

### 10 · What replaced it

```python
ABSENT = "not specified"   # one marker, every field

metric = _stated(ev, "metric");  if metric is None: metric = _stated(ev, "normalized_metric")
document, page = _stated(ev, "document_id"), _stated(ev, "page")
source = ABSENT if document is None and page is None else f"{_render(document)}:{_render(page)}"
```

* `or` is gone from every field. `_stated` returns the value or `None`, and
  treats `""` as absent -- "the evidence states the empty string" is not a thing
  anyone recorded.
* `0` survives. This is the F5 finding re-applied: a truthiness test cannot tell
  a recorded zero from an absent value.
* **`normalized_metric` is kept**, and written out as an `if` rather than an
  `or` so the distinction is visible at the point of use: this is *reading a
  second declared field*, not *inventing a value*. The canonical renderer and
  the routing policy both read the pair the same way.
* `Source` states each component independently: `doc-1:7`, `doc-1:not specified`,
  `not specified:7`, and the single marker when neither is known -- which is what
  the canonical renderer emits for it.

### 11 · The blast radius

Four of the five baseline scenarios changed, and `multi_fact` did not:

| Scenario | `prompt` / `prompt_sha256` |
| --- | --- |
| `multi_fact` | **byte-identical** |
| `temporal`, `qualitative_single`, `calculation_with_explanation`, `page_variants` | changed |

`multi_fact` is the scenario whose evidence carries every field, so no fallback
ever fired for it. That is the sharpest available statement that the change was
the *removal of defaults* and not a rewrite of the renderer, and it is asserted
rather than described.

### 12 · The baseline split

* `tests/harness/b3_legacy_context_baseline_v1.py` -- the pre-F10 record, kept
  as **history**. Its own test asserts it still shows the fabrications; if
  someone tidies it to agree with v2, the claim "F10 was real" loses its only
  artifact.
* `tests/harness/b3_legacy_context_baseline.py` -- `BASELINE_V2`, the post-F10
  capture, and **the migration target**. H2A-3B3's differential compares against
  v2 and nothing else.

A migration's prompt diff has to be attributable to the migration. It could not
be if the target still contained fabrications the migration would not introduce.

### 13 · The tests, and which kind each one is

| File | Kind |
| --- | --- |
| `test_specialist_truthfulness.py` (35) | fixture-derived correctness, at the renderer and through the boundary |
| `test_b3_support_topology.py` (20) | fixture-derived correctness, at the adapter and the pack |
| `test_b3_legacy_context_baseline.py` (31) | equivalence to v2, contract properties on the frozen record, and the v1/v2 history |
| `test_context_compiler.py` (+3) | kernel-level: projection, absence of topology, measured payload |
| `test_b3_shadow_parity.py` (updated) | equivalence to v2, plus the closed gap |

The truthfulness tests are **total**, not spot checks, because the defect was
five separate fabrications in five fields: for every field, in both directions,
the rendered line is either the authored value or the marker. A test asserting
`"Scale: 1" not in prompt` would have caught exactly one of the five.

`not specified` is written out literally in the test files rather than imported
from the renderer. Importing the constant would make "absence is stated honestly"
true by definition.

---

## 14 · Verification performed

* **Three injections, all caught**, with the sources restored byte-identically
  and the restore verified by SHA256:
  * reinstating `scope or metric` → truthfulness tests red;
  * inventing a singleton group for unplaced evidence → topology test red;
  * reordering supports within a group → order test red.
* **Full suite: 4362 passed, 143 skipped, 0 failed** (+65 over 3B1's 4297).
* **The 9 `tests/architecture` failures** (`chromadb` absent) are pre-existing:
  confirmed by stashing the changes and reproducing them on the untouched tree.
  They do not appear in a full-suite run, which is a pre-existing collection
  quirk, not a result of this phase.
* **Readiness unchanged:** `task_success 19, semantic_mismatch 0, label_alias 3,
  unexplained_mismatch 0, false_release 0, over_conservative 0`,
  `token_counts_available: False`.

---

## 15 · Decisions flagged for review

1. **Marker rather than omission** (§9). The one place this phase departed from
   the brief's lean, on the evidence that the frozen schema needs a placeholder.
2. **`canonical_slot` now travels in the pack.** It is the identity behind the
   `G1` handle, in the same way `evidence_id` is the identity behind `E1` -- and
   `evidence_id` is on the SPECIALIST disclosure profile while a slot key is not
   an evidence field at all. It is a binding-level identifier composed of
   canonical metric/period/entity/scope names, and on this profile those semantic
   *classes* are already disclosed. The pack is a governed intermediate, and the
   renderer decides what becomes text; but this is a widening of what travels,
   and H2A-3B3 should decide explicitly whether a group's slot key is rendered.
3. **`selected_context_tokens` now counts the topology** (§7), so the number is
   not comparable with a 3B1 pack's for the same input.

---

## 16 · Migration preconditions for H2A-3B3

Replaces §10 of the 3B1 note. Items 1 and 2 are now done.

> **All seven discharged by H2A-3B3.** The migration happened, the duplicated
> selection was paid off, the anti-migration guard was inverted, and the renderer
> was extended to consume the pack -- which required narrowing the specialist
> backend protocol to a single prompt, since a renderer's invocation is not
> countable from inside the torch-gated provider module. Item 6 resolved as
> *not rendered*: the prompt is byte-identical to v2. See
> `nf-v3-h2a-3b3-compiler-migration.md`.

1. ~~F10 fixed, post-F10 baseline v2 frozen.~~ **Done** (§12).
2. ~~A decision on the claim/support representation.~~ **Done** (§2).
3. **The duplicated selection is still to be paid off.** `_bound_items` in
   `src/runtime/trusted_v2_generation.py` and `admitted_specialist_evidence` in
   `rag_v2/context/specialist.py` implement the same selection. 3B3 deletes the
   former.
4. **The call site is switched in one commit** -- pack constructed, renderer fed
   from it, `_bound_items` deleted -- so the differential covers a single change.
5. **`test_the_production_generation_path_does_not_use_the_compiler` is inverted
   deliberately.** It exists to fail the moment B3 is opted in early; 3B3
   replaces it with a test that the production path *uses* the compiler, and the
   replacement is the visible act of migration.
6. **New: the renderer must decide whether it renders the topology.** The legacy
   path produces no support groups; the compiled pack does. This is the one
   *intentional* asymmetry in the differential, and 3B3 has to say which way it
   goes:
   * **not rendered** → the migration is prompt-identical to v2, and the topology
     is carried for consumers that are not the prompt;
   * **rendered** → the prompt changes, and the change is *added* information
     rather than a regression. The differential must say so explicitly instead
     of treating a diff as a failure, and the renderer earns its own baseline.
7. **New: the topology must be supplied by 3B3's adapter, not invented there.**
   `specialist_context_request` already reads `state.bound_slot_bindings`; a
   migration that rebuilt the request from `_bound_items` would silently drop it
   and compile a pack with no groups -- which is exactly the state this phase
   removed.

---

## 17 · What H2A-3B2 did not do

* no production migration; B3 still builds its context the legacy way
* no Model Provider change, no tokenizer work, no F1, no F6
* no V1, B5 or B6 integration
* no new seal; H2A-2 remains at `9497943`
* no H2A-4, and no token-efficiency number of any kind

Readiness is unchanged at `19 / 0 / 0 / 3 / 0 / 0` with `SEMANTIC_MISMATCHES = {}`.
