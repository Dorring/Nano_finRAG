# NF-V3 H2A-3 — The Harness Runtime

Status: **SEALED.** Baseline: `nf-v3-h2a2-artifact-authority` at `9497943`
(immutable, unmoved). Predecessors, in order:
`nf-v3-h2a-3-context-surface-audit.md`,
`nf-v3-h2a-3b0-context-preconditions.md`,
`nf-v3-h2a-3b1-context-compiler.md`,
`nf-v3-h2a-3b2-context-semantics.md`,
`nf-v3-h2a-3b3-compiler-migration.md`,
`nf-v3-h2a-3c-model-invocation.md`,
`nf-v3-h2a-3d-model-derived-admission.md`.

This note is the consolidation: it states what the Harness is now, which
boundaries are governed, which authorities own what, and what is deliberately
left undone. The phase notes remain the record of how each piece was built.

---

## 1 · Scope

H2A-3 turned a working single-boundary pipeline into a **model-agnostic Financial
Agent Harness** with a trust boundary at both ends: what a model is shown, and
what its output is allowed to become.

What that means concretely:

* the context a model sees is compiled, deterministically selected and disclosed
  by one authority, and no other component can widen it;
* the model is reached through a provider-neutral invocation contract, so a
  financial model or a DeepSeek endpoint is a new binding rather than a change to
  the Harness;
* model output is a candidate until a deterministic check admits it, and on the
  ingestion path it is compared against the authoritative source artifact.

H2A-3 does **not** include: any model provider being integrated, F1's tokenizer
work, the B5/B6/V1 boundaries, evaluation or ablation. Those are later.

## 2 · Final Runtime Architecture

```
Authoritative Runtime (AdaptiveRAGStateV1, EvidenceBinding, CalculationResult)
        |
        |  Context Adapter            rag_v2/context/specialist.py
        v
   ContextRequestV1
        |
        |  ContextCompilerV1          rag_v2/context/compiler.py
        |    deterministic selection -> Disclosure Authority -> ContextBudget
        v
   AgentContextPackV1
        |
        |  ContextRendererV1          src/generation/specialist_prompt.py
        v
   ModelRequestV1
        |
        |  ModelProviderV1            rag_v2/invocation/provider.py
        v
   ModelResponseV1
        |
        +--> candidate answer -> Validator / Release        (the answer path)
        |
        +--> ModelDerivedArtifactV1 -> verify -> AdmittedDerivedArtifactV1
        |                                                       (the ingestion path)
        v
   Trusted Runtime
```

### Long-term type semantics, frozen

| Type | What it is, permanently |
| --- | --- |
| `ContextRequestV1` | the authoritative-runtime → compiler input |
| `AgentContextPackV1` | one inference's governed semantic projection |
| `ContextBudgetV1` | one invocation's model-visible context budget |
| `ModelBindingV1` | renderer + provider + optional exact tokenizer |
| `ModelRequestV1` | the provider-visible execution request |
| `ModelResponseV1` | a provider-neutral execution result |
| `ModelDerivedArtifactV1` | model-produced, explicitly untrusted |
| `AdmittedDerivedArtifactV1` | a verifier-approved derived representation |

And the four that are most easily confused, written out as inequalities:

```
AgentContextPackV1      !=  ArtifactStore
AgentContextPackV1      !=  memory
AgentContextPackV1      !=  RunState
ModelResponseV1         !=  trusted evidence
AdmittedDerivedArtifactV1 != source-original evidence
```

A pack exists for one invocation and is discarded with it. A response is what a
model said. An admitted artifact is a *verified representation* of a source, and
`source_reference` travels with it so a consumer can always tell which of the two
it is holding.

## 3 · Context Boundary

**Who may read authority.** The adapter, and only the adapter:
`specialist_context_request(state)` reads `bound_evidence_ids`,
`bound_slot_bindings`, the calculation object and the query, and hands the kernel
plain artifacts. It is the only thing in `rag_v2/context` that knows what a
RunState is.

**What the compiler does.** Selection, then disclosure, then budget — in that
order, because selection has to read authoritative semantic fields to decide
anything at all, and for V1 would otherwise need a container admitted that its
profile governs key by key. It does **not** do retrieval, evidence admission,
artifact admission, or provider execution.

**What the renderer sees.** An `AgentContextPackV1` and nothing else. It cannot
reach the RunState it came from, the Disclosure Authority it already passed
through, or the capability that called it — asserted as an import rule, not as a
convention.

**What may not cross.** `source_text`, `metadata`, `temporal`, operands, error
messages. Deny by default at one authority; there is no second projection
anywhere on the path, and the production generation module no longer even imports
the Disclosure Authority.

## 4 · Invocation Boundary

```
AgentContextPackV1 -> ContextRendererV1 -> ModelRequestV1 -> ModelProviderV1 -> ModelResponseV1
```

assembled by `ModelBindingV1` and driven by `ModelInvocationRuntimeV1`.

`ModelRequestV1` is the boundary and its field set *is* the boundary:
`invocation_id`, `role`, `prompt`, `provider_id`, `model_id`. There is no pack,
no evidence, no state, no calculation and no metadata bag, and a test asserts the
exact set — because a request that grew one of those would pass every behavioural
test in the suite while handing providers a route around the renderer.

A provider owns transport, serialization, authentication and error
normalization; it does not decide what it is shown or whether its output is
trustworthy. `LegacyPromptProviderAdapterV1` wraps the existing
`generate(prompt)` backends, so eleven implementations did not each have to
learn a new interface.

## 5 · Derived Artifact Admission

```
authoritative source artifact
  -> model invocation
  -> ModelDerivedArtifactV1      (produced, unverified)
  -> deterministic verifier      oracle = the source, never another model output
  -> ArtifactAdmissionResultV1
  -> AdmittedDerivedArtifactV1   (only from a successful admission)
```

`AdmittedDerivedArtifactV1` is `init=False` with one constructor taking
`(candidate, result)`. It refuses a rejection, refuses a result for a different
artifact, refuses a bare string, and copies fields rather than holding the
candidate. **Unverified is a type, not a flag** — there is no `trusted` attribute
to flip, and reaching that class *is* the authority transition.

The verifier checks association, not a multiset: Revenue 100 / Cost 80 becoming
Revenue 80 / Cost 100 keeps every number and every count, and is refused as
`REASSIGNED` — a separate reason from `CHANGED`, because the value exists
elsewhere in the source.

## 6 · Trust Boundary Matrix

| Boundary | Input authority | Transformation | Output status |
| --- | --- | --- | --- |
| Runtime → Context Adapter | RunState authorities | projection into a request | non-model |
| Adapter → Compiler | `ContextRequestV1` | deterministic selection | governed candidate context |
| Compiler → Pack | Disclosure Authority | governed projection | model-visible semantic context |
| Pack → Renderer | `AgentContextPackV1` | presentation | rendered model input |
| Renderer → Provider | `ModelRequestV1` | transport execution | untrusted model result |
| Provider → Harness | `ModelResponseV1` | normalized execution output | untrusted |
| Model output → Derived Artifact | the model response | candidate construction | explicitly unverified |
| Derived Artifact → Admission | authoritative-source comparison | deterministic verifier | admitted / rejected |
| Admitted artifact → Retrieval | verified representation | indexing | trusted derived representation |

## 7 · Authority / Non-Authority Matrix

| Question | Authority | Notes |
| --- | --- | --- |
| Evidence | `EvidenceBinding` / admitted evidence | the Binder decides; nothing here re-decides |
| Claim support | `ClaimProvenance` / `slot_bindings` | projected into the pack, never recomputed |
| Financial semantics | H2A-2B shared semantics | `canonical_decimal`, `quantity_identity`, `text_identity` |
| Context selection | `ContextCompilerV1` policy | **not** factual authority, only relevance and order |
| Context disclosure | Disclosure Authority | the single answer to "what may this model see" |
| Model-visible representation | `AgentContextPackV1` | one invocation, discarded with it |
| Model presentation | `ContextRendererV1` | layout and static instructions; reads no runtime |
| Execution | `ModelProviderV1` | transport only; no opinion on trustworthiness |
| Model output | *nothing* | `ModelResponseV1` is untrusted by construction |
| Derived-artifact trust | `ArtifactAdmissionResultV1` | and only through it |
| Source factual authority | the original source artifact | admission never transfers it to the model |

## 8 · Provider / Renderer Separation

A provider never receives an `AgentContextPackV1`. The reason is not tidiness: a
provider that could read the pack could reach semantic fields the role's renderer
chose not to expose, and "what may this model see" would have two answers — one
governed, one depending on what a provider felt like reading.

The separation is structural, not conventional:

* `rag_v2/invocation/contracts.py` and `provider.py` import nothing from
  `rag_v2.context` and nothing from `src`;
* `rag_v2/invocation/renderer.py` does import the pack, because a renderer's
  whole job is one;
* the provider spine is forbidden `rag_v2.context`, `rag_v2.adaptive`,
  `rag_v2.derived` and `src` by an executable guard.

`src/services` and `src/retrieval` are forbidden `rag_v2.invocation` entirely —
if ingestion cannot see a `ModelResponseV1`, it cannot write one into a chunk.
That is the F6 defect stated as an import rule.

## 9 · RunBudget / ContextBudget / Provider Config

Three separate concepts, and a test asserts the field sets do not overlap:

| | Bounds | Owner |
| --- | --- | --- |
| `AdaptiveRAGBudgetV1` (RunBudget) | how much **work** the runtime performs — replans, tool calls, retries | execution policy |
| `ContextBudgetV1` | how much **information** one invocation sees | the compiler |
| `ModelBindingV1` | which model, how to reach it, and its tokenizer | deployment configuration |

A retry counter in a context budget would be one knob changing both, which is how
a context decision becomes a latency decision.

**The tokenizer seam.** A configured token bound with an exact counter is
enforced; without one it is a construction failure. No word count, character
count or substitute tokenizer appears anywhere to make a missing counter go away.
B3 configures no bound and supplies no counter — the honest state, not an
omission.

**F1 is hereby downgraded**: legacy V1 implementation debt. A future
`ModelBindingV1` supplies the exact tokenizer for its provider; nothing in the
Harness guesses one.

## 10 · Support Topology Semantics

`pack.references.support_groups` and `canonical_slot` are **Harness semantic
metadata**, permanently. They are not evidence disclosure fields, they are not
citation handles, and the current B3 renderer does not present them.

The policy is frozen rather than the implementation: a future renderer may
consume the topology only through an explicit opt-in, and it may not remove it
from the pack merely because today's renderer ignores it. Two packs differing
*only* in topology are asserted to produce identical prompts, with deliberately
different slot keys so a leak fails rather than passes by accident.

## 11 · Deferred Legacy Boundaries

Not migrated, and not H2A-3 seal blockers. Each is a legacy boundary with its own
consumers, and migrating one is a modernization rather than a correctness fix:

| Boundary | Status |
| --- | --- |
| B5 query rewrite | legacy |
| B6 conversation resolver | legacy |
| V1 answer | legacy, and the source of F1 |

`ContextRoleV1` keeps exactly one member. A role invented for symmetry would be a
field list nobody validates — the mistake `EvidenceDisclosureProfile` has already
recorded avoiding once — and the framework has to support their arrival without
`AgentContextPackV1` becoming a universal RunState.

Also deferred: provider streaming, health, cancellation, a chat/message request
shape, and rendering the support topology. Each is a capability to add when a
real provider or renderer needs it, not in anticipation.

## 12 · Independent Regression Oracles

Two kinds of evidence, and neither substitutes for the other:

**Equivalence — a change detector.** The frozen B3 baselines. `BASELINE_V2` is
the migration target and every authored prompt is byte-identical through the
whole migrated path. `b3_legacy_context_baseline_v1` is kept as history so the
F10 defect stays reproducible.

**Correctness — independently authored expectations.** Derived from the inputs,
never from the code under test:

| Oracle | What it establishes |
| --- | --- |
| `test_specialist_truthfulness` | every rendered field is the authored value or a stated absence |
| `test_b3_support_topology` | the topology is projected, not inferred |
| `test_model_derived_admission` | the adversarial numeric-fidelity set |
| `test_f6_table_cleaning_admission` | the before/after on the live ingestion path |
| `test_model_invocation` | the provider boundary, without torch |
| `test_harness_end_to_end` | both boundaries, whole, with fakes |
| `test_harness_boundaries` | the negative contracts, as executable guards |
| `rag_v2 ⇏ src` | the layering the whole shadow architecture rests on |

A differential alone would pass if the baseline and the implementation were
regenerated from each other. That is why the second column exists, and why the
phase notes each say so.

## 13 · Known Deferred Debt

Carried forward deliberately. None of it blocks the seal; each has a stated home.

1. **Ingestion's model call is outside `ModelProviderV1`** — a direct HTTP call
   with its own auth and payload, so the derived artifact carries `model_id` and
   an empty `invocation_id`. Migrating it onto the provider framework is a
   separate question from admission.
2. **Provider capabilities are one method.** Streaming, health and cancellation
   await a provider that needs them; `ModelRequestV1` carries text only, and a
   chat-shaped provider wanting a message list is a widening, not a compatible
   change.
3. **`TrustedV2RuntimeResources.specialist` keeps its name.** The capability's
   parameter was renamed to `model_backend=` in H2A-3E because it accepts a
   provider *or* a legacy backend; the resources field really does hold the
   specialist, and renaming it would be a different change with a different
   meaning.
4. **Nothing renders the support topology.** The policy is frozen and the opt-in
   path is specified; no renderer exercises it, so it is untested in use.
5. **`is_usable_table_markdown` and the verifier's structural parse were not
   unified** — deliberately, since one guards extraction and the other guards a
   transformation.
6. **B5/B6/V1 remain legacy** (§11).
7. **F1** — legacy V1 implementation debt (§9).

## 14 · Final Exit Criteria

H2A-3 closes when all of the following hold, and each is asserted rather than
claimed:

1. Authoritative artifacts are not redefined by the compiler.
2. Selection is role-specific, deterministic and inspectable.
3. Disclosure Authority is the only disclosure authority.
4. One canonical claim with N supports has explicit topology in the pack.
5. Raw authority objects cannot enter a pack.
6. The renderer consumes the pack and cannot re-read the RunState.
7. `ContextBudget` and `RunBudget` are separate, and provider configuration is a
   third thing.
8. A token bound is enforced only with an exact provider-supplied counter.
9. Model-derived artifacts have an independent admission contract.
10. Production B3 reaches a model only through the compiler.
11. A provider cannot receive an `AgentContextPackV1`, a RunState or evidence.
12. The Harness runs end to end without a checkpoint, a network or a real model.
13. Readiness is unchanged and the full regression has no unexpected failures.

Sealed as `nf-v3-h2a3-harness-runtime`. The H2A-2 tag is not moved.

## 15 · What Comes Next

Provider integration — a financial model and a DeepSeek endpoint — is now a
matter of implementing `ModelProviderV1` and constructing a `ModelBindingV1`. No
evidence, binding, context, compilation, disclosure or admission logic changes
when they arrive, which is the property the whole phase was built to establish.

After that: H2A-4 evaluation and ablation on real models, then H2B's
model-guided replanner.
