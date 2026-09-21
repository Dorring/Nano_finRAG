# NF-V3 H2A-3C — Model Invocation / Provider Boundary Consolidation

Status: **CLOSED.** Predecessors: `nf-v3-h2a-3-context-surface-audit.md`,
`nf-v3-h2a-3b0-context-preconditions.md`, `nf-v3-h2a-3b1-context-compiler.md`,
`nf-v3-h2a-3b2-context-semantics.md`, `nf-v3-h2a-3b3-compiler-migration.md`.
Baseline: sealed H2A-2 at `9497943`; this phase starts from `811f85f`.

H2A-3B3 proved the Context Runtime could take over the production B3 specialist
path. H2A-3C makes that path **provider-neutral**: the model boundary is now a
contract rather than a signature that happened to emerge from one boundary's
needs.

    AgentContextPackV1
      -> ContextRendererV1        (a role decides what the model is shown)
      -> ModelRequestV1           (text, identity, configuration)
      -> ModelProviderV1          (transport, serialization, error normalization)
      -> ModelResponseV1          (what came back)

No financial model, no DeepSeek, no F1, no F6, no H2A-4.

---

## 1 · Where the contracts live

`rag_v2/invocation/` — a new package beside `rag_v2/context/`, not inside it.

| Module | Holds |
| --- | --- |
| `contracts.py` | `ModelRequestV1`, `ModelResponseV1`, `ProviderFailureKind`, `ModelProviderError`, `InvocationIntegrityError` |
| `provider.py` | `ModelProviderV1`, `LegacyPromptProviderAdapterV1` |
| `renderer.py` | `ContextRendererV1`, `CallableContextRendererV1` |
| `binding.py` | `ModelBindingV1` |
| `runtime.py` | `ModelInvocationRuntimeV1` |

**`contracts.py` and `provider.py` import nothing from `rag_v2.context` and
nothing from `src`.** `renderer.py` does import the pack, because a renderer's
whole job is one. The split is the boundary: a provider author who wants a
semantic field has to get it through the renderer, which is the only route
Disclosure Authority governs. That is asserted on `provider.py`'s syntax tree,
not promised in a comment.

## 2 · The exact contracts

```python
ModelRequestV1
├── invocation_id: str          # from the pack
├── role: str                   # the ContextRoleV1 value
├── prompt: str                 # what the renderer produced
├── provider_id: str
├── model_id: str | None = None
└── generation: Mapping[str, Any] = {}   # carried, unpopulated (see §6)
```

```python
ModelResponseV1
├── text: str                   # may be empty; the boundary judges it
├── finish_status: str = "unknown"
├── citations: tuple[str, ...] = ()   # handles from the request, deduplicated
└── usage: Mapping[str, int] | None = None
```

```python
ModelBindingV1
├── provider: ModelProviderV1
├── renderer: ContextRendererV1
├── provider_id: str
├── model_id: str | None = None
└── exact_token_counter: ExactTokenCounterV1 | None = None
```

All three are asserted as **exact field sets** by test, so a fourth field in any
of them is a deliberate edit. The request's field set is the boundary, which is
why it is checked that way: a request that grew an `evidence` or `metadata`
field would still pass every behavioural test in the suite and would have handed
providers a route around the renderer.

`citations` is the one field beyond the obvious, and it is there because the
existing Harness needs it — a grounded generator declares which handles it used,
the capability filters that against what the pack issued, and the release path
carries the survivors. Those are *handles from the request*, so they are
provider-neutral in the sense that matters: the same vocabulary the renderer
wrote and the validator resolves.

## 3 · The final B3 call graph

```
AdaptiveRAGStateV1
  └─ specialist_context_request(state)                    rag_v2/context/specialist
       ├─ admitted_specialist_evidence(state)             (the one selection)
       ├─ state.bound_slot_bindings                       (the claim/support authority)
       └─ state._calculation_result_obj
                ↓  ContextRequestV1
       ContextCompilerV1.compile                          rag_v2/context/compiler
                ↓  AgentContextPackV1
       ModelInvocationRuntimeV1.invoke(pack)              rag_v2/invocation/runtime
            ├─ binding.renderer.render(pack) -> str       src/generation/specialist_prompt
            └─ binding.provider.invoke(request)           LegacyPromptProviderAdapterV1
                                                            └─ LocalSpecialistGenerator
                ↓  ModelResponseV1
       TrustedV2GenerationCapability reads .text, .citations
```

`model_id` is `None` and `generation` is empty for this binding: the local
specialist is a checkpoint with no registry name, and it owns its sampling
configuration in its own constructor. Both are stated rather than filled in.

## 4 · The compatibility adapter

`LegacyPromptProviderAdapterV1` wraps a `generate(prompt)` backend as a provider.
Eleven backends already satisfy that contract — two production, nine in the
suite — and requiring each to implement a new interface would have been churn
that proved nothing while leaving the boundary described by eleven signatures
instead of one.

It is a **migration adapter**, not a second source of truth. It does not render,
select, project or repair. It calls the backend with `request.prompt` and reads
what came back into a `ModelResponseV1`.

**`LocalSpecialistGenerationAdapter` is deleted.** It forwarded `generate(prompt)`
to a backend and counted calls — exactly what the new adapter does, plus the
error and response normalization that is a provider's job. Leaving it would have
meant two classes doing one thing.

One behaviour is deliberately dropped. The legacy boundary merged a
provider-declared `metadata` mapping into the candidate's `generation_metadata`,
so whatever a backend chose to put in that bag became part of the runtime's
record of the run. A provider's private bag is not runtime truth — that is the
rule this contract exists to state — so it is dropped. No production backend ever
populated it. In exchange, three fields the legacy boundary *discarded* now
reach the response: `finish_reason` becomes `finish_status`, and
`tokens_generated` / `rendered_input_length` become `usage`.

**How a backend is chosen as a provider is one `isinstance`.** `specialist=`
accepts either; an object implementing `ModelProviderV1` is used as itself, and
anything else is adapted. A future financial or DeepSeek binding arrives through
the same parameter, which is what "the Harness core does not move" has to mean
in practice.

## 5 · Renderer and provider are separate responsibilities

`ContextRendererV1` is a method protocol, `render(pack) -> str`.
`CallableContextRendererV1` adapts the function the repository already has, so
`render_specialist_prompt` becomes the first renderer without becoming a class
and without any call site changing.

What a renderer may decide: layout, static instructions, how disclosed evidence
is represented, whether an explicitly permitted support topology is rendered.
What it may not: read a RunState, retrieve raw evidence, invoke Disclosure
Authority again, reconstruct a slot binding, canonicalise a financial value, or
repair missing context.

**The separation is proved rather than described.** A provider is handed a
`ModelRequestV1` and nothing else, asserted over the object's own attributes for
any provider implementation — not only for the ones written in the test file.

## 6 · What is carried and not populated

`ModelRequestV1.generation` is empty, and that is the honest state rather than an
omission. The current provider owns its sampling configuration in its
constructor; there is no per-invocation generation configuration anywhere in the
runtime today. The field marks where a binding will supply one, in the same way
`ContextBudgetV1.reserved_output_tokens` is carried and not enforced — and
inventing values for it would be worse than leaving it empty, because a sampling
setting nobody chose is indistinguishable at the boundary from one somebody did.

## 7 · The tokenizer seam

`ModelBindingV1.exact_token_counter` is optional, and the rule it participates in
is unchanged from H2A-3B1:

```
configured token bound + exact counter available -> enforce
configured token bound + no exact counter        -> configuration failure
```

No word count, character count or substitute tokenizer appears anywhere to make
a missing counter go away. The capability builds its compiler with the binding's
counter, so a provider that can count its own tokens supplies one and the rule
starts applying to it; B3 supplies none, configures no bound, and enforces
nothing. F1 is not solved here and is not a blocker — a tokenizer is a provider
capability.

`ContextBudgetV1`, `RunBudget` and provider configuration remain three separate
concepts. A test asserts the binding carries no retry, timeout, tool, replan,
token bound or temperature field.

## 8 · `_bound_items` → `_routing_evidence_items`

Renamed, because after H2A-3B3 it no longer selects: it delegates to the
adapter's `admitted_specialist_evidence`, and it exists only so the routing
policy can decide whether a generator is needed at all — before any context
exists, and reading fields the SPECIALIST profile does not admit.

There is a second, concrete reason the old name had to go:
`src/runtime/trusted_v2_validation.py` has a `_bound_items` of its own. Two
functions of one name in two layers doing different things is how a future
contributor picks the wrong one.

Three things are asserted, and the third is the one that matters: the name
exists; the compiler's selection never reaches it (checked on `rag_v2`'s *code*,
not its prose); and it has exactly one consumer in `src/`.

## 9 · Trace counts are an observability projection

The compile, render, provider and support-group counts have no runtime consumer,
and this phase classifies them rather than inventing one. The render and provider
counts now come from the `ModelInvocationRuntimeV1` that performs the calls
rather than from counters kept beside it, so they cannot drift from what
happened.

Pinned two ways: `trace_snapshot()` rebuilds a fresh dict each call, so a
consumer holding one cannot mutate the runtime; and `generate` never reads it —
asserted by walking the function's syntax tree, so a comment naming
`trace_snapshot` cannot satisfy the check.

## 10 · Error normalization

Three kinds, and no cloud-provider taxonomy:

| Kind | When |
| --- | --- |
| `UNAVAILABLE` | the provider could not be reached or could not run |
| `TIMEOUT` | a bound elapsed — kept apart because a caller may reasonably retry one and not the other, and RunBudget owns whether it does |
| `INVALID_RESPONSE` | the provider answered, and what it answered could not be read as a model output at all |

`INVALID_RESPONSE` is deliberately **not** an empty answer. Empty text is a real
model output; classifying it in the transport layer would move a grounding
decision to the one component that does not know what the boundary asked for. An
empty answer still raises `financial_specialist_empty_candidate` where it always
did.

A provider that implements the protocol normalizes its own failures — that is
what §2 says a provider owns — and the Harness does not re-wrap a kind it chose.
A legacy backend's failures are normalized for it by the adapter, with the
original chained rather than flattened. Release behaviour is unchanged: every
normalized failure still propagates and still fails closed.

## 11 · Differential, readiness, regression

* **B3 prompt byte-identical** to `BASELINE_V2` on all five authored scenarios,
  including SHA256 — verified through `observe()` and through the new seam
  directly.
* **Readiness unchanged:** `19 / 0 / 3 / 0 / 0 / 0`, `token_counts_available:
  False`.
* **Full suite: 4434 passed, 143 skipped, 0 failed** (+47 over H2A-3B3's 4387).
* **Five injections, all caught**, sources restored byte-identically: widening
  `ModelRequestV1` with an evidence field (twice — the first attempt was caught
  by Python's dataclass rule rather than by the test, which is not the same
  thing and was redone); rendering twice per invocation; letting the provider
  spine import the pack; handing the pack itself to the provider.
* **Layering green.** The new package is inside `rag_v2`, so the existing
  `rag_v2 ⇏ src` guard covers it; a second, narrower guard covers the inversion
  that is actually plausible here — the provider spine reaching for the pack.

## 12 · The runnable proof that the checkpoint is no longer on the critical path

Every test in `tests/harness/test_model_invocation.py` runs without torch and
without a checkpoint. That is the phase's most practically useful result: the
fixed Linux checkpoint used to be required to exercise anything near the model,
and the Harness/provider boundary can now be tested on any checkout.

## 13 · Decisions flagged for review

1. **`ModelResponseV1.citations` is beyond the brief's minimum list.** It is
   there because the existing Harness needs it (§2), and the alternative was
   either an untyped metadata bag — which the same brief forbids — or changing
   release citation behaviour, which this phase must not do.
2. **A provider-declared `metadata` bag no longer reaches
   `generation_metadata`.** No production backend populated it; the change
   removes a path by which arbitrary provider data became runtime truth.
3. **`specialist=` kept its name** on `TrustedV2GenerationCapability`. It now
   accepts a provider or a legacy backend, so the name is narrower than what it
   takes. Renaming it touches 21 wiring points and nothing in the exit gate asks
   for it; recorded as debt in §14 rather than done silently.
4. **`LocalSpecialistGenerationAdapter` was deleted** rather than left as a
   second, redundant adapter. This is the 3B3 §8 rule applied to 3C's own
   compatibility surface.

## 14 · Remaining Harness debt before H2A-3D

> **Consolidated by H2A-3E.** The live list is §13 of
> `nf-v3-h2a-3-harness-runtime.md`. Item 4's `generation` field was audited (no
> producer, no consumer, one shape test) and **deleted**; item 2's `specialist=`
> was renamed to `model_backend=`. Item 6 was closed by H2A-3D. Items 1, 3, 5
> and 7 carry forward as stated debt rather than as blockers.

1. **The provider's interface is one method.** `invoke` is enough for a
   synchronous local model and for the deterministic doubles; streaming, health
   and cancellation are not specified, and a remote provider will need at least
   streaming. They should be added with the requirement that justifies them,
   not in anticipation.
2. **`specialist=` should be renamed** (`provider=` or `model=`), together with
   the router-helper naming question if the router's own interface is ever
   settled.
3. **`ModelRequestV1` carries text only.** A chat-shaped provider that wants a
   message list rather than a rendered prompt has no field for it. That is a
   deliberate widening to make when a provider needs it, not before — but it is
   a widening, not a compatible change.
4. **`generation` has no producer.** Nothing populates per-invocation sampling
   configuration; the binding's provider owns it in its constructor.
5. **`canonical_slot` / support topology are still rendered by nobody.** The
   policy is now frozen (semantic reference metadata, never an evidence
   disclosure field, opt-in for a future renderer), but no renderer exercises
   the opt-in, so the path is specified and untested in use.
6. **F6 remains the one real Harness trust debt.** Model-produced artifacts have
   no admission contract: a model's cleaned table can become trusted-looking
   context without a deterministic verifier in between. That is H2A-3D.

   > **Closed by H2A-3D.** `rag_v2/derived/` adds `ModelDerivedArtifactV1`,
   > `ArtifactAdmissionResultV1` and an `AdmittedDerivedArtifactV1` that cannot be
   > built without a successful admission, plus a deterministic numeric-fidelity
   > verifier whose oracle is the authoritative pre-model table. The ingestion
   > path no longer writes raw model output anywhere. See
   > `nf-v3-h2a-3d-model-derived-admission.md`. Items 1–5 and 7 carry forward.
7. **B5/B6/V1 still have no role.** `ContextRoleV1` keeps one member until a
   boundary is actually migrated, and the framework has to support that
   expansion without `AgentContextPackV1` becoming a universal RunState.

## 15 · What H2A-3C did not do

* no FinancialModelProvider, no DeepSeekProvider
* no F1 legacy tokenizer correction; no token bound is configured or approximated
* no F6 model-derived artifact admission
* no V1, query-rewrite or conversation-resolver migration
* no change to `ContextBudgetV1` or `RunBudget`
* no new seal; H2A-2 remains at `9497943`
* no H2A-4 and no token-efficiency number of any kind
