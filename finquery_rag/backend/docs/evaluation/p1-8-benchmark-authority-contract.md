# BENCHMARK AUTHORITY CONTRACT

**Status: DEFINED, NOT APPLIED.** Nothing in Runtime, Store, Benchmark, Gate,
Binder, Validator, Retriever or Sidecar was modified to write this. Two
call-sites the contract would need were identified and are named in §8; neither
was touched.

Every factual claim below is a measurement. The measurements, their scripts and
their inputs are in `p1-8-b0-independent-verification.md`.

---

## 0. The question this answers

> When one filing reports several real values for the same entity / metric /
> period, what source-level evidence decides which value is **eligible** to be
> that question's gold?

Three words in that sentence carry the whole contract and each is defined below:
*real*, *source-level*, *eligible*.

---

## 1. Vocabulary, made precise

The contract is written against the store's actual coordinate function, not an
idealised one. `StructuredFactStore.coordinate_key` folds:

```
coordinate(fact) = (casefold(entity), casefold(metric), casefold(period))
```

**Line numbers cited in this document are the local checkout's**
(`E:\nanochat`, `feat/nf-v3-interview-final` @ `ca3446d`). The run host is a
separate repository with a different runtime and the same symbols sit at
different lines — `coordinate_key` is `trusted_v2_production.py:506` locally and
`:471` on the host. Locate by symbol, not by line.

Casefolding matters and is load-bearing: the store holds Coca-Cola's
consolidated statement row under metric `Operating Income` and the investee
summary row under `Operating income`. **Those are the same coordinate.**

### 1.1 Real

A value is **real** when it is a stated value of a real row of the filing named
by the question's `document_id`.

*Not real:* a cell of a row that is not a quantity (a caption, a year label, a
column header). `tv2f01-s1-tsla-034` binds `5` at metric `State` against the
alternatives `21` and `5` — the store emitted a fact per table cell, so prose
produced quantities. `2025` appears as a value at `(nvidia, colette m. kress,
FY2025)` for the same reason. These are **cell-value artefacts**, not competing
values, and they must be removed at emission, not adjudicated at retrieval.

*Real but not eligible:* see §3.

### 1.2 Source-level evidence

Evidence is **source-level** when it is carried by the filing being cited — an
XBRL context, a table caption, a dimension axis — and not by the store, the
retriever, the Binder, or the benchmark. Evidence is **derived** when this
project computed it.

The distinction decides what may be used. Two derived facts do not make
source-level evidence:

- *Store entity.* The investee row's `entity` is `The Coca-Cola Company`. It is
  derived — the note belongs to Coca-Cola's filing, so the extractor inherited
  the filer — and it is what makes `entity == question_company` useless here.
- *Store page/fragment.* `page 87` and `table:9a90ac6dbf97…` distinguish the two
  rows, but they are the extractor's own labels for where a value was printed,
  not the filing's statement of what the value is about. Using them to pick a
  winner is using the store's opinion to validate the store.

### 1.3 Eligible

A value is **eligible** to be a question's gold when the question's own
authority rule (§3) selects it, and the source says it is about the entity the
question names. *Selected* and *about the right entity* are separate conditions;
the sum-007 defect is a value that satisfies the second and not the first.

---

## 2. The authority rule

The single rule the whole contract rests on:

> **A question's coordinate denotes the company-level fact unless the question
> text names the scope.**

Company-level is not an editorial judgement about importance. It is the filing's
own declaration, and it is already present, machine-readable, in the corpus:

```
company-level fact  ==  an XBRL fact whose context carries NO dimensions
                        (dimension_count == 0)
```

Measured against the corpus (`financial-facts-ixbrl-v1.jsonl`), Coca-Cola
FY2025 `us-gaap:OperatingIncomeLoss`:

```
dimension_count = 0  value 13762   -> consolidated statement of income
dimension_count = 0  value 13762   -> (the statement is tagged in two places)
dimension_count = 1  value 13426   EquityMethodInvestmentNonconsolidatedInvesteeAxis
dimension_count = 1  value 15578   ConsolidationItemsAxis / MaterialReconcilingItems
dimension_count = 1  value  1816   ConsolidationItemsAxis / CorporateNonSegment
dimension_count = 2  value  4298   OperatingSegments × EuropeMiddleEastAfrica
...                  2 values × 5 segments
```

FY2024: `9992` at dimension_count 0; `12536` carrying the investee axis.

**This is the answer to §0.** The source-level evidence that decides eligibility
is the *absence of a dimension on the fact's XBRL context*. Coca-Cola's own
operating income is `13762` and `9992` because those are the only two facts
whose context asserts the consolidated entity and nothing narrower. The gold's
`13426` and `12536` are neither less real nor mis-tagged — they are the
investee aggregate, correctly dimensioned, and therefore **not eligible** for a
question that names no scope.

### 2.1 Why not the alternatives

Each of these was considered and rejected on evidence, not taste.

| candidate rule | why it fails |
|---|---|
| `entity == question company` | Measured blind. The investee row inherits the filer's entity, so the test returns 40 consistent / 0 mismatched *including the case it is meant to catch*. A test that cannot fail is not a test. |
| primary-statement caption match | B0's classifier. Accepted only captions matching `consolidated statements of …`, flagging 68/120 — its own false-positive rate. Apple's iPad revenue sits in a *Products and Services Performance* table and is entirely Apple's. Retracted. |
| table scale / filer-total comparison | Circular. The filer's total comes from the store, which is the thing that has already mis-attributed the note table. Also insufficient: legitimate tables state no total. Retracted. |
| arithmetic total-detection | Valid **only** for the narrow class where one row at a coordinate is the sum of the others (Microsoft FY2025 `Cost of revenue` 87,831 = 22,422 + 40,171 + 25,238, confirmed by the prior-year column). It answers "which row is a total", never "which table is this filer's". |
| `row_id` / `table_fragment_id` | Distinguishes the two rows and is therefore the tempting answer — and it is derived, not source-level (§1.2). It also cannot generalise: the fragment id is a hash of the table's content, so it carries no meaning a rule could read. |

---

## 3. The authority classes

Every question resolves through exactly one of four classes. The class is a
property of the **question**, and it is decided by the question text plus its
pinned plan — never by which values retrieval happened to surface.

### C — Company-level question

*Question names an entity and a metric and no scope.* Example:
`tv2f01-s2-sum-007` — "the sum of The Coca-Cola Company's Operating income
across FY2024 and FY2025."

```
authority       the filing's company-level fact for that concept and period
eligible set    { v : fact(v) has dimension_count == 0 }
tiebreak        CONCEPT_ALIGNMENT order, then period identity (§3.5)
if empty        refuse
if |distinct values| > 1 after tiebreak   -> §4 conflict, refuse
```

§2 is this class written out.

### S — Scoped disclosure question

*The question text names the scope.* The scope may be a segment, an investee
relationship, a geography, a product line, a table, or a note.

```
authority       the fact whose context carries the named scope
eligible set    { v : fact(v)'s dimensions include the named scope }
if empty        refuse  -- never fall back to the company-level value
```

**A scoped question must not be answered by the company-level rule.** If the
question asks for the equity-method investees' operating income, `13426` is
eligible and `13762` is not — the exact mirror of sum-007. The class is
determined by the text, so a change that makes C-cases release must not make
S-cases release the wrong row. This is the failure mode the current guard's
`CONFLICTING_VALUES` prevents by refusing both; any repair has to keep it.

### M — Multi-value coordinate

*The question is well-formed but its coordinate holds several distinct values
and no rule above selects among them.*

```
status UNIQUE                 exactly one distinct comparable value   -> eligible
status CONSENSUS_SAME_VALUE   several facts, one distinct value      -> eligible
status CONFLICTING_VALUES     several distinct comparable values     -> NOT eligible
```

This is `CoordinateStatus` (`src/finance/operand_ambiguity.py`), which already
exists and already takes the right shape. Two of its decisions are contract
decisions and are adopted here:

- **Comparison is restricted to comparable quantities.** A percentage and a
  currency amount at one coordinate are not competing answers; they are
  different quantities. Counting raw distinct values would refuse
  `compare-002`'s Visa coordinate (`1926` and `21%`) and kill a correct release.
- **A value that will not canonicalise is `CONFLICTING_VALUES`.** A value that
  cannot be read cannot be shown to be unique; defaulting it to admissible would
  make the rule fail open on exactly the malformed rows it exists to catch.

`CONSENSUS_SAME_VALUE` earns its place on measurement, not symmetry. Of the 41
observations where a gold fact's coordinate holds more than one distinct raw
value, **6 are representation variants of a single comparable quantity** and are
admissible today:

```
tv2f01-s2-growth-010  stock-based compensation  fy2024  ['$10,734', '10,734']
tv2f01-s2-sum-005     tangible book value/share fy2024  ['$ 97.30', '97.3']
tv2f01-s1-v-040       net revenue               fy2025  ['$ 40,000', '$40,000', '11%']
```

Collapsing these to `CONFLICTING_VALUES` would refuse a correct release. The
remaining 35 split `CONFLICTING_VALUES` 34 / `UNIQUE` 1 — the one being a
coordinate whose extra value is incomparable rather than competing, which is
the comparable-quantity restriction doing its job.

### A — SOURCE_AMBIGUOUS

*The source genuinely does not determine one value, and no scope can be named.*

```
verdict   the value is NOT eligible to be this question's gold
disposition   the case moves to the abstention side
```

The operative word is **no scope can be named**. A coordinate where the
discriminator exists but the question omits it is class C or S, not A.

**The corpus's class-A population has not been measured, and this contract does
not claim it is empty.** An earlier draft of this section asserted zero; that
claim could not be supported and was withdrawn. What *is* measured:

- The iXBRL layer carries **125 distinct axes and 1,083 distinct members**. The
  legacy store carries none, so no case can be sorted into S or A from the store
  alone.
- Two of the 26 irreducible metrics do have a plausible member in that set —
  `Other letters of credit` → `OtherLettersOfCreditMember` (22 facts) and
  `Commercial` → `CommercialPortfolioSegmentMember` (816) — but substring
  matching over an uncontrolled member list is exactly the guess this contract
  refuses to make, and **finding the member the filing actually intended is a
  source-reading task per case.**

Class A is defined because the contract must be able to say "the question does
not determine its answer" — the sum-007 correction's second defensible
direction — not because a case currently occupies it. **Assigning a case to A to
close a gap would be a benchmark edit, not an adjudication.** Classifying the 26
is the next step's work, not this document's.

### 3.5 Period identity inside a filing

One filing reports the same metric and concept at FY2023, FY2024 and FY2025.
`period` is part of the coordinate, so these are three coordinates and do not
compete. They *do* compete when two facts carry the same period label —
measured (`resolve_canonical`): no such pair exists in this corpus at a
company-level coordinate, and the rule if one appears is period identity by
`(period_start, period_end, entry_date)` and then refusal.

---

## 4. Conflict resolution order

Applied in order; the first that selects wins. Anything that reaches the end is
refused — never guessed.

```
1. question text    names a scope?              -> class S
2. concept alignment  metric maps to a canonical quantity with an undimensioned
                      company-level fact        -> class C, resolved
3. arithmetic         one value at the coordinate is the sum of the others, and
                      the source states the total -> that value is the total
4. nothing above      distinct comparable values remain -> CONFLICTING_VALUES,
                      refuse; the case is recorded as needing source
                      adjudication, not as a system failure
```

Step 3 is the one rule that can select a winner without a dimension, and it is
admissible because it is **source-level**: the filing states the total row and
the components, and the relation holds in the source, not in the store. It is
also the only rule in this contract that a runtime can execute without the
iXBRL layer.

---

## 5. Logical-table authority

**Definition.** A **logical table** is one table element of the source filing —
one `<table>` in `primary.html`, identified by the filing, its document id, and
its position. A **physical row** is one `<tr>`. A **fact** is one quantity
asserted by one physical row.

The store holds one record per *presented value*, so a coordinate can receive
rows from several logical tables. Measured at
`(The Coca-Cola Company, operating income, FY2025)`: two rows, `table:4b97983e…`
(the consolidated statement) and `table:9a90ac6d…` (the investee summary), with
different `row_id`s and different values.

**The authority rule.**

```
logical-table authority
    A logical table is the authority for a value when the filing's own
    labelling of that table says it is about the entity, scope and period the
    question names.

    For class C the authority is the table the filing presents as the
    consolidated statement of the filer.
    For class S it is the table the question's named scope identifies.

    The store does not record this labelling. logical-table authority is
    therefore NOT ESTABLISHED IN THE LEGACY STORE, and a row's membership of a
    table is not, by itself, evidence that the row is about the filer.
```

**Consequence, stated plainly.** Any rule that would pick the consolidated row
because it lives in table `4b97983e` is circular: the store assigned that
fragment, and the store is what has to be checked. The non-circular source of
the same information is the XBRL context (§2) — which the legacy store does not
carry and the iXBRL store does.

**What would establish it, if it is ever pursued.** The minimum field set from
P1.7's audit, kept here unchanged as a *design* note and not as a migration
plan:

```
statement_or_table_scope   which statement or table the row sits in
row_role                   total | segment | member, where the source says so
```

Both are properties of the source carried forward at extraction. Neither can be
reconstructed from a hash of the table's content, which is what
`table_fragment_id` is.

---

## 6. What this contract decides about sum-007

```
question    the sum of The Coca-Cola Company's Operating income
            across FY2024 and FY2025
class       C  -- entity and metric named, no scope named
authority   the filing's company-level OperatingIncomeLoss

FY2025   dimension_count 0  ->  13,762
FY2024   dimension_count 0  ->   9,992
                                ------
                       sum  ->  23,754

current gold  13,426 + 12,536 = 25,962
              drawn from the equity-method investee summary, dimensioned with
              EquityMethodInvestmentNonconsolidatedInvesteeAxis
```

**The gold is not eligible under this contract.** 25,962 is real, correctly
tagged, and about Coca-Cola's investees; it answers a question the benchmark did
not ask. §6 of `p1-7-0a-oracle-attribution.json` proposed two directions —
re-point the gold at 23,754, or mark the case ambiguous and move it to
abstention. This contract selects the first **on the merits**: the question names
no scope, the source determines a unique company-level answer, and therefore the
case is answerable and the benchmark's number is the thing that is wrong. The
second direction is what class A is for, and this case is not class A, because a
unique eligible value exists.

**This contract does not apply that correction.** Changing the gold changes the
pinned hash `3d2a0c5b` in `benchmark-version.json`; that is a benchmark version
bump and is taken deliberately, as a separate step.

---

## 7. The measured consequence

Contract applied to all 120 cases, against the **frozen legacy store**, using
the existing `CoordinateStatus` guard as the mechanical proxy for §4:

```
CLEAN                     48    no coordinate conflict; eligible set is
                                already a single value
GUARD_REFUSES             27    a coordinate holds conflicting comparable values
ABSTENTION (no facts)     25
s3 cross-entity stratum   20    names iXBRL keys; measured separately
```

Of the **34 conflicting observations** inside the 27 refused cases:

```
resolved by canonical quantity (class C, §2)        2   both in tv2f01-s2-sum-007
no canonical quantity covers the metric            32
```

The 32 irreducible observations span 26 cases and carry metrics such as
`Deferred`, `Services`, `State`, `Colette M. Kress`, `Intersegment`,
`U.S. GSEs and government agencies`, `Cost of revenue`, `Total nominal payments
viewed`, `Foreign currency contracts`. **These metrics are absent from `METRIC_TO_CANONICAL` by the module's
own deliberate design** — its docstring says names like `Total`, `Current` and
`United States` "are the strings whose ambiguity this work exists to remove, and
mapping them to something would be a guess."

**That is the binding constraint on the coverage target, and it is a contract
consequence rather than a system failure.** Under this contract those 26 cases
are refused until the authority for their metric is defined. §4 step 3
(arithmetic) can select winners for some; the rest need a canonical quantity or
a scoped question, and either of those is a decision about the benchmark, not
about the retriever.

**Be careful how that is read.** It does *not* establish that those 26 cases are
unanswerable. Almost all of them name something that looks like a scope —
`Intersegment`, `Commercial`, `Hedge accounting`, `Stock-based compensation`,
`U.S. GSEs and government agencies` — which would make them class **S**, and
class S asks for a dimension the legacy store does not carry. Whether that
dimension exists in the filing (it likely does: the iXBRL layer holds **125 axes
and 1,083 members**) and which member the filing intended is **unmeasured**;
establishing it means reading the source per case. Class A is the residue where
no scope can be named at all, and how much of the 26 lands there is open.

**What is established is the shape of the constraint.** These 26 are refused
because their authority is undefined, not because the value could not be found.
Closing them requires adding an authority per metric — a canonical quantity or a
scope binding — and every such addition is a statement about what a question
means. That makes it a benchmark decision, not a retrieval improvement.

### 7.1 The frozen baseline, against which any change must be measured

The handoff's §1 figure is the pre-guard baseline:

```
                 handoff §1 (pre-guard)      goal statement
Release Coverage      50/95  = 52.6%          17.9%
Released Accuracy     48/50  = 96.0%         100%
Incorrect Release     0                      0
Correct Refusal       25/25 = 100%           100%
```

**These two are not the same measurement.** 52.6% is what the runtime released
*before* the operand guard; 17.9% is what a guard-ON runtime produces. The
27 refused cases above are the difference. Any coverage target must state which
runtime it is measured on, or it is measuring a guard toggle rather than
progress. This contract treats **17.9% (guard ON) as the baseline to beat**,
because the guard's refusals are the honest current answer, and because
`ffc8d834` — the commit that makes the guard ask the fact's coordinate rather
than the slot's — was authored *after* the handoff's seal.

### 7.2 The s3 stratum is not covered by this contract's measurement

The 20 cross-entity cases name iXBRL keys and resolve through a different path.
Measured: two of them collide **at the iXBRL company level** —

```
tv2f01-s3-compare-001  (Coca-Cola, net income, 2025)  {13107, 13137}
tv2f01-s3-rank-001     (Tesla,     net income, 2025)  {3794, 3855}
```

`13107` and `13137` are net income attributable to shareowners and consolidated
net income. Both are company-level, both are undimensioned, and they are
*different quantities with the same canonical quantity*. This is the case §3.5
and `CONCEPT_ALIGNMENT` ordering exist for, and it is the strongest evidence
that the company-level rule alone is necessary but not sufficient.

---

## 8. Where the contract would attach — named, not applied

Nothing was changed. These are the two call-sites the contract needs, recorded
so the next step does not have to rediscover them.

```
1. src/runtime/trusted_v2_canonical_store.py:114  resolve_canonical()
   Already implements §2 for the canonical quantities it covers: returns the
   single undimensioned value, and returns [] rather than guessing when the
   coordinate holds several. It is correct as written; what is missing is
   reach, not behaviour -- METRIC_TO_CANONICAL covers 12 quantities, and the
   26 irreducible cases are outside them.

2. src/finance/operand_ambiguity.py  coordinate_status()
   Already implements §3's M class, including comparable-quantity restriction
   and the fail-closed treatment of an uncanonicalisable value. Its
   docstring already states the contract's honest position: "This blocks
   rather than repairs... Until then the honest answer is to refuse."

3. src/runtime/trusted_v2_production.py:506  coordinate_key()
   The definition §1 is written against. Note that it is casefolded, which is
   why `Operating Income` and `Operating income` are one coordinate.
```

---

## 9. What this contract does NOT decide

- **How to make the iXBRL layer reachable from the runtime path.** §2 says what
  the authority is; it does not say how to plumb it. That is a design step.
- **Whether to add quantities to `METRIC_TO_CANONICAL`.** The 26 irreducible
  cases need an authority per metric. Deciding that `Deferred` means
  `us-gaap:DeferredRevenueCurrent` is exactly the guess the module refuses to
  make, and the contract does not relax that.
- **The gold correction itself.** §6 selects a direction; applying it changes a
  pinned hash and is a benchmark version bump.
- **Any emission-side fix for cell-value artefacts** (§1.1). Real, out of scope
  here, and it interacts with §7.1's baseline.
- **The Structural Context Sidecar.** Explicitly not started.
