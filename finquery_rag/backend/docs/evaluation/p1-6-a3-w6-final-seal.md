# P1.6-A3-W6 — final enablement, end-to-end verification, and the seal

`SEALED`. W6 adds no behaviour: it proves the authority migration is complete, verifies the
assembled system against frozen expectations, and freezes the result.

```
W1  contract                    DONE
W2  PeriodBinding producer      DONE
W3  TemporalKind producer       DONE
W4  admission + migration       DONE
W5  provenance persistence      DONE
W6  final enable + verify + seal  DONE
```

## W6-A — the A3 path is the only authoritative one

A migration fails quietly in one specific way: the new path is nominally authoritative
while some fallback still calls the old selector, and the outcome looks right because the
old rule happens to agree. Running the pipeline cannot tell those apart, so W6-A does not
rely on the outcome.

```
modules the build pulls in from this repository        952
  ...that DEFINE a legacy selector                         2   (allowed)
  ...that READ a legacy selector                           0
with both legacy selectors landmined: build succeeds, 2489 records byte-identical
with decide_emission_admission replaced by a raiser: build FAILS
```

The last two are a pair on purpose. Either alone is satisfied by a degenerate pipeline —
landmines never trip on a path that emits nothing, and a raiser "works" on a path that
calls nothing. Together they say the A3 decision is load-bearing and the legacy one is not
on the path.

The two modules that still *define* the constants are `typed_evidence_emitters.py`, which
keeps them deliberately: a rule under measurement has to stay stated, and the shadow
accounting measures the migration against the rule it replaced. Defining is not reading,
so the two are counted apart.

The modules are read from `sys.modules` rather than from a static import walk, because the
parse module is loaded by `importlib` and a walk would not see it. The first version of
this check took its snapshot *after* importing the emitter to patch it, which hid the one
module most worth looking at and reported a clean `0 readers` that meant nothing. It now
reads 952 modules.

```
the legacy whitelist admits          point, duration, comparison
the A3 routing rule withholds        segment, bucket, category, non_temporal
moved ineligible -> eligible         unknown            (over the legacy domain)
                                     UNKNOWN, YEAR      (over the whole enum)
```

One kind wide, and only one. `YEAR` appears only in the enum reading: the legacy axis never
emits it, so it cannot occur in a store, and reporting both numbers keeps that from looking
like an omission.

## W6-B — the system, end to end

**The nine tables.** The nine verified primary statements that produced no store records at
A3-0, each with a stated outcome:

```
jpm_fy2025#63193   records 120   RECOVERED
ko_fy2025#11388    records 134   RECOVERED
ko_fy2025#12533    records 162   RECOVERED
pfe_fy2024#24395   records  58   RECOVERED
v_fy2025#9951      records 146   RECOVERED

jpm_fy2025#64546   records   0   WITHHELD_ON_LEGACY_KIND
ko_fy2025#11899    records   0   WITHHELD_ON_LEGACY_KIND
nvda_fy2025#8766   records   0   WITHHELD_ON_LEGACY_KIND
tsla_fy2025#10407  records   0   WITHHELD_ON_LEGACY_KIND

tables with no outcome: 0
```

Five recovered. **The four that did not are not a scope decision, and calling them
`VALID_WITHHELD_OUT_OF_SCOPE_GEOMETRY` would have been a false label** — see below.

**Store final state**, frozen rather than predicted:

```
records 26311
added    PARTIAL / YEAR  3368      RESOLVED / DAY  3018
removed  LEGACY_FALSE_POSITIVE 691   SOURCE_AMBIGUOUS 4188
         VALID_OUT_OF_SCOPE_GEOMETRY 2173
UNCLASSIFIED_ADDED 0        UNCLASSIFIED_REMOVED 0
```

The three removal classes stay apart and are never summed. Only the first is a correction.

**Provenance:**

```
period identity + binding provenance complete   26311 / 26311
temporal provenance where kind != UNKNOWN       21303 / 21303
source cells resolvable by re-parsing            77883 / 77883
```

Resolvability is checked by re-parsing each filing and confirming the table exists and the
cell is inside its grid — not by testing the fields are non-empty.

**The 47 slots**, recomputed rather than read from a summary:

```
RESOLVED 37    NO_COMPANY_LEVEL_FACT 5    NO_FACT 5    anything else 0

existing resolved regression 0      resolved value changed 0
wrong_scope 0   wrong_metric 0   value_mismatch 0
dangerous_authority 0   scaffold_authority 0
```

**The 47 slots are a safety regression gate, not a capability-gain proof.** W4 moved 6,386
facts in and 7,052 out and this benchmark did not move at all; it is the instrument that
would have caught the migration breaking something, and it cannot show the migration
helping. Nothing here should be read as evidence that the gains were worth having.

## What W6 found: `rating` inside `operating`

The four still-empty tables are **cash flow statements** — JPMorganChase's, Coca-Cola's,
NVIDIA's (`section_type: CASH_FLOW`) and Tesla's — with columns reading
`Year ended December 31, / 2025 / 2024 / 2023`. Fiscal years are not buckets. The legacy
classifier calls them buckets anyway:

```
_BUCKET_RE = ... | range | rating | grade | tier
```

`rating` has no word boundaries, so it matches inside `ope·rating· activities`:

```
'Cash flows from operating activities:'  -> matches 'rating'
'Operating Activities'                   -> matches 'rating'
'Weighted average rating'                -> matches 'rating'
```

Every cash-flow section labelled "Operating activities" is therefore classified `bucket`,
which routes to `DISAGGREGATION_AXIS`, which withholds the entire table. All 835 withheld
cells across the four tables are withheld for this one reason, and every one of them sits
in a column that states a year.

**This is the residual cost of keeping routing a kind question.** W4 replaced the legacy
*whitelist* with a narrower negative rule, deliberately — a segment column really is a
breakdown rather than an incomplete period fact. But routing still reads the legacy kind,
so a mislabelled kind still decides. The A3 line made admission stop depending on the kind
in the ways that were wrong; this is one way that is still wrong.

It is **explained**, so it is not `UNEXPLAINED_ZERO_RECORD`. It is also **not a scope
decision**, and the verifier refuses to label it as one: it resolves a
`DISAGGREGATION_AXIS` refusal against the column's own text, and when every withheld cell
sits in a column that states a year it reports `WITHHELD_ON_LEGACY_KIND` and prints the
mechanism. Fixing the regex is a behaviour change — it would admit roughly 835 cells across
four primary statements — so it was kept out of W6 by the same rule that kept out SECTION,
the resolver, and the rest.

## W6-C — the seal

`p1.6-a3-fact-emission-seal`, at commit `27a805b`. The seal **recomputes** rather than reads:
the scalars come from the two stores and the audit, not from `store-migration-accounting.json`
which is itself a summary, and the nine-table funnel and the reachability result are
obtained by re-running both verifiers in fresh processes into a temporary directory, so the
seal checks the verification rather than a stale report of it.

```
store sha256        6ee7427f073ff04c
table-roles sha256  ca29cd40e95d0943
benchmark sha256    4260230885daac37
commit              27a805b
modules hashed      14  (the pipeline, the resolver, the benchmark, two test modules)
```

The first run of the seal returned `NOT SEALED` — one informational key was inside the
compared set — which is the seal demonstrating that the comparison reaches the verdict
rather than being decoration.

It does not supersede the table-authority seal: that one seals *which table may speak for
the company*, this one seals *whether a valued cell becomes a fact*. They are independent.

## Post-seal debt

Not to be started inside W6, and each is a behaviour change with its own delta:

* **`_BUCKET_RE`'s `rating` word boundary** — four primary statements, ~835 cells, above.
* **SECTION scope / SECTION_BRACKET** for row-major equity statements (2,173 withheld).
* **The resolver** does not consume `YEAR(2025)` or granularity; 3,368 facts wait.
* **Row-label numeric extraction** (`Balance at January 1` → `1`).
* **The duplicate `normalized_period` key** in `trusted_v2_canonical_fact_store.py`.
* **SOURCE_AMBIGUOUS** — 4,188 facts withheld because the source does not settle them.
* **`REAL_BUCKET_POSITIVE_UNOBSERVED`** — no corpus case exercises a real bucket-positive.

P1.6-A3 ends here. Capability work continues in a new phase, against this baseline rather
than by moving it.
