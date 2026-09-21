# P1.8-B0 — independent verification

Read against `p1-7-final-handoff.md` as the sole P1.7 state. Nothing here is
inherited from the P1.7 session's transcript, and every retracted item in the
handoff's §4 was treated as unavailable. **No file under `src/`, no store, no
benchmark file, no fixture was modified.**

---

## 1. Repository identity

```
handoff §Seals   repo HEAD  beae6c76749460dfc247f59c7bb7f559d2809ade
                 branch     feat/nf-v3-interview-final

local, this run  HEAD       ca3446d4bef8f50a0f32c1d8506da13d09296676
                 branch     feat/nf-v3-interview-final
```

**Not a discrepancy.** `ca3446d` is exactly one commit above the seal and its
diff is the handoff document itself plus one probe script:

```
$ git diff --stat beae6c7 ca3446d
 .../docs/evaluation/p1-7-final-handoff.md     | 233 +++++++++++++++++++++
 .../scripts/evaluation/probe_entity_reachability.py | 100 +++++++++
```

The two commits are `ca3446d` (the handoff) and `18b9ce9` (the risk-set
measurement). Neither touches the frozen runtime or the frozen benchmark.

---

## 2. Seals, recomputed

```
                                 handoff §Seals                recomputed              match
gold-evidence-v1.jsonl           3d2a0c5b7839656ce1414923ab…   3d2a0c5b7839656ce1414923ab…   yes
canonical-eval-v1.jsonl          227f0341d94ab8d4b9e7ee033f…   227f0341d94ab8d4b9e7ee033f…   yes
plan-fixtures-v8.jsonl           7463b45a5bfc309d0a5ab29cdb…   7463b45a5bfc309d0a5ab29cdb…   yes
runtime fact store               0382c6a17f7065516c0edd0d2c…   0382c6a17f7065516c0edd0d2c…   yes
```

The store hash was recomputed on the run host against the path the handoff
names, `/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl`.

The three benchmark hashes also match `benchmark-version.json` independently —
which is the authority per the standing note that the file names do **not**
imply their content:

```
migrations[0].post  (p1.6-0h)  == the three recomputed hashes
migrations[0].pre              9c8ca9ce… / 7831eef2… / 43f740f7… (plan-fixtures-v7)
```

`dataset-manifest.json` self-identifies as superseded and its
`output_files.*_sha256` still name the pre-migration content. Consistent.

---

## 3. The run host is not the repo this branch is on

New finding, required before any of §4–§7 can be trusted.

```
local  E:\nanochat\finquery_rag\backend      branch feat/nf-v3-interview-final
                                             HEAD   ca3446d
host   /disk/qh/nano-finrag                  branch p1-4g-verify
                                             HEAD   ffc8d834

git merge-base --is-ancestor ca3446d4bef8… HEAD   ->  fatal: Not a valid commit name
git merge-base --is-ancestor beae6c7674… HEAD     ->  fatal: Not a valid commit name
```

The two are **separate repositories with no shared history**. They are not
divergent clones; neither commit exists in the other.

### 3.1 The frozen benchmark is nevertheless identical

```
                              local                                host
gold-evidence-v1.jsonl        3d2a0c5b7839656ce1414923ab…          3d2a0c5b7839656ce1414923ab…
canonical-eval-v1.jsonl       227f0341d94ab8d4b9e7ee033f…          227f0341d94ab8d4b9e7ee033f…
plan-fixtures-v8.jsonl        7463b45a5bfc309d0a5ab29cdb…          7463b45a5bfc309d0a5ab29cdb…
```

Byte-identical. The oracle is one object; only the runtime differs.

### 3.2 The runtimes are not identical

The host carries three commits dated 2026-09-19 that do not exist here:

```
ffc8d834  fix(calc): the guard asks about the fact's coordinate, not the slot's
29571a1a  test(harness): the fixture store honours the coordinate contract too
ad2362c0  feat(calc): refuse an operand whose coordinate does not identify one value
```

and a module that does not exist here at all:

```
src/finance/operand_ambiguity.py       CoordinateStatus, coordinate_status()
```

`ad2362c0` refuses an operand whose coordinate does not identify one value. It is
a **coverage-reducing** change and it post-dates the handoff's seal. This is why
§7.1 of the contract distinguishes the 52.6% baseline from the 17.9% one: they
are measurements of two different runtimes, not two states of one.

`benchmark-version.json` is unchanged on both sides, so the host's modified
benchmark working tree (`git status` on the host shows the three benchmark files
as modified) is modified relative to `ffc8d834`'s tree and **identical to the
frozen bytes**. Recorded because `git status` on the host reads as a dirty
oracle and does not mean one.

---

## 4. sum-007's two source tables, re-extracted from the filing

Extracted independently from
`/disk/qh/nano-finrag/data/raw_sec_html/KO/SEC_21344_000162828026010047/primary.html`
by parsing every `<table>` element and printing each matching table's rows and
its preceding text. 117 table elements in the filing.

The filing's own metadata confirms what it is:

```
source_url        https://www.sec.gov/Archives/edgar/data/21344/000162828026010047/ko-20251231.htm
period_hits       ["2025-12-31", "December 31, 2025", "December 31, 2025"]
raw_sha256        22dd41f1aab4928f94a5f80e9497cf6203f27baa7c3f4e9088a6dd0aee929c66
recomputed        22dd41f1aab4928f94a5f80e9497cf6203f27baa7c3f4e9088a6dd0aee929c66   MATCH
```

### Table 23 — preceded by `THE COCA-COLA COMPANY AND SUBSIDIARIES / CONSOLIDATED STATEMENTS OF INCOME`

```
Year Ended December 31,          2025     2024     2023
Net Operating Revenues     $    47,941   47,061   45,754
Cost of goods sold              18,397   18,324   18,520
Gross Profit                    29,544   28,737   27,234
Selling, general and
  administrative expenses       14,521   14,582   13,972
Other operating charges          1,261    4,163    1,951
Operating Income                13,762    9,992   11,311      <-- Coca-Cola's own
...
Net Income Attributable to
  Shareowners of The Coca-Cola
  Company                  $    13,107   10,631   10,714
```

### Table 46 — preceded by `A summary of financial information for our equity method investees in the aggregate is as follows (in millions):`

```
Year Ended December 31,          2025     2024     2023
Net operating revenues     $   102,800   99,043   93,862
Cost of goods sold              60,622   58,527   55,780
Gross profit               $    42,178   40,516   38,082
Operating income           $    13,426   12,536   11,868      <-- the gold's source
Consolidated net income    $     9,355    8,439    7,657
Company equity income (loss) — net $ 2,031  1,770    1,691
```

### Corroboration from two further tables

The same filing's segment disclosure (`Information about our Company's
operations by operating segment and Corporate`) reconciles **to 13,762**:

```
                                EMEA  LatAm  N.America  AsiaPac  Bottling  SegTotal  Corporate  Elimin.  Consolidated
Operating income (loss)   $    4,298  3,742     5,070    2,042       426    15,578     (1,816)      —        13,762
```

and its FY2024 counterpart reconciles to **9,992** (`15,255` + `(5,263)`).

So Coca-Cola's own operating income for the two years is stated at
`13,762 / 9,992` in the consolidated statement, is the segment table's
`Consolidated` column, and sums to **23,754**. Confirmed three ways.

### What the gold actually is

```
gold-evidence-v1.jsonl   expected_value 25962.0000
                         operands       {a: "$ 12,536", b: "13,426"}
                         fact_ids       v2fact:ac8339670baa9e6c04907561f39317ef
                                        v2fact:b7cec5eff7725f1941a30ea541ea0929

12,536 + 13,426 = 25,962
```

Both fact ids resolve in the frozen store to **one row** —
`row:c82e0d7a3635338eb14460cd415a76fa3d1988360dfba667cef26c67d5144cc6`, content
`Operating income | $ | 13,426 $ | 12,536 $ | 11,868` — the investee summary.

**The handoff's §3 is confirmed in full, including the part that was retracted
earlier in P1.7.** B0 described `13,762` as a *segment* row; the handoff's §4
retracts that, and the source agrees with the handoff: the table holding
Coca-Cola's own `13,762` is the **consolidated statement of income**, and
`13,762` also happens to be the reconciliation table's Consolidated column. The
handoff's correction is right and B0's label was wrong.

---

## 5. Why the store cannot say which table a row came from

`p1-7-final-handoff.md` §3 asserts this. Measured against the frozen store —
field inventory over all 20,394 records:

```
candidate_id candidate_key cell_id citation_id citation_ids citation_origin
content currency document_id document_name entity evidence_id fact_id fact_type
metric page pdf_page period periods physical_source_id provenance_complete
raw_content raw_value row_id row_ids scale source_id source_text source_traceback
table_fragment_id table_id unit value values
```

`table_fragment_id` is present on every record. `table_id` is present on every
record and is `null` on every record. There is no caption, no heading, no
statement scope, no axis and no dimension on any record.

At `(The Coca-Cola Company, operating income, FY2025)` the store holds exactly
**two** records, carrying two distinct values:

```
13,762   metric "Operating Income"   row:a5da46248aad2727…   table:4b97983e2a131b0b…
         pdf_page 63   row_index 6   Net Operating Revenues row above it = 47,941
13,426   metric "Operating income"   row:c82e0d7a3635338e…   table:9a90ac6dbf971bc6…
         pdf_page 87   row_index 4   Net operating revenues row above it = 102,800
```

The two rows are distinguished by `row_id`, `table_fragment_id`, `pdf_page` and
by the metric's capitalisation. **They are the same coordinate**: the runtime's
`coordinate_key` (`trusted_v2_production.py:506`) casefolds, so `Operating
Income` and `Operating income` fold together, and `facts_at_coordinate` returns
both rows.

### 5.1 The handoff's §3 claim is correct — and its mechanism is sharper than stated

The handoff says the store "has no way to say which table a row came from". That
is right in the sense that matters: `table_fragment_id` is a **content hash**,
so although it *distinguishes* the two tables it carries nothing that says what
either table is. A rule keyed on the fragment id would be reading back a label
the extractor assigned, which is the circularity the contract's §5 rejects.

---

## 6. The source already carries the discriminator, and it is not a caption

This is the finding that makes the contract decidable, and it is not in the
handoff.

The corpus contains a second store built from the same eight filings' **iXBRL**:

```
/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl
   records 19,795   company_level 7,574   dimensioned 12,221   concepts 2,552
```

Coca-Cola `us-gaap:OperatingIncomeLoss`, every context:

```
FY2025   dim 0   13,762     consolidated
FY2025   dim 0   13,762     (tagged in two places)
FY2025   dim 1   13,426     EquityMethodInvestmentNonconsolidatedInvesteeAxis
FY2025   dim 1   15,578     ConsolidationItemsAxis / MaterialReconcilingItems
FY2025   dim 1    1,816     ConsolidationItemsAxis / CorporateNonSegment
FY2025   dim 2    {4,298 3,742 5,070 2,042 426}   OperatingSegments × segment
FY2024   dim 0    9,992     consolidated
FY2024   dim 1   12,536     EquityMethodInvestmentNonconsolidatedInvesteeAxis
...
```

**The investee row is not mislabelled. It is dimensioned
`EquityMethodInvestmentNonconsolidatedInvesteeAxis`** — the filing itself says
the value is not the consolidated entity's. The legacy store dropped that axis;
the iXBRL layer keeps it.

Applying the contract's rule (company-level = undimensioned) to this coordinate
gives exactly the contract's answer:

```
FY2024 dim 0 -> 9,992        FY2025 dim 0 -> 13,762        sum -> 23,754
```

and matches `resolve_canonical`'s proposed correction in
`p1-7-0a-oracle-attribution.json` to the digit.

### 6.1 This is already partially wired, and its reach is the binding constraint

`src/runtime/trusted_v2_canonical_store.py` implements the rule: `resolve_canonical`
returns the single undimensioned value, and returns `[]` rather than guessing
when several exist. Its own docstring states the honest position and a cost
figure:

```
"The store being replaced cannot ... which is why the operand guard has to
 refuse 18.2% of it."
```

**18.2% is the module's own figure and was not reproduced here.** This pass's
case-level measurement is 27 of 95 answerable cases refused (28.4%), which is a
different denominator — the module's may be per slot or per store coordinate.
Both are recorded rather than reconciled, because the discrepancy is a question
about which population the guard bites, not a contradiction.

Its reach is bounded by `METRIC_TO_CANONICAL`, which lists **12** quantities and
deliberately omits the ambiguous names:

```
net income, consolidated net income, total assets, total liabilities,
operating income, income from operations, research and development,
diluted earnings per share, comprehensive income, interest expense,
long-term debt, total net revenue
```

`Deferred`, `Services`, `State`, `Colette M. Kress`, `Intersegment`,
`U.S. GSEs and government agencies`, `Cost of revenue` are **not** in it, by
design — mapping them "would be a guess".

---

## 7. The risk set, re-measured

Resolving gold fact ids across **both** key spaces (the `s3` cross-entity
stratum names iXBRL keys, as `CanonicalFactStore.candidate_keys` documents):

```
MULTI_VALUE        33      the gold's coordinate holds several distinct values
SINGLE_VALUE       62
NO_FACTS           25      abstention cases
UNRESOLVED          0
```

The handoff §3 reports `32 / 43 / 45`. The difference is exactly the 20-case
`s3` stratum, which the legacy-only measurement cannot resolve at all: 25 + 20 =
45, and two of the `s3` cases collide at the **iXBRL** level:

```
tv2f01-s3-compare-001  (The Coca-Cola Company, net income, 2025)  {13107, 13137}
tv2f01-s3-rank-001     (Tesla, net income, 2025)                  {3794, 3855}
```

`13137` is consolidated net income (`us-gaap:ProfitLoss`) and `13107` is net
income attributable to shareowners (`us-gaap:NetIncomeLoss`). Tesla is the same
shape: `us-gaap:ProfitLoss` 3855 against `us-gaap:NetIncomeLoss` 3794.

**Both facts in each pair are company-level and undimensioned, and they are
*different concepts*.** So this is the case where the company-level rule of §2
is necessary but not sufficient: dimensions do not separate them, and something
else has to — the concept. That is precisely what `CONCEPT_ALIGNMENT`'s ordering
exists to settle, and it is why `resolve_canonical` never resolves through the
canonical field alone. `NetIncomeLossAttributableToNoncontrollingInterest` (KO
30, Tesla 61) sits at the same coordinate as a third undimensioned concept and
is excluded by the same mechanism.

The handoff's 32 is not wrong; it is legacy-only. 33 is the complete figure.

### 7.1 Classified with the runtime's own guard

Running the host's `coordinate_status` over the gold's own bound facts:

```
per-case verdicts            per conflicting observation
GUARD_REFUSES        27      NO_CANONICAL_QUANTITY       32
CLEAN                48      UNIQUE                       2   (both tv2f01-s2-sum-007)
ABSTENTION           25
s3 cross-entity      20      total conflicting observations 34
```

and across the **41** observations where a gold fact's coordinate holds more than
one distinct raw value:

```
conflicting_values      34
consensus_same_value     6
unique                   1
```

The 6 `CONSENSUS_SAME_VALUE` observations are representation variants of one
quantity and are admissible today — collapsing them into `CONFLICTING_VALUES`
would refuse a correct release:

```
tv2f01-s2-growth-010  stock-based compensation  ['$10,734', '10,734']
tv2f01-s2-sum-005     tangible book value/share ['$ 97.30', '97.3']
tv2f01-s1-v-040       net revenue               ['$ 40,000', '$40,000', '11%']
```

### 7.2 The finding that bounds the coverage target

Of the 34 conflicting observations, **2 are resolved by the canonical path and
they are both sum-007. The other 32 — across 26 cases — name a metric with no
canonical quantity at all.**

That is the wall for this measurement, and it is narrower than it looks. The
metrics are `Deferred`, `Services`, `State`, `Colette M. Kress`, `Intersegment`,
`Commercial(3)`, `Hedge accounting fair value`, `Cost of revenue`,
`U.S. GSEs and government agencies`, `Foreign currency contracts` and similar —
most of which name a *scope*, which would make their cases class S rather than
class A.

The iXBRL layer carries the scopes:

```
125 distinct axes      1,083 distinct members
  StatementBusinessSegmentsAxis            721
  EquitySecuritiesByIndustryAxis           725
  FinancingReceivablePortfolioSegmentAxis 1425
  FairValueByFairValueHierarchyLevelAxis  1108
  ...
```

Substring-matching the metrics against that member list gives real hits for
`Other letters of credit` → `OtherLettersOfCreditMember` (22) and `Commercial`
→ `CommercialPortfolioSegmentMember` (816), and nothing for `Colette M. Kress`,
`Ajay K. Puri`, `Intersegment` or `Hedge accounting`. **That is a signal and not
a result** — the member list is uncontrolled, the matches are loose, and
identifying the member the filing actually intended is a source-reading task per
case that this pass did not perform.

So: **how many of the 26 are class S and how many are class A is unmeasured.**
Stating either would be the guess the contract exists to prevent.

---

## 8. What was not done

```
not modified   Runtime, Store, Benchmark, Gate, Binder, Validator, Retriever,
               Sidecar   (no file under src/ was written)
not applied    migrations/p1-7-0a-oracle-attribution.json  (still PROPOSED_NOT_APPLIED)
not started    Structural Context Sidecar; benchmark migration
not used       60% or 80% coverage as an adjudication criterion
not inherited  every item in the handoff's §4 retraction list
```

Analysis scripts were written to a scratch directory on the run host
(`/tmp/p1_8/`, `/tmp/*.py`) and to no repository path.

**The host's own working tree was not clean before this work and is not
evidence of it.** `git status` there already showed modifications to
`canonical-eval-v1.jsonl`, `gold-evidence-v1.jsonl`,
`build_p1_2_plan_fixtures.py` and `run_nf_v2_17a4_parse.py` on first inspection.
Per §3.1 the two benchmark files are byte-identical to the frozen copies, so
those modifications are relative to `ffc8d834`'s tree and not to the oracle.
This pass added nothing to that list; only the temporary directory above.

---

## 9. One correction to the handoff, offered

The handoff's §1 seal is `beae6c7` and the branch is at `ca3446d`. The handoff
does not say its own seal precedes it by one commit. That is benign — the
intervening commit is the handoff itself — but a future reader comparing the
seal against `git rev-parse HEAD` will find a mismatch and has no way to know it
is expected. It is expected. Recorded here so the next reader does not spend the
hour this one did.
