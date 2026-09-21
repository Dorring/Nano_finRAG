# P1.8-B7 — BENCHMARK CHANGE PROPOSAL (NOT APPLIED)

The decision P1.8 has arrived at, written so it can be accepted or rejected case
by case. **Nothing here is applied.** `gold-evidence-v1.jsonl`,
`canonical-eval-v1.jsonl` and `benchmark-version.json` are untouched; the pinned
hash `3d2a0c5b` is intact.

Read with `p1-8-b5-source-adjudication.md` (the per-case source evidence) and
`p1-8-b6-the-real-lever.md` (why no system change substitutes for this).

---

## Why this is a benchmark change and not an engineering one

27 of the 95 answerable cases have their gold at a coordinate where the store
holds conflicting values. For each, one of three things is true, and none of
them is fixable in the runtime:

```
A. the question does not name what it asks for     the answer is undefined
B. the question names it, and the store is       needs column_identity (Store)
   ambiguous about it
C. the gold does not match the question's        needs a gold correction
   natural reading                                 (Benchmark)
```

Measured split:

```
metric IS named in the question         6   -> B or C
metric is NOT named in the question    21   -> A
                                      ---
                                       27
```

---

## Proposal 1 — re-point 1 gold (class C, highest confidence)

The only one where the source determines an answer and the gold has a different
one. Already written up as `p1-7-0a-oracle-attribution.json`, still
`PROPOSED_NOT_APPLIED`.

```
tv2f01-s2-sum-007   "the sum of The Coca-Cola Company's Operating income
                     across FY2024 and FY2025"
  current gold   25,962  = 13,426 + 12,536   <- equity-method INVESTEE table
  correct        23,754  = 13,762 +  9,992   <- Coca-Cola's own consolidated
  evidence      KO SEC_21344_000162828026010047 primary.html, table 23
                (caption "THE COCA-COLA COMPANY AND SUBSIDIARIES /
                CONSOLIDATED STATEMENTS OF INCOME"), corroborated by the
                segment table's Consolidated column (table 95)
  authority     contract section 2: company-level = dimension_count == 0
  hash impact   gold-evidence-v1.jsonl 3d2a0c5b -> new; benchmark version bump
```

## Proposal 2 — two further gold defects of the same class (class C)

Found during B5, not previously recorded. Both are the sum-007 pattern: the gold
points at a quantity the question does not denote.

```
tv2f01-s1-aapl-003   "the figure for Services in FY2025"
  current gold   75.4%              <- a GROSS MARGIN row
  natural reading 82,314            <- the Services revenue row, same page
  evidence      Apple FY2025, p.27: "Services | 82,314 | 71,050 | 60,345"
                alongside "Services | 75.4% | 73.9% | 70.8%"
  NEEDS REVIEW: "the figure for" may have been authored to mean the margin.
  If so the QUESTION should say so, not the gold.

tv2f01-s1-tsla-032   "Total automotive cost of revenues"
  current gold   (5,708)
  natural reading 57,165            <- Tesla's automotive cost of revenues
  evidence      same coordinate holds 57,165 / (5,708) / (9)%
  NEEDS REVIEW: same ambiguity as above.
```

## Proposal 3 — five cases where the store is ambiguous and the question is not

Class B. These are the only ones a `column_identity` store would fix. **They are
the argument for the store rebuild, and they are its entire yield from this
benchmark.**

```
tv2f01-s1-msft-016   "Microsoft's Cost of revenue"     gold 87,831
     87,831 is the company-level total AND equals 22,422 + 40,171 + 25,238.
     Authority confirmed twice. Recoverable WITHOUT any store change.

tv2f01-s1-pfe-030    "Pfizer's Cost of sales FY2024"   gold 17,851
     company-level total. Recoverable WITHOUT any store change.

tv2f01-s1-tsla-033   "Tesla's Revenues FY2025"         gold 82,056
     company-level total among two genuine revenue rows.

tv2f01-s2-pctshare-005  "what percentage ... was Cost of sales"  gold -0.5309
tv2f01-s2-sum-009       "sum of ... Foreign currency contracts"  gold partial
```

## Proposal 4 — twenty-one cases with no defined answer

Class A. The question names a row label that several real quantities satisfy and
names no column. **No store and no system change makes these answerable.**

```
Deferred · Services · Hedge accounting fair value adjustments ·
Common stockholders' equity(f) · U.S. GSEs and government agencies ·
Intersegment · Other contracts · Colette M. Kress · Ajay K. Puri ·
State value · U.S. Treasury securities · Total nominal payments volume(4) ·
Income tax effect · Beginning balance at January 1 · Commercial(3) ·
International-based companies · Total noninvestment-grade ·
Other letters of credit(d) · Total non-current portion of term debt ·
Foreign currency contracts
```

Two options, and only two:

```
(a) the question names its column      e.g. "what was the reported Deferred
                                       income tax expense?" -- a question edit
(b) the case moves to the abstention    the honest move when no reading exists
    side
```

**Recommendation: (b) for the ones where no column can be named from the source
row text** (`State value`, `Intersegment` — the latter is bound to a row that
reads "Total net operating revenues", i.e. a benchmark metric-label defect); and
(a) where the source row text does identify the intended column.

---

## What accepting this buys

```
                                 released   coverage
today (Gate-ON)                    46/95      48.4%
+ proposal 1 (gold re-point)       +1         49/95   51.6%
+ proposal 3 (company-level)       +4         53/95   55.8%
                                  -----       -----
target                              --         57/95   60.0%
```

**Even accepting every proposal does not reach 60%.** Proposal 4 moves 21 cases
to abstention, which changes the denominator: 53/74 = 71.6% if all 21 moved, and
that is the only arithmetic by which the target is met.

**That is the decision, stated plainly:** the 60% target is reachable only by
reclassifying 21 questions as abstentions — which is a statement that they were
never answerable — and not by any improvement to the system.

## What I need from you

```
1. proposal 1   re-point sum-007 to 23,754?          yes / no
2. proposal 2   aapl-003 and tsla-032 -- re-point, re-word the question,
                or leave?                             your call, per case
3. proposal 3   the four company-level cases: accept as recoverable?
4. proposal 4   the 21: abstention side, or question edits?  per case
```

I will not apply any of it without that. Applying any of 1–3 changes the pinned
gold hash and is a benchmark version bump; applying 4 changes the answerable
denominator and must be recorded as such in `benchmark-version.json`, not
quietly folded into a coverage figure.
