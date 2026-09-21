# P1.8-B1 — canonical-candidate retrieval: measured headroom

Answers the five questions in the P1.8-B1 brief, in order. **No code was
modified; no backend was stopped; no store, benchmark, gate, binder, validator
or finalizer was touched.** The only inputs are the frozen per-case artifacts of
the P1.7 run and the frozen stores.

## Headline

> **Canonical-candidate retrieval is worth at most 3 cases. On the Gate-ON
> baseline that is 46/95 -> 49/95 = 51.6%. It does not reach 60%, and it cannot
> be made to by this mechanism.**

The Retriever freeze should **not** be lifted for this change. The remaining
distance to 60% is elsewhere, and §5 names where.

---

## 1. Re-read of the handoff and HEAD

```
HEAD            ca3446d4bef8f50a0f32c1d8506da13d09296676
                docs(eval): P1.7 final handoff -- verified facts, and what must not be inherited
parent seal     beae6c76749460dfc247f59c7bb7f559d2809ade   (handoff §Seals)
```

Unchanged from the P1.8-B0 verification. The 120-case benchmark is
byte-identical to the frozen hashes.

---

## 2. The current baseline — and a correction to it

Your brief says the frozen baseline is 50/95 = 52.6%. **That figure is the
gate-bypass arm, not the Gate-ON configuration.** Measured from the three arms
of the P1.7 run, whose per-case artifacts are dated 2026-09-21 10:44–11:14:

```
arm                    artifact                                     released    coverage
BASE                   /tmp/g-base/A_pinned_production-cases.jsonl     20/95      21.1%
GATE_ON                /tmp/g-new/A_pinned_production-cases.jsonl      46/95      48.4%
GATE_BYPASS            /tmp/g-bypass/B_gate_bypass-cases.jsonl         50/95      52.6%
```

`p1-7-release-coverage.md` already distinguishes these: "48.4% is the
pinned-plan release rate". The handoff's §1 table prints 50/95 under the heading
"Frozen runtime ... Gate ON", but 50/95 is the third column of that document's
own comparison. **The Gate-ON release coverage is 46/95 = 48.4%.**

This matters for the target. 60% of 95 is **57 released cases**. The frozen
Gate-ON state has 46, so the distance is **11 cases**, not 7.

### 2.1 Correct refusals and accuracy are intact in every arm

```
GATE_ON   answerable 95   released 46   must_refuse 25   incorrect release 0
```

The 25 abstention cases refuse correctly in all three arms. No safety invariant
is at risk in what follows.

---

## 3. First-failure attribution of the 49 unreleased answerable cases

Computed from the frozen artifact's own fields, on the `g-new` (Gate-ON) arm:

```
GATE_BLOCKED:SEMANTIC_MISMATCH     22      terminal_state PLAN
GOLD_NOT_IN_POOL                   10
CONFLICT_IDS                        9      reason_codes ['EVIDENCE_CONFLICT']
MISSING_OPERAND                     4
VALIDATION_FAILED                   2
GOLD_NOT_BOUND                      2
                                   ---
                                    49
```

Terminal states: `PLAN` 22, `FAIL_CLOSED` 17, `READY_TO_GENERATE` 5,
`EVALUATE` 5.

**The dominant loss is the Gate, not representation.** Twenty-two cases stop in
terminal state `PLAN` with `gate_code = SEMANTIC_MISMATCH` and never reach
retrieval at all — `pool_depth` is 0 for them. A change inside the retrieval path
cannot reach them by construction.

---

## 4. The 9 `EVIDENCE_CONFLICT` cases, and what canonical retrieval recovers

These are the only cases the canonical-candidate seam can touch. All nine are
`FAIL_CLOSED` with `reason_codes = ['EVIDENCE_CONFLICT']`, gate not blocked:

```
case                     stratum                  gold_in_pool  conflict_ids
tv2f01-s1-jpm-009        factual_lookup                    1            1
tv2f01-s1-ko-011         factual_lookup                    1            1
tv2f01-s1-msft-016       factual_lookup                    1            1
tv2f01-s1-pfe-030        factual_lookup                    1            1
tv2f01-s1-tsla-031       factual_lookup                    1            1
tv2f01-s1-v-036          factual_lookup                    1            1
tv2f01-s2-pctshare-006   arithmetic_calculation            2            2
tv2f01-s2-sum-007        arithmetic_calculation            2            2
tv2f01-s3-rank-005       cross_entity_comparison           0            2
```

Gold is already in the pool for eight of nine. The conflict is what stops the
bind.

### 4.1 Recoverability, tested rather than assumed

A case is recoverable only if **every** gold value is stated by a company-level
(`dimension_count == 0`) iXBRL fact at the same entity and period. That is the
contract's authority rule, applied to the gold, independently of where the gold
currently points.

```
case                     gold value(s)              company-level fact found?
tv2f01-s1-jpm-009        $ 603,947                  yes  (1 concept)
tv2f01-s1-msft-016       87,831                     yes  (1 concept)
tv2f01-s1-pfe-030        $ 17,851                   yes  (1 concept)
tv2f01-s1-ko-011         (32)                       no
tv2f01-s1-tsla-031       $ 68,764                   no
tv2f01-s1-v-036          15                         no
tv2f01-s2-pctshare-006   $ 78,328 / $ 209,586       one of two
tv2f01-s2-sum-007        $ 12,536 / 13,426          no  (both dimensioned)
tv2f01-s3-rank-005       ixbrl keys                 not applicable
```

**3 of 9 recoverable.** And the ceiling, if every one of the nine released
without a single other case moving:

```
46 + 9 = 55/95 = 57.9%      < 57 needed?  no, 55 < 57.  Still short of 60%.
46 + 3 = 49/95 = 51.6%      the honest estimate.
```

**So the answer to your question 4 is: no.** `50/95 + N` does not reach 57/95
for any `N` this mechanism can supply — and the true base is 46/95, so even the
absolute ceiling of the mechanism (55/95) misses.

### 4.2 The three recoverable cases are not equally solid

`jpm-009` is nominally recoverable only because its metric string
`Total lending-related commitments` happens to match a filer-specific concept
`jpm:OffBalanceSheetLendingRelatedFinancialCommitmentsExcludingCommitmentsForWhichAllowanceForCreditLossNotPermittedContractualAmount`.
That is a coincidence of naming, not an authority — a metric→concept mapping
fitted to one case is the kind of hard-coding the brief forbids.

The defensible subset is **two cases**: `msft-016` (`Cost of revenue` 87,831 ->
`us-gaap:CostOfGoodsAndServicesSold`) and `pfe-030` (`Cost of sales` 17,851 ->
the same concept). Both are standard-taxonomy, both are also confirmed by the
arithmetic identity (87,831 = 22,422 + 40,171 + 25,238), and both are the kind of
company-level total the contract's §2 rule selects.

```
two defensible cases:   46 + 2 = 48/95 = 50.5%
```

---

## 5. Where the coverage actually is

Three levers, ranked by measured size. None of them is the Retriever.

```
1. THE GATE              22 cases, terminal_state PLAN, pool_depth 0
                         -- they never reach retrieval at all.
```

Two different questions live here, and they have different answers.

*How much is full gate permissiveness worth?* Compared across the three arms:

```
                     what changed                releases
BASE   -> GATE_ON    vocabulary fixed            20 -> 46    +26
GATE_ON-> GATE_BYPASS the gate is skipped        46 -> 50     +4
```

The P1.7 vocabulary fix was worth **+26**; the remaining gate headroom, measured
by removing the gate outright, is **+4**. Bypassing converts those 22 blocks into
22 `FAIL_CLOSED` cases and yields 4 net releases — the other 18 fail for reasons
a bypass does not address.

But skipping the gate is not the only way through a `SEMANTIC_MISMATCH`, and a
*further vocabulary fix* is what produced the +26. **How much a further
vocabulary change would clear is not measured here.** It needs each case's
`gate_reason_raw` read against what the question actually asks, and it is the
largest piece of unexamined headroom in the system.

```
2. THE CANONICAL-CANDIDATE SEAM      3 cases (2 defensible). Measured above.
   Not worth unfreezing the Retriever for.

3. THE OTHER 17 FAIL_CLOSED CASES    the seam does not reach them.
   GOLD_NOT_IN_POOL 10, MISSING_OPERAND 4, VALIDATION_FAILED 2, GOLD_NOT_BOUND 2.
```

**Nothing in this inventory demonstrably reaches 60%.** What is bounded is the
combination of the two mechanisms that were on the table: a fully permissive
gate (+4, measured) and canonical candidates (+3, measured) give at best
**46 + 7 = 53/95 = 55.8%** — still short. The one lever that is not bounded is
the 22 residual gate refusals, and it is the only one worth opening next.

That is a finding about the benchmark configuration, not about the Retriever, and
it is the thing to resolve before any freeze is lifted.

---

## 6. Provenance: why the seam is not a drop-in

Two structural facts, both measured, that any canonical-candidate seam has to
answer. They are the reason §7's diff is a design and not a patch to merge.

### 6.1 The iXBRL store has no page and no citation

```
legacy record           page 63, pdf_page 63, citation_id citation:v2:71c9a2af…,
                        table_fragment_id table:4b97983e…, row_id row:a5da4624…

iXBRL record            keys: candidate_key canonical_concept concept context_ref
                        dimension_count dimensions document_id document_name
                        entity fact_id ixbrl_fact_id period period_end period_start
                        provenance_complete raw_value source ticker unit
                        unknown_context value
                        NO page.  NO citation_id.  NO table or row.
```

The iXBRL layer's provenance is `document_id` + `context_ref` — a filing and an
XBRL context. That is *stronger* evidence about what a value means and *weaker*
evidence about where it is printed. Citation precision and recall are computed on
the citation space, so a candidate swapped into that space changes what a
citation means.

### 6.2 The production store loader rejects canonical candidates outright

`trusted_v2_production.py:379`:

```
if self.require_citation_id and citation_id is None:
    raise TrustedV2ProductionConfigurationError(
        f"fact store record {index} is missing structured citation_id"
    )
```

`CanonicalFactStore._as_fact` sets `"citation_id": None`. So canonical records
**cannot load into the production store today** — not because the seam was never
written, but because a fail-closed invariant would reject them. Any seam has to
either synthesise a citation id for canonical candidates or exempt them, and
either is a decision about the citation contract, which is why it belongs in
review and not in a patch I apply.

---

## 7. The seam, as a design for review

Not applied. Not written to `src/`. The shape below is what §6 forces.

```
component    CanonicalCandidateAdapter — wraps StructuredFactStore, delegates
             everything, and changes one method.

switch       TRUSTED_V2_CANONICAL_CANDIDATES=off   (default; behaviour identical)
             TRUSTED_V2_CANONICAL_CANDIDATES=on    (seam active)
             Read once at construction.  Rollback = unset the variable and
             restart; no store, index or artifact is migrated, so rollback is
             total.

identity     legacy candidate_key  ->  canonical candidate_key
             v2fact:<h>           ->  ixbrl:<h'>   and back
             Derivation is a pure function of (entity, metric, period, value,
             unit, scale, currency) — the same tuple logical_fact_id already
             uses — so it is stable across runs and independent of retrieval
             order.  Retained on the candidate as:
                 canonical_key, canonical_source, legacy_key
             Nothing is overwritten: the legacy key stays on the record.

provenance   A canonical candidate carries document_id + context_ref.  It does
             NOT carry page/table/row.  Two options, both requiring a decision:
               (a) synthesise citation_id from (document_id, context_ref), which
                   is a NEW citation space, not the physical one the benchmark
                   measures; or
               (b) keep the legacy fact as the cited support and use the
                   canonical fact only for value authority.
             (b) preserves citation P/R and is the recommended default; it also
             means the seam changes which value is bound, not what is cited.

fail-closed  The seam resolves ONLY when the canonical store returns exactly one
             undimensioned fact for the slot's coordinate.  Zero or several =>
             return the legacy candidate set unchanged.  It can therefore never
             make a coordinate less ambiguous than it is today, and the
             CONFLICTING_VALUES guard keeps firing on exactly the cases it fires
             on now.  This is the property that keeps INCORRECT_RELEASE at 0 and
             it is the first thing to test.

budget       Candidate count per slot is unchanged: the seam substitutes a
             candidate, it does not append one.  If a substitution would change
             the pool size, the substitution is skipped.
```

### 7.1 What the review has to decide before this is built

1. §6.2 — synthesise a citation for canonical candidates, or exempt them from
   `require_citation_id`? Both touch a fail-closed invariant.
2. §6.1 — (a) a new citation space, or (b) legacy support with canonical
   authority? (b) is advisory and preserves the citation metrics.
3. Whether 2–3 cases justify the change at all. My read is **no**, and that the
   effort belongs on the 22 gate-blocked cases instead.

---

## 8. What I did not do

```
did not modify        any file under src/, any store, benchmark, gate, binder,
                      validator, finalizer or sidecar
did not stop          the backend (uptime 1d12h, port 18002) or any GPU process
did not apply         the seam, or migrations/p1-7-0a-oracle-attribution.json
did not use           60% or 80% coverage as an adjudication criterion
```

All analysis ran against the frozen per-case artifacts in `/tmp/g-*/` and the two
frozen stores, read-only.

---

## 9. One thing to settle before the next round

The brief's premise — baseline 50/95 — is the gate-bypass arm. On the Gate-ON
configuration the baseline is 46/95 and the target is 11 releases away, and the
two candidate mechanisms together supply about 7. **Whether 60% is reachable at
all in the Gate-ON configuration is not established by anything I measured**, and
I would rather say that than produce a patch that cannot reach the number it was
commissioned for.
