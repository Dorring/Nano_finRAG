# P1.6-A seal — Store V2 table authority

**Verdict: SEALED.** Every number below was recomputed from the artifacts at seal time and
matched; the seal fails if any of them moves. `seal.json` carries the hashes.

```
P1.6-A authority recovery        SEALED   this
P1.6-A fact emission recovery    OPEN     nine verified primary statements produce no
                                          store records
```

The second line is why this is named for table authority and not for the whole of P1.6-A.
Source structure, table authority and the parser's cells are in place; `cells -> atomic
facts` is not. Sealing without saying so would claim a completeness the store does not have.

## What is frozen

```
filings                          8
corpus tables                 1498
  data tables                  906
  layout scaffolds             592
oracle primary                  42      read one by one, all genuine
oracle dangerous negatives     229      the safety gate
store records                26977      deterministic, rebuilt twice per filing

47-slot authority
  before (statement_type)   RESOLVED 10   NO_COMPANY_LEVEL_FACT 29
                            AMBIGUOUS 3   NO_FACT 5
  after  (table_role)       RESOLVED 37   NO_COMPANY_LEVEL_FACT 5   NO_FACT 5

  27 CAPABILITY_GAIN   19 UNCHANGED   1 REFUSAL_RETYPED   0 REGRESSION

safety gates
  wrong_scope 0   wrong_metric 0   value_mismatch 0
  dangerous_authority 0   scaffold_authority 0

under the old rule, dangerous_authority was 2
```

## The revisions

```
commit          2e492e8c      feat(evaluation): table_role reaches the resolver
                              -- 10 -> 37 resolved, gate clean

bf2c22c2939671c8  scripts/evaluation/build_store_v2.py
c0894afcbf096a5a  scripts/evaluation/build_table_authority_oracle.py
c21390a3b0847f1b  scripts/evaluation/classify_table_authority.py
f3ee85bd55f6a149  scripts/evaluation/evaluate_table_role_authority.py
7fc05790565347d5  scripts/evaluation/resolve_store_v2_slot.py
052c881264e708a4  scripts/evaluation/run_nf_v2_17a4_parse.py
310f1d353593cb6e  scripts/evaluation/score_table_authority_classifier.py
d02b91872e5837a6  scripts/evaluation/table_authority_adjudications.json
239e130f8b34a477  src/runtime/trusted_v2_canonical_fact_store.py

dccbe5d0221ccb67  artifacts/.../p1-6-a2b19a-oracle/table-authority-oracle.json
2dc6202115f2a17a  artifacts/.../p1-6-a2b19b-shadow/shadow-classification.json
ed43ddba4678fcd9  artifacts/.../p1-6-a2b20-store-v2-role/store-v2.jsonl
ca29cd40e95d0943  artifacts/.../p1-6-a2b20-store-v2-role/table-roles.json
2060a6b24499d8a2  artifacts/.../p1-6-a2b20-store-v2-role/authority-experiment.json
```

## What the seal asserts

That the data-layer repair propagates: `SEC source structure -> table_role -> Store V2 ->
company-level authority -> 27 new safe resolutions`, with every gate at zero and no
regression. And that the old rule resolved two slots out of a table the oracle calls
dangerous, which the new rule does not.

## What it does not assert

**The 37 is a ceiling set by what the store can see, not by authority.** Nine verified
primary statements produce no records at all:

```
jpm  BALANCE_SHEET  CASH_FLOW        ko   BALANCE_SHEET  CASH_FLOW  EQUITY
nvda CASH_FLOW                       pfe  EQUITY
tsla CASH_FLOW                       v    BALANCE_SHEET
```

Coca-Cola's balance sheet yields more populated cells and more iXBRL anchors than Apple's
and the store takes none of them. The loss is after `parse_table` and before
`build_semantic_corpus`'s atomic facts; only 75 `missing_period` skips are reported for the
whole of Coca-Cola. Three of the five remaining `NO_COMPANY_LEVEL_FACT` slots are blocked
on this, not on anything `table_role` decides.

**The classifier's positive rule is not independently confirmed.** 42/42 was an expected
outcome, not a finding: the oracle's positive rule and the classifier's
`EXTERNAL_STATEMENT_TITLE` are the same signal. No filing outside these eight has been run.

**Five `NO_FACT` slots are metric identity, not authority** — the filing states the figure
under a longer label than the fixture asks for.

## The residual, in buckets

```
authority                       solved
metric identity / canonicalization     5 x NO_FACT
fact emission (the blocker above)      3 x blocked on missing primary statements
correct refusal                        2 x the filing states no such company-level row
```

Nothing in the residual is a `table_role` problem, and `table_role` should not be adjusted
further. 37 safe resolutions plus 10 refusals with stated reasons is worth more than 47
answers.

## Next

```
1. holdout filing          a ninth filing, outside every rule's design, run in shadow
2. P1.6-A3                 missing primary table fact emission
3. metric identity         the 5 NO_FACT slots
4. production gate         only after 1-3, and after a full downstream regression
                           (Store V2 -> candidate views -> retrieval -> Binder ->
                           deterministic operations -> release, false_release = 0)
```
