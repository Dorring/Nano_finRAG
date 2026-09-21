# P1.6-A4 — concept alignment, validated 39 of 39

`src/finance/concept_alignment.py` maps filer-specific XBRL concepts onto canonical,
comparable quantities. `validate_concept_alignment.py` proves it against every value read
out of a filing by hand during P1.6-0D and the stratum re-derivation.

```
context-parse coverage     8 / 8 filings, all contexts parsed
alignment validation      39 / 39 MATCH
```

## Why a rule would not do

The taxonomy is shared; the filers do not use it alike, and the differences are semantic:

```
net income      Apple, JPM, MSFT, NVIDIA   us-gaap:NetIncomeLoss
                Tesla, Coca-Cola, Visa     us-gaap:ProfitLoss

interest exp.   JPMorganChase              us-gaap:InterestExpenseOperating
                Visa, Microsoft, KO        us-gaap:InterestExpenseNonoperating
```

`ProfitLoss` is the consolidated result including noncontrolling interests;
`NetIncomeLoss` is the portion attributable to the parent. Where a filer tags both they
disagree — Coca-Cola 13,137 against 13,107, Tesla 3,855 against 3,794 — and where it tags
one, that is because its noncontrolling interest is nil and the two coincide. A bank's
interest expense is an operating item and a manufacturer's is not.

So "standard concept" is not "comparable number". `net_income` therefore **prefers
`ProfitLoss`**, because the benchmark's verified golds are the consolidated figures, and
putting `NetIncomeLoss` first would silently redefine the quantity and change two of the
cross-entity golds. The ordering is part of the definition.

The alignment bridges these pairs and reproduces every verified value through them:

```
net_income           ProfitLoss, NetIncomeLoss
interest_expense     InterestExpense, InterestExpenseNonoperating, InterestExpenseOperating
comprehensive_income ComprehensiveIncomeNetOfTax, ...IncludingPortionAttributableToNoncontrollingInterest
long_term_debt       LongTermDebtNoncurrent, LongTermDebtAndCapitalLeaseObligations
```

## The context-id scheme is not uniform either

The first run resolved 32 of 39 and left **every Microsoft value unresolved**. Microsoft's
filing — produced by DFIN ActiveDisclosure — ids its contexts as UUIDs:

```xml
<xbrli:context id="C_81dc69c5-15a3-4437-a39b-38d8a7446e50">
```

where the others use `c-1`, `c-30`. A pattern matching only the first shape parsed 441
tags as zero, which left every Microsoft fact looking **undimensioned**.

That is the dangerous direction: an undimensioned-looking fact is exactly what the
resolution rule selects as company-level, so a segment figure could have been returned as
the company total. It did not, because the coverage check reports
`dimension_filter_reliable: false` and the resolver refuses — but the near miss is the
reason the check exists, and it is recorded rather than quietly fixed.

## The rule it validates

A canonical quantity resolves to the first candidate concept present at an
**undimensioned context**, at the latest period available. That is the company-level
figure by construction, not by heuristic, and it is what "the coordinate identifies one
value" should have meant all along:

```
c-1     no segment  ->  company level
c-63    StatementEquityComponentsAxis=RetainedEarningsMember
c-2179  ConsolidatedEntitiesAxis=ParentCompanyMember   -> the 55,681 parent-only variant
```

Unknown contexts are treated as **dimensioned**, not undimensioned. Assuming the opposite
would make a segment figure indistinguishable from a company total, which is the failure
the whole line of work exists to prevent.

## Status

No store changed, no fixture changed, retrieval unchanged. The alignment is a pure
module plus a validation script; both are shadow.

**Next, per the ordering agreed:** re-measure the 20 cross-entity cases against the
canonical concepts and compare coordinate ambiguity to the 18.2% baseline, then decide
whether retrieval switches.
