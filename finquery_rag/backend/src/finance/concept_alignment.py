"""Align filer-specific XBRL concepts onto canonical, comparable quantities.

The US-GAAP taxonomy is shared, but filers do not use it alike, and the
differences are semantic rather than cosmetic:

    net income   Apple, JPM, MSFT, NVIDIA  tag  us-gaap:NetIncomeLoss
                 Tesla, Coca-Cola, Visa    tag  us-gaap:ProfitLoss

Those are different quantities.  `ProfitLoss` is the consolidated result
including the noncontrolling interest; `NetIncomeLoss` is the portion
attributable to the parent.  Where a filer tags both, they disagree --
Coca-Cola 13,137 against 13,107, Tesla 3,855 against 3,794 -- and where it tags
only one, that is because its noncontrolling interest is nil and the two
coincide.  So "the same standard concept" is not "the same number", and a
retrieval keyed on a concept string would compare a consolidated figure against
a parent-only one and report the difference as fact.

    interest expense  JPMorganChase  us-gaap:InterestExpenseOperating
                      Visa, Microsoft, Coca-Cola  us-gaap:InterestExpenseNonoperating

A bank's interest expense is an operating item and a manufacturer's is not, so
the taxonomy splits them.  `rank-004` ranks four filers by interest expense and
its four golds sit on both sides of that split.

**The alignment is an ordered candidate list per canonical quantity, first
present wins**, and it is pinned by evidence: every entry below is checked
against a value read out of the filing by hand.  The ordering is part of the
definition, not a preference -- putting `NetIncomeLoss` first would silently
redefine "net income" as the parent-only figure and change two of the
cross-entity golds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: canonical quantity -> source concepts in priority order.
#:
#: `net_income` deliberately prefers `ProfitLoss`: the benchmark's verified golds
#: are the consolidated figures (Coca-Cola 13,137 rather than 13,107, Tesla 3,855
#: rather than 3,794), so that is the quantity the questions are asking about.
CONCEPT_ALIGNMENT: dict[str, tuple[str, ...]] = {
    "net_income": (
        "us-gaap:ProfitLoss",
        "us-gaap:NetIncomeLoss",
    ),
    "net_income_attributable_to_parent": (
        "us-gaap:NetIncomeLoss",
        "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic",
    ),
    "operating_income": ("us-gaap:OperatingIncomeLoss",),
    # `Assets` first on purpose: `LiabilitiesAndStockholdersEquity` carries the
    # same number by the accounting identity, so preferring it would key a
    # balance-sheet total on a concept that does not name it.
    "total_assets": ("us-gaap:Assets",),
    "total_liabilities": ("us-gaap:Liabilities",),
    "research_and_development": ("us-gaap:ResearchAndDevelopmentExpense",),
    "diluted_eps": ("us-gaap:EarningsPerShareDiluted",),
    "comprehensive_income": (
        "us-gaap:ComprehensiveIncomeNetOfTax",
        "us-gaap:ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "interest_expense": (
        "us-gaap:InterestExpense",
        "us-gaap:InterestExpenseNonoperating",
        "us-gaap:InterestExpenseOperating",
    ),
    "long_term_debt": (
        "us-gaap:LongTermDebtNoncurrent",
        "us-gaap:LongTermDebtAndCapitalLeaseObligations",
        "us-gaap:LongTermDebt",
    ),
    "revenues": (
        "us-gaap:Revenues",
        "us-gaap:RevenuesNetOfInterestExpense",
    ),
}

#: Canonical quantities the benchmark's cross-entity cases actually ask for.
BENCHMARK_QUANTITIES = (
    "net_income",
    "operating_income",
    "total_assets",
    "total_liabilities",
    "research_and_development",
    "diluted_eps",
    "comprehensive_income",
    "interest_expense",
    "long_term_debt",
)


@dataclass(frozen=True)
class Alignment:
    """One fact's canonical quantity, or why it has none."""

    canonical: str | None
    source_concept: str
    status: str  # ALIGNED | UNMAPPED
    reason: str


def align_concept(source_concept: object) -> Alignment:
    """The canonical quantity a source concept belongs to, or UNMAPPED.

    Unmapped is the common case and not a fault: most of a filing's concepts are
    line items no benchmark question asks about.  What matters is that nothing is
    mapped *approximately* -- a concept that is not in the table is reported as
    unmapped rather than matched to the nearest name.
    """

    concept = str(source_concept or "").strip()
    if not concept:
        return Alignment(None, concept, "UNMAPPED", "empty concept")
    for canonical, candidates in CONCEPT_ALIGNMENT.items():
        if concept in candidates:
            return Alignment(
                canonical, concept, "ALIGNED",
                f"{canonical} source #{candidates.index(concept) + 1} of "
                f"{len(candidates)}",
            )
    return Alignment(None, concept, "UNMAPPED", "not in the alignment table")


def resolve(facts: list[dict[str, Any]], canonical: str) -> dict[str, Any] | None:
    """Pick the company-level fact for a canonical quantity.

    Company level means an **undimensioned context** at the latest period the
    candidates offer -- the rule the JPM probe established.  Candidates are tried
    in the alignment's order, so a filer tagging both `ProfitLoss` and
    `NetIncomeLoss` yields the consolidated figure rather than whichever the
    document happened to list first.
    """

    candidates = CONCEPT_ALIGNMENT.get(canonical)
    if not candidates:
        return None
    for concept in candidates:
        pool = [
            f for f in facts
            if f.get("concept") == concept and not f.get("dimensions")
        ]
        if not pool:
            continue
        latest = max(str(f.get("period_end") or "") for f in pool)
        at_latest = [f for f in pool if str(f.get("period_end") or "") == latest]
        return {
            "canonical": canonical,
            "source_concept": concept,
            "value": at_latest[0].get("value"),
            "period_end": latest,
            "candidates_at_period": len(at_latest),
        }
    return None
