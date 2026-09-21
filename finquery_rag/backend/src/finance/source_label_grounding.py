"""Which row labels a filing can testify to, and which it cannot.

The gate's vocabulary is an ontology of financial *concepts*, and a filing
states *rows*.  A row is often not a concept: ``iPad``, ``Xtandi`` and
``Colette M. Kress`` are answer targets no financial ontology should carry, and
refusing to name them refused every question that asks for one -- 44 of the 48
cases the semantic gate blocked.

The source can testify to something narrower than meaning and stronger than a
string match: that a label it prints resolves to **one value**, for that filer
and that period.  ``iPad`` is one number in Apple's FY2025 net sales table;
``Deferred`` is three numbers in one Apple balance sheet and no reading of the
row label alone says which.  That is the same invariant the calculator already
enforces on a bound operand (``src.finance.operand_ambiguity``), asked one layer
earlier, where answering the question is still possible.

So determinacy is the entire admission rule.  There is no list of labels here,
nothing keyed by case or by question text, and no fallback to a fuzzy match: a
label is grounded exactly when the store holds it at a coordinate that
identifies one value, and everything else stays unnamed and stays refused.

Ambiguity is not a defect to be worked around.  ``Deferred`` resolving to three
values is the store telling the truth about a row label that does not identify a
quantity, and the honest response is the one the gate already gives.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rag_v2.supervisor.semantic_alignment import (
    canonical_metric_id,
    canonical_period_id,
    metric_identity,
)

from src.finance.operand_ambiguity import CoordinateStatus, coordinate_status


class SourceLabelGrounding:
    """Resolve a plan's unnameable row labels against the authoritative store.

    Callable so it can be handed to the coordinator as one function: given the
    plan, return the labels it may name, mapped to the identity each resolves
    to.  Only labels the *ontology could not name* are returned -- a label the
    ontology knows keeps the ontology's identity, because a curated equivalence
    is a better answer than a string and should not be shadowed by one.
    """

    def __init__(self, fact_store: Any) -> None:
        self._store = fact_store

    def _grounded_surface(self, slot: Any) -> str | None:
        metric = getattr(slot, "metric", None)
        if not metric or canonical_metric_id(metric) is not None:
            return None
        # The plan states the period the way a person wrote it -- `fiscal year
        # 2025` -- while the store files it as `FY2025`.  Comparing the raw
        # strings would find no facts and read that as "the source is silent"
        # rather than as "the question was spelled differently".
        period = canonical_period_id(getattr(slot, "period", None))
        entity = getattr(slot, "entity", None)
        if entity:
            facts = self._store.facts_at_coordinate(entity, metric, period)
        else:
            # A slot may carry no entity, and looking that up as the empty one
            # finds nothing -- which reads as *the source has no such row* when
            # the source has it for exactly one filer.  Asking across filers and
            # then requiring determinacy keeps the rule honest: two companies
            # reporting the same label differently is still not grounded.
            facts = self._store.facts_for_label(metric, period)
        if not facts:
            return None
        if coordinate_status(facts[0], facts) is CoordinateStatus.CONFLICTING_VALUES:
            return None
        return str(metric)

    def __call__(self, plan: Any) -> Mapping[str, str]:
        grounded: dict[str, str] = {}
        for slot in getattr(plan, "required_slots", ()) or ():
            surface = self._grounded_surface(slot)
            if surface is None:
                continue
            identity = metric_identity(surface)
            if identity:
                grounded[surface] = identity
        return grounded


__all__ = ["SourceLabelGrounding"]
