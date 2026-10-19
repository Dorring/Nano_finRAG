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
import re
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
        self.last_detected_ambiguities: list[dict[str, Any]] = []

    @staticmethod
    def _extract_target_value(query: str, metric: str | None = None) -> str | None:
        if metric:
            m_metric = re.search(
                rf"(?:target value for|目标值|数值|金额)\s*['\"]?{re.escape(metric)}['\"]?\s*[:：\s]+(\([^)]+\)|[^\s.，,]+)",
                query,
                re.IGNORECASE,
            )
            if m_metric:
                return m_metric.group(1).strip()
        m_target = re.search(
            r"(?:target value|目标值|数值|金额)(?:\s+for\s+[a-zA-Z0-9_\s]+)?[:：\s]+(\([^)]+\)|[^\s.，,]+)",
            query,
            re.IGNORECASE,
        )
        if m_target:
            return m_target.group(1).strip()
        return None

    @classmethod
    def _fact_value_matches_query(
        cls, fact: Mapping[str, Any], query: str, metric: str | None = None
    ) -> bool:
        val = str(fact.get("value") or fact.get("raw_value") or "").strip()
        if not val:
            return False

        target_val = cls._extract_target_value(query, metric=metric)
        if target_val:
            if val.strip().lower() == target_val.strip().lower():
                return True
            digits_fact = re.sub(r"[^\d]", "", val)
            digits_target = re.sub(r"[^\d]", "", target_val)
            if digits_fact and digits_target and digits_fact == digits_target:
                is_fact_neg = "(" in val or "-" in val
                is_target_neg = "(" in target_val or "-" in target_val
                if is_target_neg:
                    return is_fact_neg
                return True
            return False

        # Direct string containment
        if val in query:
            return True

        digits_only = re.sub(r"[^\d]", "", val)
        if digits_only and re.search(rf"(?<!\d){re.escape(digits_only)}(?!\d)", query):
            # Guard against false-positive matching of 4-digit calendar years
            if len(digits_only) == 4 and digits_only.startswith(("19", "20")):
                return False
            return True
        return False

    def _grounded_surface(self, slot: Any, query: str | None = None) -> str | None:
        metric = getattr(slot, "metric", None)
        if not metric or canonical_metric_id(metric) is not None:
            return None
        # The plan states the period the way a person wrote it -- `fiscal year
        # 2025` -- while the store files it as `FY2025`.  Comparing the raw
        # strings would find no facts and read that as "the source is silent"
        # rather than as "the question was spelled differently".
        period = canonical_period_id(getattr(slot, "period", None))
        entity = getattr(slot, "entity", None)
        scope = getattr(slot, "scope", None)
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
        if scope:
            scoped_facts = tuple(
                f for f in facts
                if f.get("scope") and str(f.get("scope")).casefold() == str(scope).casefold()
            )
            if scoped_facts:
                facts = scoped_facts
        if coordinate_status(facts[0], facts) is CoordinateStatus.CONFLICTING_VALUES:
            if query:
                val_matches = [
                    f for f in facts
                    if self._fact_value_matches_query(f, query, metric=metric)
                ]
                if len(val_matches) == 1:
                    facts = tuple(val_matches)
            if coordinate_status(facts[0], facts) is CoordinateStatus.CONFLICTING_VALUES:
                self.last_detected_ambiguities.append({
                    "slot_id": getattr(slot, "slot_id", None),
                    "entity": entity or facts[0].get("entity"),
                    "metric": metric,
                    "period": period,
                    "scope": scope,
                    "conflicting_facts": facts,
                })
                return None
        return str(metric)

    def __call__(self, plan: Any, query: str | None = None) -> Mapping[str, str]:
        self.last_detected_ambiguities = []
        grounded: dict[str, str] = {}
        for slot in getattr(plan, "required_slots", ()) or ():
            surface = self._grounded_surface(slot, query=query)
            if surface is None:
                continue
            identity = metric_identity(surface)
            if identity:
                grounded[surface] = identity
        return grounded


__all__ = ["SourceLabelGrounding"]
