"""Query the store by canonical quantity, falling back to the stored metric.

The rebuilt iXBRL store can answer "Apple's FY2025 net income" with one value,
because the filings say so through a standard concept and a context that carries
the scope.  The store being replaced cannot: its coordinate holds whatever the
breadcrumb happened to flatten together, which is why the operand guard has to
refuse 18.2% of it.

This wraps both.  A slot's metric is mapped to a canonical quantity -- `Net
income` to `net_income` -- and resolved through `concept_alignment`, which
applies the candidate ordering and selects the undimensioned context.  Anything
that does not map, or that the iXBRL store cannot answer, goes to the legacy
store exactly as before.

**The fallback is not a courtesy.**  The fixtures currently name metrics the old
store was built around (`United States`, `Current`, `Total`), and those have no
canonical quantity.  Without the fallback, switching would break every case whose
fixture has not been re-derived yet -- so the switch can be made and measured
before the fixtures move, which keeps step 2 and step 3 separable.

**Ordering at query time is load-bearing.**  The store preserves the source
concept; `canonical_concept` alone collapses `ProfitLoss` and `NetIncomeLoss`
into `net_income` and cannot say which wins, which leaves Tesla holding both
3,794 and 3,855.  Resolution goes through `CONCEPT_ALIGNMENT` in order, never
through the canonical field alone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from src.finance.concept_alignment import CONCEPT_ALIGNMENT

#: slot metric -> canonical quantity.  Only names that unambiguously denote one
#: quantity are listed.  `Total`, `Current` and `United States` are deliberately
#: absent: they are the strings whose ambiguity this work exists to remove, and
#: mapping them to something would be a guess.
METRIC_TO_CANONICAL: dict[str, str] = {
    "net income": "net_income",
    "consolidated net income": "net_income",
    "total assets": "total_assets",
    "total liabilities": "total_liabilities",
    "operating income": "operating_income",
    "income from operations": "operating_income",
    "research and development": "research_and_development",
    "diluted earnings per share": "diluted_eps",
    "comprehensive income": "comprehensive_income",
    "interest expense": "interest_expense",
    "long-term debt": "long_term_debt",
    "total net revenue": "revenues",
}


def _fold(text: object) -> str:
    return " ".join(str(text or "").split()).casefold()


def _period_year(period: object) -> str:
    match = re.search(r"(\d{4})", str(period or ""))
    return match.group(1) if match else ""


class CanonicalFactStore:
    """A store that answers canonical quantities, and otherwise does not change.

    Implements the `StructuredFactStore` surface the runtime uses, so it can be
    substituted wherever that object is passed.
    """

    def __init__(
        self,
        legacy: Any,
        ixbrl_path: Path | str,
        *,
        enabled: bool = True,
    ) -> None:
        self.legacy = legacy
        self.path = Path(ixbrl_path)
        self.enabled = bool(enabled)
        self._records: list[dict] = []
        # (entity, concept, period) -> distinct values at company level.
        self._company_level: dict[tuple, set] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            self._records.append(record)
            if record.get("dimension_count") != 0:
                continue
            key = (
                str(record.get("entity")),
                str(record.get("concept")),
                str(record.get("period")),
            )
            self._company_level.setdefault(key, set()).add(str(record.get("value")))

    # --- the canonical path ---------------------------------------------------

    @property
    def canonical_ready(self) -> bool:
        return bool(self._records)

    def canonical_quantity(self, metric: object) -> str | None:
        return METRIC_TO_CANONICAL.get(_fold(metric))

    def resolve_canonical(
        self, entity: object, canonical: str, period: object
    ) -> list[dict]:
        """Company-level facts for a canonical quantity, in alignment order.

        Returns at most one logical fact -- but **only if exactly one value is
        present**.  More than one means the quantity is still ambiguous for this
        entity and the honest answer is to return nothing, letting the caller
        fall back, rather than to pick one and look certain.
        """

        if not self.enabled or not self._records:
            return []
        year = _period_year(period)
        for concept in CONCEPT_ALIGNMENT.get(canonical, ()):
            values = {
                value
                for (e, c, p), found in self._company_level.items()
                if e == str(entity) and c == concept and _period_year(p) == year
                for value in found
            }
            if not values:
                continue
            if len(values) != 1:
                return []
            value = next(iter(values))
            matching = [
                r for r in self._records
                if str(r.get("entity")) == str(entity)
                and r.get("concept") == concept
                and r.get("dimension_count") == 0
                and _period_year(r.get("period")) == year
                and str(r.get("value")) == value
            ]
            return matching[:1]
        return []

    # --- the store surface ----------------------------------------------------

    def facts_at_coordinate(
        self, entity: Any, metric: Any, period: Any
    ) -> tuple[Mapping[str, Any], ...]:
        canonical = self.canonical_quantity(metric)
        if canonical:
            resolved = self.resolve_canonical(entity, canonical, period)
            if resolved:
                return tuple(self._as_fact(record) for record in resolved)
        return self.legacy.facts_at_coordinate(entity, metric, period)

    def _as_fact(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Shape an iXBRL record like a store fact, so callers do not branch."""

        return {
            **record,
            "evidence_id": record.get("fact_id"),
            "citation_id": None,
            "metric": record.get("canonical_concept") or record.get("concept"),
            "normalized_metric": record.get("concept"),
            "source_text": record.get("raw_value"),
            "parsed_numeric_value": record.get("value"),
        }

    def materialize(self, candidate_key: str) -> Mapping[str, Any]:
        for record in self._records:
            if record.get("candidate_key") == candidate_key:
                return self._as_fact(record)
        return self.legacy.materialize(candidate_key)

    def iter_records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.legacy.iter_records())

    def coordinate_key(self, record: Mapping[str, Any]) -> tuple[str, str, str]:
        if record.get("source") == "ixbrl":
            return (
                str(record.get("entity")),
                str(record.get("canonical_concept") or record.get("concept")),
                str(record.get("period")),
            )
        return self.legacy.coordinate_key(record)

    @property
    def candidate_count(self) -> int:
        return self.legacy.candidate_count

    @property
    def coordinate_count(self) -> int:
        return self.legacy.coordinate_count

    @property
    def candidate_keys(self) -> tuple[str, ...]:
        return self.legacy.candidate_keys
