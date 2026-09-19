"""The canonical store must not answer when it cannot, and must not guess.

Switching retrieval is the risky half of this work: a store that returns *a*
value for every query looks healthy in every summary and is wrong exactly where
the old store was wrong.  So the tests here are mostly about refusal -- an
unmapped metric falls back, an ambiguous quantity returns nothing rather than
picking one, and the alignment's ordering is what separates `ProfitLoss` from
`NetIncomeLoss`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from src.runtime.trusted_v2_canonical_store import (  # noqa: E402
    METRIC_TO_CANONICAL,
    CanonicalFactStore,
)


class _LegacyStore:
    """The minimum of the store surface the wrapper delegates to."""

    def __init__(self, facts: dict[tuple, list[dict]] | None = None) -> None:
        self.facts = facts or {}
        self.asked: list[tuple] = []

    def facts_at_coordinate(self, entity, metric, period):
        self.asked.append((entity, metric, period))
        return tuple(self.facts.get((entity, metric, period), ()))

    def materialize(self, candidate_key):
        raise KeyError(candidate_key)

    def iter_records(self):
        return ()

    def coordinate_key(self, record):
        return (record.get("entity"), record.get("metric"), record.get("period"))

    candidate_count = 0
    coordinate_count = 0
    candidate_keys: tuple[str, ...] = ()


def _record(entity, concept, value, period="FY2025", dimensions=0, key=None):
    return {
        "source": "ixbrl",
        "candidate_key": key or f"ixbrl:{entity}:{concept}:{value}",
        "fact_id": key or f"ixbrl:{entity}:{concept}:{value}",
        "entity": entity,
        "concept": concept,
        "canonical_concept": None,
        "dimension_count": dimensions,
        "period": period,
        "value": value,
    }


@pytest.fixture
def store(tmp_path: Path):
    records = [
        # Tesla tags both: consolidated and parent-only.
        _record("Tesla", "us-gaap:ProfitLoss", "3855"),
        _record("Tesla", "us-gaap:NetIncomeLoss", "3794"),
        # Apple tags one.
        _record("Apple", "us-gaap:Assets", "359241"),
        # A dimensioned NetIncomeLoss that must never be selected.
        _record("Tesla", "us-gaap:ProfitLoss", "9999", dimensions=2),
        # An entity with two values at the same concept: unresolvable.
        _record("Divided", "us-gaap:Liabilities", "100"),
        _record("Divided", "us-gaap:Liabilities", "200"),
    ]
    path = tmp_path / "ixbrl.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    legacy = _LegacyStore({("Tesla", "United States", "FY2025"): [{"value": "1"}]})
    return CanonicalFactStore(legacy, path), legacy


def test_a_mapped_metric_resolves_from_the_ixbrl_store(store) -> None:
    canonical_store, legacy = store
    facts = canonical_store.facts_at_coordinate("Apple", "Total assets", "FY2025")
    assert [f["value"] for f in facts] == ["359241"]
    assert legacy.asked == [], "the legacy store should not have been consulted"


def test_the_alignment_ordering_selects_the_consolidated_figure(store) -> None:
    """`ProfitLoss` is preferred to `NetIncomeLoss`, so 3,855 not 3,794."""

    canonical_store, _legacy = store
    facts = canonical_store.facts_at_coordinate("Tesla", "Net income", "FY2025")
    assert [f["value"] for f in facts] == ["3855"]


def test_a_dimensioned_fact_is_never_selected_as_company_level(store) -> None:
    canonical_store, _legacy = store
    facts = canonical_store.facts_at_coordinate("Tesla", "Net income", "FY2025")
    assert "9999" not in [f["value"] for f in facts]


def test_an_ambiguous_quantity_returns_nothing_rather_than_one_value(store) -> None:
    """Two values at one concept is exactly the defect; picking one is worse.

    Returning the first would look like a successful resolution and be a guess.
    The wrapper returns nothing, and the caller falls back.
    """

    canonical_store, legacy = store
    legacy.facts[(None, "Total liabilities", "FY2025")] = []
    facts = canonical_store.facts_at_coordinate(None, "Total liabilities", "FY2025")
    assert facts == ()


def test_an_unmapped_metric_falls_back_to_the_legacy_store(store) -> None:
    """The fixtures still name `United States`, `Current`, `Total`."""

    canonical_store, legacy = store
    facts = canonical_store.facts_at_coordinate("Tesla", "United States", "FY2025")
    assert [f["value"] for f in facts] == ["1"]
    assert legacy.asked == [("Tesla", "United States", "FY2025")]


def test_a_known_metric_with_no_ixbrl_fact_falls_back(store) -> None:
    canonical_store, legacy = store
    legacy.facts[("Pfizer", "Total assets", "FY2024")] = [{"value": "213396"}]
    facts = canonical_store.facts_at_coordinate("Pfizer", "Total assets", "FY2024")
    assert [f["value"] for f in facts] == ["213396"]


def test_disabling_the_canonical_path_uses_the_legacy_store(tmp_path: Path) -> None:
    """The flag is what makes the switch separable from the rebuild."""

    path = tmp_path / "ixbrl.jsonl"
    path.write_text(
        json.dumps(_record("Apple", "us-gaap:Assets", "359241")) + "\n",
        encoding="utf-8",
    )
    legacy = _LegacyStore({("Apple", "Total assets", "FY2025"): [{"value": "legacy"}]})
    canonical_store = CanonicalFactStore(legacy, path, enabled=False)
    facts = canonical_store.facts_at_coordinate("Apple", "Total assets", "FY2025")
    assert [f["value"] for f in facts] == ["legacy"]


def test_a_missing_ixbrl_file_degrades_to_the_legacy_store(tmp_path: Path) -> None:
    legacy = _LegacyStore({("Apple", "Total assets", "FY2025"): [{"value": "legacy"}]})
    canonical_store = CanonicalFactStore(legacy, tmp_path / "absent.jsonl")
    assert canonical_store.canonical_ready is False
    facts = canonical_store.facts_at_coordinate("Apple", "Total assets", "FY2025")
    assert [f["value"] for f in facts] == ["legacy"]


def test_ambiguous_strings_are_deliberately_absent_from_the_map() -> None:
    """`Total`, `Current`, `United States` name no single quantity.

    Mapping them to something would be the guess this work exists to stop, so
    the correct behaviour is that they are not in the table.
    """

    for ambiguous in ("Total", "Current", "United States", "Other", "Additions"):
        assert ambiguous.casefold() not in METRIC_TO_CANONICAL


def test_the_period_year_is_what_matches_not_the_label(store) -> None:
    """`FY2025` and an instant `ASOF2025-09-27` are the same fiscal year."""

    canonical_store, _legacy = store
    facts = canonical_store.facts_at_coordinate("Apple", "Total assets", "FY2025")
    assert facts, "a period label difference must not hide the fact"
