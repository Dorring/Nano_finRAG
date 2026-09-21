#!/usr/bin/env python3
"""C0.3: is a fixture's stale recorded question cosmetic, or does it lie about the plan?

C5 found rows whose fixture `question` is not the benchmark's canonical question.
Those are two different things and only one of them is dangerous:

  METADATA_ONLY_DRIFT   the fixture's *slots* still say what the canonical
                        question asks; only the recorded sentence is a fossil of
                        an earlier wording
  SEMANTIC_SLOT_DRIFT   the slots disagree with the canonical question on entity,
                        metric, period or operand order -- the fixture would
                        answer a question that is not being asked

The distinction is read from the plan, never from the sentence: the canonical
question is parsed for its metric, its companies and its period, and those are
compared with the fixture's `required_slots` as they now stand.

**Order is semantic only where the operation makes it so.** `difference` has a
minuend and a subtrahend, so the order of the two mentions decides the sign and
must match. `comparison` asks which company is higher -- a set question, and the
order the question happens to enumerate them in is not part of what is asked.
`ranking` enumerates the companies to rank; the answer is the ordering, which
the gold carries separately.

    python scripts/evaluation/classify_p1_8_question_drift.py \
        --eval-set <canonical-eval-v1.jsonl> --fixture <plan-fixtures-v9.jsonl>

Exits non-zero if anything is UNCLASSIFIED, or if any SEMANTIC_SLOT_DRIFT
survives.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixture_integrity as integrity  # noqa: E402

METADATA_ONLY_DRIFT = "METADATA_ONLY_DRIFT"
SEMANTIC_SLOT_DRIFT = "SEMANTIC_SLOT_DRIFT"
UNCLASSIFIED = "UNCLASSIFIED"

#: The canonical templates, one per operation.  Parsing the *canonical* sentence
#: is enough: the fixture's sentence is the thing under test.
_COMPARISON_RE = re.compile(
    r"which company had a higher\s+(?P<metric>.+?)\s+in\s+(?P<period>FY\s?\d{4})\s*,\s*"
    r"(?P<a>.+?)\s+or\s+(?P<b>.+?)\s*\?",
    re.IGNORECASE,
)
_RANKING_RE = re.compile(
    r"rank the following companies by\s+(?P<metric>.+?)\s+in\s+(?P<period>FY\s?\d{4})\s+"
    r"from highest to lowest\s*:\s*(?P<entities>.+?)\.?\s*$",
    re.IGNORECASE,
)
_DIFFERENCE_RE = re.compile(
    r"difference\s+in\s+(?P<metric>.+?)\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)"
    r"\s+in\s+(?P<period>FY\s?\d{4})",
    re.IGNORECASE,
)

#: Operations for which the order of the mentions carries meaning.
_ORDER_SEMANTIC = frozenset({"difference", "cross_entity_difference"})


def _norm(text: object) -> str:
    return " ".join(str(text or "").strip().casefold().split())


def parse_canonical(question: str, operation: str) -> dict | None:
    """The canonical question's (metric, entities in mention order, period)."""

    if operation == "comparison":
        match = _COMPARISON_RE.search(question)
        if not match:
            return None
        entities = [match.group("a"), match.group("b")]
    elif operation == "ranking":
        match = _RANKING_RE.search(question)
        if not match:
            return None
        entities = [part.strip() for part in match.group("entities").split(",")]
    elif operation in _ORDER_SEMANTIC:
        match = _DIFFERENCE_RE.search(question)
        if not match:
            return None
        entities = [match.group("a"), match.group("b")]
    else:
        return None

    resolved = [integrity.resolve_mention(entity) for entity in entities]
    if any(entity is None for entity in resolved):
        return None
    return {
        "metric": match.group("metric").strip(),
        "entities": resolved,
        "period": _norm(match.group("period")).replace(" ", "").upper(),
    }


def classify(row: dict, canonical_question: str) -> dict:
    plan = row.get("plan") or {}
    operation = str(plan.get("operation") or "")
    slots = list(plan.get("required_slots") or [])
    parsed = parse_canonical(canonical_question, operation)

    if parsed is None:
        return {
            "id": row.get("id"),
            "classification": UNCLASSIFIED,
            "reason": f"canonical question for operation {operation!r} did not parse",
            "canonical_question": canonical_question,
        }

    slot_entities = [integrity.resolve_mention(s.get("entity")) for s in slots]
    slot_metrics = [_norm(s.get("metric")) for s in slots]
    slot_periods = [_norm(s.get("period")).replace(" ", "").upper() for s in slots]

    disagreements = []
    if set(slot_entities) != set(parsed["entities"]):
        disagreements.append(
            {"field": "entity_set", "fixture": slot_entities, "canonical": parsed["entities"]}
        )
    if any(metric != _norm(parsed["metric"]) for metric in slot_metrics):
        disagreements.append(
            {"field": "metric", "fixture": slot_metrics, "canonical": _norm(parsed["metric"])}
        )
    if any(period != parsed["period"] for period in slot_periods):
        disagreements.append(
            {"field": "period", "fixture": slot_periods, "canonical": parsed["period"]}
        )
    if operation in _ORDER_SEMANTIC and slot_entities[: len(parsed["entities"])] != parsed["entities"]:
        disagreements.append(
            {
                "field": "operand_order",
                "fixture": slot_entities,
                "canonical": parsed["entities"],
                "note": "order decides the sign for this operation",
            }
        )

    return {
        "id": row.get("id"),
        "classification": METADATA_ONLY_DRIFT if not disagreements else SEMANTIC_SLOT_DRIFT,
        "operation": operation,
        "canonical_question": canonical_question,
        "fixture_question": row.get("question"),
        "canonical_parsed": parsed,
        "slot_entities": slot_entities,
        "slot_metrics": slot_metrics,
        "slot_periods": slot_periods,
        "disagreements": disagreements,
        "order_is_semantic": operation in _ORDER_SEMANTIC,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    questions = integrity.questions_by_id(args.eval_set)
    rows = integrity.load_rows(args.fixture)

    drifted = [r for r in rows if str(r.get("question") or "") != questions.get(str(r.get("id")), "")]
    results = [classify(row, questions[str(row["id"])]) for row in drifted]

    counts: dict[str, int] = {}
    for result in results:
        counts[result["classification"]] = counts.get(result["classification"], 0) + 1

    print(f"drifted rows: {len(drifted)}")
    for name in (METADATA_ONLY_DRIFT, SEMANTIC_SLOT_DRIFT, UNCLASSIFIED):
        print(f"  {name:<22} {counts.get(name, 0)}")
    print()
    for result in results:
        if result["classification"] == METADATA_ONLY_DRIFT:
            continue
        print(f"  {result['classification']:<20} {result['id']}")
        print(f"      canonical : {result.get('canonical_question')}")
        for disagreement in result.get("disagreements", []):
            print(f"      {disagreement['field']}: fixture={disagreement['fixture']} canonical={disagreement['canonical']}")
        if result.get("reason"):
            print(f"      reason    : {result['reason']}")
    print()
    print("  METADATA_ONLY_DRIFT rows (recorded sentence is a fossil; slots agree):")
    for result in results:
        if result["classification"] == METADATA_ONLY_DRIFT:
            print(f"    {result['id']:<26} op={result['operation']:<12} {str(result['fixture_question'])[:60]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"counts": counts, "results": results},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nwritten to {args.out}")

    return 1 if (counts.get(UNCLASSIFIED, 0) or counts.get(SEMANTIC_SLOT_DRIFT, 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
