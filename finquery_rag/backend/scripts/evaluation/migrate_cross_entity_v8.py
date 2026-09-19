"""P1.6-0H: write the re-derived cross-entity stratum into the benchmark.

One batch.  The specification is `docs/evaluation/p1-6-0g-rederivation.md`; this
applies it and records what it changed.

Three files move together and must not drift apart:

* `plan-fixtures-v7 -> v8` -- the per-slot metric, which is what retrieval
  matches on
* `gold-evidence-v1` -- the values, the expected outcome and the `fact_ids`
* `canonical-eval-v1` -- the question text, which names the metric

The values are **not** taken from the store.  Every one was read out of a filing
with its page recorded in the specification, because the store's coordinate is
what P1.6-A has to repair and a value taken from it inherits the defect.  The
store is used only to resolve `fact_id`, and where the new value's coordinate is
ambiguous the resolver says so rather than picking one -- `fact_ids` is
provenance, and a wrong one is worse than a flagged one.

  python migrate_cross_entity_v8.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

BENCH = _BACKEND_DIR / "benchmarks/tv2_canonical_v1"
KO = "The Coca-Cola Company"
JPM = "JPMorganChase"

#: case_id -> (metric, {entity: value}, expected outcome keys).
#: Every value below was read from the filing page named in the specification.
SPEC: dict[str, dict] = {
    "tv2f01-s3-compare-001": {
        "metric": "Net income",
        "values": {KO: "13,137", "Tesla": "3,855"},
        "expected_higher": KO,
    },
    "tv2f01-s3-compare-002": {
        "metric": "Net income",
        "values": {"Apple": "112,010", "Visa": "20,058"},
        "expected_higher": "Apple",
    },
    "tv2f01-s3-compare-003": {
        "metric": "Total assets",
        "values": {"Tesla": "137,806", KO: "104,816"},
        "expected_higher": "Tesla",
    },
    "tv2f01-s3-compare-004": {
        "metric": "Net income",
        "values": {"Apple": "112,010", "Microsoft": "101,832"},
        "expected_higher": "Apple",
    },
    "tv2f01-s3-compare-005": {
        "metric": "Diluted earnings per share",
        "values": {JPM: "20.02", "Apple": "$ 7.46"},
        "expected_higher": JPM,
    },
    "tv2f01-s3-compare-006": {
        "metric": "Comprehensive income",
        "values": {JPM: "$ 65,214", "Tesla": "4,886"},
        "expected_higher": JPM,
    },
    "tv2f01-s3-compare-007": {
        "metric": "Net income",
        "values": {JPM: "$ 57,048", "Pfizer": "8,062"},
        "expected_higher": JPM,
        # The corpus holds one FY2024 filing and it is Pfizer's.  Comparing it
        # against a FY2025 filer is legitimate only if the case says so.
        "fiscal_year_by_entity": {JPM: "FY2025", "Pfizer": "FY2024"},
    },
    "tv2f01-s3-compare-008": {
        "metric": "Diluted earnings per share",
        "values": {JPM: "20.02", "Apple": "$ 7.46"},
        "expected_higher": JPM,
    },
    "tv2f01-s3-compare-009": {
        "metric": "Net income",
        "values": {"Apple": "112,010", JPM: "$ 57,048"},
        "expected_higher": "Apple",
    },
    "tv2f01-s3-compare-010": {
        "metric": "Operating income",
        "values": {"Apple": "133,050", "Microsoft": "128,528"},
        "expected_higher": "Apple",
    },
    "tv2f01-s3-crossdiff-001": {
        "metric": "Net income",
        "values": {"NVIDIA": "$ 72,880", "Tesla": "3,855"},
        "expected_value": "69025",
    },
    "tv2f01-s3-crossdiff-002": {
        "metric": "Net income",
        "values": {"Apple": "112,010", "Microsoft": "101,832"},
        "expected_value": "10178",
    },
    "tv2f01-s3-crossdiff-003": {
        "metric": "Total liabilities",
        "values": {"Apple": "285,508", "Tesla": "54,941"},
        "expected_value": "230567",
    },
    "tv2f01-s3-crossdiff-004": {
        "metric": "Long-term debt",
        "values": {KO: "42,119", "Visa": "19,602"},
        "expected_value": "22517",
    },
    "tv2f01-s3-crossdiff-005": {
        "metric": "Total assets",
        "values": {"NVIDIA": "$ 111,601", KO: "104,816"},
        "expected_value": "6785",
    },
    "tv2f01-s3-rank-001": {
        "metric": "Net income",
        "values": {"Tesla": "3,855", KO: "13,137", "Visa": "20,058"},
        "expected_ranking": ["Visa", KO, "Tesla"],
    },
    "tv2f01-s3-rank-002": {
        "metric": "Research and development",
        "values": {"Apple": "34,550", "Microsoft": "32,488", "Tesla": "$ 6,411"},
        "expected_ranking": ["Apple", "Microsoft", "Tesla"],
    },
    "tv2f01-s3-rank-003": {
        "metric": "Comprehensive income",
        "values": {
            JPM: "$ 65,214", "Microsoft": "$ 104,075",
            "Tesla": "4,886", "Visa": "$ 20,614",
        },
        "expected_ranking": ["Microsoft", JPM, "Visa", "Tesla"],
    },
    "tv2f01-s3-rank-004": {
        "metric": "Interest expense",
        "values": {JPM: "97,898", KO: "1,654", "Visa": "(589)", "Microsoft": "(2,385)"},
        "expected_ranking": [JPM, KO, "Visa", "Microsoft"],
    },
    "tv2f01-s3-rank-005": {
        "metric": "Operating income",
        "values": {"Apple": "133,050", "Microsoft": "128,528", KO: "13,762"},
        "expected_ranking": ["Apple", "Microsoft", KO],
    },
}

_QUESTION_TEMPLATES = {
    "comparison": "Which company had a higher {metric} in FY2025, {a} or {b}?",
    # The plan calls it `difference`; the gold calls the same operation
    # `cross_entity_difference`.  Both names are in the corpus.
    "difference": (
        "What is the difference in {metric} between {a} and {b} in FY2025?"
    ),
    "cross_entity_difference": (
        "What is the difference in {metric} between {a} and {b} in FY2025?"
    ),
    "ranking": (
        "Rank the following companies by {metric} in FY2025 from highest to "
        "lowest: {entities}."
    ),
}


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _dump(rows: list[dict]) -> str:
    return (
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
        + "\n"
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _numeric(text: object) -> str | None:
    import re

    match = re.search(r"\(?\$?\s*-?[\d][\d,\.]*\)?", str(text or ""))
    if not match:
        return None
    digits = re.sub(r"[^\d]", "", match.group(0))
    return digits or None


def _resolve_fact(store, entity: str, metric: str, period: str, value: str) -> dict:
    """The store candidate whose value matches, or why one could not be chosen."""

    facts = store.facts_at_coordinate(entity, metric, period)
    wanted = _numeric(value)
    matches = [
        f for f in facts
        if _numeric(f.get("value")) == wanted
        and (f.get("value") or "").strip().startswith("(") == value.strip().startswith("(")
    ]
    if len(matches) == 1:
        return {"fact_id": matches[0].get("fact_id"), "status": "resolved"}
    if not matches:
        return {"fact_id": None, "status": "NO_MATCHING_FACT",
                "coordinate_facts": len(facts)}
    return {"fact_id": None, "status": "AMBIGUOUS_SAME_VALUE",
            "candidates": [m.get("fact_id") for m in matches]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the v8 files; without it, report only")
    parser.add_argument("--fact-store", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)

    import os

    from src.runtime.trusted_v2_production import StructuredFactStore

    store_path = args.fact_store or Path(os.environ["TRUSTED_V2_FACT_STORE_PATH"])
    store = StructuredFactStore(store_path)

    gold_rows = _load(BENCH / "gold-evidence-v1.jsonl")
    eval_rows = _load(BENCH / "canonical-eval-v1.jsonl")
    plan_rows = _load(BENCH / "plan-fixtures-v7.jsonl")

    gold_by_id = {row["id"]: row for row in gold_rows}
    eval_by_id = {row["id"]: row for row in eval_rows}
    plan_by_id = {row["id"]: row for row in plan_rows}

    before = {
        "gold_evidence": _sha256((BENCH / "gold-evidence-v1.jsonl").read_text(encoding="utf-8")),
        "canonical_eval": _sha256((BENCH / "canonical-eval-v1.jsonl").read_text(encoding="utf-8")),
        "plan_fixtures_v7": _sha256((BENCH / "plan-fixtures-v7.jsonl").read_text(encoding="utf-8")),
    }

    changes: list[dict] = []
    fact_status: dict[str, int] = {}

    for case_id, spec in sorted(SPEC.items()):
        gold = gold_by_id[case_id]
        evaluation = eval_by_id[case_id]
        plan = plan_by_id[case_id]
        slot_period = str(plan["plan"]["required_slots"][0]["period"])
        operation = str(plan["plan"]["operation"])

        record = {
            "case_id": case_id,
            "old_metric": gold.get("metric"),
            "new_metric": spec["metric"],
            "old_values": gold.get("values"),
            "new_values": spec["values"],
            "fact_ids": {},
        }

        # --- gold: values, expectation, and provenance ------------------------
        gold["metric"] = spec["metric"]
        gold["values"] = dict(spec["values"])
        for key in ("expected_higher", "expected_ranking", "expected_value"):
            if key in spec:
                gold[key] = spec[key]
            elif key in gold and key != "expected_value":
                gold.pop(key, None)
        if "expected_value" in gold and "expected_value" not in spec:
            gold["expected_value"] = None
        if "fiscal_year_by_entity" in spec:
            gold["fiscal_year_by_entity"] = spec["fiscal_year_by_entity"]

        fact_ids = []
        for entity, value in spec["values"].items():
            resolved = _resolve_fact(store, entity, spec["metric"], slot_period, value)
            fact_status[resolved["status"]] = fact_status.get(resolved["status"], 0) + 1
            record["fact_ids"][entity] = resolved
            if resolved["fact_id"]:
                fact_ids.append(resolved["fact_id"])
        gold["fact_ids"] = fact_ids

        # --- plan: the metric every slot retrieves on --------------------------
        for slot in plan["plan"]["required_slots"]:
            slot["metric"] = spec["metric"]

        # --- eval: the question names the metric -------------------------------
        entities = list(spec["values"])
        if operation == "ranking":
            question = _QUESTION_TEMPLATES["ranking"].format(
                metric=spec["metric"], entities=", ".join(entities)
            )
        else:
            question = _QUESTION_TEMPLATES[operation].format(
                metric=spec["metric"], a=entities[0], b=entities[1]
            )
        record["old_question"] = evaluation.get("question")
        record["new_question"] = question
        evaluation["question"] = question
        if "fiscal_year_by_entity" in spec:
            evaluation["fiscal_year_by_entity"] = spec["fiscal_year_by_entity"]
        changes.append(record)

    v8_text = _dump(plan_rows)
    gold_text = _dump(gold_rows)
    eval_text = _dump(eval_rows)

    after = {
        "gold_evidence": _sha256(gold_text),
        "canonical_eval": _sha256(eval_text),
        "plan_fixtures_v8": _sha256(v8_text),
    }

    print(f"=== P1.6-0H migration {'APPLIED' if args.apply else 'DRY RUN'} ===")
    print(f"  cases changed: {len(changes)}")
    print(f"  fact_id resolution: {fact_status}")
    print()
    for record in changes:
        print(f"  {record['case_id']}")
        print(f"      {record['old_metric']!r} -> {record['new_metric']!r}")
        for entity, resolved in record["fact_ids"].items():
            print(f"      {entity[:20]:20} {resolved['status']}")
    print()
    print(f"  gold   {before['gold_evidence'][:12]} -> {after['gold_evidence'][:12]}")
    print(f"  eval   {before['canonical_eval'][:12]} -> {after['canonical_eval'][:12]}")
    print(f"  plan   {before['plan_fixtures_v7'][:12]} -> {after['plan_fixtures_v8'][:12]}")

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps({"phase": "P1.6-0H", "applied": bool(args.apply),
                        "before": before, "after": after,
                        "fact_id_status": fact_status,
                        "changes": changes},
                       ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if not args.apply:
        print("\n  (dry run -- pass --apply to write)")
        return 0

    (BENCH / "plan-fixtures-v8.jsonl").write_text(v8_text, encoding="utf-8")
    (BENCH / "plan-fixtures-v8.jsonl.sha256").write_text(after["plan_fixtures_v8"] + "\n", encoding="utf-8")
    (BENCH / "gold-evidence-v1.jsonl").write_text(gold_text, encoding="utf-8")
    (BENCH / "canonical-eval-v1.jsonl").write_text(eval_text, encoding="utf-8")
    (BENCH / "canonical-eval-v1.jsonl.sha256").write_text(after["canonical_eval"] + "\n", encoding="utf-8")

    old_manifest = json.loads((BENCH / "plan-fixtures-v7.manifest.json").read_text(encoding="utf-8"))
    manifest = dict(old_manifest)
    manifest["fixture"] = "plan-fixtures-v8"
    manifest["fixture_sha256"] = after["plan_fixtures_v8"]
    manifest["gold_sha256"] = after["gold_evidence"]
    manifest["eval_set_sha256"] = after["canonical_eval"]
    manifest["stage"] = "P1-6-0H-REDERIVED-CROSS-ENTITY"
    manifest["cross_entity_rederivation"] = {
        "spec": "docs/evaluation/p1-6-0g-rederivation.md",
        "cases_changed": sorted(SPEC),
        "previous": before,
        "fact_id_status": fact_status,
    }
    (BENCH / "plan-fixtures-v8.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n  wrote plan-fixtures-v8.jsonl (+ .sha256, .manifest.json), "
          "gold-evidence-v1.jsonl, canonical-eval-v1.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
