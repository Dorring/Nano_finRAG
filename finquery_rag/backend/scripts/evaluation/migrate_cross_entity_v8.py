"""P1.6-0H: write the re-derived cross-entity stratum into the benchmark (v7 -> v8).

One batch.  The specification is `docs/evaluation/p1-6-0g-rederivation.md`, with the
correction in `p1-6-0a10-expense-sign.md`; this applies it and records what it changed.

Three files move together and must not drift apart:

* `plan-fixtures-v7 -> v8` -- the per-slot metric, which is what retrieval matches on
* `gold-evidence-v1` -- the values, the expected outcome and the `fact_ids`
* `canonical-eval-v1` -- the question text, which names the metric

The values are **not** taken from a store.  Every one was read out of a filing and
verified against it (P1.6-A6, 31 of 31 pairs).  The store is used only to resolve
`fact_id`, and it is the rebuilt iXBRL store, because that is the one whose concept
carries the fact being asserted -- resolving against the old store would attach
provenance to whichever flattened row happened to hold the number.

  python migrate_cross_entity_v8.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

BENCH = _BACKEND_DIR / "benchmarks/tv2_canonical_v1"
IXBRL_STORE = Path(
    "/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl"
)
KO = "The Coca-Cola Company"
JPM = "JPMorganChase"

#: case -> metric, values, and the expectation.  Every value read from a filing.
SPEC: dict[str, dict] = {
    "tv2f01-s3-compare-001": {
        "metric": "Net income",
        "values": {KO: "13137", "Tesla": "3855"},
        "expected_higher": KO,
    },
    "tv2f01-s3-compare-002": {
        "metric": "Net income",
        "values": {"Apple": "112010", "Visa": "20058"},
        "expected_higher": "Apple",
    },
    "tv2f01-s3-compare-003": {
        "metric": "Total assets",
        "values": {"Tesla": "137806", KO: "104816"},
        "expected_higher": "Tesla",
    },
    "tv2f01-s3-compare-004": {
        "metric": "Net income",
        "values": {"Apple": "112010", "Microsoft": "101832"},
        "expected_higher": "Apple",
    },
    "tv2f01-s3-compare-005": {
        "metric": "Diluted earnings per share",
        "values": {JPM: "20.02", "Apple": "7.46"},
        "expected_higher": JPM,
    },
    "tv2f01-s3-compare-006": {
        "metric": "Comprehensive income",
        "values": {JPM: "65214", "Tesla": "4886"},
        "expected_higher": JPM,
    },
    "tv2f01-s3-compare-007": {
        "metric": "Net income",
        "values": {JPM: "57048", "Pfizer": "8062"},
        "expected_higher": JPM,
        # The corpus holds one FY2024 filing and it is Pfizer's.  The comparison
        # is legitimate only if the case says so.
        "fiscal_year_by_entity": {JPM: "FY2025", "Pfizer": "FY2024"},
    },
    "tv2f01-s3-compare-008": {
        "metric": "Diluted earnings per share",
        "values": {JPM: "20.02", "Apple": "7.46"},
        "expected_higher": JPM,
    },
    "tv2f01-s3-compare-009": {
        "metric": "Net income",
        "values": {"Apple": "112010", JPM: "57048"},
        "expected_higher": "Apple",
    },
    "tv2f01-s3-compare-010": {
        "metric": "Operating income",
        "values": {"Apple": "133050", "Microsoft": "128528"},
        "expected_higher": "Apple",
    },
    "tv2f01-s3-crossdiff-001": {
        "metric": "Net income",
        "values": {"NVIDIA": "72880", "Tesla": "3855"},
        "expected_value": "69025",
    },
    "tv2f01-s3-crossdiff-002": {
        "metric": "Net income",
        "values": {"Apple": "112010", "Microsoft": "101832"},
        "expected_value": "10178",
    },
    "tv2f01-s3-crossdiff-003": {
        "metric": "Total liabilities",
        "values": {"Apple": "285508", "Tesla": "54941"},
        "expected_value": "230567",
    },
    "tv2f01-s3-crossdiff-004": {
        "metric": "Long-term debt",
        "values": {KO: "42119", "Visa": "19602"},
        "expected_value": "22517",
    },
    "tv2f01-s3-crossdiff-005": {
        "metric": "Total assets",
        "values": {"NVIDIA": "111601", KO: "104816"},
        "expected_value": "6785",
    },
    "tv2f01-s3-rank-001": {
        "metric": "Net income",
        "values": {"Tesla": "3855", KO: "13137", "Visa": "20058"},
        "expected_ranking": ["Visa", KO, "Tesla"],
    },
    "tv2f01-s3-rank-002": {
        "metric": "Research and development",
        "values": {"Apple": "34550", "Microsoft": "32488", "Tesla": "6411"},
        "expected_ranking": ["Apple", "Microsoft", "Tesla"],
    },
    "tv2f01-s3-rank-003": {
        "metric": "Comprehensive income",
        "values": {JPM: "65214", "Microsoft": "104075", "Tesla": "4886",
                   "Visa": "20614"},
        "expected_ranking": ["Microsoft", JPM, "Visa", "Tesla"],
    },
    "tv2f01-s3-rank-004": {
        "metric": "Interest expense",
        "values": {JPM: "97898", KO: "1654", "Visa": "589", "Microsoft": "2385"},
        # Tagged values, not presented ones: US-GAAP expense concepts carry a
        # debit balance type, so all four are positive and the ranking is by
        # expense magnitude.  See p1-6-0a10-expense-sign.md.
        "expected_ranking": [JPM, "Microsoft", KO, "Visa"],
        "question_note": "largest first",
    },
    "tv2f01-s3-rank-005": {
        "metric": "Operating income",
        "values": {"Apple": "133050", "Microsoft": "128528", KO: "13762"},
        "expected_ranking": ["Apple", "Microsoft", KO],
    },
}

_QUESTION_TEMPLATES = {
    "comparison": "Which company had a higher {metric} in FY2025, {a} or {b}?",
    "difference": "What is the difference in {metric} between {a} and {b} in FY2025?",
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
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n"
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digits(text: object) -> str:
    cleaned = re.sub(r"[^\d.]", "", str(text or ""))
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned


def _ixbrl_index(path: Path) -> dict[tuple, list[str]]:
    """(entity, source concept, year, value) -> candidate keys at company level.

    Keyed on the **source** concept, not `canonical_concept`.  The resolution
    walks `CONCEPT_ALIGNMENT`'s candidate list -- `us-gaap:ProfitLoss` then
    `us-gaap:NetIncomeLoss` -- so the index has to answer for those strings;
    keying on the canonical name instead matches nothing and reports every
    fact_id as unresolvable.
    """

    index: dict[tuple, list[str]] = {}
    if not path.is_file():
        return index
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("dimension_count") != 0:
            continue
        year = re.search(r"(\d{4})", str(record.get("period") or ""))
        if not year:
            continue
        key = (
            str(record.get("entity")),
            str(record.get("concept")),
            year.group(1),
            _digits(record.get("value")),
        )
        index.setdefault(key, []).append(str(record.get("candidate_key")))
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--store", type=Path, default=IXBRL_STORE)
    args = parser.parse_args(argv)

    from src.finance.concept_alignment import CONCEPT_ALIGNMENT
    from src.runtime.trusted_v2_canonical_store import METRIC_TO_CANONICAL

    index = _ixbrl_index(args.store)
    if not index:
        print(f"iXBRL store not readable at {args.store}; cannot resolve fact_ids")
        return 1

    gold_rows = _load(BENCH / "gold-evidence-v1.jsonl")
    eval_rows = _load(BENCH / "canonical-eval-v1.jsonl")
    plan_rows = _load(BENCH / "plan-fixtures-v7.jsonl")
    gold_by_id = {r["id"]: r for r in gold_rows}
    eval_by_id = {r["id"]: r for r in eval_rows}
    plan_by_id = {r["id"]: r for r in plan_rows}

    before = {
        name: _sha256((BENCH / name).read_text(encoding="utf-8"))
        for name in ("gold-evidence-v1.jsonl", "canonical-eval-v1.jsonl",
                     "plan-fixtures-v7.jsonl")
    }

    changes: list[dict] = []
    fact_status: dict[str, int] = {}

    for case_id, spec in sorted(SPEC.items()):
        gold, evaluation, plan = gold_by_id[case_id], eval_by_id[case_id], plan_by_id[case_id]
        canonical = METRIC_TO_CANONICAL.get(spec["metric"].casefold())
        record = {
            "case_id": case_id,
            "old_metric": gold.get("metric"),
            "new_metric": spec["metric"],
            "old_values": gold.get("values"),
            "new_values": spec["values"],
            "old_expected": {k: gold.get(k) for k in
                             ("expected_higher", "expected_ranking", "expected_value")},
            "new_expected": {k: spec.get(k) for k in
                             ("expected_higher", "expected_ranking", "expected_value")},
            "fact_ids": {},
        }

        gold["metric"] = spec["metric"]
        gold["values"] = dict(spec["values"])
        for key in ("expected_higher", "expected_ranking", "expected_value"):
            gold[key] = spec.get(key)
        if "fiscal_year_by_entity" in spec:
            gold["fiscal_year_by_entity"] = spec["fiscal_year_by_entity"]

        fact_ids = []
        for entity, value in spec["values"].items():
            year = str(spec.get("fiscal_year_by_entity", {}).get(
                entity, "FY2025"))[-4:]
            keys: list[str] = []
            for concept in CONCEPT_ALIGNMENT.get(canonical or "", ()):
                keys = index.get((entity, concept, year, _digits(value)), [])
                if keys:
                    record["fact_ids"][entity] = {
                        "status": "resolved", "concept": concept, "fact_id": keys[0]}
                    fact_ids.append(keys[0])
                    break
            if not keys:
                record["fact_ids"][entity] = {"status": "NOT_FOUND",
                                              "canonical": canonical}
                fact_status["NOT_FOUND"] = fact_status.get("NOT_FOUND", 0) + 1
            else:
                fact_status["resolved"] = fact_status.get("resolved", 0) + 1
        gold["fact_ids"] = fact_ids

        for slot in plan["plan"]["required_slots"]:
            slot["metric"] = spec["metric"]

        entities = list(spec["values"])
        operation = str(plan["plan"]["operation"])
        if operation == "ranking":
            question = _QUESTION_TEMPLATES["ranking"].format(
                metric=spec["metric"], entities=", ".join(entities))
        else:
            question = _QUESTION_TEMPLATES[operation].format(
                metric=spec["metric"], a=entities[0], b=entities[1])
        record["old_question"] = evaluation.get("question")
        record["new_question"] = question
        evaluation["question"] = question
        if "fiscal_year_by_entity" in spec:
            evaluation["fiscal_year_by_entity"] = spec["fiscal_year_by_entity"]
        changes.append(record)

    v8_text, gold_text, eval_text = _dump(plan_rows), _dump(gold_rows), _dump(eval_rows)
    after = {
        "gold-evidence-v1.jsonl": _sha256(gold_text),
        "canonical-eval-v1.jsonl": _sha256(eval_text),
        "plan-fixtures-v8.jsonl": _sha256(v8_text),
    }

    print(f"=== P1.6-0H migration {'APPLIED' if args.apply else 'DRY RUN'} ===")
    print(f"  cases changed {len(changes)}   fact_ids {fact_status}")
    print()
    for record in changes:
        expected = {k: v for k, v in record["new_expected"].items() if v is not None}
        if expected != {k: v for k, v in record["old_expected"].items() if v is not None}:
            print(f"  {record['case_id']}")
            print(f"      expectation {record['old_expected']} -> {record['new_expected']}")
    print()
    for name, digest in before.items():
        print(f"  {name:28} {digest[:12]} -> "
              f"{after.get(name, after.get('plan-fixtures-v8.jsonl'))[:12]}")

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps({"phase": "P1.6-0H", "applied": bool(args.apply),
                        "before": before, "after": after,
                        "fact_id_status": fact_status, "changes": changes},
                       ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    if not args.apply:
        print("\n  (dry run -- pass --apply to write)")
        return 0

    (BENCH / "plan-fixtures-v8.jsonl").write_text(v8_text, encoding="utf-8")
    (BENCH / "plan-fixtures-v8.jsonl.sha256").write_text(
        after["plan-fixtures-v8.jsonl"] + "\n", encoding="utf-8")
    (BENCH / "gold-evidence-v1.jsonl").write_text(gold_text, encoding="utf-8")
    (BENCH / "canonical-eval-v1.jsonl").write_text(eval_text, encoding="utf-8")
    (BENCH / "canonical-eval-v1.jsonl.sha256").write_text(
        after["canonical-eval-v1.jsonl"] + "\n", encoding="utf-8")

    manifest = dict(json.loads(
        (BENCH / "plan-fixtures-v7.manifest.json").read_text(encoding="utf-8")))
    manifest.update({
        "fixture": "plan-fixtures-v8",
        "fixture_sha256": after["plan-fixtures-v8.jsonl"],
        "gold_sha256": after["gold-evidence-v1.jsonl"],
        "eval_set_sha256": after["canonical-eval-v1.jsonl"],
        "stage": "P1-6-0H-REDERIVED-CROSS-ENTITY",
        "cross_entity_rederivation": {
            "spec": "docs/evaluation/p1-6-0g-rederivation.md",
            "corrections": "docs/evaluation/p1-6-0a10-expense-sign.md",
            "cases_changed": sorted(SPEC),
            "previous": before,
            "fact_id_status": fact_status,
        },
    })
    (BENCH / "plan-fixtures-v8.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("\n  wrote plan-fixtures-v8.jsonl (+ .sha256, .manifest.json), "
          "gold-evidence-v1.jsonl, canonical-eval-v1.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
