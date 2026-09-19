"""P1.6-A11: does the migrated fixture agree with the store it will be answered from?

Reads `plan-fixtures-v8`, `gold-evidence-v1` and `canonical-eval-v1` from disk, and
for every entity of every cross-entity case asks `CanonicalFactStore` for the value
-- the same call retrieval will make.  The fixture and the store were produced by
different scripts from different sources, so agreement between them is evidence
rather than a tautology.

It also checks the three things a fixture migration can get wrong quietly:

* the plan's metric, the gold's metric and the question naming the same quantity
* the `fact_ids` pointing at records that exist in the store
* `compare-007` carrying its fiscal years, since it compares a FY2025 filer
  against a FY2024 one

  python verify_v8_migration.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

BENCH = _BACKEND_DIR / "benchmarks/tv2_canonical_v1"
LEGACY_STORE = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl")
IXBRL_STORE = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl")
STRATUM_PREFIX = "tv2f01-s3-"


def _digits(text: object) -> str:
    cleaned = re.sub(r"[^\d.]", "", str(text or ""))
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from src.runtime.trusted_v2_canonical_store import CanonicalFactStore
    from src.runtime.trusted_v2_production import StructuredFactStore

    def load(name: str) -> dict[str, dict]:
        return {
            row["id"]: row
            for row in (
                json.loads(line)
                for line in (BENCH / name).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }

    gold = load("gold-evidence-v1.jsonl")
    evaluation = load("canonical-eval-v1.jsonl")
    plan = load("plan-fixtures-v8.jsonl")

    store = CanonicalFactStore(StructuredFactStore(LEGACY_STORE), IXBRL_STORE)
    ixbrl_keys = {
        json.loads(line).get("candidate_key")
        for line in IXBRL_STORE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    rows = []
    tally = {"AGREE": 0, "MISMATCH": 0, "EMPTY": 0}
    problems = []
    for case_id in sorted(k for k in gold if k.startswith(STRATUM_PREFIX)):
        row = gold[case_id]
        plan_row = plan[case_id]
        evaluation_row = evaluation[case_id]
        metric = row.get("metric")
        slot_metrics = {s.get("metric") for s in plan_row["plan"]["required_slots"]}
        fiscal = row.get("fiscal_year_by_entity") or {}

        if slot_metrics != {metric}:
            problems.append((case_id, "PLAN_METRIC", f"{slot_metrics} vs {metric!r}"))
        if metric and metric.casefold() not in str(evaluation_row.get("question", "")).casefold():
            problems.append((case_id, "QUESTION_METRIC",
                             evaluation_row.get("question", "")[:60]))

        for entity, value in (row.get("values") or {}).items():
            period = fiscal.get(entity, "FY2025")
            facts = store.facts_at_coordinate(entity, metric, period)
            resolved = sorted({_digits(f.get("value")) for f in facts})
            status = "AGREE" if resolved == [_digits(value)] else (
                "EMPTY" if not resolved else "MISMATCH")
            tally[status] += 1
            if status != "AGREE":
                problems.append((case_id, status, f"{entity}: store={resolved} "
                                                     f"fixture={value}"))
            rows.append({"case_id": case_id, "entity": entity, "metric": metric,
                         "period": period, "fixture": value, "store": resolved,
                         "status": status})

        for fact_id in row.get("fact_ids") or ():
            if fact_id not in ixbrl_keys:
                problems.append((case_id, "FACT_ID_MISSING", str(fact_id)))

    report = {"phase": "P1.6-A11", "tally": tally,
              "facts_checked": len(rows), "problems": problems, "rows": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v8-migration-verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print("=== fixture vs store ===")
    for row in rows:
        mark = "ok " if row["status"] == "AGREE" else "!! "
        print(f"  {mark}{row['case_id'][-12:]:12} {row['entity'][:22]:22} "
              f"{str(row['metric'])[:24]:24} {row['period']:8} "
              f"fixture={row['fixture']:>10} store={str(row['store'])[:14]}")
    print(f"\n  {tally}")
    if problems:
        print("\n=== problems ===")
        for case_id, kind, detail in problems:
            print(f"  {kind:16} {case_id[-12:]:12} {detail}")
    else:
        print("\n  no problems")
    print(f"  written to {args.out / 'v8-migration-verification.json'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
