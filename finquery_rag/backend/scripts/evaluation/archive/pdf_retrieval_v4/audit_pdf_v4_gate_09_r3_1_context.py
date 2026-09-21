#!/usr/bin/env python3
"""Post-seal context disambiguatability audit for Gate 09 R3."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.evidence_slot_matcher_v3 import match_slot  # noqa: E402

R3 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09-r3"
R7 = (
    ROOT
    / "artifacts/evaluation/pdf-retrieval-v4-gate-09/evidence-set-predictions.jsonl.gz"
)
PLANS = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07/query-plan-predictions.json"
)
GOV = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09-r3-1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path, field: str | None = None):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            item["case_id"]: item[field] if field else item
            for item in (json.loads(line) for line in handle if line.strip())
        }


def load_jsonl(path: Path):
    return {item["case_id"]: item for item in map(json.loads, path.open())}


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads((R3 / "prediction-seal.json").read_text())
    attachment = R3 / "authoritative-attachments.jsonl.gz"
    if not seal.get("sealed") or seal["attachment_sha256"] != sha(attachment):
        raise RuntimeError("r3_seal_invalid")
    evidence = load_gzip(attachment, "authoritative_evidence")
    pools = load_gzip(R7)
    plans = {
        item["case_id"]: item["plan"] for item in json.loads(PLANS.read_text())["plans"]
    }
    governance = load_jsonl(GOV)
    records = []
    for case_id, plan in plans.items():
        if len(plan.get("operand_slots") or []) <= 1:
            continue
        expected = set(governance[case_id]["strict_gold_identities"])
        pool = {item["candidate_key"] for item in pools[case_id]["candidate_pool"]}
        if not expected.issubset(pool):
            continue
        classifications = []
        for slot in plan["operand_slots"]:
            matches = [
                match
                for item in evidence[case_id]
                if (match := match_slot(plan, slot, item))
            ]
            gold_matches = [
                item
                for item in matches
                if expected.intersection(
                    item.get("supporting_candidate_keys") or [item["candidate_key"]]
                )
            ]
            if not gold_matches:
                category = "context_unresolved"
            else:
                gold_ids = {item["evidence_id"] for item in gold_matches}
                competitors = [
                    item for item in matches if item["evidence_id"] not in gold_ids
                ]
                equivalent = {
                    item.get("equivalent_group_id")
                    for item in gold_matches
                    if item.get("equivalent_group_id")
                }
                if (
                    competitors
                    and equivalent
                    and all(
                        item.get("equivalent_group_id") in equivalent
                        for item in competitors
                    )
                ):
                    category = "frozen_equivalent"
                elif not competitors:
                    category = "runtime_context_distinguishable"
                elif any(
                    item["statement_context"] == "compatible" for item in gold_matches
                ) and not any(
                    item["statement_context"] == "compatible" for item in competitors
                ):
                    category = "runtime_context_distinguishable"
                else:
                    gold_context = {
                        (item.get("row_id"), item.get("table_fragment_id"))
                        for item in gold_matches
                    }
                    competitor_context = {
                        (item.get("row_id"), item.get("table_fragment_id"))
                        for item in competitors
                    }
                    category = (
                        "query_silent_source_ambiguity"
                        if gold_context != competitor_context
                        else "context_unresolved"
                    )
            classifications.append({"slot_id": slot["slot_id"], "category": category})
        runtime = all(
            item["category"] in {"frozen_equivalent", "runtime_context_distinguishable"}
            for item in classifications
        )
        records.append(
            {
                "case_id": case_id,
                "classifications": classifications,
                "runtime_disambiguatable": runtime,
            }
        )
    counter = Counter(
        item["category"] for record in records for item in record["classifications"]
    )
    runtime_cases = {
        item["case_id"] for item in records if item["runtime_disambiguatable"]
    }
    benchmark = {
        case for case in runtime_cases if governance[case]["requires_multiple_sources"]
    }
    calculation = {
        case
        for case in runtime_cases
        if governance[case]["query_type"] == "calculation_multi_operand"
    }
    write("collision-groups.json", {"counts": dict(counter), "records": records})
    write(
        "context-disambiguatability.json",
        {
            "pool_complete_multi_slot": len(records),
            "runtime_disambiguatable_cases": sorted(runtime_cases),
            "runtime_disambiguatable_count": len(runtime_cases),
        },
    )
    write(
        "multislot-ceiling.json",
        {"formal_denominator": 12, "runtime_disambiguatable": len(runtime_cases)},
    )
    write(
        "benchmark-multievidence-ceiling.json",
        {"formal_pool_complete": 10, "runtime_disambiguatable": len(benchmark)},
    )
    write(
        "calculation-ceiling.json",
        {"formal_pool_complete": 8, "runtime_disambiguatable": len(calculation)},
    )
    write(
        "acceptance.json",
        {
            "decision": "context_disambiguatability_audited",
            "next_gate": "context_aware_evidence_set_replay",
            "runtime_disambiguatable_multi_slot_cases": sorted(runtime_cases),
            "gold_reads_after_r3_seal": True,
            "attachment_mutations": 0,
            "production_switch_allowed": False,
        },
    )
    print(
        json.dumps(
            {
                "counts": dict(counter),
                "runtime_multi": len(runtime_cases),
                "runtime_benchmark": len(benchmark),
                "runtime_calculation": len(calculation),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
