#!/usr/bin/env python3
"""Gate 08 R3: Multi-Evidence Slot Metrics.

Analyzes multi-slot cases in the R3 predictions to report per-slot candidate
availability and budget truncation.
"""

from __future__ import annotations
import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate 08 R3 Multi-Evidence Slot Metrics"
    )
    parser.parse_args()

    base = ROOT

    r3_dir = base / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
    predictions_path = r3_dir / "predictions.jsonl.gz"
    seal_path = r3_dir / "prediction-seal.json"
    gold_path = base / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
    universe_path = (
        base / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/universe-scoring.json"
    )
    plans_path = (
        base
        / "artifacts/evaluation/pdf-retrieval-v4-gate-07/query-plan-predictions.json"
    )
    out_path = r3_dir / "scoring/multi-evidence-metrics.json"

    # 1. Verify seal
    seal = load_json(seal_path)
    sealed_ok = seal.get("sealed") is True and seal.get("gold_reads_before_seal") == 0
    if not sealed_ok:
        print(
            "ERROR: R3 seal verification failed: sealed="
            f"{seal.get('sealed')}, gold_reads_before_seal="
            f"{seal.get('gold_reads_before_seal')}",
            file=sys.stderr,
        )
        return 1
    print(
        f"seal ok: sealed={seal.get('sealed')}, "
        f"gold_reads_before_seal={seal.get('gold_reads_before_seal')}"
    )

    # 2. Load data
    predictions = [
        r for r in load_jsonl(predictions_path) if r.get("stream") != "header"
    ]
    gold_rows = load_jsonl(gold_path)
    universe = load_json(universe_path)
    plans_doc = load_json(plans_path)

    # gold labels by case_id
    gold_by_case: dict[str, dict] = {}
    for row in gold_rows:
        gold_by_case[row["case_id"]] = row

    # universe details grouped by case_id (positional order preserved)
    universe_by_case: dict[str, list[dict]] = {}
    for det in universe.get("details", []):
        universe_by_case.setdefault(det["case_id"], []).append(det)

    # query plans by case_id
    plans_by_case: dict[str, dict] = {}
    for item in plans_doc.get("plans", []):
        cid = item.get("case_id")
        if cid is not None:
            plans_by_case[cid] = item.get("plan", {}) or {}

    POOL_BUDGET = 40
    records: list[dict] = []
    multi_count = 0
    complete_count = 0
    budget_truncated_count = 0

    for pred in predictions:
        if not pred.get("is_multi_slot"):
            continue
        multi_count += 1
        case_id = pred["case_id"]

        gold_row = gold_by_case.get(case_id, {})
        gold_sources = gold_row.get("expected_sources", []) or []

        # operand slots from query plan
        plan = plans_by_case.get(case_id, {})
        operand_slots = plan.get("operand_slots", []) or []
        required_slot_count = len(operand_slots)

        # e3 expanded pool
        e3_pool = pred.get("e3_expanded_pool", []) or []
        pool_keys: set[str] = set()
        for item in e3_pool:
            ck = item.get("candidate_key")
            if ck is not None:
                pool_keys.add(ck)

        # structured expanded fused -> rank by key
        structured = pred.get("structured_expanded", {}) or {}
        structured_fused = structured.get("fused", []) or []
        structured_rank_by_key: dict[str, int] = {}
        for item in structured_fused:
            ck = item.get("candidate_key")
            rk = item.get("rank")
            if ck is not None and rk is not None:
                structured_rank_by_key[ck] = rk

        # universe details for this case (positional)
        univ_details = universe_by_case.get(case_id, [])

        budget_truncated = len(e3_pool) == POOL_BUDGET
        if budget_truncated:
            budget_truncated_count += 1

        gold_details: list[dict] = []
        gold_in_pool = 0
        for idx, gs in enumerate(gold_sources):
            ck = gs.get("candidate_key")
            in_pool = ck in pool_keys if ck is not None else False
            in_universe = False
            if idx < len(univ_details):
                det = univ_details[idx]
                in_universe = det.get("was_in_structured_universe") is True
            structured_rank = structured_rank_by_key.get(ck) if ck is not None else None
            if in_pool:
                gold_in_pool += 1
            gold_details.append(
                {
                    "gold_index": idx,
                    "candidate_key": ck,
                    "in_universe": in_universe,
                    "in_e3_pool": in_pool,
                    "structured_fused_rank": structured_rank,
                }
            )

        gold_source_count = len(gold_sources)
        complete_evidence_available = (
            gold_source_count > 0 and gold_in_pool == gold_source_count
        )
        if complete_evidence_available:
            complete_count += 1

        records.append(
            {
                "case_id": case_id,
                "required_slot_count": required_slot_count,
                "gold_source_count": gold_source_count,
                "gold_in_e3_expanded_pool": gold_in_pool,
                "complete_evidence_available": complete_evidence_available,
                "budget_truncated": budget_truncated,
                "gold_details": gold_details,
            }
        )

    output = {
        "gate": "pdf_retrieval_v4_gate_08_r3",
        "multi_evidence_case_count": multi_count,
        "complete_evidence_availability": f"{complete_count}/{multi_count}",
        "multi_slot_budget_truncated": budget_truncated_count,
        "records": records,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
