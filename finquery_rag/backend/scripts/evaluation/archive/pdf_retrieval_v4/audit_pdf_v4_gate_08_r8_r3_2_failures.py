#!/usr/bin/env python3
"""Post-seal 4B Top5 failure attribution without changing predictions."""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3-2"
PRED = OUT / "rerank-predictions.jsonl.gz"
VIEWS = BASE / "pdf-retrieval-v4-gate-08-r8-r3-1a/rerank-input-views-v2.jsonl.gz"
PLAN = BASE / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
SIDECAR = BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
GOV = BASE / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"


def load_gzip(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {item["case_id"]: item for item in map(json.loads, handle)}


def normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def main() -> int:
    predictions, views = map(load_gzip, (PRED, VIEWS))
    plans = {item["case_id"]: item["plan"] for item in json.loads(PLAN.read_text())["plans"]}
    bindings = [json.loads(line) for line in SIDECAR.open(encoding="utf-8")]
    governance = {item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))}
    view_maps = {case_id: {item["candidate_key"]: item for item in record["candidates"]} for case_id, record in views.items()}
    counts: Counter[str] = Counter()
    records = []
    for binding in bindings:
        case_id, key = binding["case_id"], binding["candidate_key"]
        ranked = predictions[case_id]["ranked_candidates"]
        gold = next((item for item in ranked if item["candidate_key"] == key), None)
        if gold is None or gold["post_rerank_rank"] <= 5:
            continue
        plan = plans[case_id]
        top5_docs = [view_maps[case_id][item["candidate_key"]]["document_view"] for item in ranked[:5]]
        gold_doc = view_maps[case_id][key]["document_view"]
        metrics = [normalize(slot.get("raw_metric_phrase") or "") for slot in plan.get("operand_slots", [])]
        periods = [normalize(slot.get("period") or "") for slot in plan.get("operand_slots", [])]
        metric_competitors = [doc for doc in top5_docs if any(metric and metric in normalize(doc) for metric in metrics)]
        if governance[case_id]["query_type"] == "calculation_multi_operand":
            category = "calculation_operand_competition"
        elif governance[case_id]["requires_multiple_sources"]:
            category = "multi_slot_competition"
        elif any(normalize(doc) == normalize(gold_doc) for doc in top5_docs):
            category = "duplicate_semantic_source"
        elif metric_competitors and any(
            periods and all(period not in normalize(doc) for period in periods)
            for doc in metric_competitors
        ):
            category = "same_metric_wrong_period"
        elif metric_competitors and any("statement:" in doc.lower() for doc in metric_competitors):
            category = "same_metric_wrong_statement"
        elif metric_competitors:
            category = "same_metric_source_competition"
        else:
            category = "generic_financial_similarity"
        counts[category] += 1
        records.append({**binding, "first_failure": category, "pre_rerank_rank": gold["pre_rerank_rank"], "post_rerank_rank": gold["post_rerank_rank"], "context_status": gold["context_status"], "top5_candidate_keys": [item["candidate_key"] for item in ranked[:5]]})
    payload = {"scope": "top100_gold_below_top5", "record_count": len(records), "counts": dict(counts), "records": records, "prediction_mutation": 0, "next_gate": "4b_failure_attribution_review"}
    (OUT / "failure-attribution.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record_count": len(records), "counts": dict(counts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
