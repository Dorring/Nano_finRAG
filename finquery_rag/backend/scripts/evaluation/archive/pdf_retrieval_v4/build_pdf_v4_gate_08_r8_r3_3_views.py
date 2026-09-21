#!/usr/bin/env python3
"""Build and seal statement-aware main and operand-focused slot query views."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.structure_aware_rerank_view import (  # noqa: E402
    build_rerank_query_view,
    build_slot_rerank_query_view,
    sha256_text,
)

BASE = ROOT / "artifacts/evaluation"
R31A = BASE / "pdf-retrieval-v4-gate-08-r8-r3-1a"
V2 = R31A / "rerank-input-views-v2.jsonl.gz"
PLAN = BASE / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
R32 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-2"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3-3-p0"
VIEWS = OUT / "queryplan-rerank-input-views.jsonl.gz"
EXPECTED_V2_SHA = "82ea6c75dae8607e7bda462c39745abcff9ac991611c271c70e34fa318fc6dc1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if sha(V2) != EXPECTED_V2_SHA:
        raise RuntimeError("r3_1_v2_views_sha_mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    plans = {item["case_id"]: item["plan"] for item in json.loads(PLAN.read_text())["plans"]}
    records = []
    changed_main_cases = statement_hint_cases = slot_count = 0
    with gzip.open(V2, "rt", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            plan = plans[source["case_id"]]
            main_view = build_rerank_query_view(plan)
            old_main_sha = source["query_view_sha256"]
            main_sha = sha256_text(main_view)
            changed_main_cases += main_sha != old_main_sha
            statement_hint_cases += bool(plan.get("statement_hint"))
            slots = []
            if len(plan.get("operand_slots", [])) > 1:
                for slot in plan["operand_slots"]:
                    query = build_slot_rerank_query_view(plan, slot)
                    slots.append({"slot_id": slot["slot_id"], "query_view": query, "query_view_sha256": sha256_text(query)})
                    slot_count += 1
            records.append({"case_id": source["case_id"], "query_plan_id": source["query_plan_id"], "main_query_view": main_view, "main_query_view_sha256": main_sha, "previous_main_query_view_sha256": old_main_sha, "main_score_reusable": main_sha == old_main_sha, "slot_query_views": slots, "candidates": [{"candidate_key": item["candidate_key"], "pre_rerank_rank": item["pre_rerank_rank"], "document_view": item["document_view"], "document_view_sha256": item["document_view_sha256"], "context_status": item["context_status"]} for item in source["candidates"]]})
    if len(records) != 72 or slot_count != 36:
        raise RuntimeError("r3_3_view_count_contract_failed")
    with VIEWS.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for record in records:
                zipped.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    manifest = {"cases": 72, "candidate_occurrences": 7200, "multi_slot_cases": 18, "slot_count": 36, "slot_pair_count": 3600, "statement_hint_nonempty_cases": statement_hint_cases, "main_query_changed_cases": changed_main_cases, "main_query_reused_cases": 72 - changed_main_cases, "candidate_mutation": 0, "document_view_mutation": 0, "r3_1_v2_views_sha256": EXPECTED_V2_SHA, "r3_2_prediction_sha256": sha(R32 / "rerank-predictions.jsonl.gz"), "query_plan_sha256": sha(PLAN), "queryplan_views_sha256": sha(VIEWS)}
    protocol = {"gate": "pdf_retrieval_v4_gate_08_r8_r3_3_p0", "gold_reads_before_seal": 0, "governance_reads_before_seal": 0, "model_calls": 0, "retrieval_runs": 0, "candidate_added": 0, "candidate_removed": 0, "model_scan": False, "prompt_scan": False, "slot_quota_scan": False}
    write("input-integrity.json", manifest)
    write("view-manifest.json", manifest)
    write("input-seal.json", {**manifest, **protocol, "sealed": True})
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
