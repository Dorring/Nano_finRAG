#!/usr/bin/env python3
"""Build the immutable 80-binding strict source scoring sidecar."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
GOV = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-strict-source-contract"
FIELDS = ("candidate_key", "document_id", "page", "evidence_id")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding_id(case_id: str, source_index: int, candidate_key: str, evidence_id: str) -> str:
    raw = f"{case_id}\x1f{source_index}\x1f{candidate_key}\x1f{evidence_id}"
    return "binding:v1:" + hashlib.sha256(raw.encode()).hexdigest()


def main() -> int:
    labels = {item["case_id"]: item for item in map(json.loads, LABELS.open(encoding="utf-8"))}
    governance = {item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))}
    if set(labels) != set(governance) or len(labels) != 72:
        raise RuntimeError("strict_source_case_contract_blocked")
    sidecar = []
    mismatches = []
    reordered = 0
    parity = {"source_index": 0, **{field: 0 for field in FIELDS}}
    for case_id in sorted(labels):
        expected = labels[case_id].get("expected_sources") or []
        governed = governance[case_id].get("strict_gold_source_bindings") or []
        if len(expected) != len(governed):
            mismatches.append({"case_id": case_id, "field": "binding_count", "labels": len(expected), "governance": len(governed)})
            continue
        for source_index, (source, governed_binding) in enumerate(zip(expected, governed, strict=True)):
            if governed_binding.get("source_index") != source_index:
                reordered += 1
            else:
                parity["source_index"] += 1
            for field in FIELDS:
                if source.get(field) == governed_binding.get(field):
                    parity[field] += 1
                else:
                    mismatches.append({"case_id": case_id, "source_index": source_index, "field": field, "labels": source.get(field), "governance": governed_binding.get(field)})
            candidate_key = str(source["candidate_key"])
            evidence_id = str(source["evidence_id"])
            sidecar.append(
                {
                    "binding_id": binding_id(case_id, source_index, candidate_key, evidence_id),
                    "case_id": case_id,
                    "source_index": source_index,
                    "candidate_key": candidate_key,
                    "document_id": source.get("document_id"),
                    "page": source.get("page"),
                    "evidence_id": evidence_id,
                    "evidence_family_id": governed_binding.get("evidence_family_id"),
                }
            )
    closure = len(sidecar) == 80 and not mismatches and reordered == 0 and all(value == 80 for value in parity.values())
    OUT.mkdir(parents=True, exist_ok=True)
    sidecar_path = OUT / "strict-gold-source-bindings.jsonl"
    sidecar_path.write_text("".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in sidecar))
    audit = {
        "schema": "pdf-retrieval-v4/strict-gold-source-binding/v1",
        "case_count": f"{len(labels)}/72",
        "source_binding_count": f"{len(sidecar)}/80",
        "case_id_parity": "72/72",
        **{f"{field}_parity": f"{value}/80" for field, value in parity.items()},
        "missing_bindings": max(0, 80 - len(sidecar)),
        "extra_bindings": max(0, len(sidecar) - 80),
        "reordered_bindings": reordered,
        "mismatches": mismatches,
        "labels_sha256": sha(LABELS),
        "governance_sha256": sha(GOV),
        "sidecar_sha256": sha(sidecar_path),
        "strict_gold_identities_status": "strict_gold_unique_candidate_set_diagnostic_only",
        "retrieval_recall_scoring_unit": "case_id_source_index_candidate_key",
        "decision": "strict_gold_source_binding_contract_closed" if closure else "strict_gold_source_binding_contract_blocked",
        "next_gate": "unified_80_binding_rescore" if closure else "stop_and_audit_benchmark_mutation",
        "prediction_reruns": 0,
        "retrieval_runs": 0,
        "gold_reads_before_prediction_seal": 0,
        "production_switch_allowed": False,
    }
    (OUT / "binding-parity.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (OUT / "acceptance.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2))
    return 0 if closure else 1


if __name__ == "__main__":
    raise SystemExit(main())
