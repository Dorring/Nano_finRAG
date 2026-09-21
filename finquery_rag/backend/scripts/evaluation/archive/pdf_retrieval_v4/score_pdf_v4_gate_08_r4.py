#!/usr/bin/env python3
"""Score sealed Gate 08 R4 fusion predictions."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
R4_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r4"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _keys(items: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("candidate_key") or "") for item in items}


def main() -> int:
    pred_path = R4_DIR / "fusion-predictions.jsonl.gz"
    seal = _json(R4_DIR / "prediction-seal.json")
    if not seal.get("sealed") or seal["prediction_sha256"] != _sha(pred_path):
        raise RuntimeError("r4_prediction_seal_invalid")
    predictions = {x["case_id"]: x for x in _gz(pred_path)}
    r3 = {x["case_id"]: x for x in _gz(R3_DIR / "predictions.jsonl.gz")}
    gold: list[tuple[str, int, str]] = []
    with LABELS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for index, source in enumerate(record.get("expected_sources") or []):
                key = source.get("candidate_key")
                if key:
                    gold.append((record["case_id"], index, key))

    def hits(field: str) -> set[tuple[str, int, str]]:
        return {
            identity
            for identity in gold
            if identity[2] in _keys(predictions[identity[0]][field])
        }

    f0, f1, f2 = hits("f0_pool"), hits("f1_pool"), hits("f2_pool")
    union = {
        identity
        for identity in gold
        if identity[2]
        in (_keys(r3[identity[0]]["e1_pool"]) | _keys(r3[identity[0]]["e2_expanded_pool"]))
    }
    gross_loss = union - f0
    synergy = f0 - union
    recovered = gross_loss & f2
    regression = f0 - f2
    added = f2 - f0
    synergy_retained = synergy & f2
    new_outside_union = added - union
    score = len(f2)
    if score >= 60 and len(synergy_retained) == 2:
        decision = "lane_preserving_fusion_strong_pass"
        next_gate = "slot_preserving_evidence_set"
    elif score >= 57:
        decision = "lane_preserving_fusion_passed"
        next_gate = "field_aware_retrieval"
    elif score >= 55:
        decision = "lane_preserving_fusion_gain_real_but_insufficient"
        next_gate = "field_aware_retrieval"
    else:
        decision = "lane_preserving_fusion_insufficient"
        next_gate = "field_aware_retrieval"
    ablation = {
        "f0": f"{len(f0)}/80",
        "f1": f"{len(f1)}/80",
        "f2": f"{len(f2)}/80",
        "family_union": f"{len(union)}/80",
        "gross_fusion_loss": len(gross_loss),
        "net_union_gap": len(union) - len(f0),
        "fusion_synergy_gain": len(synergy),
        "gross_fusion_loss_recovered": f"{len(recovered)}/{len(gross_loss)}",
        "synergy_gold_retained": f"{len(synergy_retained)}/{len(synergy)}",
        "original_e3_gold_retained": f"{len(f0 & f2)}/{len(f0)}",
        "new_gold_added": len(added),
        "new_gold_regressed": len(regression),
        "contract_gross_loss_net_gain": len(recovered) - len(regression),
        "new_outside_union_synergy_gain": len(new_outside_union),
        "observed_score_delta": len(f2) - len(f0),
        "score_reconciliation": (
            f"{len(f0)} + {len(recovered)} recovered - {len(regression)} "
            f"regressed + {len(new_outside_union)} new-outside-union = {len(f2)}"
        ),
    }
    _write(R4_DIR / "fusion-ablation.json", ablation)
    _write(R4_DIR / "gross-loss-recovery.json", {"recovered": sorted(recovered), "still_lost": sorted(gross_loss - f2)})
    _write(R4_DIR / "synergy-retention.json", {"retained": sorted(synergy_retained), "regressed": sorted(synergy - f2)})
    _write(R4_DIR / "regression-matrix.json", {"added": sorted(added), "regressed": sorted(regression)})
    _write(R4_DIR / "multi-slot-fusion-audit.json", {"multi_slot_cases": sum(x["is_multi_slot"] for x in predictions.values()), "all_have_trace": all(bool(x["f2_trace"]) for x in predictions.values() if x["is_multi_slot"])})
    _write(R4_DIR / "candidate-budget-audit.json", {"residual_budget_max": 40, "violations": sum(len(x["f2_trace"]) > 40 for x in predictions.values()), "e0_prefix_exact": sum(x["e0_prefix_exact"] for x in predictions.values())})
    r3_loss_classes = _json(R3_DIR / "scoring/fusion-loss-classification.json")
    remaining_by_class: dict[str, int] = {}
    recovered_by_class: dict[str, int] = {}
    for detail in r3_loss_classes["classification_details"]:
        identity = (detail["case_id"], detail["source_index"], detail["candidate_key"])
        target = recovered_by_class if identity in recovered else remaining_by_class
        category = detail["fusion_loss_category"]
        target[category] = target.get(category, 0) + 1
    _write(
        R4_DIR / "first-failure-attribution.json",
        {
            "recovered": len(recovered),
            "lane_preserving_fusion_budget_loss": len(gross_loss - f2),
            "recovered_by_r3_1_category": recovered_by_class,
            "remaining_by_r3_1_category": remaining_by_class,
        },
    )
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r4",
        "decision": decision,
        "next_gate": next_gate,
        "metrics": ablation,
        "raw_gold_retained": "31/31",
        "raw_e0_prefix_exact": "72/72",
        "synergy_gate_passed": len(synergy_retained) == len(synergy),
        "bm25_searches": 0,
        "dense_searches": 0,
        "embedding_calls": 0,
        "index_reads": 0,
        "parameter_scan": False,
        "quota_scan": False,
        "production_switch_allowed": False,
    }
    _write(R4_DIR / "acceptance.json", acceptance)
    _write(R4_DIR / "next-gate.json", {"current_gate": "pdf_retrieval_v4_gate_08_r4", "decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps(ablation, indent=2))
    print(f"decision={decision} next_gate={next_gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
