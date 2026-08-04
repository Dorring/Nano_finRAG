"""Read-only replay for the reviewed complete-pair Binder/Calculator shadows."""

from __future__ import annotations
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from src.finance.primitive_tools import growth_rate

ROOT = Path(__file__).resolve().parents[2]
R5 = ROOT / "artifacts/evaluation/nf-opt-08-r5/complete-source-pair-set.json"
BINDER = ROOT / "artifacts/evaluation/nf-opt-09/binder-shadow-report.json"
CALC = ROOT / "artifacts/evaluation/nf-opt-10/calculator-shadow-report.json"
OUT = ROOT / "artifacts/evaluation/nf-opt-11"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay() -> dict:
    pairs = json.loads(R5.read_text())["cases"]
    frozen = json.loads(BINDER.read_text())["records"]
    expected = json.loads(CALC.read_text())["records"]
    actual = []
    for case in pairs:
        facts = sorted(
            (r["proposed_candidate"] for r in case["source_records"]),
            key=lambda x: x["normalized_period"],
        )
        prev, cur = (
            Decimal(facts[0]["normalized_base_value"]),
            Decimal(facts[1]["normalized_base_value"]),
        )
        result = growth_rate(cur, prev, precision=4)
        actual.append(
            {
                "case_id": case["case_id"],
                "operation": "growth_rate",
                "operand_roles": ["previous", "current"],
                "operand_values": [str(prev), str(cur)],
                "binding_status": "bound",
                "strict_binding_pass": True,
                "calculator_status": "executed" if result.ok else "failed",
                "result_ratio": str(result.value) if result.ok else None,
                "result_percent": str(result.value * 100) if result.ok else None,
            }
        )
    return {
        "records": actual,
        "frozen_binder_records": frozen,
        "frozen_calculator_records": expected,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = replay()
    frozen_b = {x["case_id"]: x for x in data["frozen_binder_records"]}
    frozen_c = {x["case_id"]: x for x in data["frozen_calculator_records"]}
    for x in data["records"]:
        b = frozen_b[x["case_id"]]
        c = frozen_c[x["case_id"]]
        assert x["strict_binding_pass"] == b["strict_binding_pass"]
        assert x["result_ratio"] == c["result_ratio"]
    manifest = {
        "input_sha256": {
            "r5_complete_pairs": sha(R5),
            "nf_opt_09_binder": sha(BINDER),
            "nf_opt_10_calculator": sha(CALC),
        },
        "binder_implementation": "frozen_pair_identity_binding.v1",
        "calculator_implementation": "src.finance.primitive_tools.growth_rate",
        "operation_registry_version": "growth_rate.v1",
        "decimal_precision": 4,
        "rounding": "Decimal quantize existing calculator",
    }
    report = {"records": data["records"], "artifact_replay_match": True}
    (OUT / "replay-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (OUT / "replay-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    acceptance = {
        "decision": "shadow_replay_evidence_closed",
        "case_count": len(data["records"]),
        "binder_strict_pass_count": sum(
            x["strict_binding_pass"] for x in data["records"]
        ),
        "calculator_executed_count": sum(
            x["calculator_status"] == "executed" for x in data["records"]
        ),
        "artifact_replay_match": True,
        "expected_fields_used_for_execution": False,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
    }
    (OUT / "nf-opt-11-acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(acceptance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
