"""Gate 02 R3: Finalize - acceptance gate and next-gate decision.

Loads all previously generated R3 artifacts and builds the final
acceptance gate check and next-gate decision.

Acceptance gates:
  - Coverage gate:     Documents 8/8, Pages 1348/1348, Missing 0
  - Identity gate:     Table/Row/Cell ID Conflict = 0, Broken FK = 0
  - Oracle gate:       Table 22/22, Row 22/22, Numeric 22/22,
                       Scale 22/22, Source Traceback 22/22
  - Regression gate:   Old Probe True Regression = 0
  - Safety gate:       Question Reads = 0, Gold Reads = 0,
                       Governance Reads = 0, Index Builds = 0,
                       Retrieval = 0, Production Writes = 0

Next-gate decision:
  - All pass  → full_corpus_unified_structured_adapter_passed
                next_gate = full_corpus_financial_semantic_graph
  - Oracle regression → full_corpus_adapter_oracle_regression_blocked
  - Old probe regression → full_document_context_adapter_regression_blocked
  - production_switch_allowed = false (always)

Outputs:
  - acceptance.json
  - next-gate.json

Reads ONLY previously generated R3 evaluation artifacts.
No questions, gold, or governance data is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R3_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"
SHARED_NANOCHAT_ROOT = ROOT.parents[4]

EXPECTED_DOCUMENTS = 8
EXPECTED_PAGES = 1348


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Acceptance gate builders
# ---------------------------------------------------------------------------


def _build_coverage_gate(
    seal: dict[str, Any],
    integrity: dict[str, Any],
    structure_metrics: dict[str, Any],
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    """Coverage gate: Documents 8/8, Pages 1348/1348, Missing 0."""
    document_count = int(integrity.get("document_count", 0))
    page_count = int(structure_metrics.get("page_count", 0)) or int(seal.get("page_count", 0))
    missing_pages = int(reconciliation.get("true_missing_page_count", 0))

    documents_ok = document_count == EXPECTED_DOCUMENTS
    pages_ok = page_count == EXPECTED_PAGES
    missing_ok = missing_pages == 0

    return {
        "documents": f"{document_count}/{EXPECTED_DOCUMENTS}",
        "pages": f"{page_count}/{EXPECTED_PAGES}",
        "missing_pages": missing_pages,
        "passed": documents_ok and pages_ok and missing_ok,
    }


def _build_identity_gate(
    structure_metrics: dict[str, Any],
    identity_integrity: dict[str, Any],
) -> dict[str, Any]:
    """Identity gate: Table/Row/Cell ID Conflict = 0, Broken FK = 0."""
    table_conflict = int(structure_metrics.get("duplicate_table_id_count", 0))
    row_conflict = int(structure_metrics.get("duplicate_row_id_count", 0))
    cell_conflict = int(structure_metrics.get("duplicate_cell_id_count", 0))

    id_integrity = identity_integrity.get("identity_integrity", {})
    broken_fk = (
        int(id_integrity.get("row_to_table_missing", 0))
        + int(id_integrity.get("cell_to_row_missing", 0))
    )

    return {
        "table_id_conflict": table_conflict,
        "row_id_conflict": row_conflict,
        "cell_id_conflict": cell_conflict,
        "broken_fk": broken_fk,
        "passed": (
            table_conflict == 0
            and row_conflict == 0
            and cell_conflict == 0
            and broken_fk == 0
        ),
    }


def _build_oracle_gate(oracle_regression: dict[str, Any]) -> dict[str, Any]:
    """Oracle gate: Table/Row/Numeric/Scale/Source Traceback all 22/22."""
    gate_checks = oracle_regression.get("gate_checks", {})
    table_ok = bool(gate_checks.get("table_22_22", False))
    row_ok = bool(gate_checks.get("row_22_22", False))
    numeric_ok = bool(gate_checks.get("numeric_22_22", False))
    scale_ok = bool(gate_checks.get("scale_22_22", False))
    source_ok = bool(gate_checks.get("source_traceback_22_22", False))

    return {
        "table": oracle_regression.get("table_recovery", "?"),
        "row": oracle_regression.get("row_recovery", "?"),
        "numeric": oracle_regression.get("numeric_exact", "?"),
        "scale": oracle_regression.get("scale_recoverability", "?"),
        "source_traceback": oracle_regression.get("source_traceback", "?"),
        "table_passed": table_ok,
        "row_passed": row_ok,
        "numeric_passed": numeric_ok,
        "scale_passed": scale_ok,
        "source_traceback_passed": source_ok,
        "passed": table_ok and row_ok and numeric_ok and scale_ok and source_ok,
    }


def _build_regression_gate(
    legacy_continuity: dict[str, Any],
    context_diff: dict[str, Any],
) -> dict[str, Any]:
    """Regression gate: Old Probe True Regression = 0."""
    true_regression = int(legacy_continuity.get("true_regression_count", 0))
    actual_regression = int(context_diff.get("actual_regression_count", 0))

    return {
        "old_probe_true_regression": true_regression,
        "full_document_context_actual_regression": actual_regression,
        "passed": true_regression == 0 and actual_regression == 0,
    }


def _build_safety_gate(seal: dict[str, Any]) -> dict[str, Any]:
    """Safety gate: no question/gold/governance/index/retrieval/production reads."""
    question_reads = int(seal.get("question_reads_before_seal", 0))
    gold_reads = int(seal.get("gold_reads_before_seal", 0))
    governance_reads = int(seal.get("governance_reads_before_seal", 0))
    index_builds = int(seal.get("index_builds", 0))
    retrieval_runs = int(seal.get("retrieval_runs", 0))
    production_writes = int(seal.get("production_index_writes", 0))

    return {
        "question_reads": question_reads,
        "gold_reads": gold_reads,
        "governance_reads": governance_reads,
        "index_builds": index_builds,
        "retrieval": retrieval_runs,
        "production_writes": production_writes,
        "passed": (
            question_reads == 0
            and gold_reads == 0
            and governance_reads == 0
            and index_builds == 0
            and retrieval_runs == 0
            and production_writes == 0
        ),
    }


# ---------------------------------------------------------------------------
# Next-gate decision
# ---------------------------------------------------------------------------


def _build_next_gate_decision(
    coverage_gate: dict[str, Any],
    identity_gate: dict[str, Any],
    oracle_gate: dict[str, Any],
    regression_gate: dict[str, Any],
    safety_gate: dict[str, Any],
) -> tuple[str, str]:
    """Determine the decision and next gate."""
    all_passed = (
        coverage_gate["passed"]
        and identity_gate["passed"]
        and oracle_gate["passed"]
        and regression_gate["passed"]
        and safety_gate["passed"]
    )
    if all_passed:
        return (
            "full_corpus_unified_structured_adapter_passed",
            "full_corpus_financial_semantic_graph",
        )
    if not oracle_gate["passed"]:
        return (
            "full_corpus_adapter_oracle_regression_blocked",
            "stop_and_fix_oracle_regression",
        )
    if not regression_gate["passed"]:
        return (
            "full_document_context_adapter_regression_blocked",
            "stop_and_fix_probe_regression",
        )
    return (
        "full_corpus_unified_structured_adapter_blocked",
        "stop_and_fix",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=R3_OUT)
    args = parser.parse_args()

    # 1. Load all previously generated artifacts
    artifact_specs = [
        ("gate-02-r3-protocol.json", "protocol"),
        ("input-integrity.json", "integrity"),
        ("probe-structural-diff-reconciliation.json", "reconciliation"),
        ("adapter-prediction-manifest.json", "manifest"),
        ("adapter-prediction-seal.json", "seal"),
        ("full-corpus-structure-metrics.json", "structure_metrics"),
        ("identity-integrity.json", "identity_integrity"),
        ("legacy-probe-identity-continuity.json", "legacy_continuity"),
        ("full-document-context-diff-audit.json", "context_diff"),
        ("post-seal-oracle-regression.json", "oracle_regression"),
        ("d-class-structural-presence.json", "d_class_presence"),
    ]

    artifacts: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for filename, key in artifact_specs:
        path = args.out_dir / filename
        if not path.is_file():
            missing.append(filename)
            continue
        artifacts[key] = _load_json(path)

    if missing:
        print("ERROR: The following artifacts are missing:")
        for name in missing:
            print(f"  {name}")
        print("\nRun the preceding gate scripts first.")
        return 1

    seal = artifacts["seal"]
    if not seal.get("sealed"):
        print("ERROR: Seal is not valid (sealed != true).")
        return 1

    # 2. Build acceptance gates
    coverage_gate = _build_coverage_gate(
        seal,
        artifacts["integrity"],
        artifacts["structure_metrics"],
        artifacts["reconciliation"],
    )
    identity_gate = _build_identity_gate(
        artifacts["structure_metrics"],
        artifacts["identity_integrity"],
    )
    oracle_gate = _build_oracle_gate(artifacts["oracle_regression"])
    regression_gate = _build_regression_gate(
        artifacts["legacy_continuity"],
        artifacts["context_diff"],
    )
    safety_gate = _build_safety_gate(seal)

    all_passed = (
        coverage_gate["passed"]
        and identity_gate["passed"]
        and oracle_gate["passed"]
        and regression_gate["passed"]
        and safety_gate["passed"]
    )

    # 3. Build next-gate decision
    decision, next_gate = _build_next_gate_decision(
        coverage_gate,
        identity_gate,
        oracle_gate,
        regression_gate,
        safety_gate,
    )

    # 4. Write acceptance.json
    acceptance = {
        "schema": "pdf-retrieval-v4/gate-02-r3/acceptance/v1",
        "gate": "pdf_retrieval_v4_gate_02_r3",
        "coverage_gate": coverage_gate,
        "identity_gate": identity_gate,
        "oracle_gate": oracle_gate,
        "regression_gate": regression_gate,
        "safety_gate": safety_gate,
        "d_class_presence_summary": {
            "d_class_total": artifacts["d_class_presence"].get("d_class_total", 0),
            "d_class_page_present": artifacts["d_class_presence"].get("d_class_page_present", 0),
            "d_class_table_present": artifacts["d_class_presence"].get("d_class_table_present", 0),
            "d_class_row_present": artifacts["d_class_presence"].get("d_class_row_present", 0),
            "b_class_total": artifacts["d_class_presence"].get("b_class_total", 0),
            "b_class_row_cell_exists": artifacts["d_class_presence"].get("b_class_row_cell_exists", 0),
        },
        "all_passed": all_passed,
        "decision": decision,
    }
    _write_json(args.out_dir / "acceptance.json", acceptance)

    # 5. Write next-gate.json
    next_gate_output = {
        "schema": "pdf-retrieval-v4/gate-02-r3/next-gate/v1",
        "current_gate": "pdf_retrieval_v4_gate_02_r3",
        "decision": decision,
        "next_gate": next_gate,
        "production_switch_allowed": False,
    }
    _write_json(args.out_dir / "next-gate.json", next_gate_output)

    # 6. Print summary
    print("Acceptance gate summary:")
    print(f"  Coverage gate:   {'PASS' if coverage_gate['passed'] else 'FAIL'}"
          f"  (docs {coverage_gate['documents']}, pages {coverage_gate['pages']},"
          f" missing {coverage_gate['missing_pages']})")
    print(f"  Identity gate:   {'PASS' if identity_gate['passed'] else 'FAIL'}"
          f"  (table={identity_gate['table_id_conflict']},"
          f" row={identity_gate['row_id_conflict']},"
          f" cell={identity_gate['cell_id_conflict']},"
          f" fk={identity_gate['broken_fk']})")
    print(f"  Oracle gate:     {'PASS' if oracle_gate['passed'] else 'FAIL'}"
          f"  (table={oracle_gate['table']}, row={oracle_gate['row']},"
          f" numeric={oracle_gate['numeric']}, scale={oracle_gate['scale']},"
          f" source={oracle_gate['source_traceback']})")
    print(f"  Regression gate: {'PASS' if regression_gate['passed'] else 'FAIL'}"
          f"  (true_regression={regression_gate['old_probe_true_regression']},"
          f" actual_regression={regression_gate['full_document_context_actual_regression']})")
    print(f"  Safety gate:     {'PASS' if safety_gate['passed'] else 'FAIL'}"
          f"  (q={safety_gate['question_reads']},"
          f" gold={safety_gate['gold_reads']},"
          f" gov={safety_gate['governance_reads']},"
          f" idx={safety_gate['index_builds']},"
          f" retr={safety_gate['retrieval']},"
          f" prod={safety_gate['production_writes']})")
    print(f"\n  All passed: {all_passed}")
    print(f"  Decision: {decision}")
    print(f"  Next gate: {next_gate}")
    print("  Production switch allowed: False")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
