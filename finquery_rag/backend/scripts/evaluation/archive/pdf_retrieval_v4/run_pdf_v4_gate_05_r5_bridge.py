#!/usr/bin/env python3
"""Gate 05 R5 — Full-corpus Candidate Evidence Bridge Builder.

Loads all 38,319 Production Candidates + Full-corpus Semantic Graph,
bridges each candidate to semantic evidence, and generates structured views.

NO Retrieval, NO Question/Gold access, NO Index builds.

Usage:
    python3 scripts/evaluation/run_pdf_v4_gate_05_r5_bridge.py

Outputs (artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/):
    bridge-results.jsonl      — one BridgeResult per candidate
    structured-views.jsonl    — one CandidateStructuredView per Grade-A candidate
    bridge-summary.json       — aggregate statistics
    bridge-validation.json    — pre-seal validation results
    bridge-eligibility.json   — eligibility breakdown
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure backend is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.pdf_retrieval_v4.candidate_bridge_models import (  # noqa: E402
    is_structured_eligible,
)
from src.pdf_retrieval_v4.candidate_evidence_bridge import CandidateEvidenceBridge  # noqa: E402
from src.pdf_retrieval_v4.candidate_signature import build_candidate_signature  # noqa: E402
from src.pdf_retrieval_v4.candidate_structured_view import StructuredViewBuilder  # noqa: E402
from src.pdf_retrieval_v4.bridge_equivalence import BridgeEquivalence  # noqa: E402
from src.pdf_retrieval_v4.bridge_validator import BridgeValidator  # noqa: E402
from src.pdf_retrieval_v4.semantic_evidence_catalog import load_catalog  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GATE_03_R2_DIR = BACKEND_DIR / "artifacts/evaluation/pdf-retrieval-v4-gate-03-r2"
GATE_08_R2_DIR = BACKEND_DIR / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2"
CANDIDATE_VIEWS_PATH = GATE_08_R2_DIR / "candidate-views/view-pairs.jsonl"
OUTPUT_DIR = BACKEND_DIR / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5"


def read_jsonl(path: Path):
    """Yield JSON objects from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: list) -> None:
    """Write records to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(path: Path, data) -> None:
    """Write JSON to a file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    print("=" * 70)
    print("Gate 05 R5 — Full-corpus Candidate Evidence Bridge Builder")
    print("=" * 70)

    start_time = time.time()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load Semantic Evidence Catalog
    # ------------------------------------------------------------------
    print("\n[1/6] Loading Semantic Evidence Catalog from Gate 03 R2...")
    t0 = time.time()
    catalog = load_catalog(GATE_03_R2_DIR)
    print(f"  Loaded {catalog.total_count} evidence units in {time.time() - t0:.1f}s")
    print(f"  Stats: {json.dumps(catalog.stats(), indent=2)}")

    # ------------------------------------------------------------------
    # Step 2: Load Production Candidates and Build Signatures
    # ------------------------------------------------------------------
    print("\n[2/6] Loading Production Candidates...")
    t0 = time.time()
    signatures = []
    candidate_count = 0
    for record in read_jsonl(CANDIDATE_VIEWS_PATH):
        sig = build_candidate_signature(record)
        signatures.append(sig)
        candidate_count += 1
    print(f"  Loaded {candidate_count} candidates in {time.time() - t0:.1f}s")

    # Eligibility breakdown
    eligible_count = sum(1 for s in signatures if is_structured_eligible(s.block_type))
    print(f"  Structured eligible: {eligible_count}")
    print(f"  Raw-only: {candidate_count - eligible_count}")

    # ------------------------------------------------------------------
    # Step 3: Bridge All Candidates
    # ------------------------------------------------------------------
    print("\n[3/6] Bridging candidates to semantic evidence...")
    t0 = time.time()
    bridge = CandidateEvidenceBridge(catalog)
    results = bridge.bridge_all(signatures)
    elapsed = time.time() - t0
    print(
        f"  Bridged {len(results)} candidates in {elapsed:.1f}s ({len(results) / elapsed:.0f}/s)"
    )

    # Apply equivalence check
    equivalence = BridgeEquivalence(catalog)
    print(f"  Equivalent groups: {equivalence.group_count}")
    results = [equivalence.check_equivalent_bridge(r) for r in results]

    # Summary
    summary = bridge.build_summary(results)
    print("\n  Bridge Summary:")
    print(f"    Total candidates:     {summary['total_candidates']}")
    print(f"    Grade-A count:        {summary['grade_a_count']}")
    print(f"    Grade-B count:        {summary['grade_b_count']}")
    print(f"    Unmapped count:       {summary['unmapped_count']}")
    print(f"    Grade breakdown:      {json.dumps(summary['grade_counts'])}")

    # ------------------------------------------------------------------
    # Step 4: Build Structured Views for Grade-A Candidates
    # ------------------------------------------------------------------
    print("\n[4/6] Building Structured Views for Grade-A candidates...")
    t0 = time.time()
    view_builder = StructuredViewBuilder(catalog)
    views = view_builder.build_all_views(signatures, results)
    print(f"  Built {len(views)} structured views in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: Pre-seal Validation
    # ------------------------------------------------------------------
    print("\n[5/6] Running pre-seal validation...")
    validator = BridgeValidator(equivalence)
    validation = validator.validate(signatures, results, views)

    # Cross-page validation
    evidence_lookup: dict[str, tuple[str, int]] = {}
    for ev in catalog.get_all():
        evidence_lookup[ev.evidence_id] = (ev.document_id, ev.pdf_page)
    cross_violations = validator.validate_cross_page(
        signatures, results, evidence_lookup
    )
    if cross_violations:
        validation["violations"].extend(cross_violations)
        validation["passed"] = False

    print(f"  Validation passed: {validation['passed']}")
    print(f"  Violations: {len(validation['violations'])}")
    if validation["metrics"]:
        for k, v in validation["metrics"].items():
            if k != "top_fanout":
                print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # Step 6: Write Artifacts
    # ------------------------------------------------------------------
    print("\n[6/6] Writing artifacts...")

    # Bridge results
    write_jsonl(OUTPUT_DIR / "bridge-results.jsonl", [r.to_dict() for r in results])
    print(f"  Wrote bridge-results.jsonl ({len(results)} records)")

    # Structured views
    write_jsonl(OUTPUT_DIR / "structured-views.jsonl", [v.to_dict() for v in views])
    print(f"  Wrote structured-views.jsonl ({len(views)} records)")

    # Summary
    write_json(OUTPUT_DIR / "bridge-summary.json", summary)
    print("  Wrote bridge-summary.json")

    # Validation
    write_json(OUTPUT_DIR / "bridge-validation.json", validation)
    print("  Wrote bridge-validation.json")

    # Eligibility
    eligibility = {
        "total_candidate": candidate_count,
        "structured_eligible": eligible_count,
        "raw_only": candidate_count - eligible_count,
        "block_type_breakdown": {},
    }
    from collections import Counter

    bt_counts = Counter(s.block_type for s in signatures)
    eligibility["block_type_breakdown"] = dict(bt_counts)
    write_json(OUTPUT_DIR / "bridge-eligibility.json", eligibility)
    print("  Wrote bridge-eligibility.json")

    # Equivalence stats
    write_json(OUTPUT_DIR / "bridge-equivalence.json", equivalence.stats())
    print("  Wrote bridge-equivalence.json")

    elapsed_total = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Gate 05 R5 Bridge Build complete in {elapsed_total:.1f}s")
    print(f"  Candidates:   {candidate_count}")
    print(f"  Grade-A:      {summary['grade_a_count']}")
    print(f"  Views:        {len(views)}")
    print(f"  Validation:   {'PASSED' if validation['passed'] else 'FAILED'}")
    print(f"{'=' * 70}")

    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
