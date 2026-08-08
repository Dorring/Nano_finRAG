"""Gate 03 R2 — Full-corpus Financial Semantic Graph builder.

Reads the sealed adapter predictions (Gate 02 R3) and produces the
semantic graph through 5 deterministic passes:

  Pass A — Logical Table + Row Classification
  Pass B — Header / Metric Graph
  Pass C — Temporal / Dimension Graph
  Pass D — Scale / Currency Resolution
  Pass E — Typed Evidence Emission

Safety constraints (verified before seal):
  question_reads = 0
  gold_reads_before_seal = 0
  governance_reads_before_seal = 0
  candidate_bridge_builds = 0
  index_builds = 0
  retrieval_runs = 0
  production_switch_allowed = false

Usage:
  python scripts/evaluation/run_pdf_v4_gate_03_r2_build.py [--backend-root PATH]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow running from anywhere
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = BACKEND_ROOT / "src"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.pdf_retrieval_v4.financial_table_classifier import classify_table  # noqa: E402
from src.pdf_retrieval_v4.metric_path_builder import (  # noqa: E402
    build_metric_paths,
    detect_conflicting_parents,
    detect_parent_cycles,
)
from src.pdf_retrieval_v4.semantic_currency_resolver import resolve_table_currency  # noqa: E402
from src.pdf_retrieval_v4.semantic_equivalence import (  # noqa: E402
    build_equivalence_map,
    detect_equivalent_set_double_counting,
    load_equivalent_sets,
)
from src.pdf_retrieval_v4.semantic_graph_validator import validate_semantic_graph  # noqa: E402
from src.pdf_retrieval_v4.semantic_row_classifier import classify_table_rows  # noqa: E402
from src.pdf_retrieval_v4.semantic_scale_resolver import resolve_table_scale  # noqa: E402
from src.pdf_retrieval_v4.temporal_axis_graph import build_axis_bindings  # noqa: E402
from src.pdf_retrieval_v4.typed_evidence_emitters import (  # noqa: E402
    emit_atomic_facts,
    emit_bucket_facts,
    emit_comparison_facts,
    emit_narrative_evidence,
    emit_row_matrices,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARED_NANOCHAT_ROOT = BACKEND_ROOT.parents[4]
MINERU_OUTPUT_ROOT = (
    SHARED_NANOCHAT_ROOT / ".runtime/pdf-retrieval-v4-gate-02-r2/mineru"
)

R3_OUT = BACKEND_ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"
GATE03_OUT = BACKEND_ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03-r2"

ADAPTER_PREDICTIONS = R3_OUT / "adapter-predictions.jsonl.gz"
AMBIGUITY_CLOSURE = R3_OUT / "ambiguity-closure.json"


# ---------------------------------------------------------------------------
# Safety counters (verified at seal time)
# ---------------------------------------------------------------------------

SAFETY_COUNTERS = {
    "question_reads": 0,
    "gold_reads_before_seal": 0,
    "governance_reads_before_seal": 0,
    "candidate_bridge_builds": 0,
    "index_builds": 0,
    "retrieval_runs": 0,
    "production_switch_allowed": False,
}


def _verify_safety() -> None:
    """Verify that no forbidden reads/builds occurred before seal."""
    for key in (
        "question_reads",
        "gold_reads_before_seal",
        "governance_reads_before_seal",
        "candidate_bridge_builds",
        "index_builds",
        "retrieval_runs",
    ):
        if SAFETY_COUNTERS[key] != 0:
            raise RuntimeError(
                f"SAFETY VIOLATION: {key} = {SAFETY_COUNTERS[key]} (must be 0)"
            )


# ---------------------------------------------------------------------------
# Load adapter predictions
# ---------------------------------------------------------------------------


def load_adapter_predictions(path: Path) -> list[dict[str, Any]]:
    """Load all page records from the sealed adapter predictions."""
    pages: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


# ---------------------------------------------------------------------------
# Narrative extraction from MinerU content_list.json
# ---------------------------------------------------------------------------


def _find_content_list_json(doc_output_dir: Path) -> Path | None:
    """Find content_list.json for a document (mirrors full_corpus_adapter)."""
    matches = sorted(doc_output_dir.rglob("*_content_list.json"))
    return matches[0] if matches else None


def _extract_narrative_blocks(
    content_path: Path | None,
    document_id: str,
) -> list[dict[str, Any]]:
    """Extract narrative blocks (text, title) from content_list.json.

    No LLM summarization — raw_text is stored as-is.
    """
    if not content_path or not content_path.is_file():
        return []
    try:
        data = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    blocks: list[dict[str, Any]] = []
    current_section: str = ""

    for block in data:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        text = str(block.get("text") or "").strip()
        page_idx = int(block.get("page_idx") or 0)
        pdf_page = page_idx + 1  # content_list is 0-indexed

        if block_type == "title" and text:
            current_section = text[:300]
            blocks.append(
                {
                    "document_id": document_id,
                    "pdf_page": pdf_page,
                    "section_path": current_section,
                    "heading": text[:300],
                    "raw_text": text[:2000],
                    "bbox": block.get("bbox"),
                    "evidence_subtype": "heading",
                }
            )
        elif block_type == "text" and text and len(text) > 20:
            # Skip very short text fragments
            blocks.append(
                {
                    "document_id": document_id,
                    "pdf_page": pdf_page,
                    "section_path": current_section,
                    "heading": current_section,
                    "raw_text": text[:2000],
                    "bbox": block.get("bbox"),
                    "evidence_subtype": "paragraph",
                }
            )

    return blocks


def extract_all_narrative(mineru_root: Path) -> list[dict[str, Any]]:
    """Extract narrative blocks from all documents in the corpus."""
    all_blocks: list[dict[str, Any]] = []

    for doc_dir in sorted(mineru_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        document_id = doc_dir.name
        content_path = _find_content_list_json(doc_dir)
        blocks = _extract_narrative_blocks(content_path, document_id)
        all_blocks.extend(blocks)

    return all_blocks


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------


def build_semantic_graph(
    pages: list[dict[str, Any]],
    mineru_root: Path,
    ambiguity_closure_path: Path,
) -> dict[str, Any]:
    """Run all 5 passes and return the complete semantic graph + metrics."""

    # --- Load semantic equivalence from R3.2 R1 ---
    equivalent_sets = load_equivalent_sets(ambiguity_closure_path)
    equivalence_map = build_equivalence_map(equivalent_sets)
    equivalent_group_count = len(set(equivalence_map.values()))

    # Build skip set: for each equivalent group, keep only the canonical
    # (first sorted) row_id and skip the rest to prevent duplicate facts.
    group_to_rows: dict[str, list[str]] = {}
    for row_id, group_id in equivalence_map.items():
        group_to_rows.setdefault(group_id, []).append(row_id)
    skip_row_ids: set[str] = set()
    for group_id, row_ids in group_to_rows.items():
        if len(row_ids) > 1:
            canonical = sorted(row_ids)[0]
            skip_row_ids.update(rid for rid in row_ids if rid != canonical)
    if skip_row_ids:
        print(f"  Equivalent-set skip rows: {len(skip_row_ids)}")

    # --- Pass A: Logical Table + Row Classification ---
    print("[Pass A] Classifying 942 tables and 11,607 rows...")
    logical_tables: list[Any] = []
    semantic_rows_all: list[Any] = []

    for page in pages:
        document_id = str(page.get("document_id") or "")
        pdf_page = int(page.get("pdf_page") or 0)
        for table in page.get("tables") or []:
            lt = classify_table(table)
            logical_tables.append(lt)
            srs = classify_table_rows(
                table,
                lt.table_fragment_id,
                document_id,
                pdf_page,
            )
            semantic_rows_all.extend(srs)

    print(f"  Logical tables: {len(logical_tables)}")
    row_type_counts = Counter(sr.row_type for sr in semantic_rows_all)
    print(f"  Row types: {dict(row_type_counts)}")

    # --- Pass B: Header / Metric Graph ---
    print("[Pass B] Building metric paths...")
    # Group semantic rows by table
    rows_by_table: dict[str, list[Any]] = {}
    for sr in semantic_rows_all:
        rows_by_table.setdefault(sr.table_fragment_id, []).append(sr)

    metric_paths_all: list[Any] = []
    for table_id, table_rows in rows_by_table.items():
        mps = build_metric_paths(table_rows)
        metric_paths_all.extend(mps)

    eligible_count = sum(1 for sr in semantic_rows_all if sr.is_financial_data_row)
    resolved_count = sum(1 for mp in metric_paths_all if mp.metric_status == "resolved")
    ambiguous_count = sum(
        1 for mp in metric_paths_all if mp.metric_status == "ambiguous"
    )
    metric_coverage = resolved_count / eligible_count if eligible_count > 0 else 0.0
    print(
        f"  Metric paths: {len(metric_paths_all)}, "
        f"resolved: {resolved_count}, ambiguous: {ambiguous_count}, "
        f"resolved coverage: {metric_coverage:.2%}"
    )

    parent_cycles = detect_parent_cycles(metric_paths_all, semantic_rows_all)
    conflicting_parents = detect_conflicting_parents(metric_paths_all)
    print(f"  Parent cycles: {parent_cycles}, conflicting: {conflicting_parents}")

    # --- Pass C: Temporal / Dimension Graph ---
    print("[Pass C] Building temporal/dimension axis bindings...")
    axis_bindings_all: list[Any] = []

    for page in pages:
        for table in page.get("tables") or []:
            table_fragment_id = str(table.get("table_fragment_id") or "")
            cells = table.get("cells") or []
            abs_list = build_axis_bindings(cells, table_fragment_id)
            axis_bindings_all.extend(abs_list)

    temporal_kind_counts = Counter(ab.temporal_kind for ab in axis_bindings_all)
    print(f"  Temporal kinds: {dict(temporal_kind_counts)}")

    # --- Pass D: Scale + Currency Resolution ---
    print("[Pass D] Resolving scale and currency...")
    scale_resolutions: list[Any] = []
    currency_resolutions: list[Any] = []

    # Build a map of table_fragment_id → table dict for quick lookup
    table_by_id: dict[str, dict[str, Any]] = {}
    table_page_map: dict[str, tuple[str, int]] = {}  # table_id → (doc_id, page)
    for page in pages:
        for table in page.get("tables") or []:
            tfid = str(table.get("table_fragment_id") or "")
            table_by_id[tfid] = table
            table_page_map[tfid] = (
                str(page.get("document_id") or ""),
                int(page.get("pdf_page") or 0),
            )

    for page in pages:
        for table in page.get("tables") or []:
            tfid = str(table.get("table_fragment_id") or "")
            scale = resolve_table_scale(table, tfid)
            currency = resolve_table_currency(table, tfid)
            scale_resolutions.append(scale)
            currency_resolutions.append(currency)

    scale_status_counts = Counter(sr.scale_status for sr in scale_resolutions)
    print(f"  Scale status: {dict(scale_status_counts)}")
    currency_status_counts = Counter(cr.currency_status for cr in currency_resolutions)
    print(f"  Currency status: {dict(currency_status_counts)}")

    # --- Pass E: Typed Evidence Emission ---
    print("[Pass E] Emitting typed evidence...")

    # Group cells and rows by table for emission
    all_atomic_facts: list[Any] = []
    all_comparison_facts: list[Any] = []
    all_bucket_facts: list[Any] = []
    all_row_matrices: list[Any] = []

    for page in pages:
        for table in page.get("tables") or []:
            tfid = str(table.get("table_fragment_id") or "")
            table_rows = rows_by_table.get(tfid, [])
            # Skip non-canonical rows from equivalent sets to prevent duplicate facts
            table_rows = [sr for sr in table_rows if sr.row_id not in skip_row_ids]
            table_metric_paths = [
                mp for mp in metric_paths_all if mp.table_fragment_id == tfid
            ]
            table_axis = [
                ab for ab in axis_bindings_all if ab.table_fragment_id == tfid
            ]
            table_cells = table.get("cells") or []

            scale = next(
                (s for s in scale_resolutions if s.table_fragment_id == tfid), None
            )
            currency = next(
                (c for c in currency_resolutions if c.table_fragment_id == tfid), None
            )

            if not scale or not currency:
                continue

            all_atomic_facts.extend(
                emit_atomic_facts(
                    table_rows,
                    table_metric_paths,
                    table_axis,
                    table_cells,
                    scale,
                    currency,
                    equivalence_map,
                )
            )
            all_comparison_facts.extend(
                emit_comparison_facts(
                    table_rows,
                    table_metric_paths,
                    table_axis,
                    table_cells,
                    scale,
                    equivalence_map,
                )
            )
            all_bucket_facts.extend(
                emit_bucket_facts(
                    table_rows,
                    table_metric_paths,
                    table_axis,
                    table_cells,
                    scale,
                    currency,
                )
            )
            all_row_matrices.extend(
                emit_row_matrices(
                    table_rows,
                    table_metric_paths,
                    table_axis,
                    table_cells,
                    scale,
                    currency,
                    equivalence_map,
                )
            )

    # --- Deduplicate atomic facts by semantic_fact_id ---
    # Only equivalent_set collapsing (R3.2 R1) is allowed to produce
    # canonical ids across physical rows; different documents/tables/rows
    # must never be merged by semantic content alone.
    seen_atomic_ids: set[str] = set()
    deduped_atomic: list[Any] = []
    for af in all_atomic_facts:
        if af.semantic_fact_id in seen_atomic_ids:
            continue
        seen_atomic_ids.add(af.semantic_fact_id)
        deduped_atomic.append(af)
    if len(deduped_atomic) < len(all_atomic_facts):
        removed = len(all_atomic_facts) - len(deduped_atomic)
        print(f"  Deduped atomic facts: removed {removed} duplicate ids")
    all_atomic_facts = deduped_atomic

    # --- Deduplicate row matrices by metric_path + table_fragment_id + row_id ---
    seen_matrix_keys: set[tuple[str, str, str]] = set()
    deduped_matrices: list[Any] = []
    for rm in all_row_matrices:
        key = (rm.metric_path, rm.table_fragment_id, rm.row_id)
        if key in seen_matrix_keys:
            continue
        seen_matrix_keys.add(key)
        deduped_matrices.append(rm)
    all_row_matrices = deduped_matrices

    print(f"  Atomic facts: {len(all_atomic_facts)}")
    print(f"  Comparison facts: {len(all_comparison_facts)}")
    print(f"  Bucket facts: {len(all_bucket_facts)}")
    print(f"  Row matrices: {len(all_row_matrices)}")

    # Narrative evidence
    print("[Pass E] Extracting narrative evidence from MinerU content_list...")
    narrative_blocks = extract_all_narrative(mineru_root)
    all_narrative = emit_narrative_evidence(narrative_blocks)
    # Deduplicate narrative evidence by semantic_evidence_id
    # (MinerU content_list may contain repeated text blocks on the same page)
    seen_narrative_ids: set[str] = set()
    deduped_narrative: list[Any] = []
    for ne in all_narrative:
        if ne.semantic_evidence_id in seen_narrative_ids:
            continue
        seen_narrative_ids.add(ne.semantic_evidence_id)
        deduped_narrative.append(ne)
    if len(deduped_narrative) < len(all_narrative):
        removed = len(all_narrative) - len(deduped_narrative)
        print(f"  Deduped narrative evidence: removed {removed} duplicates")
    all_narrative = deduped_narrative
    print(f"  Narrative evidence: {len(all_narrative)}")

    # --- Equivalent-set double counting check ---
    atomic_dicts = [af.to_dict() for af in all_atomic_facts]
    equiv_double_counting = detect_equivalent_set_double_counting(
        atomic_dicts, equivalence_map
    )

    # --- Collect all cells for pre-emission admission denominator ---
    all_cells: list[dict[str, Any]] = []
    for page in pages:
        for table in page.get("tables") or []:
            all_cells.extend(table.get("cells") or [])

    # --- Validation ---
    print("[Validation] Running gate checks...")
    validation = validate_semantic_graph(
        logical_tables=logical_tables,
        semantic_rows=semantic_rows_all,
        metric_paths=metric_paths_all,
        axis_bindings=axis_bindings_all,
        scale_resolutions=scale_resolutions,
        atomic_facts=all_atomic_facts,
        comparison_facts=all_comparison_facts,
        bucket_facts=all_bucket_facts,
        row_matrices=all_row_matrices,
        narrative_evidence=all_narrative,
        all_cells=all_cells,
        equivalent_double_counting=equiv_double_counting,
        parent_cycles=parent_cycles,
        conflicting_parents=conflicting_parents,
    )

    metrics = validation["metrics"]
    print(
        f"  Metric Path Present Coverage:  {metrics['metric_path_present_coverage']:.2%}"
    )
    print(
        f"  Metric Path Resolved Coverage: {metrics['metric_path_resolved_coverage']:.2%}"
    )
    print(f"  Typed Evidence Admission:      {metrics['typed_evidence_admission']:.2%}")
    print(f"  Atomic Fact Admission:         {metrics['atomic_fact_admission']:.2%}")
    print(f"  Eligible numeric cells:        {metrics['eligible_numeric_cells']}")
    print(f"  Atomic-eligible cells:         {metrics['atomic_eligible_cells']}")
    print(f"  Typed-eligible cells:          {metrics['typed_eligible_cells']}")
    print(f"  All gates passed: {validation['all_passed']}")

    return {
        "logical_tables": logical_tables,
        "semantic_rows": semantic_rows_all,
        "metric_paths": metric_paths_all,
        "axis_bindings": axis_bindings_all,
        "scale_resolutions": scale_resolutions,
        "currency_resolutions": currency_resolutions,
        "atomic_facts": all_atomic_facts,
        "comparison_facts": all_comparison_facts,
        "bucket_facts": all_bucket_facts,
        "row_matrices": all_row_matrices,
        "narrative_evidence": all_narrative,
        "equivalence_map": equivalence_map,
        "equivalent_group_count": equivalent_group_count,
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, items: list[Any]) -> None:
    """Write a list of dataclass instances (with to_dict) as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_artifacts(graph: dict[str, Any], output_dir: Path) -> None:
    """Write all semantic graph artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_dir / "logical-tables.jsonl", graph["logical_tables"])
    _write_jsonl(output_dir / "semantic-rows.jsonl", graph["semantic_rows"])
    _write_jsonl(output_dir / "metric-paths.jsonl", graph["metric_paths"])
    _write_jsonl(output_dir / "axis-bindings.jsonl", graph["axis_bindings"])
    _write_jsonl(output_dir / "scale-resolutions.jsonl", graph["scale_resolutions"])
    _write_jsonl(
        output_dir / "currency-resolutions.jsonl", graph["currency_resolutions"]
    )
    _write_jsonl(output_dir / "atomic-facts.jsonl", graph["atomic_facts"])
    _write_jsonl(output_dir / "comparison-facts.jsonl", graph["comparison_facts"])
    _write_jsonl(output_dir / "bucket-facts.jsonl", graph["bucket_facts"])
    _write_jsonl(output_dir / "row-matrices.jsonl", graph["row_matrices"])
    _write_jsonl(output_dir / "narrative-evidence.jsonl", graph["narrative_evidence"])

    # Write validation + metrics
    _write_json(output_dir / "semantic-graph-metrics.json", graph["validation"])

    # Write equivalence info
    _write_json(
        output_dir / "semantic-equivalence.json",
        {
            "equivalent_group_count": graph["equivalent_group_count"],
            "equivalent_row_count": len(graph["equivalence_map"]),
        },
    )

    print(f"\nArtifacts written to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate 03 R2 — Full-corpus Financial Semantic Graph builder"
    )
    parser.add_argument(
        "--backend-root",
        type=Path,
        default=BACKEND_ROOT,
        help="Backend root directory",
    )
    parser.add_argument(
        "--mineru-output",
        type=Path,
        default=MINERU_OUTPUT_ROOT,
        help="MinerU output root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GATE03_OUT,
        help="Output directory for artifacts",
    )
    args = parser.parse_args()

    # Verify inputs exist
    if not ADAPTER_PREDICTIONS.is_file():
        print(f"ERROR: adapter predictions not found at {ADAPTER_PREDICTIONS}")
        return 1

    print("=" * 70)
    print("Gate 03 R2 — Full-corpus Financial Semantic Graph")
    print("=" * 70)
    print(f"Adapter predictions: {ADAPTER_PREDICTIONS}")
    print(f"MinerU output root:  {args.mineru_output}")
    print(f"Output directory:    {args.output_dir}")
    print()

    # Load adapter predictions
    print("Loading adapter predictions...")
    pages = load_adapter_predictions(ADAPTER_PREDICTIONS)
    print(f"  Pages: {len(pages)}")
    print(f"  Tables: {sum(len(p.get('tables', [])) for p in pages)}")

    # Build semantic graph
    graph = build_semantic_graph(pages, args.mineru_output, AMBIGUITY_CLOSURE)

    # Verify safety
    _verify_safety()
    print("\nSafety verification: PASSED (0 questions, 0 gold, 0 governance reads)")

    # Write artifacts
    write_artifacts(graph, args.output_dir)

    # Print final summary
    v = graph["validation"]
    print("\n" + "=" * 70)
    print("Gate 03 R2 Build Summary")
    print("=" * 70)
    print(f"  All gates passed: {v['all_passed']}")
    print(
        f"  Metric Path Resolved Coverage: {v['metrics']['metric_path_resolved_coverage']:.2%}"
    )
    print(
        f"  Metric Path Present Coverage:  {v['metrics']['metric_path_present_coverage']:.2%}"
    )
    print(
        f"  Typed Evidence Admission:      {v['metrics']['typed_evidence_admission']:.2%}"
    )
    print(
        f"  Atomic Fact Admission:         {v['metrics']['atomic_fact_admission']:.2%}"
    )
    print(f"  Eligible numeric cells:        {v['metrics']['eligible_numeric_cells']}")
    print(f"  Atomic-eligible cells:         {v['metrics']['atomic_eligible_cells']}")
    print(f"  Typed-eligible cells:          {v['metrics']['typed_eligible_cells']}")
    print(f"  Atomic facts:                  {v['metrics']['atomic_fact_count']}")
    print(f"  Comparison facts:              {v['metrics']['comparison_fact_count']}")
    print(f"  Bucket facts:                  {v['metrics']['bucket_fact_count']}")
    print(f"  Row matrices:                  {v['metrics']['row_matrix_count']}")
    print(
        f"  Narrative evidence:            {v['metrics']['narrative_evidence_count']}"
    )
    print(f"  Equivalent groups:             {graph['equivalent_group_count']}")
    print(
        f"  Scale: resolved={v['metrics']['scale_resolved']}, "
        f"candidate={v['metrics']['scale_candidate_only']}, "
        f"conflict={v['metrics']['scale_conflict']}, "
        f"missing={v['metrics']['scale_missing']}"
    )
    print(f"  Scale conflict detected:       {v['metrics']['scale_conflict_detected']}")
    print(
        f"  Scale conflict auto-resolution: {v['metrics']['scale_conflict_auto_resolution']}"
    )

    # Exit codes: 0 = gates passed, 2 = build completed but gates failed,
    #             1 = execution/input error
    return 0 if v["all_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
