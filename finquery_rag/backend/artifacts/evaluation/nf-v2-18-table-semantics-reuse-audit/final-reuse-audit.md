# NF-V2-18A-R2P0 Existing Table Semantics Reuse Audit

## Scope and freeze

Base: d1dc63fdb666959b0f57a4ce8786a579ecc8be8e

This was an audit-only pass. No new table-semantics layer was implemented; no retrieval behavior, parser behavior, model, benchmark, or production configuration was changed. The NF-V2-18A-R1 120-question development regression was not rerun.

## What is already present

The repository contains a real PDF physical adapter in src/pdf_retrieval_v4/full_corpus_adapter.py. It turns MinerU table markup plus PyMuPDF words/geometry into table fragments, rows and cells. table_html_parser.py expands rowspan/colspan grids, and adapter_identity.py produces deterministic table/row/cell identities.

The semantic boundary is the Gate 03 graph:

- semantic_graph_models.py: LogicalTable, SemanticRow, MetricPath, SemanticAxisBinding, AtomicFact, RowMatrix, and source traceback.
- metric_path_builder.py: ordered parent/leaf metric paths.
- temporal_axis_graph.py: per-cell point/duration/comparison/bucket/segment/category/unknown axes.
- semantic_scale_resolver.py and semantic_currency_resolver.py: explicit, hierarchical and fail-closed unit metadata.
- typed_evidence_emitters.py and semantic_evidence_catalog.py: canonical typed evidence and bridges.
- structural_joint_binder_v2.py, joint_operand_binder.py, and src/domain/financial_fact.py: existing Binder-compatible metric/period/value/unit/provenance contracts.

The canonical transition is therefore:

PDF physical extraction -> canonical physical fragment -> shared semantic graph -> typed evidence/catalog -> Binder/calculator.

## NF-OPT-08 reconstruction

NF-OPT-08 is located at src/evaluation/nf_opt_08.py and scripts/evaluation/run_nf_opt_08_r2_mapping_package.py. Its shadow schema preserves cell coordinates, spans, raw/normalized text, numeric value, bbox, parser provenance, table/page identity, header path and scale context. It has explicit manual-verification and hierarchy checks. The checked-in artifact reports 22 source records, 22 detected tables, 84 parsed pages, but zero required-row/cell/period/scale/currency recoveries and a blocked shadow-only decision; production indexes were not written. The three NF-OPT-08 contract test modules pass 24 tests.

NF-OPT-08 itself does not provide a completed production continuation merge. Later Gate 04B adds the reusable conservative logical-table/continuation contract. Its sealed artifact reports 36 candidate pairs, 2 positives, 34 negatives, 0 automatic merges, 14 blocked ambiguous, 22 do-not-merge, 2 missed, 0 false merges, zero source loss and 100% traceback.

## Header, period, and units

Multi-level headers already have two implementations: PDF cells receive header_path in full_corpus_adapter.py, while the historical mapping package has resolve_header(..., matrix_multilevel). Existing temporal code binds headers to cells, but its common vocabulary is point/duration plus normalized periods. A4 HTML is richer for project semantics (INSTANT, QUARTER, YTD, ANNUAL, UNKNOWN) and must not be downgraded or guessed. A small adapter should map these values conservatively while retaining the subtype.

Scale/currency are already separate contracts. The scale resolver has S0-S6 resolution levels and makes page/adjacent-page evidence candidate-only; the currency resolver does not infer currency from domicile. This is directly reusable by HTML after source fields are normalized.

## A4 HTML/iXBRL comparison

The A4 parser (scripts/evaluation/run_nf_v2_17a4_parse.py) emits table_id, row_id, cell_id, header_rows, flattened column_headers, period_columns, raw/normalized values, unit/currency/scale and source provenance. It also preserves ixbrl_contexts and ixbrl_facts with fact_id, concept, context_ref, unit_ref, decimals, raw value and period context.

The semantic overlap is strong, but A4 does not currently emit table_fragment_id, logical_table_id, per-column header_path, MetricPath, SemanticAxisBinding, or typed AtomicFact/row-matrix objects. This is why direct reuse is not enough. The safe solution is one HTML physical adapter at the existing boundary, then the same semantic passes.

iXBRL should be optional metadata on the canonical fact/cell provenance. It should not become a second evidence universe. Disagreement between table and iXBRL values must be retained and fail closed.

## R1 failure mapping

All 14 multi-level-header failures, all 5 period-column mismatches, and all 3 child-rank failures are at least partially addressable by existing capabilities. The first two map to header-path/axis reconstruction. The last maps to the existing table -> row -> cell/fact hierarchy and binder provenance, but not to a ready-made generic fine-ranker. This conclusion is a mapping of known R1 counts, not a rerun.

## Minimal R2 plan (not executed)

1. Add a single HTML physical adapter that emits the existing physical fragment/row/cell shape and attaches document/accession/raw-SHA provenance.
2. Run the existing row, metric-path, axis, scale/currency and typed-evidence passes.
3. Attach iXBRL fields as optional source provenance.
4. Reuse Gate 04B's conservative continuation contract for HTML only where structure supports it.
5. Feed the resulting canonical evidence to the existing catalog/bridge/Binder.
6. Add adapter fixtures for rowspan/colspan, compound three/six-month headers, instant/duration, QUARTER/YTD, units, iXBRL context and provenance.

No schema migration or new FinancialColumnContextV1/FinancialCellEvidenceV1 layer is justified.

## Decision

EXTEND_EXISTING_TABLE_SEMANTICS

There is substantial reusable implementation, but the A4 HTML representation needs an adapter and a careful period/iXBRL mapping. Production remains V1, and R2P0 stops at the audit.

