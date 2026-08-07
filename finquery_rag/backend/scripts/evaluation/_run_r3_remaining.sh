#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/pdf-sr-v2/finquery_rag/backend

PY=/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/pdf-sr-v2/finquery_rag/backend/.venv/bin/python
SCRIPTS=scripts/evaluation

echo "========================================"
echo "Step 1/7: Reconcile probe structural diff"
echo "========================================"
$PY $SCRIPTS/reconcile_probe_structural_diff_r3.py
echo ""

echo "========================================"
echo "Step 2/7: Audit full-document context diff"
echo "========================================"
$PY $SCRIPTS/audit_full_document_context_diff_r3.py
echo ""

echo "========================================"
echo "Step 3/7: Audit legacy identity continuity"
echo "========================================"
$PY $SCRIPTS/audit_legacy_identity_continuity_r3.py
echo ""

echo "========================================"
echo "Step 4/7: Seal predictions"
echo "========================================"
$PY $SCRIPTS/seal_pdf_v4_gate_02_r3.py
echo ""

echo "========================================"
echo "Step 5/7: Score Oracle regression (post-seal)"
echo "========================================"
$PY $SCRIPTS/score_pdf_v4_gate_02_r3_oracle.py || true
echo ""

echo "========================================"
echo "Step 6/7: Observe D-class presence (post-seal)"
echo "========================================"
$PY $SCRIPTS/observe_d_class_presence_r3.py
echo ""

echo "========================================"
echo "Step 7/7: Finalize acceptance"
echo "========================================"
$PY $SCRIPTS/finalize_pdf_v4_gate_02_r3.py || true
echo ""

echo "========================================"
echo "All steps complete."
echo "========================================"
ls -la artifacts/evaluation/pdf-retrieval-v4-gate-02-r3/
