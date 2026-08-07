#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/pdf-sr-v2/finquery_rag/backend

PY=/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/pdf-sr-v2/finquery_rag/backend/.venv/bin/python

echo "=== Python version ==="
$PY --version
echo "=== PyMuPDF check ==="
$PY -c "import fitz; print('PyMuPDF', fitz.__version__)"
echo "=== Running Gate 02 R3 Adapter ==="
$PY scripts/evaluation/run_pdf_v4_gate_02_r3_adapter.py \
    --r2-commit 6f5990f \
    --code-commit working-tree 2>&1
echo "=== Adapter done (exit $?) ==="
