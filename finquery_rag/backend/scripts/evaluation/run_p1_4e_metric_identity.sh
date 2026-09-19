#!/usr/bin/env bash
# The three percentage_share cases the metric-identity change is about, x3.
#
# 002 and 004 failed with MISSING_OPERAND because their slots name metrics the
# ontology could not resolve, which made `_fact_matches_slot` return False for
# both sides.  003 is the case whose calculation was already proven correct
# (-0.0432 in P1.4c) and which has not bound since -- if it binds here and
# computes -0.0432, the parser/calculator path is confirmed end to end; if it
# stays MISSING_OPERAND, the remaining question is the binder's own candidate
# selection and output, and the enrichment stays as it is.
#
# Nine requests, not 360.  Rule B: in-process, so the backend stops first and
# restarts at the end.
set -u

REPO=/disk/qh/nano-finrag
BACKEND=$REPO/finquery_rag/backend
OUT=$REPO/artifacts/evaluation/p1-4e-metric-identity
FIXTURES=$BACKEND/benchmarks/tv2_canonical_v1/plan-fixtures-v5.jsonl
PY=$BACKEND/.venv/bin/python
LOG=$OUT/campaign.log
CASES=${CASES:-"tv2f01-s2-pctshare-002 tv2f01-s2-pctshare-003 tv2f01-s2-pctshare-004"}

BACKEND_PID=$(pgrep -f 'src.main:app' | head -1)
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== P1.4e metric identity start $(date -Is) ==="
echo "HEAD=$(cd "$REPO" && git rev-parse --short HEAD)  cases=$CASES"

restore() {
  if [ -n "${BACKEND_PID:-}" ]; then
    cd "$BACKEND" && nohup "$PY" -m uvicorn src.main:app --host 127.0.0.1 --port 18002 --workers 1 \
      >/tmp/backend-p14e.log 2>&1 &
    sleep 8
  fi
  echo "=== P1.4e complete $(date -Is) ==="
}
trap restore EXIT

if [ -n "${BACKEND_PID:-}" ]; then
  echo "stopping backend $BACKEND_PID"
  kill "$BACKEND_PID"
  for _ in $(seq 1 30); do
    pgrep -f 'src.main:app' >/dev/null || break
    sleep 1
  done
  pgrep -f 'src.main:app' >/dev/null && { echo "FATAL: backend still up"; exit 1; }
fi

CASES="$CASES" "$PY" - <<'PY'
import json, os
from pathlib import Path
base = Path("/disk/qh/nano-finrag/finquery_rag/backend/benchmarks/tv2_canonical_v1")
keep = set(os.environ["CASES"].split())
rows = [json.loads(l) for l in (base / "canonical-eval-v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
sel = [r for r in rows if r["id"] in keep]
Path("/tmp/pctshare-subset.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in sel) + "\n",
    encoding="utf-8", newline="\n")
print(f"  {len(sel)} questions: {[r['id'] for r in sel]}", flush=True)
PY

for i in 1 2 3; do
  echo "--- run $i ---"
  ( cd "$BACKEND" && "$PY" scripts/evaluation/run_p1_2_dual_track_benchmark.py \
      --track replay --fixtures "$FIXTURES" --eval-set /tmp/pctshare-subset.jsonl \
      --out-dir "$OUT/run$i" ) || echo "RUN FAILED: $i"
done

echo "=== P1.4e metric identity complete $(date -Is) ==="
