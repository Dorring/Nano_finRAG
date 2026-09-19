#!/usr/bin/env bash
# P1.4F-1: which half of 2713f8f turned pctshare-001 from RELEASED to NOT_RELEASED?
#
# 2713f8f contains two changes that can reach 001, and they are toggled
# independently here so the four combinations are measured rather than argued:
#
#   P14F_VOCAB=new|old        whether total_operating_expenses is in the ontology
#   P14F_MATCHING=identity|old  whether the matching sites use metric_identity
#
# The packet, fixture and provider are identical across all four; only these two
# switches move.  Each combination runs three times, because a single run of an
# LLM-mediated path cannot separate a 1-case effect from repeat noise -- which is
# the mistake this bisect exists to stop repeating.
#
# Rule B: in-process, so the backend stops first and restarts at the end.
set -u

REPO=/disk/qh/nano-finrag
BACKEND=$REPO/finquery_rag/backend
OUT=$REPO/artifacts/evaluation/p1-4f1-bisect-001
FIXTURES=$BACKEND/benchmarks/tv2_canonical_v1/plan-fixtures-v5.jsonl
PY=$BACKEND/.venv/bin/python
LOG=$OUT/bisect.log
CASES=${CASES:-"tv2f01-s2-pctshare-001"}

BACKEND_PID=$(pgrep -f 'src.main:app' | head -1)
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== P1.4F-1 bisect start $(date -Is) ==="
echo "HEAD=$(cd "$REPO" && git rev-parse --short HEAD)  case=$CASES"

restore() {
  if [ -n "${BACKEND_PID:-}" ]; then
    cd "$BACKEND" && nohup "$PY" -m uvicorn src.main:app --host 127.0.0.1 --port 18002 --workers 1 \
      >/tmp/backend-p14f1.log 2>&1 &
    sleep 8
  fi
  echo "=== P1.4F-1 bisect end $(date -Is) ==="
}
trap restore EXIT

if [ -n "${BACKEND_PID:-}" ]; then
  echo "stopping backend $BACKEND_PID"
  kill "$BACKEND_PID"
  for _ in $(seq 1 60); do
    pgrep -f 'src.main:app' >/dev/null || break
    sleep 1
  done
  pgrep -f 'src.main:app' >/dev/null && { echo "FATAL: backend still up after 60s"; exit 1; }
  echo "backend stopped"
fi

CASES="$CASES" "$PY" - <<'PY'
import json, os
from pathlib import Path
base = Path("/disk/qh/nano-finrag/finquery_rag/backend/benchmarks/tv2_canonical_v1")
keep = set(os.environ["CASES"].split())
rows = [json.loads(l) for l in (base / "canonical-eval-v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
sel = [r for r in rows if r["id"] in keep]
Path("/tmp/p14f1-eval.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in sel) + "\n",
    encoding="utf-8", newline="\n")
print(f"  {len(sel)} questions: {[r['id'] for r in sel]}", flush=True)
PY

for VOCAB in new old; do
  for MATCHING in identity old; do
    label="${VOCAB}-vocab__${MATCHING}-matching"
    echo "--- $label ---"
    for i in 1 2 3; do
      ( cd "$BACKEND" && P14F_VOCAB=$VOCAB P14F_MATCHING=$MATCHING \
          "$PY" scripts/evaluation/run_p1_2_dual_track_benchmark.py \
          --track replay --fixtures "$FIXTURES" --eval-set /tmp/p14f1-eval.jsonl \
          --out-dir "$OUT/$label/run$i" ) || echo "RUN FAILED: $label/$i"
    done
  done
done

echo "=== P1.4F-1 bisect complete $(date -Is) ==="
