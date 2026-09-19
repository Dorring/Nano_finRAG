#!/usr/bin/env bash
# The eight percentage_share cases, three times each -- the step that costs 24
# model calls instead of a 120-case campaign.
#
# Only run once the static build passes: the build already proves every
# answerable slot selects at least one fact and every abstention slot selects
# none, so a failure here is the binder's reachability rather than a coordinate
# that cannot exist.
#
# Rule B: in-process, so the backend is stopped first and restarted at the end.
set -u

REPO=/disk/qh/nano-finrag
BACKEND=$REPO/finquery_rag/backend
OUT=$REPO/artifacts/evaluation/p1-4d-pctshare
FIXTURES=$BACKEND/benchmarks/tv2_canonical_v1/plan-fixtures-v5.jsonl
PY=$BACKEND/.venv/bin/python
LOG=$OUT/campaign.log

BACKEND_PID=$(pgrep -f 'src.main:app' | head -1)
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== P1.4d pctshare start $(date -Is) ==="
echo "HEAD=$(cd "$REPO" && git rev-parse --short HEAD)  fixtures=$FIXTURES"

restore() {
  if [ -n "${BACKEND_PID:-}" ]; then
    cd "$BACKEND" && nohup "$PY" -m uvicorn src.main:app --host 127.0.0.1 --port 18002 --workers 1 \
      >/tmp/backend-p14d.log 2>&1 &
    sleep 8
  fi
  echo "=== P1.4d pctshare end $(date -Is) ==="
}
trap restore EXIT

if [ -n "${BACKEND_PID:-}" ]; then
  echo "stopping backend $BACKEND_PID"
  kill "$BACKEND_PID"
  # The backend builds a model at startup and takes its time shutting down, so
  # this waits a full minute rather than thirty seconds.  A premature "still up"
  # aborts the run and the exit trap restarts what was never stopped, which is
  # how a healthy backend reads as an outage.
  for _ in $(seq 1 60); do
    pgrep -f 'src.main:app' >/dev/null || break
    sleep 1
  done
  pgrep -f 'src.main:app' >/dev/null && { echo "FATAL: backend still up after 60s"; exit 1; }
  echo "backend stopped"
fi

# A questions file holding just these eight, built from the real eval set so the
# runner sees the same rows it always does.
"$PY" - <<'PY'
import json
from pathlib import Path
base = Path("/disk/qh/nano-finrag/finquery_rag/backend/benchmarks/tv2_canonical_v1")
rows = [json.loads(l) for l in (base / "canonical-eval-v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
keep = [r for r in rows if r["id"].startswith("tv2f01-s2-pctshare")]
out = Path("/tmp/pctshare-eval.jsonl")
out.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in keep) + "\n", encoding="utf-8", newline="\n")
print(f"  {len(keep)} questions written to {out}", flush=True)
PY

for i in 1 2 3; do
  echo "--- run $i ---"
  ( cd "$BACKEND" && "$PY" scripts/evaluation/run_p1_2_dual_track_benchmark.py \
      --track replay --fixtures "$FIXTURES" --eval-set /tmp/pctshare-eval.jsonl \
      --out-dir "$OUT/run$i" ) || echo "RUN FAILED: $i"
done

echo "=== P1.4d pctshare complete $(date -Is) ==="
