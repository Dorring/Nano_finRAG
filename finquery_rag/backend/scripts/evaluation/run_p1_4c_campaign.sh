#!/usr/bin/env bash
# P1.4c: three ALIGNED-REPLAY runs on the repaired fixture (v4).
#
# The "before" column already exists and is not re-run: the three p14b runs in
# the P1.4 campaign were taken at this same commit (8551135) on plan-fixtures-v3,
# and their seal records that fixture's digest.  Re-running them would add
# nothing and would risk the code moving between the two columns, which is the
# one thing this measurement cannot tolerate.
#
# So the only variable across the two columns is the fixture, and commit_sha,
# config_fingerprint, eval_set_sha256 and gold_sha256 must all come out
# identical to the p14b runs.
#
# Rule B: the REPLAY track is in-process.  The backend is stopped first and
# restarted at the end.
set -u

REPO=/disk/qh/nano-finrag
BACKEND=$REPO/finquery_rag/backend
OUT=$REPO/artifacts/evaluation/p1-4c-percentage-fixture
FIXTURES=$BACKEND/benchmarks/tv2_canonical_v1/plan-fixtures-v4.jsonl
LOG=$OUT/campaign.log
PY=$BACKEND/.venv/bin/python

BACKEND_PID=$(pgrep -f 'src.main:app' | head -1)

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== P1.4c campaign start $(date -Is) ==="
echo "HEAD=$(cd "$REPO" && git rev-parse HEAD)  fixtures=$FIXTURES"
echo "backend_pid=${BACKEND_PID:-none}"

restore() {
  echo "=== restoring ==="
  if [ -n "${BACKEND_PID:-}" ]; then
    cd "$BACKEND" && nohup "$PY" -m uvicorn src.main:app --host 127.0.0.1 --port 18002 --workers 1 \
      >/tmp/backend-p14c.log 2>&1 &
    sleep 8
    echo "backend restarted pid=$!"
  fi
  echo "=== P1.4c campaign end $(date -Is) ==="
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
  echo "backend stopped"
fi

for i in 1 2 3; do
  dir="$OUT/v4-run$i"
  echo "--- v4 run $i -> $dir ---"
  ( cd "$BACKEND" && "$PY" scripts/evaluation/run_p1_2_dual_track_benchmark.py \
      --track replay --fixtures "$FIXTURES" --out-dir "$dir" ) || echo "RUN FAILED: v4/$i"
done

echo "=== P1.4c campaign complete $(date -Is) ==="
