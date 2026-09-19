#!/usr/bin/env bash
# P1.4 benchmark campaign: three ALIGNED-REPLAY runs at each of two commits.
#
# P1.4a (the shared canonicaliser) and P1.4b (the calculator's lexer) are
# measured separately so a movement can be attributed to one of them.  The
# question the campaign answers is not "did the numbers go up" -- both changes
# were expected to be near-zero on the benchmark -- but "did anything outside
# the percentage cases move", which needs the repeat spread, not a single run.
#
# Rule B: the REPLAY track is in-process, so the backend must not hold a CUDA
# context while this runs.  It is stopped first and restarted at the end, and
# the checkout is restored to the branch it started on.
#
# Usage: bash run_p1_4_percentage_campaign.sh
set -u

REPO=/disk/qh/nano-finrag
BACKEND=$REPO/finquery_rag/backend
OUT=$REPO/artifacts/evaluation/p1-4-percentage-campaign
LOG=$OUT/campaign.log
PY=$BACKEND/.venv/bin/python

START_REF=$(cd "$REPO" && git rev-parse --abbrev-ref HEAD)
P14A=821ad33   # feat(semantics): give a percentage both its readings
P14B=8551135   # fix(finance): read a percentage the same way
BACKEND_PID=$(pgrep -f 'src.main:app' | head -1)

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== campaign start $(date -Is) ==="
echo "branch=$START_REF  a=$P14A  b=$P14B  backend_pid=${BACKEND_PID:-none}"

restore() {
  echo "=== restoring ==="
  cd "$REPO" && git checkout "$START_REF" >/dev/null 2>&1
  if [ -n "${BACKEND_PID:-}" ]; then
    cd "$BACKEND" && nohup "$PY" -m uvicorn src.main:app --host 127.0.0.1 --port 18002 --workers 1 \
      >/tmp/backend-p14.log 2>&1 &
    sleep 8
    echo "backend restarted pid=$!"
  fi
  echo "=== campaign end $(date -Is) ==="
}
trap restore EXIT

# --- Rule B: exactly one PyTorch process ------------------------------------
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

run_three() {
  local ref=$1 label=$2
  cd "$REPO" && git checkout "$ref" >/dev/null 2>&1 || { echo "FATAL: checkout $ref"; return 1; }
  echo "--- $label at $ref ($(cd "$REPO" && git log --oneline -1)) ---"
  for i in 1 2 3; do
    local dir="$OUT/$label-run$i"
    echo "--- $label run $i -> $dir ---"
    ( cd "$BACKEND" && "$PY" scripts/evaluation/run_p1_2_dual_track_benchmark.py \
        --track replay --out-dir "$dir" ) || echo "RUN FAILED: $label/$i"
  done
}

run_three "$P14A" p14a
run_three "$P14B" p14b

echo "=== campaign complete $(date -Is) ==="
