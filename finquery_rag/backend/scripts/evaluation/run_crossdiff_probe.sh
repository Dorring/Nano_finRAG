#!/usr/bin/env bash
# crossdiff-004: ten runs at each of the two commits, with the retrieval
# exception captured before it is swallowed.
#
# p14a (821ad33) is the shared canonicaliser; p14b (8551135) adds the calculator
# lexer.  The case is `cross_entity_difference`, so it is not a
# percentage_share case and its fixture is identical in v3 and v4 -- the
# default (v3) is used so the probe sees what the original campaign saw.
#
# Rule B: in-process, so the backend is stopped first and restarted at the end.
set -u

REPO=/disk/qh/nano-finrag
BACKEND=$REPO/finquery_rag/backend
OUT=$REPO/artifacts/evaluation/p1-4c-crossdiff-004
LOG=$OUT/probe.log
PY=$BACKEND/.venv/bin/python
CASE=tv2f01-s3-crossdiff-004

START_REF=$(cd "$REPO" && git rev-parse --abbrev-ref HEAD)
BACKEND_PID=$(pgrep -f 'src.main:app' | head -1)

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== crossdiff-004 probe start $(date -Is) ==="
echo "branch=$START_REF backend_pid=${BACKEND_PID:-none}"

restore() {
  echo "=== restoring ==="
  cd "$REPO" && git checkout "$START_REF" >/dev/null 2>&1
  if [ -n "${BACKEND_PID:-}" ]; then
    cd "$BACKEND" && nohup "$PY" -m uvicorn src.main:app --host 127.0.0.1 --port 18002 --workers 1 \
      >/tmp/backend-crossdiff.log 2>&1 &
    sleep 8
    echo "backend restarted pid=$!"
  fi
  echo "=== probe end $(date -Is) ==="
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

for pair in "821ad33 p14a" "8551135 p14b"; do
  set -- $pair
  ref=$1; label=$2
  cd "$REPO" && git checkout "$ref" >/dev/null 2>&1 || { echo "FATAL checkout $ref"; exit 1; }
  echo "--- $label at $ref ($(cd "$REPO" && git log --oneline -1)) ---"
  ( cd "$BACKEND" && "$PY" scripts/evaluation/probe_crossdiff_004.py \
      --case "$CASE" --runs 10 --out-dir "$OUT/$label" ) || echo "PROBE FAILED: $label"
done

echo "=== probe complete $(date -Is) ==="
