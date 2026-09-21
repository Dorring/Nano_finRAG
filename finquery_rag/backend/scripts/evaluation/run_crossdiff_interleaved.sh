#!/usr/bin/env bash
# crossdiff-004, commits ALTERNATED run by run.
#
# The sequential design cannot separate a commit effect from drift in the
# binder's failure rate over wall-clock time: whichever commit runs second
# absorbs all of it.  That is how the original campaign produced "0/3 at p14a,
# 2/3 at p14b" -- an ordering, read as a cause.
#
# Here the two commits alternate, so both see the same time window and a
# difference has to be a commit difference.  It costs a checkout and a resource
# load per run, which is the price of the comparison being interpretable.
set -u

REPO=/disk/qh/nano-finrag
BACKEND=$REPO/finquery_rag/backend
OUT=$REPO/artifacts/evaluation/p1-4c-crossdiff-interleaved
LOG=$OUT/probe.log
PY=$BACKEND/.venv/bin/python
CASE=tv2f01-s3-crossdiff-004
PAIRS=${PAIRS:-10}

START_REF=$(cd "$REPO" && git rev-parse --abbrev-ref HEAD)
BACKEND_PID=$(pgrep -f 'src.main:app' | head -1)

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=== interleaved probe start $(date -Is) pairs=$PAIRS ==="
echo "branch=$START_REF backend_pid=${BACKEND_PID:-none}"

restore() {
  cd "$REPO" && git checkout "$START_REF" >/dev/null 2>&1
  if [ -n "${BACKEND_PID:-}" ]; then
    cd "$BACKEND" && nohup "$PY" -m uvicorn src.main:app --host 127.0.0.1 --port 18002 --workers 1 \
      >/tmp/backend-interleaved.log 2>&1 &
    sleep 8
  fi
  echo "=== interleaved probe end $(date -Is) ==="
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

for i in $(seq 1 "$PAIRS"); do
  for pair in "821ad33 p14a" "8551135 p14b"; do
    set -- $pair
    ref=$1; label=$2
    cd "$REPO" && git checkout "$ref" >/dev/null 2>&1 || { echo "FATAL checkout $ref"; exit 1; }
    echo "--- pair $i  $label ($ref) $(date +%H:%M:%S) ---"
    ( cd "$BACKEND" && "$PY" scripts/evaluation/probe_crossdiff_004.py \
        --case "$CASE" --runs 1 \
        --out-dir "$OUT/$label/run$i" ) || echo "PROBE FAILED: $label/$i"
  done
done

echo "=== interleaved probe complete $(date -Is) ==="
