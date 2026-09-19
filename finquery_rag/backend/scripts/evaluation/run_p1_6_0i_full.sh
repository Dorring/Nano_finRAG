#!/usr/bin/env bash
# P1.6-0I: the full 120-case run on the re-derived stratum and the rebuilt store.
#
# What changed since the last full campaign, and why a new one is needed rather
# than a re-score of the old:
#
#   fixtures   plan-fixtures-v8, the cross-entity stratum re-derived onto
#              canonical concepts -- eleven cases changed their expected answer
#   store      the rebuilt iXBRL store, now what `_build_fact_store` returns,
#              so retrieval answers by canonical quantity
#
# The backend is stopped for the run and restored through the deploy script,
# because the store is loaded and cached at resource-build time: a process that
# was already up holds the old one and would measure the previous system.
set -u

REPO=/disk/qh/nano-finrag
BACKEND=$REPO/finquery_rag/backend
OUT=$REPO/artifacts/evaluation/p1-6-0i-full
FIXTURES=$BACKEND/benchmarks/tv2_canonical_v1/plan-fixtures-v8.jsonl
PY=$BACKEND/.venv/bin/python
LOG=$OUT/campaign.log

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

BACKEND_PID=$(pgrep -f 'src.main:app' | head -1)
echo "=== P1.6-0I full 120 start $(date -Is) ==="
echo "HEAD=$(cd "$REPO" && git rev-parse --short HEAD)"
echo "fixtures=$(basename "$FIXTURES") sha256=$(sha256sum "$FIXTURES" | cut -c1-64)"
echo "gold_sha256=$(sha256sum "$BACKEND/benchmarks/tv2_canonical_v1/gold-evidence-v1.jsonl" | cut -c1-64)"
echo "eval_sha256=$(sha256sum "$BACKEND/benchmarks/tv2_canonical_v1/canonical-eval-v1.jsonl" | cut -c1-64)"
echo "ixbrl_store_sha256=$(sha256sum "$REPO/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl" | cut -c1-64)"

restore() {
  if [ -n "${BACKEND_PID:-}" ]; then
    cd "$REPO" && bash scripts/deploy/start_backend.sh
  fi
  echo "=== P1.6-0I full 120 end $(date -Is) ==="
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

for i in 1 2 3; do
  echo "--- run $i ---"
  ( cd "$BACKEND" && "$PY" scripts/evaluation/run_p1_2_dual_track_benchmark.py \
      --track replay --fixtures "$FIXTURES" \
      --out-dir "$OUT/run$i" ) || echo "RUN FAILED: $i"
done

echo "=== P1.6-0I full 120 complete $(date -Is) ==="
