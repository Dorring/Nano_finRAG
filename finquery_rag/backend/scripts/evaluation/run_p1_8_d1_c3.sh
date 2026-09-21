#!/usr/bin/env bash
# P1.8-D1-C3: does the corrected fixture, with the narrow patch OFF, give the
# right sign?
#
# Staged exactly as the brief asks -- one case, then both, then the stratum,
# then the whole benchmark -- so that a negative result is learned on one
# question rather than after a full run.
#
#   bash scripts/evaluation/run_p1_8_d1_c3.sh
#
# Every stage replays against the same V2 benchmark (gold a3d17211, eval set
# 227f0341) and the same v9 fixture, so the only thing that varies between the
# stages is how many questions are asked.

set -euo pipefail
cd /disk/qh/nano-finrag/finquery_rag/backend

PY=.venv/bin/python
C3=/disk/qh/nano-finrag/artifacts/evaluation/p1-8-d1-c3
V2=/disk/qh/nano-finrag/artifacts/evaluation/p1-8-c-v2
FIX="$C3/plan-fixtures-v9.jsonl"
GOLD="$V2/gold-evidence-v1.jsonl"
EVAL="$V2/canonical-eval-v1.jsonl"
SUBSET="$C3/subset"

# --- preflight ---------------------------------------------------------------------------------
# The whole point of C3 is that the *fixture* was wrong, not the runtime.  If the
# narrow operand-authority patch is present the experiment measures the patch
# instead, so refuse to run rather than report a number that means something else.
echo "== preflight: runtime source state =="
sha256sum src/finance/structured_operand_binding.py
if grep -q "_cross_entity_difference_specs" src/finance/structured_operand_binding.py; then
  echo "REFUSING: the narrow cross-entity patch is applied; C3 must run with it OFF" >&2
  exit 3
fi
echo "narrow cross-entity patch: ABSENT (flag OFF)"
echo "fixture under test: $(sha256sum "$FIX")"
echo

# --- subsets -----------------------------------------------------------------------------------
mkdir -p "$SUBSET"
"$PY" - "$EVAL" "$FIX" "$SUBSET" <<'PYEOF'
import json, sys
from pathlib import Path

eval_path, fix_path, out_dir = (Path(a) for a in sys.argv[1:4])

def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def dump(rows, path):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )

eval_rows = {r["id"]: r for r in load(eval_path)}
fix_rows = {r["id"]: r for r in load(fix_path)}

subsets = {
    "one-001": ["tv2f01-s3-crossdiff-001"],
    "one-002": ["tv2f01-s3-crossdiff-002"],
    "crossdiff": [f"tv2f01-s3-crossdiff-00{i}" for i in range(1, 6)],
}
for name, ids in subsets.items():
    dump([eval_rows[i] for i in ids], out_dir / f"{name}.eval.jsonl")
    dump([fix_rows[i] for i in ids], out_dir / f"{name}.fixtures.jsonl")
    print(f"  subset {name}: {len(ids)} question(s)")
PYEOF
echo

# --- stages ------------------------------------------------------------------------------------
run_stage () {  # <name> <eval-file> <fixture-file>
  local name="$1" eval_file="$2" fixture_file="$3"
  echo "== stage $name =="
  "$PY" scripts/evaluation/run_p1_2_dual_track_benchmark.py \
    --track replay \
    --eval-set "$eval_file" \
    --fixtures "$fixture_file" \
    --gold-evidence "$GOLD" \
    --out-dir "$C3/$name" \
    --timeout-per-query 150
  echo
}

run_stage c3-probe-001       "$SUBSET/one-001.eval.jsonl"        "$SUBSET/one-001.fixtures.jsonl"
run_stage c3-probe-002       "$SUBSET/one-002.eval.jsonl"        "$SUBSET/one-002.fixtures.jsonl"
run_stage c3-crossdiff-r1    "$SUBSET/crossdiff.eval.jsonl"      "$SUBSET/crossdiff.fixtures.jsonl"
run_stage c3-crossdiff-r2    "$SUBSET/crossdiff.eval.jsonl"      "$SUBSET/crossdiff.fixtures.jsonl"
run_stage c3-crossdiff-r3    "$SUBSET/crossdiff.eval.jsonl"      "$SUBSET/crossdiff.fixtures.jsonl"
run_stage c3-full-v9         "$EVAL"                             "$FIX"

echo "== D1-C3 stages complete =="
