"""Build an NF37 ranking-only report from frozen candidate rankings."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from src.evaluation.evaluation import load_jsonl_cases  # noqa: E402
from src.evaluation.nf37_metrics import ranking_metrics  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases = load_jsonl_cases(args.cases)
    pool = json.loads(Path(args.candidate_pool).read_text(encoding="utf-8"))
    rankings = {row["case_id"]: row.get("candidates", []) for row in pool}
    report = ranking_metrics(cases, rankings)
    report["case_count"] = len(cases)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
