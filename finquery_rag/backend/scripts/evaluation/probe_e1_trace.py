"""E1 probe: what does one in-process run actually expose per stage?

The funnel needs per-stage outcomes, and the map says they live in
`FinancialQueryResult.debug_metadata["trace"]`.  Before building anything on top
of that, look at it: which keys exist, which are populated on a real question,
and whether the retrieval candidate ids are candidate keys (comparable with gold)
or something else.

A funnel built on assumed field names would report zeros for missing keys, and a
zero that means "the key is absent" is indistinguishable from one that means
"nothing was retrieved".  So this runs one case — plan pinned, production
retriever — and prints what is really there.

  .venv/bin/python scripts/evaluation/probe_e1_trace.py --case tv2f01-s1-aapl-001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

BENCH_DIR = _BACKEND_DIR / "benchmarks" / "tv2_canonical_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="tv2f01-s1-aapl-001")
    parser.add_argument("--pinned", type=Path,
                        default=Path("/disk/qh/nano-finrag/artifacts/evaluation/"
                                     "e1-pinned-plans/pinned-plans-v1.jsonl"))
    parser.add_argument("--eval-set", type=Path, default=BENCH_DIR / "canonical-eval-v1.jsonl")
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()

    from src.runtime.query_lifecycle import QueryExecutionService
    from src.runtime.runtime_contract import FinancialQueryRequest
    from src.runtime.trusted_v2_production import (
        _cached_resources,
        build_trusted_v2_runtime_for_request,
    )

    pinned = {json.loads(line)["case_id"]: json.loads(line)
              for line in args.pinned.read_text(encoding="utf-8").splitlines()
              if line.strip()}
    questions = {json.loads(line)["id"]: json.loads(line)
                 for line in args.eval_set.read_text(encoding="utf-8").splitlines()
                 if line.strip()}
    question = questions[args.case]
    entry = pinned[args.case]

    print("=" * 78)
    print("E1 PROBE -- one in-process run")
    print("=" * 78)
    print(f"  case      {args.case}")
    print(f"  question  {question['question']}")
    print(f"  pinned plan {json.dumps(entry['plan'], ensure_ascii=False)[:300]}")
    print()

    environ = dict(os.environ)
    resources = _cached_resources(environ)
    print(f"  resources built: index_reader={type(resources.index_reader).__name__}"
          f" binder={type(resources.binder).__name__}")
    print()

    request = FinancialQueryRequest(
        request_id=f"e1-probe-{args.case}",
        user_id="e1",
        session_id=f"__stateless__:e1-probe-{args.case}",
        original_query=question["question"],
        standalone_query=question["question"],
        query_as_resolved=True,
        conversation_metadata={},
        request_metadata={"document_names": [question["document_id"]]
                          if question.get("document_id") else [],
                          "n_results": 5,
                          "conversation_history": None,
                          "memory_profile": None},
    )

    runtime = build_trusted_v2_runtime_for_request(None, request, resources=resources)
    service = QueryExecutionService(runtime)
    result = asyncio.run(service.execute(request))

    print(f"  status          {result.status}")
    print(f"  release_status  {result.release_status}")
    print(f"  reason_codes    {list(result.reason_codes)}")
    print()

    debug = getattr(result, "debug_metadata", None) or {}
    print(f"  debug_metadata keys: {sorted(debug)}")
    trace = debug.get("trace")
    if trace is None:
        print("  !! no trace in debug_metadata -- the funnel cannot be built from it")
        print(f"  result type: {type(result).__name__}")
        attrs = [a for a in dir(result) if not a.startswith("_")]
        print(f"  result attrs: {attrs}")
        return 1

    payload = trace.to_dict() if hasattr(trace, "to_dict") else dict(trace)
    print(f"  trace type      {type(trace).__name__}")
    print(f"  trace keys      {sorted(payload)}")
    print()
    print("  populated trace values:")
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, (list, tuple)):
            rendered = f"{len(value)} items"
            if value and len(value) <= 4:
                rendered += f" {list(value)[:4]}"
        elif isinstance(value, dict):
            rendered = f"{len(value)} keys {sorted(value)[:6]}"
        else:
            rendered = repr(value)[:120]
        print(f"    {key:32} {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
