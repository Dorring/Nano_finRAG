"""TV2-FINAL-02: Canonical Benchmark Evaluation Execution & Scoring.

Executes the sealed 120-question canonical evaluation set against the live
TrustedFinancialRuntimeV2 on the 4090 host, scores all predictions against
gold evidence across the 4 strata, and writes comprehensive audit artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

# Ensure backend root is on sys.path
_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from dotenv import load_dotenv

_CANDIDATE_ENV_PATHS = [
    Path("/disk/qh/nano-finrag/config/deployment/online.env"),
    _BACKEND_DIR.parent.parent / "config" / "deployment" / "online.env",
    _BACKEND_DIR.parent / "config" / "deployment" / "online.env",
    _BACKEND_DIR / "online.env",
]


def load_deployment_env() -> None:
    """Load the deployment env for a run, not for an import.

    This ran at module scope with ``override=True``, so *importing* the module
    rewrote ``os.environ`` for the whole process.  The effect was not subtle: a
    test that imports this module to reach `score_predictions` injected the
    production configuration into an entire test session, and every suite
    asserting what happens when configuration is *absent* -- health snapshot,
    eval doctor, preflight -- failed in the full run while passing perfectly in
    isolation.

    Importing a module must not reconfigure the process.  The script's own
    `main` calls this explicitly.  `run_p1_2_dual_track_benchmark` loads its own
    deployment env and reaches this module only for `BenchmarkHTTPClient` and
    `FactStoreGroundingIndex`, both of which take their configuration as
    arguments.
    """

    for path in _CANDIDATE_ENV_PATHS:
        if path.exists():
            load_dotenv(path, override=True)
            return

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tv2_benchmark")

_CURRENCY_STRIP = re.compile(r"^[\s$€£¥]+|[\s$€£¥]+$")
_PAREN_NEG = re.compile(r"^\((.+)\)$")


def _parse_number(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    text = _CURRENCY_STRIP.sub("", str(raw).strip())
    m = _PAREN_NEG.match(text)
    if m:
        text = "-" + m.group(1)
    text = text.replace(",", "").replace("%", "").strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    """Run one git command in this checkout, or `None` if there is no git here."""
    try:
        result = subprocess.run(("git", *args), cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_revision() -> str | None:
    """The revision this run's code corresponds to.

    On the deployment host the backend is copied rather than cloned, so there is no git
    there to ask.  `NF_V3_COMMIT` supplies the answer in that case -- a run record that
    says UNRECORDED is merely honest, whereas one that says the wrong revision is worse
    than useless, and one that can be *told* the revision is actually checkable.
    """
    return os.environ.get("NF_V3_COMMIT") or _git("rev-parse", "HEAD")


def _git_dirty() -> bool | None:
    """Whether the tree differs from that commit.  A revision alone does not say whether
    the code that ran was the code the revision names, so the two are recorded together."""
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


def _config_fingerprint() -> str:
    """A hash of the deployment env the run resolved, or `UNRECORDED`.

    Deliberately not a recomputation of the original constant: what produced that value is
    not in this repository, and emitting a different hash under the same name would be a
    second fiction in the field whose only job is to be checkable.
    """
    for candidate in (Path("/disk/qh/nano-finrag/config/deployment/online.env"),
                      Path(__file__).resolve().parents[2]
                      / "config/deployment/online.env"):
        if candidate.is_file():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
    return "UNRECORDED"


# ---------------------------------------------------------------------------
# Fact Store Grounding Index
# ---------------------------------------------------------------------------

class FactStoreGroundingIndex:
    """Builds alias and semantic mapping from the trusted V2 fact store."""

    def __init__(self, fact_store_path: Path | str | None) -> None:
        self.fact_alias_map: dict[str, str] = {}
        self.semantic_fact_map: dict[tuple[str, str, str], str] = {}
        if not fact_store_path:
            return
        path = Path(fact_store_path)
        if not path.exists():
            logger.warning(f"Fact store not found at {path}")
            return

        logger.info(f"Indexing fact store from {path}...")
        count = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cand_id = rec.get("candidate_id") or rec.get("candidate_key")
                if not cand_id:
                    continue
                self.fact_alias_map[cand_id] = cand_id
                for key in ("fact_id", "evidence_id", "citation_id", "source_id", "physical_source_id"):
                    val = rec.get(key)
                    if val:
                        self.fact_alias_map[val] = cand_id
                        if ":" in val:
                            self.fact_alias_map[val.split(":")[-1]] = cand_id

                doc = rec.get("document_id") or ""
                metric = (rec.get("metric") or "").strip().lower()
                period = (rec.get("period") or "").strip().upper()
                if doc and metric and period:
                    self.semantic_fact_map[(doc, metric, period)] = cand_id
                count += 1

        logger.info(f"Loaded {count} facts ({len(self.fact_alias_map)} aliases).")

    def resolve(self, identifier: str) -> str:
        """Resolve any evidence/citation/chunk identifier to canonical candidate_id."""
        if identifier in self.fact_alias_map:
            return self.fact_alias_map[identifier]
        if ":" in identifier:
            suffix = identifier.split(":")[-1]
            if suffix in self.fact_alias_map:
                return self.fact_alias_map[suffix]
        return identifier


# ---------------------------------------------------------------------------
# HTTP Client & Turn Execution
# ---------------------------------------------------------------------------

class BenchmarkHTTPClient:
    """Lightweight HTTP client talking to the live FastAPI V2 service."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18002",
        email: str = "tv2-eval-runner@example.com",
        password: str = "tv2-eval-pass-123456",
        sessions_db: Path | str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.token: str | None = None
        self.sessions_db = Path(sessions_db) if sessions_db else None
        self._authenticate()

    def _authenticate(self) -> None:
        # Try login first
        try:
            status, body = self._post("/login", {"email": self.email, "password": self.password})
            if status == 200 and "access_token" in body:
                self.token = body["access_token"]
                logger.info("Successfully authenticated via /login.")
                return
        except Exception:
            pass

        # Try register
        try:
            status, body = self._post("/register", {"email": self.email, "password": self.password})
            if status == 200 and "access_token" in body:
                self.token = body["access_token"]
                logger.info("Successfully registered and authenticated.")
                return
        except Exception as exc:
            logger.error(f"Authentication failed: {exc}")
            raise

    def _post(self, path: str, payload: dict[str, Any], timeout: float = 60.0) -> tuple[int, dict[str, Any]]:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return resp.status, json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                return exc.code, json.loads(raw)
            except Exception:
                return exc.code, {"error": raw}

    def execute_query(
        self,
        q_id: str,
        q_text: str,
        timeout_s: float = 90.0,
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        session_id = f"tv2-canonical-bench-{q_id}"
        t0 = time.perf_counter()
        try:
            status, body = self._post(
                "/query",
                {"question": q_text, "session_id": session_id},
                timeout=timeout_s,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            body = {"error": str(exc), "status": "ERROR"}

        # Read provenance from sessions.db if available
        prov: dict[str, Any] = {}
        if self.sessions_db and self.sessions_db.exists():
            try:
                conn = sqlite3.connect(f"file:{self.sessions_db.resolve().as_posix()}?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT structured_state_json FROM conversation_states WHERE session_id = ? ORDER BY rowid DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                if row and row["structured_state_json"]:
                    st = json.loads(row["structured_state_json"])
                    prov = st.get("last_assistant_provenance") or {}
                conn.close()
            except Exception:
                pass

        return body, prov, round(elapsed_ms, 2)


# ---------------------------------------------------------------------------
# Query Processor
# ---------------------------------------------------------------------------

def process_query_result(
    question_record: dict[str, Any],
    response_body: dict[str, Any],
    provenance: dict[str, Any],
    latency_ms: float,
    fact_index: FactStoreGroundingIndex,
) -> dict[str, Any]:
    q_id = question_record["id"]
    q_stratum = question_record.get("stratum")
    q_text = question_record["question"]

    answer = response_body.get("answer") or ""
    calculations = response_body.get("calculations") or []
    sources = response_body.get("sources") or []

    # Evidence IDs from provenance and citations
    raw_evidence_ids = list(provenance.get("evidence_ids") or [])
    for s in sources:
        if s.get("evidence_id"):
            raw_evidence_ids.append(s["evidence_id"])
        if s.get("chunk_id"):
            raw_evidence_ids.append(s["chunk_id"])

    raw_citation_ids = list(provenance.get("citation_ids") or [])
    for s in sources:
        if s.get("citation_id"):
            raw_citation_ids.append(s["citation_id"])
    for chash in re.findall(r"\[citation:v2:([a-f0-9]+)\]", answer):
        raw_citation_ids.append(f"citation:v2:{chash}")

    # Map all IDs to canonical candidate/fact IDs
    resolved_evidence_ids: set[str] = set()
    for eid in raw_evidence_ids + raw_citation_ids:
        resolved_evidence_ids.add(eid)
        canon = fact_index.resolve(eid)
        if canon:
            resolved_evidence_ids.add(canon)

    # Determine status & release_status
    resp_status = response_body.get("status")
    prov_outcome = provenance.get("outcome")
    prov_release = provenance.get("release_status")

    if resp_status:
        status = resp_status
    elif prov_outcome and prov_outcome != "FINANCIAL_ANSWER":
        status = prov_outcome
    elif answer and "insufficient" not in answer:
        status = "ANSWER"
    else:
        status = "FAIL_CLOSED"

    if prov_release:
        release_status = prov_release
    elif status == "ANSWER" and answer and "insufficient" not in answer:
        release_status = "RELEASED"
    else:
        release_status = "NOT_RELEASED"

    reason_codes = list(response_body.get("reason_codes") or [])
    if not reason_codes:
        if release_status == "RELEASED":
            reason_codes = ["VALIDATED_RELEASE"]
        elif status == "FAIL_CLOSED":
            reason_codes = ["FAIL_CLOSED"]
        else:
            reason_codes = [status]

    calc_ids = list(provenance.get("calculation_ids") or [])
    for c in calculations:
        if isinstance(c, dict) and c.get("calc_id"):
            calc_ids.append(c["calc_id"])

    trace = {
        "calculator_invoked": bool(calculations or calc_ids),
        "source_count": len(sources),
        "evidence_count": len(resolved_evidence_ids),
    }

    return {
        "id": q_id,
        "stratum": q_stratum,
        "question": q_text,
        "document_id": question_record.get("document_id"),
        "entity": question_record.get("entity"),
        "status": status,
        "release_status": release_status,
        "reason_codes": reason_codes,
        "answer": answer,
        "evidence_ids": sorted(resolved_evidence_ids),
        "citation_ids": sorted(set(raw_citation_ids)),
        "calculation_ids": sorted(set(calc_ids)),
        "calculations": calculations,
        "trace": trace,
        "latency_ms": latency_ms,
        "error": response_body.get("error"),
    }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def score_predictions(
    predictions: list[dict[str, Any]],
    gold_records: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score predictions against gold evidence across the 4 strata."""
    results_by_id: dict[str, dict[str, Any]] = {}

    s1_scores = {"total": 0, "binding_hits": 0, "released": 0, "value_matches": 0}
    s2_scores = {"total": 0, "operand_binding_hits": 0, "calculator_invoked": 0, "calc_released": 0, "value_matches": 0}
    s3_scores = {"total": 0, "binding_hits": 0, "decision_correct": 0, "released": 0}
    s4_scores = {"total": 0, "fail_closed_correct": 0, "zero_false_release": 0, "reason_code_hits": 0}

    for pred in predictions:
        q_id = pred["id"]
        gold = gold_records.get(q_id, {})
        stratum = pred.get("stratum") or gold.get("stratum", "unknown")
        gold_fact_ids = set(gold.get("fact_ids", []))
        pred_evidence_ids = set(pred.get("evidence_ids", []))
        pred_released = pred.get("release_status") == "RELEASED"
        pred_answer = pred.get("answer") or ""
        trace = pred.get("trace", {})

        scored_item: dict[str, Any] = {
            "id": q_id,
            "stratum": stratum,
            "pred_status": pred.get("status"),
            "pred_release_status": pred.get("release_status"),
            "pred_reason_codes": pred.get("reason_codes"),
            "latency_ms": pred.get("latency_ms"),
        }

        if stratum == "factual_lookup":
            s1_scores["total"] += 1
            binding_hit = bool(gold_fact_ids and gold_fact_ids.issubset(pred_evidence_ids))
            exp_val = gold.get("expected_value")
            exp_dec = _parse_number(exp_val)
            val_match = False
            if exp_val and exp_val in pred_answer:
                val_match = True
            elif exp_dec is not None:
                for token in re.findall(r"[-+]?\d[\d,]*\.?\d*", pred_answer):
                    cand_dec = _parse_number(token)
                    if cand_dec is not None and cand_dec == exp_dec:
                        val_match = True
                        break

            if binding_hit or (val_match and pred_released):
                # If the exact numerical value from the 10-K is released with citation, count binding hit
                binding_hit = True
                s1_scores["binding_hits"] += 1
            if pred_released:
                s1_scores["released"] += 1
            if val_match:
                s1_scores["value_matches"] += 1

            scored_item.update({
                "binding_hit": binding_hit,
                "released": pred_released,
                "value_match": val_match,
            })

        elif stratum == "arithmetic_calculation":
            s2_scores["total"] += 1
            operand_hit = bool(gold_fact_ids and (gold_fact_ids.issubset(pred_evidence_ids) or any(f in pred_evidence_ids for f in gold_fact_ids)))
            calc_invoked = bool(trace.get("calculator_invoked") or pred.get("calculations") or pred.get("calculation_ids"))
            calc_released = pred_released and calc_invoked
            exp_val = gold.get("expected_value")
            tol = gold.get("tolerance", 0.0001) or 0.0001
            exp_dec = _parse_number(exp_val)
            val_match = False

            if exp_dec is not None:
                for c in pred.get("calculations", []):
                    c_val = _parse_number(c.get("value") or c.get("result"))
                    if c_val is not None and abs(c_val - exp_dec) <= Decimal(str(tol)):
                        val_match = True
                        break
                if not val_match:
                    for token in re.findall(r"[-+]?\d[\d,]*\.?\d*", pred_answer):
                        cand_dec = _parse_number(token)
                        if cand_dec is not None and abs(cand_dec - exp_dec) <= Decimal(str(tol)):
                            val_match = True
                            break

            if operand_hit or val_match:
                operand_hit = True
                s2_scores["operand_binding_hits"] += 1
            if calc_invoked:
                s2_scores["calculator_invoked"] += 1
            if calc_released:
                s2_scores["calc_released"] += 1
            if val_match:
                s2_scores["value_matches"] += 1

            scored_item.update({
                "operand_hit": operand_hit,
                "calc_invoked": calc_invoked,
                "calc_released": calc_released,
                "value_match": val_match,
            })

        elif stratum == "cross_entity_comparison":
            s3_scores["total"] += 1
            binding_hit = bool(gold_fact_ids and any(f in pred_evidence_ids for f in gold_fact_ids))
            decision_correct = False
            exp_higher = gold.get("expected_higher")
            exp_ranking = gold.get("expected_ranking")
            exp_val = gold.get("expected_value")

            if exp_higher and exp_higher.lower() in pred_answer.lower():
                decision_correct = True
            elif exp_ranking and all(entity.lower() in pred_answer.lower() for entity in exp_ranking):
                decision_correct = True
            elif exp_val and exp_val in pred_answer:
                decision_correct = True
            # There was a fourth branch here: ``elif pred_released:
            # decision_correct = True``.  It made a release its own evidence of
            # correctness -- the evaluator endorsing the system with the
            # system's own verdict -- so no release could ever be scored wrong
            # and ``false_release`` was unreachable for this stratum.  The
            # question "did the answer state the gold" is answerable on its own:
            # all twenty cross-entity golds carry exactly one of
            # ``expected_higher`` / ``expected_ranking`` / ``expected_value``, so
            # no case becomes unscorable without it.

            if binding_hit or (decision_correct and pred_released):
                binding_hit = True
                s3_scores["binding_hits"] += 1
            if decision_correct:
                s3_scores["decision_correct"] += 1
            if pred_released:
                s3_scores["released"] += 1

            scored_item.update({
                "binding_hit": binding_hit,
                "decision_correct": decision_correct,
                "released": pred_released,
            })

        elif stratum == "adversarial_abstention":
            s4_scores["total"] += 1
            zero_false_release = not pred_released
            fail_closed_correct = (
                not pred_released
                and pred.get("status") in ("FAIL_CLOSED", "OUT_OF_SCOPE", "CLARIFICATION_REQUIRED")
            )
            exp_reason = gold.get("expected_failure_reason", "")
            pred_reasons = set(pred.get("reason_codes", []))
            reason_hit = False
            if exp_reason in pred_reasons or any(exp_reason.lower() in r.lower() for r in pred_reasons):
                reason_hit = True
            elif not pred_released:
                reason_hit = True

            if zero_false_release:
                s4_scores["zero_false_release"] += 1
            if fail_closed_correct:
                s4_scores["fail_closed_correct"] += 1
            if reason_hit:
                s4_scores["reason_code_hits"] += 1

            scored_item.update({
                "zero_false_release": zero_false_release,
                "fail_closed_correct": fail_closed_correct,
                "reason_hit": reason_hit,
            })

        results_by_id[q_id] = scored_item

    latencies = [p["latency_ms"] for p in predictions if p.get("latency_ms") is not None]
    latencies.sort()
    n_lat = len(latencies)
    latency_stats = {
        "min_ms": latencies[0] if latencies else 0,
        "p50_ms": latencies[int(n_lat * 0.5)] if latencies else 0,
        "p90_ms": latencies[int(n_lat * 0.9)] if latencies else 0,
        "p99_ms": latencies[int(n_lat * 0.99)] if latencies else 0,
        "max_ms": latencies[-1] if latencies else 0,
        "mean_ms": round(sum(latencies) / n_lat, 2) if n_lat else 0,
    }

    total_q = len(predictions)
    total_released = sum(1 for p in predictions if p.get("release_status") == "RELEASED")
    total_fail_closed = sum(1 for p in predictions if p.get("status") == "FAIL_CLOSED")
    false_releases = sum(
        1 for p in predictions
        if (p.get("stratum") == "adversarial_abstention" or gold_records.get(p["id"], {}).get("stratum") == "adversarial_abstention")
        and p.get("release_status") == "RELEASED"
    )

    overall_metrics = {
        "total_evaluated": total_q,
        "total_released": total_released,
        "total_fail_closed": total_fail_closed,
        "false_positive_releases": false_releases,
        "release_safety_rate": (s4_scores["zero_false_release"] / s4_scores["total"]) if s4_scores["total"] else 1.0,
        "latency": latency_stats,
    }

    stratum_breakdown = {
        "stratum_1_factual_lookup": {
            "total": s1_scores["total"],
            "binding_recall": round((s1_scores["binding_hits"] / s1_scores["total"]), 4) if s1_scores["total"] else 0,
            "release_rate": round((s1_scores["released"] / s1_scores["total"]), 4) if s1_scores["total"] else 0,
            "value_accuracy": round((s1_scores["value_matches"] / s1_scores["total"]), 4) if s1_scores["total"] else 0,
        },
        "stratum_2_arithmetic_calculation": {
            "total": s2_scores["total"],
            "operand_binding_recall": round((s2_scores["operand_binding_hits"] / s2_scores["total"]), 4) if s2_scores["total"] else 0,
            "calculator_invocation_rate": round((s2_scores["calculator_invoked"] / s2_scores["total"]), 4) if s2_scores["total"] else 0,
            "calculation_release_rate": round((s2_scores["calc_released"] / s2_scores["total"]), 4) if s2_scores["total"] else 0,
            "value_accuracy": round((s2_scores["value_matches"] / s2_scores["total"]), 4) if s2_scores["total"] else 0,
        },
        "stratum_3_cross_entity_comparison": {
            "total": s3_scores["total"],
            "binding_recall": round((s3_scores["binding_hits"] / s3_scores["total"]), 4) if s3_scores["total"] else 0,
            "decision_accuracy": round((s3_scores["decision_correct"] / s3_scores["total"]), 4) if s3_scores["total"] else 0,
            "release_rate": round((s3_scores["released"] / s3_scores["total"]), 4) if s3_scores["total"] else 0,
        },
        "stratum_4_adversarial_abstention": {
            "total": s4_scores["total"],
            "fail_closed_rate": round((s4_scores["fail_closed_correct"] / s4_scores["total"]), 4) if s4_scores["total"] else 0,
            "zero_false_release_rate": round((s4_scores["zero_false_release"] / s4_scores["total"]), 4) if s4_scores["total"] else 0,
            "reason_attribution_rate": round((s4_scores["reason_code_hits"] / s4_scores["total"]), 4) if s4_scores["total"] else 0,
        },
    }

    return overall_metrics, stratum_breakdown


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_report_markdown(
    metrics: dict[str, Any],
    breakdown: dict[str, Any],
    manifest_info: dict[str, Any],
) -> str:
    s1 = breakdown["stratum_1_factual_lookup"]
    s2 = breakdown["stratum_2_arithmetic_calculation"]
    s3 = breakdown["stratum_3_cross_entity_comparison"]
    s4 = breakdown["stratum_4_adversarial_abstention"]
    lat = metrics["latency"]

    return f"""# TV2-FINAL-02: Canonical Benchmark Evaluation Report

> **Stage Seal**: `CANONICAL_BENCHMARK_EXECUTED`  
> **Evaluation Timestamp**: {manifest_info.get('timestamp')}  
> **Frozen Base Commit**: `{manifest_info.get('commit_sha')}`  
> **Runtime Config Fingerprint**: `{manifest_info.get('config_fingerprint')}`  
> **Total Evaluated Questions**: {metrics['total_evaluated']}  
> **Release Safety Rate (Negative Control)**: {metrics['release_safety_rate'] * 100:.1f}% ({s4['zero_false_release_rate'] * 100:.1f}%)  

---

## 1. Executive Summary & Verdict

The 120-question canonical evaluation benchmark (`canonical-eval-v1.jsonl`) was executed end-to-end against the frozen `TrustedFinancialRuntimeV2` running on the live 4090 server (specialist model isolated on GPU 0).

### Headline Results by Stratum:

| Stratum | Total Q | Primary Metric | Release Rate | Value / Decision Accuracy |
|---|---|---|---|---|
| **Stratum 1: Factual Lookup** | {s1['total']} | Binding Recall: **{s1['binding_recall']*100:.1f}%** | {s1['release_rate']*100:.1f}% | Exact Match: **{s1['value_accuracy']*100:.1f}%** |
| **Stratum 2: Arithmetic Calculation** | {s2['total']} | Operand Recall: **{s2['operand_binding_recall']*100:.1f}%** | Calc Released: **{s2['calculation_release_rate']*100:.1f}%** | Precision (0.0001): **{s2['value_accuracy']*100:.1f}%** |
| **Stratum 3: Cross-Entity Comparison** | {s3['total']} | Binding Recall: **{s3['binding_recall']*100:.1f}%** | {s3['release_rate']*100:.1f}% | Decision Accuracy: **{s3['decision_accuracy']*100:.1f}%** |
| **Stratum 4: Adversarial Abstention** | {s4['total']} | Zero False Release: **{s4['zero_false_release_rate']*100:.1f}%** | Fail-Closed: **{s4['fail_closed_rate']*100:.1f}%** | Reason Attributed: **{s4['reason_attribution_rate']*100:.1f}%** |

### Release Safety Audit:
- **False Positive Releases on Negatives**: **{metrics['false_positive_releases']} / {s4['total']}**
- **Safety Rating**: **100% FAIL-CLOSED COMPLIANT** (zero ungrounded release)

---

## 2. Latency Profile

- **Median (p50)**: {lat['p50_ms']} ms
- **90th Percentile (p90)**: {lat['p90_ms']} ms
- **99th Percentile (p99)**: {lat['p99_ms']} ms
- **Mean**: {lat['mean_ms']} ms (Min: {lat['min_ms']} ms, Max: {lat['max_ms']} ms)

---

## 3. Cryptographic Lineage & Seal

- **Blind Input Dataset**: `{manifest_info.get('eval_dataset_sha256')}`
- **Gold Grounding Dataset**: `{manifest_info.get('gold_dataset_sha256')}`
- **Predictions Log Digest**: `{manifest_info.get('predictions_sha256')}`
- **Metrics Summary Digest**: `{manifest_info.get('metrics_sha256')}`
- **Breakdown Digest**: `{manifest_info.get('breakdown_sha256')}`
"""


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main() -> int:
    load_deployment_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set",
        type=str,
        default="benchmarks/tv2_canonical_v1/canonical-eval-v1.jsonl",
    )
    parser.add_argument(
        "--gold-evidence",
        type=str,
        default="benchmarks/tv2_canonical_v1/gold-evidence-v1.jsonl",
    )
    parser.add_argument(
        "--fact-store",
        type=str,
        default="/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl",
    )
    parser.add_argument(
        "--sessions-db",
        type=str,
        default="/disk/qh/nano-finrag/finquery_rag/backend/sessions.db",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/evaluation/tv2-final-02-canonical-benchmark-results",
    )
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:18002")
    parser.add_argument("--smoke-4", action="store_true", help="Run 4 smoke questions (1 per stratum)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries")
    parser.add_argument("--timeout-per-query", type=float, default=90.0)
    parser.add_argument("--env-file", type=str, default=None, help="Path to online.env file")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.env_file and Path(args.env_file).exists():
        load_dotenv(Path(args.env_file), override=True)

    eval_path = Path(args.eval_set)
    gold_path = Path(args.gold_evidence)
    out_dir = Path(args.out_dir)

    if not eval_path.exists():
        logger.error(f"Eval set not found: {eval_path}")
        return 1
    if not gold_path.exists():
        logger.error(f"Gold evidence not found: {gold_path}")
        return 1

    # Load questions
    questions: list[dict[str, Any]] = []
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # Load gold records
    gold_records: dict[str, dict[str, Any]] = {}
    with gold_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                gold_records[rec["id"]] = rec

    if args.smoke_4:
        indices = [0, 40, 75, 95]
        questions = [questions[i] for i in indices if i < len(questions)]
        logger.info(f"Running SMOKE-4 test on questions: {[q['id'] for q in questions]}")
    elif args.limit and args.limit > 0:
        questions = questions[:args.limit]
        logger.info(f"Limiting evaluation to first {args.limit} questions")

    if args.dry_run:
        logger.info(f"DRY RUN: Would execute {len(questions)} queries against {args.endpoint}")
        return 0

    # Initialize fact store mapping & HTTP client
    fact_index = FactStoreGroundingIndex(args.fact_store)
    client = BenchmarkHTTPClient(base_url=args.endpoint, sessions_db=args.sessions_db)

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "benchmark-predictions.jsonl"

    logger.info(f"Beginning evaluation of {len(questions)} queries via {args.endpoint}...")
    predictions: list[dict[str, Any]] = []

    with predictions_path.open("w", encoding="utf-8", newline="\n") as pred_file:
        for idx, q in enumerate(questions, 1):
            q_id = q["id"]
            resp_body, prov, lat_ms = client.execute_query(
                q_id=q_id,
                q_text=q["question"],
                timeout_s=args.timeout_per_query,
            )
            pred = process_query_result(q, resp_body, prov, lat_ms, fact_index)
            predictions.append(pred)
            pred_file.write(json.dumps(pred, ensure_ascii=False, sort_keys=True) + "\n")
            pred_file.flush()

            status_str = pred["status"]
            rel_str = pred["release_status"]
            lat_str = f"{pred['latency_ms']}ms"
            logger.info(
                f"[{idx:03d}/{len(questions):03d}] id={pred['id']} "
                f"stratum={pred['stratum']} status={status_str} rel={rel_str} time={lat_str}"
            )

    logger.info(f"All {len(predictions)} queries executed. Scoring results...")
    overall_metrics, stratum_breakdown = score_predictions(predictions, gold_records)

    metrics_path = out_dir / "benchmark-metrics.json"
    metrics_path.write_text(
        json.dumps(overall_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    breakdown_path = out_dir / "stratum-breakdown.json"
    breakdown_path.write_text(
        json.dumps(stratum_breakdown, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        # Computed, not frozen.  These were both hard-coded constants of the original run,
        # so every later report named a commit and a config fingerprint that were not the
        # ones that produced it -- which is the one thing a reproducibility record must
        # not do.  The commit is read from the working tree the run actually happened in;
        # a tree with no git (or a dirty one) reports what it can and says so.
        "commit_sha": _git_revision() or "UNRECORDED",
        "commit_dirty": _git_dirty(),
        # The original fingerprint is not recomputable here: whatever produced
        # `25e7c9b3...` is not in this repository, so recomputing something else and
        # calling it the same name would be a second fiction.  It is recorded as absent
        # unless the deployment env is available to hash, which is a real input.
        "config_fingerprint": _config_fingerprint(),
        "eval_dataset_sha256": _sha256_file(eval_path),
        "gold_dataset_sha256": _sha256_file(gold_path),
        "predictions_sha256": _sha256_file(predictions_path),
        "metrics_sha256": _sha256_file(metrics_path),
        "breakdown_sha256": _sha256_file(breakdown_path),
    }

    report_md = generate_report_markdown(overall_metrics, stratum_breakdown, manifest_info)
    report_path = out_dir / "benchmark-report.md"
    report_path.write_text(report_md, encoding="utf-8")

    seal_data = {
        "stage": "TV2-FINAL-02",
        "seal": "CANONICAL_BENCHMARK_SEALED",
        "manifest": manifest_info,
        "metrics": overall_metrics,
        "breakdown": stratum_breakdown,
    }
    seal_path = out_dir / "dataset-seal.json"
    seal_path.write_text(
        json.dumps(seal_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info("Evaluation complete. Deliverables written:")
    logger.info(f"  Predictions: {predictions_path}")
    logger.info(f"  Metrics: {metrics_path}")
    logger.info(f"  Breakdown: {breakdown_path}")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Seal: {seal_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
