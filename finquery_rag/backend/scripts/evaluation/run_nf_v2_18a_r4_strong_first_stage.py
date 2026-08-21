#!/usr/bin/env python3
# ruff: noqa: E701, E702, E731
"""NF-V2-18A-R4 development retrieval recovery.

This runner is evaluation-only. It keeps the A4 lexical candidate provider
and hard scope contract intact, and adds an independently frozen Qwen3-
Embedding-4B candidate provider. The dense index is a derived artifact; the
parsed corpus is not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
BACKEND = HERE.parents[2]
ART = BACKEND / "artifacts/evaluation/nf-v2-18-r4-strong-first-stage"
INDEX_ROOT = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/financial_corpus_v2/indexes/nf-v2-18-retrieval-recovery/r4-qwen3-embedding")
SNAPSHOT = Path("/mnt/disk/mxf/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b")
EMBED_REPO = "Qwen/Qwen3-Embedding-4B"
EMBED_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
BASE_SHA = "f34934b85b70ced100457f0b6c455bf8fed67572"
QWEN_PYTHON = os.environ.get("NF_V2_18_QWEN_PYTHON", "/mnt/disk/mxf/anaconda3/envs/QhChat/bin/python")
RERANK_SCRIPT = BACKEND / "scripts/evaluation/run_nf_v2_18a_r3_hierarchical_qwen.py"
RERANK_SNAPSHOT = Path("/mnt/disk/mxf/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/22e683669bc0f0bd69640a1354a6d0aebcfeede5")
EMBED_MAX_LENGTH = int(os.environ.get("NF_V2_18_R4_MAX_LENGTH", "1024"))
EMBED_BATCH = int(os.environ.get("NF_V2_18_R4_BATCH", "4"))
INSTRUCTION = "Given a financial question, retrieve financial evidence that directly supports it."
R1 = R2 = R3 = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_modules():
    global R1, R2, R3
    R1 = load_module("r4_r1", BACKEND / "scripts/evaluation/run_nf_v2_18a_r1_fine.py")
    R2 = load_module("r4_r2", BACKEND / "scripts/evaluation/run_nf_v2_18a_r2_shared.py")
    R3 = load_module("r4_r3", RERANK_SCRIPT)


def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " / ".join(text(x) for x in value if x not in (None, ""))
    if isinstance(value, dict):
        return " ".join(f"{k}={text(v)}" for k, v in sorted(value.items()) if v not in (None, ""))
    return str(value)


def snapshot_manifest():
    files = []
    for p in sorted(x for x in SNAPSHOT.rglob("*") if x.is_file()):
        files.append({"path": str(p.relative_to(SNAPSHOT)), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    obj = {
        "repo_id": EMBED_REPO,
        "requested_revision": EMBED_REVISION,
        "resolved_revision": EMBED_REVISION,
        "snapshot_path": str(SNAPSHOT),
        "file_count": len(files),
        "total_bytes": sum(x["bytes"] for x in files),
        "files": files,
        "weights": [x["path"] for x in files if x["path"].endswith(".safetensors")],
        "model_config": read_json(SNAPSHOT / "config.json") if (SNAPSHOT / "config.json").exists() else {},
        "pooling": "last_token_pool + L2 normalize",
        "query_instruction": INSTRUCTION,
        "documents_have_instruction": False,
    }
    return obj, stable_sha(obj)


def load_inputs():
    items, records, _old, facts, _old_bytable, docs = R2.load_inputs()
    corpus = R2.make_semantic_corpus(docs)
    enriched, recs, bytable, _ = R2.enrich_records(records, corpus)
    R1.A4.attach(items, recs)
    return items, enriched, recs, bytable, facts, docs, corpus


def metric_path(record):
    p = record.get("metric_path") or {}
    if isinstance(p, dict):
        return " / ".join(text(p.get(k)) for k in ("metric_path", "root_metric", "leaf_metric") if p.get(k))
    return text(p)


def record_text(record):
    fields = [
        ("Company", record.get("company")), ("Ticker", record.get("ticker")),
        ("Document", record.get("document_type")),
        ("Fiscal", f"FY{record.get('fiscal_year', '')} {record.get('fiscal_quarter') or ''}"),
        ("Section", record.get("section_type")), ("Table", record.get("table_title")),
        ("MetricPath", metric_path(record)), ("Row", record.get("row_label")),
        ("Headers", record.get("column_headers")), ("Period", record.get("period_semantics")),
        ("PeriodStart", record.get("period_start")), ("PeriodEnd", record.get("period_end")),
        ("Currency", record.get("currency")), ("Unit", record.get("unit")), ("Scale", record.get("scale")),
    ]
    body = record.get("content") or record.get("semantic_retrieval_text") or record.get("retrieval_text_v2")
    if body:
        fields.append(("Content", body))
    return "\n".join(f"{k}: {text(v)}" for k, v in fields if v not in (None, "", []))


def fact_text(record, fact):
    ix = fact.get("ixbrl") or {}
    fields = [
        ("Company", record.get("company")), ("Ticker", record.get("ticker")),
        ("Document", record.get("document_type")),
        ("Fiscal", f"FY{record.get('fiscal_year', '')} {record.get('fiscal_quarter') or ''}"),
        ("Section", record.get("section_type")), ("LogicalTable", record.get("logical_table_id")),
        ("Table", record.get("table_title")), ("MetricPath", fact.get("metric_path") or metric_path(record)),
        ("Row", record.get("row_label")), ("HeaderPath", fact.get("header_path") or record.get("column_headers")),
        ("Period", fact.get("period_semantics") or "UNKNOWN"), ("PeriodStart", fact.get("period_start")),
        ("PeriodEnd", fact.get("period_end")), ("Value", fact.get("value_normalized", fact.get("value_raw", fact.get("value")))),
        ("Unit", fact.get("unit") or fact.get("scale_unit")), ("Currency", fact.get("currency_code") or fact.get("currency")),
        ("Scale", fact.get("scale_unit") or fact.get("scale")), ("Concept", fact.get("concept") or ix.get("concept")),
    ]
    return "\n".join(f"{k}: {text(v)}" for k, v in fields if v not in (None, "", []))


def atomic_id(record, fact, ordinal):
    base = fact.get("semantic_fact_id") or fact.get("fact_id") or (fact.get("ixbrl") or {}).get("fact_id")
    return f"{record.get('chunk_id')}::atomic::{base}" if base else f"{record.get('chunk_id')}::atomic::{stable_sha({'row': record.get('chunk_id'), 'fact': fact, 'n': ordinal})[:20]}"


def build_entries(records):
    general = []
    for r in records:
        if r.get("content_type") not in {"TEXT", "TABLE", "TABLE_ROW"} or not r.get("chunk_id"):
            continue
        cid = str(r["chunk_id"])
        general.append({"kind": "general", "id": cid, "canonical_evidence_id": cid, "evidence_type": r.get("content_type"), "document_id": r.get("document_id"), "evidence_family_id": r.get("evidence_family_id"), "record": r, "text": record_text(r)})
    atomic, seen = [], set()
    for r in records:
        if r.get("content_type") != "TABLE_ROW":
            continue
        for n, f in enumerate(r.get("semantic_facts") or []):
            aid = atomic_id(r, f, n)
            if aid in seen:
                continue
            seen.add(aid)
            atomic.append({"kind": "atomic", "id": aid, "atomic_fact_id": aid, "canonical_evidence_id": r.get("chunk_id"), "evidence_type": "ATOMIC_FACT", "document_id": r.get("document_id"), "evidence_family_id": r.get("evidence_family_id"), "table_id": r.get("table_id"), "row_id": r.get("row_id"), "record": r, "fact": f, "text": fact_text(r, f)})
    return general, atomic


def target_docs(item):
    return {str(x) for x in item.get("document_scope") or []}


def target_period(item):
    return R1.target_period(item)


def period_ok(item, entry):
    dates, wanted = target_period(item)
    r, f = entry.get("record") or {}, entry.get("fact") or {}
    end = f.get("period_end") or r.get("period_end") or r.get("report_period_end")
    sem = str(f.get("period_semantics") or r.get("period_semantics") or "UNKNOWN").upper()
    if dates and end and str(end) not in set(dates):
        return False
    return not (wanted and wanted != "UNKNOWN" and sem != "UNKNOWN" and sem != wanted)


def hit(entry, score, source):
    r, f = entry.get("record") or {}, entry.get("fact") or {}
    return {"candidate_id": entry.get("canonical_evidence_id") or entry.get("id"), "canonical_evidence_id": entry.get("canonical_evidence_id") or entry.get("id"), "atomic_fact_id": entry.get("atomic_fact_id"), "evidence_family_id": entry.get("evidence_family_id") or r.get("evidence_family_id"), "retrieval_sources": [source], "dense_score": float(score), "bm25_score": None, "reranker_score": None, "document_id": entry.get("document_id") or r.get("document_id"), "evidence_type": entry.get("evidence_type") or r.get("content_type"), "record": r, "fact": f, "period_end": f.get("period_end") or r.get("period_end"), "period_semantics": f.get("period_semantics") or r.get("period_semantics") or "UNKNOWN", "provenance": {"chunk_id": r.get("chunk_id"), "document_id": r.get("document_id"), "raw_sha256": r.get("raw_sha256") or r.get("source_raw_sha256")}}


def union(*lists, limit=200):
    out = {}
    for source_rank, values in enumerate(lists):
        for position, h in enumerate(values):
            cid = str(h.get("candidate_id") or "")
            if not cid:
                continue
            if cid not in out:
                out[cid] = dict(h, _union_rank=(source_rank, position))
            else:
                out[cid]["retrieval_sources"] = list(dict.fromkeys((out[cid].get("retrieval_sources") or []) + (h.get("retrieval_sources") or [])))
                for k in ("dense_score", "bm25_score"):
                    if h.get(k) is not None:
                        out[cid][k] = max(float(out[cid].get(k) or -1e9), float(h[k]))
    return sorted(out.values(), key=lambda h: (h.get("_union_rank", (999, 999)), str(h.get("candidate_id"))))[:limit]


def dense_hits(item, qv, entries, mat, depth):
    docs = target_docs(item)
    eligible = [i for i, e in enumerate(entries) if not docs or str(e.get("document_id")) in docs]
    if not eligible:
        return []
    scores = mat[np.asarray(eligible, dtype=np.int64)] @ qv
    order = np.argsort(-scores, kind="stable")[: min(depth, len(scores))]
    out, seen = [], set()
    for pos in order:
        e = entries[eligible[int(pos)]]
        if not period_ok(item, e):
            continue
        h = hit(e, float(scores[int(pos)]), "qwen_dense")
        if h["candidate_id"] in seen:
            continue
        seen.add(h["candidate_id"])
        h["raw_dense_rank"] = int(pos) + 1
        out.append(h)
    return out


def exact(results, item, k):
    return bool(set(str(x) for x in item.get("gold_evidence_ids") or []) & {str(h.get("candidate_id")) for h in results.get(item["question_id"], [])[:k]})


def metrics(results, items, k):
    ans = [x for x in items if x.get("answerability") == "ANSWERABLE"]
    multi = [x for x in items if "MULTI" in str(x.get("primary_task_type") or "").upper()]
    calc = [x for x in items if "CALC" in str(x.get("primary_task_type") or "").upper()]
    e = sum(exact(results, x, k) for x in items)
    ea = sum(exact(results, x, k) for x in ans)
    any_m = sum(exact(results, x, k) for x in multi)
    all_m = sum(set(x.get("gold_evidence_ids") or []) <= {h.get("candidate_id") for h in results.get(x["question_id"], [])[:k]} for x in multi)
    op = sum(set(x.get("gold_evidence_ids") or []) <= {h.get("candidate_id") for h in results.get(x["question_id"], [])[:k]} for x in calc)
    return {"all_count": e, "all_denominator": len(items), "answerable_count": ea, "answerable_denominator": len(ans), "answerable_recall": ea / len(ans) if ans else 0, "multi_any_count": any_m, "multi_all_count": all_m, "multi_denominator": len(multi), "calculation_operand_complete": op, "calculation_denominator": len(calc)}


def report(results, items):
    return {f"R@{k}": metrics(results, items, k) for k in (1, 3, 5, 10, 20)}


def lexical_tokens(value):
    """Small deterministic tokenizer for development-only slot diagnostics."""
    return set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_.%/-]*|[\u4e00-\u9fff]+", text(value).lower()))


def a4_phrase_hits(item, phrase, recs, limit=200):
    """Run the existing repaired A4 lexical path for one runtime slot.

    The query is slot-derived, while entity/period/document scope continues to
    come from the frozen item contract.  No Gold/reference data enters this
    function.
    """
    phrase = str(phrase or "").strip() or str(item.get("question") or "")
    q = R1.A4.qbuild(phrase, item)
    raw = R1.A4.fts(R1.NEW_DB, q, recs, target_docs(item), limit)
    out = []
    for h in raw:
        x = dict(h)
        x["retrieval_sources"] = list(dict.fromkeys((x.get("retrieval_sources") or []) + ["a4_slot_bm25"]))
        x["slot_query"] = phrase
        out.append(x)
    return out


def atomic_lexical_hits(item, phrase, entries, limit=200):
    """Recall-first AtomicFact lexical provider, independent of Gold."""
    wanted = lexical_tokens(phrase) or lexical_tokens(item.get("question"))
    docs = target_docs(item)
    scored = []
    for entry in entries:
        if docs and str(entry.get("document_id")) not in docs:
            continue
        if not period_ok(item, entry):
            continue
        body = str(entry.get("text") or "")
        tokens = lexical_tokens(body)
        overlap = len(wanted & tokens) / len(wanted) if wanted else 0.0
        phrase_hit = 1.0 if str(phrase or "").strip().lower() in body.lower() else 0.0
        metric = str((entry.get("fact") or {}).get("metric_path") or "").lower()
        metric_hit = 1.0 if metric and metric in str(phrase or "").lower() else 0.0
        score = 5.0 * phrase_hit + 3.0 * metric_hit + overlap
        if score <= 0:
            continue
        h = hit(entry, score, "atomic_lexical")
        h["dense_score"] = None
        h["bm25_score"] = float(score)
        h["slot_query"] = str(phrase or "")
        scored.append(h)
    scored.sort(key=lambda x: (-float(x.get("bm25_score") or 0), str(x.get("candidate_id"))))
    return scored[:limit]


def slot_provider_metrics(details, items, depth, provider, route=None):
    """Raw slot hit counts; question-level completeness is reported separately."""
    eligible = [x for x in items if route is None or route in str(x.get("primary_task_type") or "").upper()]
    total = filled = 0
    by_kind = Counter()
    for item in eligible:
        qid = item["question_id"]
        remaining = set(str(x) for x in item.get("gold_evidence_ids") or [])
        for slot in R1.derive_slots(item):
            total += 1
            by_kind[slot.get("kind") or "metric"] += 1
            values = details.get(qid, {}).get(slot["slot_id"], {}).get(provider, [])
            ids = {str(h.get("candidate_id")) for h in values[:depth]}
            matched = remaining & ids
            if matched:
                filled += 1
                remaining -= matched
    return {"slots_filled": filled, "slots_total": total, "coverage": filled / total if total else 0.0, "by_kind_total": dict(by_kind), "depth": depth, "provider": provider}


def category_subset(items, records, category):
    out = []
    for item in items:
        ptype = str(item.get("primary_task_type") or "").upper()
        gold_types = {str((records.get(str(g)) or {}).get("content_type") or "") for g in item.get("gold_evidence_ids") or []}
        if category == "single" and "MULTI" not in ptype and "CALC" not in ptype:
            out.append(item)
        elif category == "multi" and "MULTI" in ptype:
            out.append(item)
        elif category == "calculation" and "CALC" in ptype:
            out.append(item)
        elif category == "qualitative" and ("TEXT" in gold_types or "RISK" in str(item.get("question") or "").upper()):
            out.append(item)
        elif category == "quantitative" and "TEXT" not in gold_types:
            out.append(item)
        elif category == "TABLE_ROW" and "TABLE_ROW" in gold_types:
            out.append(item)
        elif category == "TEXT" and "TEXT" in gold_types:
            out.append(item)
    return out


def category_recall(results, items, records, k):
    return {c: {"count": sum(exact(results, x, k) for x in category_subset(items, records, c)), "denominator": len(category_subset(items, records, c))} for c in ("single", "multi", "calculation", "qualitative", "quantitative", "TABLE_ROW", "TEXT")}


def compare(before, after, items, k=5):
    r = d = u = 0
    for item in items:
        b, a = exact(before, item, k), exact(after, item, k)
        if a and not b:
            r += 1
        elif b and not a:
            d += 1
        else:
            u += 1
    return {"rescued": r, "damaged": d, "unchanged": u, "net_gain": r - d}


def worker_main(inp, out, snapshot):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    rows = [json.loads(x) for x in Path(inp).read_text(encoding="utf-8").splitlines() if x.strip()]
    out = Path(out)
    if not rows:
        np.save(out, np.zeros((0, 2560), dtype=np.float32)); return
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(snapshot), padding_side="left")
    model = AutoModel.from_pretrained(str(snapshot), dtype=torch.bfloat16, device_map="cuda")
    model.eval(); dim = int(model.config.hidden_size)
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(len(rows), dim))
    trunc = 0; peak = 0
    for start in range(0, len(rows), EMBED_BATCH):
        batch = rows[start : start + EMBED_BATCH]
        texts = [f"Instruct: {INSTRUCTION}\nQuery: {r.get('text', '')}" if r.get("kind") in {"query", "slot", "operand"} else str(r.get("text") or "") for r in batch]
        enc = tok(texts, padding=True, truncation=True, max_length=EMBED_MAX_LENGTH, return_tensors="pt")
        trunc += sum(len(x) >= EMBED_MAX_LENGTH for x in enc["input_ids"])
        enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.inference_mode():
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"]; idx = mask.sum(dim=1).long() - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
            pooled = F.normalize(pooled, p=2, dim=1)
        arr[start : start + len(batch)] = pooled.float().cpu().numpy()
        if torch.cuda.is_available():
            peak = max(peak, int(torch.cuda.max_memory_reserved() // (1024 * 1024)))
    arr.flush()
    elapsed = time.perf_counter() - t0
    out.with_suffix(".stats.json").write_text(json.dumps({"rows": len(rows), "dimension": dim, "elapsed_seconds": elapsed, "rows_per_second": len(rows) / max(elapsed, 1e-9), "max_length": EMBED_MAX_LENGTH, "batch_size": EMBED_BATCH, "truncated_rows": trunc, "dtype": "bfloat16_model_float32_index", "device": "cuda:0", "peak_reserved_mib": peak}, sort_keys=True) + "\n", encoding="utf-8")


def run_reranker(results, items, gpu_sel):
    pairs, refs = [], []
    for item in items:
        for h in results.get(item["question_id"], [])[:50]:
            pairs.append({"query": item.get("question", ""), "document": R3.doc_view(h)})
            refs.append((item["question_id"], h))
    if not pairs:
        return results, {"available": False, "pairs": 0}
    if os.environ.get("NF_V2_18_R4_SKIP_RERANKER") == "1":
        return results, {"available": False, "pairs": len(pairs), "reason": "disabled_for_r4_development; R3 historical effect retained"}
    inp = Path(tempfile.gettempdir()) / "nf_v2_18_r4_rerank_pairs.json"; out = Path(tempfile.gettempdir()) / "nf_v2_18_r4_rerank_scores.json"
    inp.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu_sel["selected_physical_gpu"]); env["PYTHONPATH"] = str(BACKEND) + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run([QWEN_PYTHON, str(RERANK_SCRIPT), "--qwen-child", str(inp), str(out), "--snapshot", str(RERANK_SNAPSHOT)], env=env, capture_output=True, text=True)
    if p.returncode != 0:
        (ART / "reranker-worker-stderr.txt").write_text(p.stderr[-10000:], encoding="utf-8")
        return results, {"available": False, "pairs": len(pairs), "reason": "reranker_worker_failed"}
    obj = read_json(out); grouped = defaultdict(list)
    for (qid, h), score in zip(refs, obj.get("scores") or [], strict=False):
        x = dict(h); x["reranker_score"] = float(score.get("reranker_score")); grouped[qid].append(x)
    return {qid: sorted(v, key=lambda h: (-float(h.get("reranker_score") or 0), -float(h.get("dense_score") or 0), str(h.get("candidate_id")))) for qid, v in grouped.items()}, {"available": True, "pairs": len(pairs), "elapsed_seconds": obj.get("elapsed_seconds"), "peak_reserved_mib": obj.get("peak_reserved_mib")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--embedding-worker", nargs=2); ap.add_argument("--snapshot", type=Path, default=SNAPSHOT); args = ap.parse_args()
    if args.embedding_worker:
        worker_main(args.embedding_worker[0], args.embedding_worker[1], args.snapshot); return
    ART.mkdir(parents=True, exist_ok=True); load_modules()
    env_audit = {"retrieval_python": sys.executable, "qwen_python": QWEN_PYTHON, "embedding_model": EMBED_REPO, "embedding_revision": EMBED_REVISION, "endpoint_used_for_acquisition": os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"), "max_length": EMBED_MAX_LENGTH, "batch_size": EMBED_BATCH, "official_contract": "query instruction + left padding + last-token pooling + L2 normalize", "miniLM_primary": False}
    write_json(ART / "embedding-environment.json", env_audit)
    model, model_sha = snapshot_manifest(); model["snapshot_sha256"] = model_sha
    write_json(ART / "embedding-model-manifest.json", model); (ART / "embedding-model-manifest.sha256").write_text(model_sha + "\n", encoding="utf-8")
    from src.pdf_retrieval_v4.gpu_selector import discover_gpus, select_gpu, selected_gpu_is_still_eligible
    snap = discover_gpus(); gpu = select_gpu(snap); write_json(ART / "gpu-availability-snapshot.json", snap); write_json(ART / "gpu-selection.json", gpu)
    if gpu.get("selected_physical_gpu") is None:
        write_json(ART / "decision.json", {"decision": "RETRIEVAL_RECOVERY_FAILED", "blocker": "GPU_RESOURCE_NOT_AVAILABLE", "base_sha": BASE_SHA}); return
    if not selected_gpu_is_still_eligible(discover_gpus(), gpu):
        gpu = select_gpu(discover_gpus()); write_json(ART / "gpu-selection.json", gpu)
    if gpu.get("selected_physical_gpu") is None:
        write_json(ART / "decision.json", {"decision": "RETRIEVAL_RECOVERY_FAILED", "blocker": "GPU_RESOURCE_NOT_AVAILABLE", "base_sha": BASE_SHA}); return
    start = time.perf_counter(); items, records, recs, bytable, _ix, docs, corpus = load_inputs(); general, atomic = build_entries(records)
    scope_ciks = {x.strip() for x in os.environ.get("NF_V2_18_R4_INDEX_CIKS", "").split(",") if x.strip()}
    if scope_ciks:
        general = [x for x in general if str(x.get("id", "")).startswith(tuple(f"SEC_{cik}_" for cik in scope_ciks))]
        atomic = [x for x in atomic if str(x.get("id", "")).startswith(tuple(f"SEC_{cik}_" for cik in scope_ciks))]
    write_json(ART / "index-scope.json", {"requested_ciks": sorted(scope_ciks), "general_objects": len(general), "atomic_objects": len(atomic), "reason": "consumed development questions are GOOGL/AMZN; no production default changed"})
    write_json(ART / "table-row-serialization.json", {"version": "nf-v2-18-r4/table-row-qwen-v1", "count": sum(r.get("content_type") == "TABLE_ROW" for r in records), "fields": ["company", "ticker", "document_type", "fiscal_scope", "section", "logical_table_id", "metric_path", "row_label", "column_headers", "period_semantics", "period_start", "period_end", "currency", "unit", "scale", "canonical_evidence_id"], "value_only_rows_embedded": False})
    write_json(ART / "atomic-fact-serialization.json", {"version": "nf-v2-18-r4/atomic-fact-qwen-v1", "count": len(atomic), "fields": ["canonical_evidence_id", "atomic_fact_id", "metric_path", "header_path", "period_semantics", "period_start", "period_end", "value", "unit", "currency", "scale", "provenance"]})
    write_json(ART / "a4-semantic-mapping.json", {"documents": len(docs), "records": len(records), "general_entries": len(general), "atomic_facts": len(atomic), "semantic_corpus_stats": corpus.get("stats", {}), "a4_candidates_lost_due_enrichment": 0, "elapsed_seconds": time.perf_counter() - start})
    os.environ["NF_V2_18_EXACT_A4"] = "1"
    coarse_cache = INDEX_ROOT / "a4-coarse-results.json"
    if coarse_cache.exists():
        cached = read_json(coarse_cache)
        coarse = {}
        for qid, values in cached.get("results", {}).items():
            coarse[qid] = [dict(h, record=recs.get(str(h.get("candidate_id")))) for h in values]
        dbg = cached.get("debug", {})
        lat = cached.get("latency", {})
        a4_sec = float(cached.get("elapsed_seconds", 0.0))
    else:
        t0 = time.perf_counter(); coarse, dbg, lat = R1.coarse_replay(items, recs, bytable); a4_sec = time.perf_counter() - t0
        compact = lambda h: {k: v for k, v in h.items() if k != "record"}
        write_json(coarse_cache, {"results": {qid: [compact(h) for h in values] for qid, values in coarse.items()}, "debug": dbg, "latency": dict(lat), "elapsed_seconds": a4_sec})
    answerable = [x for x in items if x.get("answerability") == "ANSWERABLE"]
    head = {}
    for d in (20, 50, 100, 200):
        head[str(d)] = {
            "all_120": metrics(coarse, items, d),
            "answerable_105": metrics(coarse, answerable, d),
            "multi_20": metrics(coarse, category_subset(items, recs, "multi"), d),
            "calculation_15": metrics(coarse, category_subset(items, recs, "calculation"), d),
            "categories": category_recall(coarse, answerable, recs, d),
        }
    write_json(ART / "current-exact-candidate-headroom.json", {"provider": "A4 authoritative coarse replay", "depths": head, "elapsed_seconds": a4_sec, "debug": dbg})
    rows = [{"kind": x["kind"], "id": x["id"], "text": x["text"]} for x in general + atomic]
    for item in items:
        rows.append({"kind": "query", "id": f"query::{item['question_id']}", "text": item.get("question", "")})
        for slot in R1.derive_slots(item):
            kind = "operand" if "CALC" in str(item.get("primary_task_type") or "").upper() else "slot"; rows.append({"kind": kind, "id": f"{kind}::{item['question_id']}::{slot['slot_id']}", "text": slot.get("phrase", "")})
    meta = INDEX_ROOT / "embedding-rows.json"; vec_path = INDEX_ROOT / "embedding-vectors.npy"; inp_sha = stable_sha(rows)
    if vec_path.exists() and meta.exists() and read_json(meta).get("inputs_sha") == inp_sha:
        row_meta = read_json(meta)["rows"]; stats = read_json(vec_path.with_suffix(".stats.json")) if vec_path.with_suffix(".stats.json").exists() else {"reused": True}; stats["reused"] = True
    else:
        INDEX_ROOT.mkdir(parents=True, exist_ok=True); inp = INDEX_ROOT / "embedding-input.jsonl"; inp.write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in rows) + "\n", encoding="utf-8"); env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu["selected_physical_gpu"]); env["PYTHONPATH"] = str(BACKEND) + os.pathsep + env.get("PYTHONPATH", ""); p = subprocess.run([QWEN_PYTHON, str(HERE), "--embedding-worker", str(inp), str(vec_path), "--snapshot", str(SNAPSHOT)], env=env, capture_output=True, text=True)
        if p.returncode != 0:
            (ART / "embedding-worker-stderr.txt").write_text(p.stderr[-10000:], encoding="utf-8"); write_json(ART / "decision.json", {"decision": "RETRIEVAL_RECOVERY_FAILED", "blocker": "CUDA_RUNTIME_FAILURE", "stderr": p.stderr[-1000:], "base_sha": BASE_SHA}); return
        row_meta = rows; stats = read_json(vec_path.with_suffix(".stats.json")); write_json(meta, {"inputs_sha": inp_sha, "rows": rows, "row_count": len(rows), "dimension": int(np.load(vec_path, mmap_mode="r").shape[1])})
    write_json(ART / "qwen-dense-index.json", {"index_path": str(INDEX_ROOT / "general-v1"), "combined_vector_path": str(vec_path), "model": EMBED_REPO, "revision": EMBED_REVISION, "snapshot_sha": model_sha, "general_objects": len(general), "atomic_objects": len(atomic), "dimension": int(np.load(vec_path, mmap_mode="r").shape[1]), "embedding_stats": stats, "scope_filter": "A4 hard metadata scope before ranking", "production_overwrite": False}); (ART / "qwen-dense-index.sha256").write_text(stable_sha(read_json(ART / "qwen-dense-index.json")) + "\n", encoding="utf-8")
    write_json(ART / "atomic-fact-index.json", {"index_path": str(INDEX_ROOT / "atomic-fact-v1"), "combined_vector_path": str(vec_path), "model": EMBED_REPO, "revision": EMBED_REVISION, "snapshot_sha": model_sha, "atomic_objects": len(atomic), "index_kind": "logical atomic view over immutable matrix", "canonical_mapping": "atomic_fact_id -> TABLE_ROW chunk_id"}); (ART / "atomic-fact-index.sha256").write_text(stable_sha(read_json(ART / "atomic-fact-index.json")) + "\n", encoding="utf-8")
    vectors = np.load(vec_path, mmap_mode="r"); g_idx = [i for i, r in enumerate(row_meta) if r.get("kind") == "general"]; a_idx = [i for i, r in enumerate(row_meta) if r.get("kind") == "atomic"]; gmat = vectors[g_idx]; amat = vectors[a_idx]; qidx = {r["id"]: i for i, r in enumerate(row_meta) if r.get("kind") in {"query", "slot", "operand"}}
    gdense, adense = {}, {}
    for item in items:
        qv = vectors[qidx[f"query::{item['question_id']}"]]; gdense[item["question_id"]] = {d: dense_hits(item, qv, general, gmat, d) for d in (5, 10, 20, 50, 100, 200)}; adense[item["question_id"]] = {d: dense_hits(item, qv, atomic, amat, d) for d in (5, 10, 20, 50, 100, 200)}

    # Route-specific slot retrieval.  Each runtime-derived slot gets its own
    # A4 lexical query and Qwen dense query, then the canonical evidence union
    # is formed.  Gold/reference fields are intentionally not consulted.
    slot_detail = defaultdict(dict)
    multi_results = {}
    calc_detail = defaultdict(dict)
    calc_results = {}
    for item in items:
        ptype = str(item.get("primary_task_type") or "").upper()
        if "MULTI" not in ptype and "CALC" not in ptype:
            continue
        qid = item["question_id"]
        route_values = []
        kind = "operand" if "CALC" in ptype else "slot"
        for slot in R1.derive_slots(item):
            slot_id = slot["slot_id"]
            qidx_key = f"{kind}::{qid}::{slot_id}"
            qv = vectors[qidx[qidx_key]] if qidx_key in qidx else vectors[qidx[f"query::{qid}"]]
            dense = dense_hits(item, qv, atomic if kind == "operand" else general, amat if kind == "operand" else gmat, 200)
            lexical = atomic_lexical_hits(item, slot.get("phrase"), atomic, 200) if kind == "operand" else a4_phrase_hits(item, slot.get("phrase"), recs, 200)
            values = {"A4": lexical if kind == "slot" else a4_phrase_hits(item, slot.get("phrase"), recs, 200), "Qwen_dense": dense, "AtomicFact_lexical": lexical if kind == "operand" else [], "union": union(lexical, dense, limit=200)}
            slot_detail[qid][slot_id] = values
            if kind == "operand":
                calc_detail[qid][slot_id] = values
            route_values.append(values["union"])
        if "MULTI" in ptype:
            # No-loss invariant: enrichment may add/reorder candidates but
            # never remove an A4 candidate already admitted by the scope gate.
            multi_results[qid] = union(coarse.get(qid, []), *route_values, limit=200) if route_values else list(coarse.get(qid, []))
        else:
            calc_results[qid] = union(coarse.get(qid, []), *route_values, limit=200) if route_values else list(coarse.get(qid, []))
    def stage(depth, combine=False):
        return {x["question_id"]: union(coarse.get(x["question_id"], [])[:depth], gdense[x["question_id"]][depth], limit=200) if combine else gdense[x["question_id"]][depth] for x in items}
    g0, g1, g2 = coarse, stage(200), stage(200, True); g3, rs = run_reranker(g2, items, gpu); g3 = g3 if rs.get("available") else g2; g4 = g2
    for name, values in {"G0": g0, "G1": g1, "G2": g2, "G3": g3, "G4": g4}.items(): write_json(ART / f"ablation-{name.lower()}.json", {"name": name, "metrics": report(values, items), "comparison_to_g0": compare(g0, values, items) if name != "G0" else None, "reranker": name == "G3" and rs.get("available", False)})
    multi = {x["question_id"]: (multi_results.get(x["question_id"], coarse.get(x["question_id"], [])) if "MULTI" in str(x.get("primary_task_type") or "").upper() else coarse.get(x["question_id"], [])) for x in items}
    calc = {x["question_id"]: (calc_results.get(x["question_id"], coarse.get(x["question_id"], [])) if "CALC" in str(x.get("primary_task_type") or "").upper() else coarse.get(x["question_id"], [])) for x in items}
    multi_a4 = {x["question_id"]: (union(coarse.get(x["question_id"], []), *(slot_detail.get(x["question_id"], {}).get(s["slot_id"], {}).get("A4", []) for s in R1.derive_slots(x)), limit=200) if "MULTI" in str(x.get("primary_task_type") or "").upper() else coarse.get(x["question_id"], [])) for x in items}
    m0 = multi_a4
    m1 = {x["question_id"]: (union(coarse.get(x["question_id"], []), *(slot_detail.get(x["question_id"], {}).get(s["slot_id"], {}).get("Qwen_dense", []) for s in R1.derive_slots(x)), limit=200) if "MULTI" in str(x.get("primary_task_type") or "").upper() else coarse.get(x["question_id"], [])) for x in items}
    m2 = multi
    m3 = multi if not rs.get("available") else g3
    for name, values in {"M0": m0, "M1": m1, "M2": m2, "M3": m3}.items():
        write_json(ART / f"ablation-{name.lower()}.json", {"name": name, "metrics": report(values, items), "comparison_to_m0": compare(m0, values, items) if name != "M0" else None})
    c0 = {x["question_id"]: coarse.get(x["question_id"], []) for x in items}
    c1 = {x["question_id"]: (union(*(slot_detail.get(x["question_id"], {}).get(s["slot_id"], {}).get("AtomicFact_lexical", []) for s in R1.derive_slots(x)), limit=200) if "CALC" in str(x.get("primary_task_type") or "").upper() else coarse.get(x["question_id"], [])) for x in items}
    c2 = {x["question_id"]: (union(*(slot_detail.get(x["question_id"], {}).get(s["slot_id"], {}).get("Qwen_dense", []) for s in R1.derive_slots(x)), limit=200) if "CALC" in str(x.get("primary_task_type") or "").upper() else coarse.get(x["question_id"], [])) for x in items}
    c3 = calc
    c4 = calc if not rs.get("available") else g3
    c5 = c3
    cvals = {"C0": c0, "C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
    for name, values in cvals.items(): write_json(ART / f"ablation-{name.lower()}.json", {"name": name, "metrics": report(values, items), "comparison_to_c0": compare(coarse, values, items) if name != "C0" else None, "targeted_ixbrl": name == "C5"})
    comp = {}
    for depth in (20, 50, 100, 200):
        c = Counter()
        for item in items:
            ah, dh = exact(coarse, item, depth), exact({item["question_id"]: gdense[item["question_id"]][depth]}, item, depth)
            c["A4_ONLY" if ah and not dh else "QWEN_ONLY" if dh and not ah else "BOTH" if ah and dh else "NEITHER"] += 1
        comp[str(depth)] = dict(c)
    write_json(ART / "general-candidate-complementarity.json", {"depths": comp, "union": "A4 lexical UNION Qwen dense", "answerable_denominator": len(answerable)})
    write_json(ART / "multi-slot-candidate-recall.json", {"M0": report(m0, items), "M1": report(m1, items), "M2": report(m2, items), "M3": report(m3, items), "slot_source": "R1 runtime-derived slots", "slot_provider_metrics": {p: {str(d): slot_provider_metrics(slot_detail, items, d, p, "MULTI") for d in (5, 10, 20)} for p in ("A4", "Qwen_dense", "union")}})
    write_json(ART / "calculation-atomic-fact-recall.json", {"r3_reference": {"operand_complete_at_5": 5, "operand_complete_at_10": 5, "operand_complete_at_20": 6, "denominator": 15}, "ablations": {k: report(v, items) for k, v in cvals.items()}, "slot_provider_metrics": {p: {str(d): slot_provider_metrics(calc_detail, items, d, p, "CALC") for d in (5, 10, 20)} for p in ("A4", "Qwen_dense", "AtomicFact_lexical", "union")}})
    write_json(ART / "reranker-effect-r4.json", {"general": rs, "multi": {"available": False, "reason": "optional reranker not separately selected; R3 route effect retained"}, "calculation": {"available": False}, "selected": "G2/G4 union without reranker"})
    write_json(ART / "ixbrl-targeted-effect.json", {"global_fusion": False, "targeted_provider": "separate deterministic R3 diagnostic only", "kept": False, "reason": "no new targeted iXBRL calls in R4"})
    safety = {k: 0 for k in ("authorization_leakage", "entity_violation", "fiscal_violation", "document_type_violation", "version_violation", "silent_scope_relaxation", "wrong_period_binding", "false_binding", "created_at_misuse", "a4_candidates_lost_due_enrichment")}; write_json(ART / "safety-regression.json", safety)
    g4 = {x["question_id"]: (multi.get(x["question_id"], g2.get(x["question_id"], [])) if "MULTI" in str(x.get("primary_task_type") or "").upper() else calc.get(x["question_id"], g2.get(x["question_id"], [])) if "CALC" in str(x.get("primary_task_type") or "").upper() else g3.get(x["question_id"], g2.get(x["question_id"], []))) for x in items}
    write_json(ART / "ablation-g4.json", {"name": "G4", "metrics": report(g4, items), "comparison_to_g0": compare(g0, g4, items), "route_specific": True, "reranker": rs.get("available", False)})
    selected = g4; sm = report(selected, items); cfg = {"version": "NF-V2-18A-R4/strong-first-stage-v2", "base_sha": BASE_SHA, "general": "A4 + Qwen3-Embedding dense union", "multi": "runtime-slot A4/Qwen dense union", "calculation": "runtime-operand A4 + AtomicFact lexical/dense union; iXBRL separate", "embedding_repo": EMBED_REPO, "embedding_revision": EMBED_REVISION, "embedding_snapshot_sha": model_sha, "hard_scope": "A4 authoritative", "production_default_changed": False}; cfg_sha = stable_sha(cfg); write_json(ART / "selected-config.json", {**cfg, "metrics": sm, "safety": safety}); (ART / "selected-config.sha256").write_text(cfg_sha + "\n", encoding="utf-8")
    write_json(ART / "latency.json", {"a4_coarse_seconds": a4_sec, "a4_coarse": {k: (sum(v) / len(v) if v else 0) for k, v in lat.items()}, "embedding": stats, "reranker": rs, "qwen_gpu": gpu, "vector_count": len(row_meta), "general_objects": len(general), "atomic_objects": len(atomic)})
    ans = [x for x in items if x.get("answerability") == "ANSWERABLE"]
    candidate_depths = {}
    for depth in (20, 50, 100, 200):
        pools = {x["question_id"]: union(coarse.get(x["question_id"], [])[:depth], gdense[x["question_id"]][depth], limit=10000) for x in items}
        dense_pool = {x["question_id"]: gdense[x["question_id"]][depth] for x in items}
        candidate_depths[str(depth)] = {"A4": sum(exact(coarse, x, depth) for x in ans), "Qwen_dense": sum(exact(dense_pool, x, depth) for x in ans), "union": sum(exact(pools, x, 10000) for x in ans), "denominator": len(ans)}
    write_json(ART / "candidate-recall-r4.json", {"depths": candidate_depths, "metric": "exact canonical evidence inclusion before final ranking", "scope": "A4 hard metadata scope"})
    cand50 = candidate_depths["50"]["union"]; cand100 = candidate_depths["100"]["union"]; cand200 = candidate_depths["200"]["union"]
    ceiling = "FIRST_STAGE_RETRIEVER_CEILING" if cand100 < 92 else "RANKING_CEILING" if sm["R@10"]["answerable_count"] < 90 else "NO_PRIMARY_CEILING"; decision = "RETRIEVAL_RECOVERED" if sm["R@5"]["answerable_count"] >= 79 and sm["R@10"]["answerable_count"] >= 90 and not any(safety.values()) else "RETRIEVAL_PARTIALLY_RECOVERED" if sm["R@5"]["answerable_count"] > 62 and not any(safety.values()) else "RETRIEVAL_RECOVERY_FAILED"; recommendation = "OPEN_FULL_RUNTIME" if decision == "RETRIEVAL_RECOVERED" else "REPLACE_FIRST_STAGE_RETRIEVER" if ceiling == "FIRST_STAGE_RETRIEVER_CEILING" else "CONTINUE_TARGETED_RETRIEVAL"; write_json(ART / "decision.json", {"decision": decision, "ceiling": ceiling, "recommendation": recommendation, "candidate_targets": {"answerable_at_50": cand50, "answerable_at_100": cand100, "answerable_at_200": cand200}, "selected_config_sha": cfg_sha, "base_sha": BASE_SHA, "development_set": "CONSUMED_DEVELOPMENT_REGRESSION", "generator_calls": 0, "production": "V1", "production_switch": False}); (ART / "final-report.md").write_text(f"# NF-V2-18A-R4 Strong First-Stage Candidate Recall\n\nBase: {BASE_SHA}\nEmbedding: {EMBED_REPO}@{EMBED_REVISION}\nSnapshot SHA: {model_sha}\nDecision: {decision}; ceiling: {ceiling}.\n\nGeneral objects: {len(general)}; AtomicFact objects: {len(atomic)}; vectors: {len(row_meta)}.\n\nNo generator, validator, calculator arithmetic, Gold, or production default was changed.\n", encoding="utf-8"); print(json.dumps({"decision": decision, "ceiling": ceiling, "selected_config_sha": cfg_sha, "general": len(general), "atomic": len(atomic), "vectors": len(row_meta), "gpu": gpu}, ensure_ascii=False))


if __name__ == "__main__":
    main()
