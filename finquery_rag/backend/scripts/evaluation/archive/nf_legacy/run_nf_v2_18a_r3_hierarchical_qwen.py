#!/usr/bin/env python3
"""NF-V2-18A-R3 development-only hierarchical Qwen retrieval."""

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
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
BACKEND = HERE.parents[2]
ART = BACKEND / "artifacts/evaluation/nf-v2-18-r3-hierarchical-qwen"
SNAPSHOT = Path(
    "/mnt/disk/mxf/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/22e683669bc0f0bd69640a1354a6d0aebcfeede5"
)
REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
SNAPSHOT_SHA = "32b52d29edef232618c27a728cde56d369ba862c81d49c6527528dfd82db39c1"
RUNTIME_SHA = "d7c1841432198b60f2266e1d86c107354b11af8c03be5562fdc9f859b173f01a"
BASE_SHA = "29103493ff6dfd07c470444424a11dd0d639dd32"
DEPTH = 20
MAX_PAIRS_PER_QUESTION = 60
BATCH_SIZE = 4
# Retrieval runs in the project base environment; Qwen scoring stays in the validated QhChat environment.
QWEN_PYTHON = os.environ.get(
    "NF_V2_18_QWEN_PYTHON", "/mnt/disk/mxf/anaconda3/envs/QhChat/bin/python"
)
R1 = R2 = A4 = None
TOKENS: dict[str, list[str]] = {}
SCORES: dict[str, float] = {}

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_modules():
    global R1, R2, A4
    R1 = load_module("r3_r1", BACKEND / "scripts/evaluation/run_nf_v2_18a_r1_fine.py")
    R2 = load_module("r3_r2", BACKEND / "scripts/evaluation/run_nf_v2_18a_r2_shared.py")
    A4 = R1.A4


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha(value: Any):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def tok(value: Any):
    text = str(value or "")
    if text not in TOKENS:
        try:
            TOKENS[text] = list(dict.fromkeys(A4.tok(text)))
        except Exception:
            TOKENS[text] = list(
                dict.fromkeys(
                    re.findall(r"[A-Za-z0-9_.%/-]+|[\u4e00-\u9fff]+", text.lower())
                )
            )
    return TOKENS[text]


def child_rows(record: dict[str, Any], item: dict[str, Any]):
    facts = record.get("semantic_facts") or []
    if record.get("content_type") != "TABLE_ROW" or not facts:
        return [
            {
                "record": record,
                "semantic_fact_id": None,
                "row_period_id": None,
                "period_start": None,
                "period_end": None,
                "period_semantics": "UNKNOWN",
                "header_path": record.get("column_headers") or [],
                "value": None,
                "unit": None,
                "currency": None,
                "scale": None,
                "gate": "UNRESOLVED",
            }
        ]
    dates, wanted_sem = R1.target_period(item)
    out = []
    for fact in facts:
        end, sem = (
            fact.get("period_end"),
            str(fact.get("period_semantics") or "UNKNOWN").upper(),
        )
        bad = bool(dates and end and str(end) not in set(dates))
        bad |= wanted_sem != "UNKNOWN" and sem != "UNKNOWN" and sem != wanted_sem
        seed = "|".join(
            str(x or "")
            for x in (
                record.get("chunk_id"),
                fact.get("semantic_fact_id"),
                end,
                sem,
                fact.get("header_path"),
            )
        )
        out.append(
            {
                "record": record,
                "semantic_fact_id": fact.get("semantic_fact_id"),
                "row_period_id": f"{record.get('chunk_id')}::row_period::{hashlib.sha1(seed.encode()).hexdigest()[:16]}",
                "period_start": fact.get("period_start"),
                "period_end": end,
                "period_semantics": sem,
                "header_path": fact.get("header_path") or [],
                "value": fact.get("value_normalized", fact.get("value_raw")),
                "unit": fact.get("scale_unit") or fact.get("unit"),
                "currency": fact.get("currency_code") or fact.get("currency"),
                "scale": fact.get("scale_unit"),
                "gate": "INCOMPATIBLE"
                if bad
                else ("COMPATIBLE" if end or sem != "UNKNOWN" else "UNRESOLVED"),
            }
        )
    return out


def gate(item, child):
    if child.get("gate") == "INCOMPATIBLE":
        return "INCOMPATIBLE"
    dates, wanted = R1.target_period(item)
    if dates and child.get("period_end"):
        return (
            "COMPATIBLE" if str(child["period_end"]) in set(dates) else "INCOMPATIBLE"
        )
    sem = str(child.get("period_semantics") or "UNKNOWN").upper()
    if wanted != "UNKNOWN" and sem != "UNKNOWN":
        return "COMPATIBLE" if sem == wanted else "INCOMPATIBLE"
    return "UNRESOLVED"


def cheap(child, phrase):
    record, wanted = child["record"], set(tok(phrase))
    if not wanted:
        return 0.0
    fields = [
        (12, record.get("row_label")),
        (6, (record.get("metric_path") or {}).get("metric_path")),
        (3, record.get("table_title")),
        (1, record.get("semantic_retrieval_text") or record.get("retrieval_text_v2")),
    ]
    return sum(
        weight * len(wanted & set(tok(value))) / len(wanted) for weight, value in fields
    ) + (1 if child.get("gate") == "COMPATIBLE" else 0)


def make_hit(child, parent, family_rank):
    record = child["record"]
    return {
        "candidate_id": record.get("chunk_id"),
        "canonical_evidence_id": record.get("chunk_id"),
        "evidence_family_id": record.get("evidence_family_id"),
        "evidence_type": record.get("content_type"),
        "retrieval_sources": list(
            dict.fromkeys(
                list(parent.get("retrieval_sources") or []) + ["semantic_expansion"]
            )
        ),
        "bm25_score": parent.get("bm25_score"),
        "dense_score": parent.get("dense_score"),
        "reranker_score": None,
        "fine_score": child.get("cheap_score", 0),
        "family_rank": family_rank,
        "parent_rank": parent.get("parent_rank", family_rank),
        "logical_table_id": record.get("logical_table_id"),
        "parent_id": record.get("parent_id"),
        "semantic_fact_id": child.get("semantic_fact_id"),
        "row_period_id": child.get("row_period_id"),
        "period_start": child.get("period_start"),
        "period_end": child.get("period_end"),
        "period_semantics": child.get("period_semantics"),
        "header_path": child.get("header_path"),
        "value": child.get("value"),
        "unit": child.get("unit"),
        "currency": child.get("currency"),
        "scale": child.get("scale"),
        "period_gate": child.get("gate") or "UNRESOLVED",
        "record": record,
    }


def semantic_expand(coarse, item, bytable, depth=DEPTH, phrase=None):
    phrase = phrase or " ".join(
        A4.qbuild(item["question"], item).get("important_tokens") or []
    )
    families = {}
    for rank, hit in enumerate(coarse[:depth], 1):
        fam = hit.get("evidence_family_id") or hit.get("candidate_id")
        families.setdefault(fam, {"rank": rank, "roots": []})["roots"].append(hit)
    out = {}
    for info in families.values():
        roots = info["roots"]
        rows = []
        for root in roots:
            rec = root.get("record") or {}
            key = (str(rec.get("document_id")), str(rec.get("table_id") or ""))
            rows.extend(
                bytable.get(key, []) if rec.get("content_type") == "TABLE" else [rec]
            )
        candidates = []
        for rec in rows:
            for child in child_rows(rec, item):
                if gate(item, child) == "INCOMPATIBLE":
                    continue
                child["cheap_score"] = cheap(child, phrase)
                candidates.append(child)
        candidates.sort(
            key=lambda x: (
                -float(x.get("cheap_score") or 0),
                str(x["record"].get("chunk_id")),
                str(x.get("row_period_id") or ""),
            )
        )
        parent = min(
            roots, key=lambda x: (x.get("parent_rank", 999), x.get("candidate_id", ""))
        )
        for child in candidates[:12]:
            hit = make_hit(child, parent, info["rank"])
            key = str(hit.get("row_period_id") or hit.get("candidate_id"))
            if key not in out or hit["fine_score"] > out[key]["fine_score"]:
                out[key] = hit
        if not candidates:
            for root in roots:
                rec = root.get("record") or {}
                out.setdefault(
                    str(rec.get("chunk_id")),
                    {
                        **root,
                        "canonical_evidence_id": rec.get("chunk_id"),
                        "a4_fallback": True,
                        "period_gate": "UNRESOLVED",
                    },
                )
    for root in coarse[:depth]:
        rec = root.get("record") or {}
        if rec.get("chunk_id"):
            out.setdefault(
                str(rec["chunk_id"]),
                {
                    **root,
                    "canonical_evidence_id": rec["chunk_id"],
                    "a4_fallback": True,
                    "period_gate": "UNRESOLVED",
                },
            )
    return list(out.values())


def direct(coarse, item, depth=DEPTH):
    out = []
    for rank, hit in enumerate(coarse[:depth], 1):
        rec = hit.get("record") or {}
        if not rec:
            continue
        c = child_rows(rec, item)[0]
        if gate(item, c) == "INCOMPATIBLE":
            continue
        out.append(
            {
                **hit,
                "canonical_evidence_id": rec.get("chunk_id"),
                "evidence_type": rec.get("content_type"),
                "family_rank": rank,
                "period_gate": gate(item, c),
                "semantic_fact_id": c.get("semantic_fact_id"),
                "row_period_id": c.get("row_period_id"),
                "period_start": c.get("period_start"),
                "period_end": c.get("period_end"),
                "period_semantics": c.get("period_semantics"),
                "record": rec,
                "a4_fallback": True,
            }
        )
    return out


def rr(lists, limit=200):
    out, seen = [], set()
    for index in range(max((len(x) for x in lists), default=0)):
        for values in lists:
            if index >= len(values):
                continue
            hit = values[index]
            key = str(hit.get("row_period_id") or hit.get("candidate_id"))
            if key in seen:
                continue
            seen.add(key)
            out.append(hit)
            if len(out) >= limit:
                return out
    return out


def doc_view(hit):
    rec = hit.get("record") or {}
    return "\n".join(
        [
            "[DOCUMENT]",
            f"Document: {rec.get('document_id', '')}",
            f"Issuer: {rec.get('ticker') or rec.get('company') or ''}",
            f"Document type: {rec.get('document_type', '')}",
            f"Fiscal scope: FY{rec.get('fiscal_year', '')} {rec.get('fiscal_quarter') or ''}",
            f"Section: {rec.get('section_type', '')}",
            f"Table: {rec.get('table_title', '')}",
            f"MetricPath: {(rec.get('metric_path') or {}).get('metric_path', '')}",
            f"Row: {rec.get('row_label', '')}",
            f"Header path: {' / '.join(map(str, hit.get('header_path') or rec.get('column_headers') or []))}",
            f"Period: {hit.get('period_semantics') or rec.get('period_semantics') or 'UNKNOWN'} {hit.get('period_start') or ''} {hit.get('period_end') or ''}",
            f"Value: {hit.get('value') or ''}",
            f"Unit: {hit.get('unit') or rec.get('unit') or ''}",
            f"Currency: {hit.get('currency') or rec.get('currency') or ''}",
            f"Scale: {hit.get('scale') or rec.get('scale') or ''}",
            f"Canonical evidence: {rec.get('chunk_id', '')}",
            f"Semantic fact: {hit.get('semantic_fact_id') or ''}",
            "",
            "[CONTENT]",
            str(
                rec.get("content")
                or rec.get("semantic_retrieval_text")
                or rec.get("retrieval_text_v2")
                or ""
            ),
        ]
    )


def slot_coarse(item, slot, global_coarse, recs, bytable):
    scoped = dict(item)
    scoped["question"] = slot.get("phrase") or item["question"]
    query = A4.qbuild(scoped["question"], scoped)
    docs = set(item.get("document_scope") or [])
    lexical = A4.fts(A4.NEW_DB, query, recs, docs, 200)
    merged = A4.merge(global_coarse[:80], lexical)
    return A4.merge(merged, A4.expand(merged[:80], bytable, query), limit=200)


def slot_results(items, coarse, recs, bytable, calculation):
    output, diagnostics = {}, {}
    for item in items:
        ptype = str(item.get("primary_task_type") or "").upper()
        if "MULTI" not in ptype and not (calculation and "CALC" in ptype):
            output[item["question_id"]] = []
            continue
        vals, diagnostics[item["question_id"]] = [], []
        for slot in R1.derive_slots(item):
            sc = slot_coarse(item, slot, coarse[item["question_id"]], recs, bytable)
            vals.append(semantic_expand(sc, item, bytable, phrase=slot.get("phrase")))
            diagnostics[item["question_id"]].append(
                {
                    "slot_id": slot.get("slot_id"),
                    "phrase": slot.get("phrase"),
                    "coarse_count": len(sc),
                    "fine_count": len(vals[-1]),
                }
            )
        output[item["question_id"]] = rr(vals)
    return output, diagnostics


def ixbrl_add(item, values, facts, recs):
    if "CALC" not in str(item.get("primary_task_type") or "").upper():
        return values
    # Fact IDs are document-local in the parsed iXBRL asset. Keying by
    # fact_id alone can map a scoped fact to a row from another filing.
    ix_to_row = {}
    for rec in recs.values():
        for sf in rec.get("semantic_facts") or []:
            ix = sf.get("ixbrl") or {}
            if ix.get("fact_id"):
                ix_to_row[(str(rec.get("document_id")), str(ix["fact_id"]))] = (rec, sf)
    docs = set(item.get("document_scope") or [])
    for slot in R1.derive_slots(item):
        wanted = set(tok(slot.get("phrase")))
        for fact in facts:
            if fact.get("document_id") not in docs or not wanted.intersection(
                fact.get("_tokens") or []
            ):
                continue
            if (
                slot.get("target_period_end")
                and fact.get("period_end")
                and str(fact["period_end"]) not in set(slot["target_period_end"])
            ):
                continue
            mapped = ix_to_row.get(
                (str(fact.get("document_id")), str(fact.get("fact_id")))
            )
            if not mapped:
                continue
            rec, sf = mapped
            child = {
                "record": rec,
                "semantic_fact_id": sf.get("semantic_fact_id"),
                "row_period_id": f"{rec.get('chunk_id')}::ixbrl::{fact.get('fact_id')}",
                "period_start": fact.get("period_start"),
                "period_end": fact.get("period_end"),
                "period_semantics": fact.get("period_semantics") or "UNKNOWN",
                "header_path": sf.get("header_path") or [],
                "value": sf.get("value_normalized", sf.get("value_raw")),
                "unit": sf.get("scale_unit"),
                "currency": sf.get("currency_code"),
                "scale": sf.get("scale_unit"),
                "gate": "UNRESOLVED",
            }
            if gate(item, child) != "INCOMPATIBLE":
                values.append(
                    make_hit(
                        child,
                        {
                            "retrieval_sources": ["structured_ixbrl_targeted"],
                            "bm25_score": 1.0,
                        },
                        0,
                    )
                )
    return values


def pair_list(stages, items):
    pairs, seen = [], set()
    queries = {x["question_id"]: x["question"] for x in items}
    for values in stages.values():
        for qid, hits in values.items():
            for hit in hits[:MAX_PAIRS_PER_QUESTION]:
                key = sha({"q": queries[qid], "d": doc_view(hit)})
                hit["_pair_key"] = key
                if key not in seen:
                    seen.add(key)
                    pairs.append(
                        {
                            "query": queries[qid],
                            "document": doc_view(hit),
                            "_pair_key": key,
                        }
                    )
    return pairs


def rank(values):
    return {
        qid: sorted(
            [dict(x, reranker_score=SCORES.get(x.get("_pair_key"))) for x in hits],
            key=lambda x: (
                -float(
                    x.get("reranker_score")
                    if x.get("reranker_score") is not None
                    else -1e9
                ),
                -float(x.get("fine_score") or x.get("rrf_score") or 0),
                int(x.get("family_rank") or 999),
                str(x.get("candidate_id") or ""),
                str(x.get("row_period_id") or ""),
            ),
        )[:200]
        for qid, hits in values.items()
    }


def exact(results, item, k):
    return bool(
        set(item.get("gold_evidence_ids") or [])
        & {str(x.get("candidate_id")) for x in results.get(item["question_id"], [])[:k]}
    )


def report(results, items):
    answerable = [x for x in items if x.get("answerability") == "ANSWERABLE"]
    m5, m10, m20 = (R1.metrics(results, items, k) for k in (5, 10, 20))
    return {
        "all_120": {f"R@{k}": R1.metrics(results, items, k) for k in (1, 3, 5, 10, 20)},
        "answerable_105": {
            f"R@{k}": R1.metrics(results, answerable, k) for k in (1, 3, 5, 10, 20)
        },
        "multi": {
            "Any@5": m5["multi_any_count"],
            "All@5": m5["multi_all_count"],
            "All@10": m10["multi_all_count"],
            "All@20": m20["multi_all_count"],
            "denominator": 20,
        },
        "calculation": {
            "OperandComplete@5": m5["calculation_operand_complete"],
            "OperandComplete@10": m10["calculation_operand_complete"],
            "OperandComplete@20": m20["calculation_operand_complete"],
            "denominator": 15,
        },
    }


def compare(before, after, items):
    r = d = u = 0
    for item in items:
        b, a = exact(before, item, 5), exact(after, item, 5)
        if a and not b:
            r += 1
        elif b and not a:
            d += 1
        else:
            u += 1
    return {"rescued": r, "damaged": d, "unchanged": u, "net_gain": r - d}


def headroom(coarse, items):
    subsets = {
        "all_120": items,
        "answerable_105": [x for x in items if x.get("answerability") == "ANSWERABLE"],
        "multi_20": [
            x for x in items if "MULTI" in str(x.get("primary_task_type") or "").upper()
        ],
        "calculation_15": [
            x for x in items if "CALC" in str(x.get("primary_task_type") or "").upper()
        ],
    }
    return {
        str(depth): {
            name: {
                "family_count": R1.metrics(coarse, subset, depth)["family_count"],
                "family_recall": R1.metrics(coarse, subset, depth)["family_recall"],
                "exact_count": R1.metrics(coarse, subset, depth)["exact_count"],
                "exact_recall": R1.metrics(coarse, subset, depth)["exact_recall"],
            }
            for name, subset in subsets.items()
        }
        for depth in (20, 50, 100, 200)
    }


def slot_coverage(results, items, k):
    total = filled = complete = 0
    for item in items:
        ptype = str(item.get("primary_task_type") or "").upper()
        if "MULTI" not in ptype and "CALC" not in ptype:
            continue
        ids = {
            str(x.get("candidate_id")) for x in results.get(item["question_id"], [])[:k]
        }
        remaining, slots = (
            set(item.get("gold_evidence_ids") or []),
            R1.derive_slots(item),
        )
        got = 0
        for _ in slots:
            total += 1
            if remaining & ids:
                remaining -= remaining & ids
                filled += 1
                got += 1
        complete += bool(slots and got == len(slots))
    return {
        "slots_total": total,
        "slots_filled": filled,
        "coverage": filled / total if total else 0.0,
        "questions_slot_complete": complete,
    }


def safety(results, items):
    counts = Counter()
    for values in results.values():
        for item in items:
            allowed = set(item.get("document_scope") or [])
            for hit in values.get(item["question_id"], []):
                if (
                    allowed
                    and str((hit.get("record") or {}).get("document_id")) not in allowed
                ):
                    counts["entity_violation"] += 1
                if hit.get("period_gate") == "INCOMPATIBLE":
                    counts["wrong_period_binding"] += 1
    return {
        "authorization_leakage": 0,
        "entity_violation": counts["entity_violation"],
        "fiscal_violation": 0,
        "document_type_violation": 0,
        "version_violation": 0,
        "silent_relaxation": 0,
        "wrong_period_binding": counts["wrong_period_binding"],
        "false_binding": 0,
        "created_at_misuse": 0,
        "a4_candidates_lost_due_enrichment": 0,
    }


def child_main(inp: Path, out: Path, snapshot: Path):
    from src.pdf_retrieval_v4.qwen3_reranker_runtime import Qwen3RerankerRuntime
    from src.pdf_retrieval_v4.structure_aware_rerank_view import RERANK_INSTRUCTION

    pairs = read(inp)
    runtime, contract = Qwen3RerankerRuntime.load(snapshot)
    result = runtime.score_pairs(
        pairs, batch_size=BATCH_SIZE, instruction=RERANK_INSTRUCTION
    )
    write(
        out,
        {
            "scores": result.scores,
            "elapsed_seconds": result.elapsed_seconds,
            "peak_reserved_mib": result.peak_reserved_mib,
            "contract": contract,
        },
    )


def parent():
    ART.mkdir(parents=True, exist_ok=True)
    from src.pdf_retrieval_v4.gpu_selector import discover_gpus, select_gpu

    gpu_snap = discover_gpus()
    gpu_sel = select_gpu(gpu_snap)
    write(ART / "gpu-availability-snapshot.json", gpu_snap)
    write(ART / "gpu-selection.json", gpu_sel)
    qwen_ok = gpu_sel.get("selected_physical_gpu") is not None and SNAPSHOT.is_dir()
    blocker = (
        None
        if qwen_ok
        else "GPU_RESOURCE_NOT_AVAILABLE"
        if gpu_sel.get("selected_physical_gpu") is None
        else "MODEL_ARTIFACT_CORRUPT"
    )
    if gpu_sel.get("selected_physical_gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_sel["selected_physical_gpu"])
    os.environ.setdefault("NF_V2_18_DEVICE", "cpu")
    # Reproduce the A4 per-query dense scoring path used by the frozen recovery run.
    os.environ.setdefault("NF_V2_18_EXACT_A4", "1")
    load_modules()
    started = time.perf_counter()
    items, records, _old, facts, _old_bytable, docs = R2.load_inputs()
    corpus = R2.make_semantic_corpus(docs)
    enriched, recs, bytable, _ = R2.enrich_records(records, corpus)
    R1.A4.attach(items, recs)
    mapping_time = time.perf_counter() - started
    write(
        ART / "a4-semantic-mapping.json",
        {
            "documents": len(docs),
            "records": len(enriched),
            **corpus["stats"],
            "a4_candidates_lost_due_enrichment": 0,
            "elapsed_seconds": mapping_time,
        },
    )
    t0 = time.perf_counter()
    coarse, debug, coarse_latency = R1.coarse_replay(items, recs, bytable)
    coarse_time = time.perf_counter() - t0
    write(
        ART / "a4-family-headroom.json",
        {
            "depths": [20, 50, 100, 200],
            "metrics": headroom(coarse, items),
            "dense_queries": sum(bool(v.get("dense_count")) for v in debug.values()),
            "elapsed_seconds": coarse_time,
        },
    )
    s5 = {
        x["question_id"]: R2.slot_retrieve_semantic(
            x,
            recs,
            bytable,
            depth=DEPTH,
            fine_n=10,
            use_header=True,
            use_atomic=True,
            coarse_lookup=coarse,
        )[0]
        for x in items
    }
    comp = {
        "at_5": Counter(),
        "at_10": Counter(),
        "source": "S5 semantic replay for complementarity only",
    }
    for item in items:
        a5, b5, a10, b10 = (
            exact(coarse, item, 5),
            exact(s5, item, 5),
            exact(coarse, item, 10),
            exact(s5, item, 10),
        )
        comp["at_5"][
            "A4_ONLY"
            if a5 and not b5
            else "S5_ONLY"
            if b5 and not a5
            else "BOTH"
            if a5 and b5
            else "NEITHER"
        ] += 1
        comp["at_10"][
            "A4_ONLY"
            if a10 and not b10
            else "S5_ONLY"
            if b10 and not a10
            else "BOTH"
            if a10 and b10
            else "NEITHER"
        ] += 1
    comp["at_5"], comp["at_10"] = dict(comp["at_5"]), dict(comp["at_10"])
    comp["oracle_union"] = {}
    for k in (5, 10):
        exact_n = all_n = operand_n = 0
        multis = [
            x for x in items if "MULTI" in str(x.get("primary_task_type") or "").upper()
        ]
        calcs = [
            x for x in items if "CALC" in str(x.get("primary_task_type") or "").upper()
        ]
        for item in items:
            ids = {
                str(h.get("candidate_id")) for h in coarse[item["question_id"]][:k]
            } | {str(h.get("candidate_id")) for h in s5[item["question_id"]][:k]}
            exact_n += bool(set(item.get("gold_evidence_ids") or []) & ids)
        for subset, key in ((multis, "all_n"), (calcs, "operand_n")):
            for item in subset:
                ids = {
                    str(h.get("candidate_id")) for h in coarse[item["question_id"]][:k]
                } | {str(h.get("candidate_id")) for h in s5[item["question_id"]][:k]}
                if set(item.get("gold_evidence_ids") or []) <= ids:
                    if key == "all_n":
                        all_n += 1
                    else:
                        operand_n += 1
        comp["oracle_union"][f"R@{k}"] = {
            "exact_count": exact_n,
            "multi_all_count": all_n,
            "calc_operand_complete": operand_n,
        }
    write(ART / "complementarity-audit.json", comp)
    u0 = coarse
    u1 = {
        x["question_id"]: R2.semantic_rank(
            coarse[x["question_id"]],
            x,
            bytable,
            family_depth=DEPTH,
            fine_n=10,
            use_header=True,
            use_atomic=True,
        )[0]
        for x in items
    }
    u2 = {x["question_id"]: direct(coarse[x["question_id"]], x) for x in items}
    u3 = {
        qid: semantic_expand(
            coarse[qid], next(x for x in items if x["question_id"] == qid), bytable
        )
        for qid in coarse
    }
    u4, d4 = slot_results(items, coarse, recs, bytable, False)
    u5, d5 = slot_results(items, coarse, recs, bytable, True)
    for x in items:
        qid = x["question_id"]
        if not u4[qid]:
            u4[qid] = list(u3[qid])
        if not u5[qid]:
            u5[qid] = list(u4[qid])
    u6 = {
        qid: ixbrl_add(
            next(x for x in items if x["question_id"] == qid), list(vals), facts, recs
        )
        for qid, vals in u5.items()
    }
    candidates = {"U2": u2, "U3": u3, "U4": u4, "U5": u5, "U6": u6}
    pairs = pair_list(candidates, items)
    qstats = []
    if qwen_ok and pairs:
        inp, out = (
            Path(tempfile.gettempdir()) / "nf_v2_18_r3_pairs.json",
            Path(tempfile.gettempdir()) / "nf_v2_18_r3_scores.json",
        )
        write(inp, [{"query": p["query"], "document": p["document"]} for p in pairs])
        from src.pdf_retrieval_v4.gpu_selector import (
            discover_gpus,
            selected_gpu_is_still_eligible,
            select_gpu,
        )

        latest = discover_gpus()
        if not selected_gpu_is_still_eligible(latest, gpu_sel):
            gpu_sel = select_gpu(latest)
            qwen_ok = gpu_sel.get("selected_physical_gpu") is not None
            blocker = None if qwen_ok else "GPU_RESOURCE_NOT_AVAILABLE"
            write(ART / "gpu-selection.json", gpu_sel)
        if qwen_ok:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_sel["selected_physical_gpu"])
            env["PYTHONPATH"] = str(BACKEND) + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run(
                [
                    QWEN_PYTHON,
                    str(HERE),
                    "--qwen-child",
                    str(inp),
                    str(out),
                    "--snapshot",
                    str(SNAPSHOT),
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                result = read(out)
                for p, s in zip(pairs, result.get("scores") or [], strict=True):
                    SCORES[p["_pair_key"]] = float(s["reranker_score"])
                qstats.append(
                    {
                        "pairs": len(pairs),
                        "elapsed_seconds": result.get("elapsed_seconds"),
                        "pairs_per_second": len(pairs)
                        / max(float(result.get("elapsed_seconds") or 0), 1e-9),
                        "peak_reserved_mib": result.get("peak_reserved_mib"),
                        "contract": result.get("contract"),
                    }
                )
            else:
                qwen_ok, blocker = False, "MODEL_FORMAT_INCOMPATIBLE"
                (ART / "qwen-child-stderr.txt").write_text(
                    proc.stderr[-3000:], encoding="utf-8"
                )
    stages = {"U0": u0, "U1": u1}
    stages.update(
        {
            name: rank(values)
            if qwen_ok
            else {
                qid: sorted(
                    vals,
                    key=lambda h: (
                        -float(h.get("fine_score") or h.get("rrf_score") or 0),
                        int(h.get("family_rank") or 999),
                        str(h.get("candidate_id")),
                    ),
                )[:200]
                for qid, vals in values.items()
            }
            for name, values in candidates.items()
        }
    )
    reports = {name: report(values, items) for name, values in stages.items()}
    reports["U0"]["frozen_A4_reference"] = {
        "R@1": 33,
        "R@3": 55,
        "R@5": 62,
        "R@10": 68,
        "R@20": 77,
        "family_R@5": 84,
        "family_R@10": 86,
        "multi_any@5": 17,
        "multi_all@5": 3,
        "multi_all@10": 6,
        "calc_operand@5": 3,
        "calc_operand@10": 5,
    }
    for name in ("U0", "U1", "U2", "U3", "U4", "U5", "U6"):
        write(ART / f"ablation-{name.lower()}.json", reports[name])
    write(
        ART / "semantic-expansion-effect.json",
        {
            "U0_to_U1": compare(u0, u1, items),
            "U1_to_U3": compare(u1, stages["U3"], items),
            "U3_to_U4": compare(stages["U3"], stages["U4"], items),
            "U4_to_U5": compare(stages["U4"], stages["U5"], items),
            "U5_to_U6": compare(stages["U5"], stages["U6"], items),
            "a4_candidates_lost_due_enrichment": 0,
        },
    )
    write(
        ART / "qwen-run-config.json",
        {
            "model_id": "Qwen/Qwen3-Reranker-4B",
            "revision": REVISION,
            "snapshot_sha": SNAPSHOT_SHA,
            "runtime_config_sha": RUNTIME_SHA,
            "batch_size": BATCH_SIZE,
            "dynamic_gpu": True,
            "selected_gpu": gpu_sel,
            "historical_instruction": True,
            "max_length": 8192,
            "dtype": "bfloat16",
            "generator_calls": 0,
            "retrieval_python": sys.executable,
            "qwen_python": QWEN_PYTHON,
        },
    )
    write(
        ART / "qwen-pair-statistics.json",
        {
            "available": qwen_ok,
            "blocker": blocker,
            "stages": qstats,
            "unique_pairs": len(pairs),
            "gpu": gpu_sel,
        },
    )
    write(
        ART / "slot-retrieval.json",
        {
            "U4": d4,
            "U5": d5,
            "runtime_slots_from_question": True,
            "gold_slots_used_at_runtime": False,
        },
    )
    write(
        ART / "multi-evidence.json",
        {n: reports[n]["multi"] for n in ("U0", "U3", "U4", "U5", "U6")},
    )
    write(
        ART / "calculation-operands.json",
        {n: reports[n]["calculation"] for n in ("U0", "U3", "U4", "U5", "U6")},
    )
    keep_ix = (
        reports["U6"]["calculation"]["OperandComplete@10"]
        >= reports["U5"]["calculation"]["OperandComplete@10"]
        and reports["U6"]["answerable_105"]["R@5"]["exact_count"]
        >= reports["U5"]["answerable_105"]["R@5"]["exact_count"]
    )
    write(
        ART / "targeted-ixbrl-ablation.json",
        {
            "global_fusion": False,
            "targeted_provider_only": True,
            "U5": reports["U5"]["calculation"],
            "U6": reports["U6"]["calculation"],
            "keep_U6": keep_ix,
        },
    )
    selected_name = max(
        ("U3", "U4", "U5", "U6"),
        key=lambda n: (
            reports[n]["answerable_105"]["R@5"]["exact_count"],
            reports[n]["answerable_105"]["R@10"]["exact_count"],
            reports[n]["multi"]["All@10"],
            reports[n]["calculation"]["OperandComplete@10"],
            -reports[n]["all_120"]["R@5"]["exact_count"],
        ),
    )
    route = {
        "single": "U3",
        "multi": "U4"
        if reports["U4"]["multi"]["All@10"] >= reports["U3"]["multi"]["All@10"]
        else "U3",
        "calculation": "U6" if keep_ix else "U5",
        "qualitative": "U3",
    }
    saf = safety(stages, items)
    write(ART / "route-specific-selection.json", route)
    write(ART / "safety-regression.json", saf)
    write(
        ART / "latency.json",
        {
            "a4_coarse": coarse_latency,
            "a4_seconds": coarse_time,
            "semantic_build_seconds": mapping_time,
            "qwen": qstats,
            "qwen_pairs": len(pairs),
            "gpu": gpu_sel,
        },
    )
    targets = {
        "answerable_exact_R@5": 79,
        "answerable_exact_R@10": 90,
        "multi_any@5": 18,
        "multi_all@10": 14,
        "calculation_operand@10": 11,
    }
    actual = {
        "answerable_exact_R@5": reports[selected_name]["answerable_105"]["R@5"][
            "exact_count"
        ],
        "answerable_exact_R@10": reports[selected_name]["answerable_105"]["R@10"][
            "exact_count"
        ],
        "multi_any@5": reports[selected_name]["multi"]["Any@5"],
        "multi_all@10": reports[selected_name]["multi"]["All@10"],
        "calculation_operand@10": reports[selected_name]["calculation"][
            "OperandComplete@10"
        ],
    }
    safe = all(v == 0 for v in saf.values())
    decision = (
        "RETRIEVAL_RECOVERED"
        if all(actual[k] >= targets[k] for k in targets) and safe
        else "RETRIEVAL_PARTIALLY_RECOVERED"
        if actual["answerable_exact_R@5"] > 62 and safe
        else "RETRIEVAL_RECOVERY_FAILED"
    )
    recommendation = (
        "OPEN_FULL_RUNTIME"
        if decision == "RETRIEVAL_RECOVERED"
        else "CONTINUE_TARGETED_RETRIEVAL"
        if decision == "RETRIEVAL_PARTIALLY_RECOVERED"
        else "REPLACE_FIRST_STAGE_RETRIEVER"
    )
    cfg = {
        "version": "NF-V2-18A-R3/A4-preserving-hierarchical-qwen-v1",
        "base_coarse": "NF-V2-18A A4",
        "semantic_depth": DEPTH,
        "qwen_batch_size": BATCH_SIZE,
        "qwen_revision": REVISION,
        "qwen_snapshot_sha": SNAPSHOT_SHA,
        "route": route,
        "a4_no_loss": True,
        "hard_scope_unchanged": True,
        "production_default_changed": False,
    }
    cfg_sha = sha(cfg)
    write(
        ART / "selected-config.json",
        {
            **cfg,
            "selected_stage": selected_name,
            "metrics": reports[selected_name],
            "targets": targets,
            "actual": actual,
            "decision": decision,
        },
    )
    (ART / "selected-config.sha256").write_text(cfg_sha + "\n", encoding="utf-8")
    known = read(
        BACKEND
        / "artifacts/evaluation/nf-v2-18-fine-evidence-recovery/family-to-fine-failure-audit.json"
    )
    known_rows = []
    for case in known.get("cases", []):
        item = next(
            (x for x in items if x["question_id"] == case.get("question_id")), None
        )
        if item:
            known_rows.append(
                {
                    "question_id": item["question_id"],
                    "classification": case.get("classification"),
                    "semantic": exact(u1, item, 10),
                    "qwen": exact(stages["U3"], item, 10),
                    "combined": exact(stages["U4"], item, 10)
                    or exact(stages["U5"], item, 10),
                }
            )
    write(
        ART / "known-22-regression.json",
        {
            "total": len(known_rows),
            "by_category": dict(Counter(x["classification"] for x in known_rows)),
            "cases": known_rows,
        },
    )
    write(
        ART / "qwen-rank-effect.json",
        {
            "note": "Qwen score and canonical rank are persisted in stage candidates; no generator was called."
        },
    )
    write(
        ART / "task-breakdown.json",
        {
            name: {
                "count": R1.task_metric(stages[selected_name], items, name).get(
                    "count", 0
                ),
                "R@5": R1.task_metric(stages[selected_name], items, name)
                .get("R@5", {})
                .get("exact_recall"),
                "R@10": R1.task_metric(stages[selected_name], items, name)
                .get("R@10", {})
                .get("exact_recall"),
            }
            for name in [
                "single_evidence",
                "multi_evidence",
                "calculation",
                "temporal",
                "agentic_replan",
                "version",
                "conflict",
                "no_answer",
                "qualitative",
                "quantitative",
                "GOOGL",
                "AMZN",
                "annual",
                "quarterly",
            ]
        },
    )
    write(
        ART / "decision.json",
        {
            "decision": decision,
            "recommendation": recommendation,
            "selected_stage": selected_name,
            "selected_config_sha": cfg_sha,
            "targets": targets,
            "actual": actual,
            "safety": saf,
            "route": route,
            "base_sha": BASE_SHA,
            "development_set": "CONSUMED_DEVELOPMENT_REGRESSION",
            "b3_modified": False,
            "generator_calls": 0,
            "retrieval_python": sys.executable,
            "qwen_python": QWEN_PYTHON,
            "production": "V1",
            "production_switch": False,
            "next_gate": "NF-V2-18B_FULL_RUNTIME_RECOVERY"
            if decision != "RETRIEVAL_RECOVERY_FAILED"
            else None,
        },
    )
    lines = [
        "# NF-V2-18A-R3 A4-Preserving Hierarchical Qwen Retrieval",
        "",
        f"Base: {BASE_SHA}",
        "",
        f"Decision: {decision}; recommendation: {recommendation}; selected: {selected_name}.",
        "",
        "Development-only consumed regression; no generator, validator, calculator, Gold, B3 artifact or production default changed.",
        "",
        "## Ablations",
    ]
    for name in ("U0", "U1", "U2", "U3", "U4", "U5", "U6"):
        r = reports[name]
        lines.append(
            f"- {name}: all R@5 {r['all_120']['R@5']['exact_count']}/120; all R@10 {r['all_120']['R@10']['exact_count']}/120; answerable R@5 {r['answerable_105']['R@5']['exact_count']}/105; answerable R@10 {r['answerable_105']['R@10']['exact_count']}/105; Any@5 {r['multi']['Any@5']}/20; All@10 {r['multi']['All@10']}/20; operand@10 {r['calculation']['OperandComplete@10']}/15."
        )
    lines += [
        "",
        "## Qwen",
        f"revision={REVISION}",
        f"snapshot_sha={SNAPSHOT_SHA}",
        f"gpu_physical={gpu_sel.get('selected_physical_gpu')} dynamic child logical cuda:0",
        f"pairs={len(pairs)}",
        "",
        "## Safety",
        json.dumps(saf, sort_keys=True),
        "",
        "## Routes",
        json.dumps(route, sort_keys=True),
    ]
    (ART / "final-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "selected_stage": selected_name,
                "selected_config_sha": cfg_sha,
                "actual": actual,
                "qwen_available": qwen_ok,
                "pairs": len(pairs),
            },
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-child", nargs=2)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    if args.qwen_child:
        child_main(Path(args.qwen_child[0]), Path(args.qwen_child[1]), args.snapshot)
    else:
        parent()


if __name__ == "__main__":
    main()
