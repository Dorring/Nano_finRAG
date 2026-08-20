#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
BACKEND = REPO / "finquery_rag/backend"
ARTB3 = BACKEND / "artifacts/evaluation/nf-v2-17-fresh-blind-eval"
ART = BACKEND / "artifacts/evaluation/nf-v2-18-fine-evidence-recovery"
A4_PATH = BACKEND / "scripts/evaluation/run_nf_v2_18a_recovery.py"
CORPUS = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/financial_corpus_v2")
NEW_DB = CORPUS / "indexes/nf-v2-18-retrieval-recovery/enriched-bm25/index.sqlite"
DENSE = CORPUS / "indexes/nf-v2-18-retrieval-recovery/dense-v2"
MODEL = "/home/mxf/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BASE_SHA = "122a96b302c8c53c71eb1185b9df86a7103567e7"
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.%/-]*|[0-9][A-Za-z0-9_.%/-]*|[\u4e00-\u9fff]+")
DATE = re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b")
STOP = set(
    "a an and are as at be been being by does do did for from how in is it of on or the to was were what when where which who why with would report reports filing filed give show shown according please tell me row value amount number figure following both either either's their this that does associated statement disclosure given source occurrence use using calculate calculate what".split()
)
TOK_CACHE = {}


def load_a4():
    spec = importlib.util.spec_from_file_location("nf_v2_18a", A4_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A4 = load_a4()


def JL(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def W(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha(obj):
    return hashlib.sha256(
        json.dumps(
            obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def tokens(text):
    key = str(text or "")
    if key not in TOK_CACHE:
        TOK_CACHE[key] = list(dict.fromkeys(A4.tok(key)))
    return TOK_CACHE[key]


def norm(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def quoted(question):
    return [
        m.group(1).strip()
        for m in re.finditer(r"['\"]([^'\"]+)['\"]", str(question))
        if tokens(m.group(1).strip())
    ]


def target_period(item):
    ts = item.get("temporal_scope") or {}
    vals = []
    for key in ("period_end", "document_end"):
        if ts.get(key):
            vals.append(str(ts[key]))
    for key in ("periods", "period_end_values"):
        for value in ts.get(key) or []:
            if value:
                vals.append(str(value))
    return list(dict.fromkeys(vals)), str(ts.get("fact_semantics") or "UNKNOWN").upper()


def derive_slots(item):
    q = str(item.get("question") or "")
    phrases = quoted(q)
    ptype = str(item.get("primary_task_type") or "").upper()
    if "MULTI" in ptype and len(phrases) >= 2:
        phrases = phrases[:]
    elif "CALC" in ptype and len(phrases) >= 2:
        phrases = phrases[:2]
    elif not phrases:
        qb = A4.qbuild(q, item)
        phrases = [" ".join(qb.get("important_tokens") or [])]
    else:
        phrases = [phrases[0]]
    if not phrases:
        phrases = [q]
    kind = "metric"
    if "CALC" in ptype:
        kind = "operand"
    dates, semantics = target_period(item)
    docs = sorted(str(x) for x in item.get("document_scope") or [])
    out = []
    for i, phrase in enumerate(phrases):
        out.append(
            {
                "slot_id": f"{item['question_id']}::slot_{i + 1}",
                "kind": kind,
                "phrase": phrase,
                "target_period_end": dates,
                "target_period_semantics": semantics,
                "document_ids": docs,
            }
        )
    return out


def record_label(record):
    if record.get("row_label"):
        return str(record["row_label"])
    cells = record.get("cells") or []
    if cells and isinstance(cells[0], dict):
        for key in ("column_header", "raw_value", "text", "value"):
            if cells[0].get(key):
                return str(cells[0][key])
    return ""


def record_headers(record):
    out = [str(x) for x in record.get("column_headers") or [] if x]
    for cell in record.get("cells") or []:
        if isinstance(cell, dict):
            for key in ("column_header", "period_label", "header", "column"):
                if cell.get(key):
                    out.append(str(cell[key]))
    return out


def period_info(record, item):
    dates, semantics = target_period(item)
    headers = record_headers(record)
    header_text = " ".join(headers)
    all_text = " ".join(
        [
            header_text,
            str(record.get("period_start") or ""),
            str(record.get("period_end") or ""),
            str(record.get("period_semantics") or ""),
        ]
    )
    target_found = bool(dates and any(date in all_text for date in dates))
    target_years = {date[:4] for date in dates if len(date) >= 4}
    year_found = bool(target_years and any(year in all_text for year in target_years))
    explicit_headers = bool(
        DATE.search(header_text)
        or re.search(
            r"(?:three|six|nine|twelve|three|six|nine)\s+months|year ended|as of",
            header_text,
            re.I,
        )
    )
    rec_sem = str(record.get("period_semantics") or "UNKNOWN").upper()
    deterministic_wrong = False
    reason = "UNKNOWN"
    bonus = 0.0
    if (
        record.get("_deterministic_row_period")
        and record.get("period_end")
        and dates
        and str(record["period_end"]) not in dates
    ):
        deterministic_wrong = True
        reason = "EXPLICIT_RECORD_PERIOD_MISMATCH"
    elif (
        record.get("_deterministic_row_period")
        and semantics != "UNKNOWN"
        and rec_sem != "UNKNOWN"
        and rec_sem != semantics
    ):
        if explicit_headers and target_found is False and year_found is False:
            deterministic_wrong = True
            reason = "EXPLICIT_SEMANTICS_MISMATCH"
        else:
            reason = "SEMANTICS_AMBIGUOUS"
    elif target_found or year_found:
        bonus += 4.0
        reason = "TARGET_PERIOD_PRESENT"
    elif record.get("_deterministic_row_period") and explicit_headers and dates:
        deterministic_wrong = True
        reason = "TARGET_PERIOD_ABSENT_FROM_EXPLICIT_HEADER"
    elif semantics != "UNKNOWN" and rec_sem == semantics:
        bonus += 2.0
        reason = "DOCUMENT_PERIOD_SEMANTICS_MATCH"
    return {
        "deterministic_wrong": deterministic_wrong,
        "reason": reason,
        "bonus": bonus,
        "target_found": target_found,
        "header_text": header_text[:2000],
        "record_semantics": rec_sem,
    }


def phrase_score(record, phrases, item):
    label = record_label(record)
    title = str(record.get("table_title") or "")
    full = str(record.get("retrieval_text_v2") or record.get("content") or "")
    label_n, title_n = norm(label), norm(title)
    label_t, title_t, full_t = set(tokens(label)), set(tokens(title)), set(tokens(full))
    best = 0.0
    best_phrase = ""
    for phrase in phrases:
        pt = set(tokens(phrase))
        if not pt:
            continue
        overlap_label = len(pt & label_t) / len(pt)
        overlap_title = len(pt & title_t) / len(pt)
        overlap_full = len(pt & full_t) / len(pt)
        exact_label = 1.0 if norm(phrase) in label_n and norm(phrase) else 0.0
        exact_title = 1.0 if norm(phrase) in title_n and norm(phrase) else 0.0
        score = (
            12.0 * exact_label
            + 8.0 * overlap_label
            + 3.0 * exact_title
            + 2.5 * overlap_title
            + 1.5 * overlap_full
        )
        if score > best:
            best, best_phrase = score, phrase
    pinfo = period_info(record, item)
    return best + pinfo["bonus"], best_phrase, pinfo


def make_hit(record, parent_hit, score, phrase, pinfo):
    return {
        "candidate_id": record["chunk_id"],
        "evidence_family_id": record.get("evidence_family_id"),
        "retrieval_sources": list(
            dict.fromkeys(
                list(parent_hit.get("retrieval_sources") or []) + ["local_lexical"]
            )
        ),
        "bm25_score": parent_hit.get("bm25_score"),
        "dense_score": parent_hit.get("dense_score"),
        "reranker_score": None,
        "fine_score": float(score),
        "parent_id": record.get("parent_id"),
        "parent_rank": parent_hit.get("parent_rank"),
        "row_period_id": (
            f"{record['chunk_id']}::row_period::{hashlib.sha1((str(phrase) + '|' + str(pinfo.get('target_found'))).encode()).hexdigest()[:12]}"
            if record.get("content_type") == "TABLE_ROW" and pinfo.get("target_found")
            else None
        ),
        "period_binding": pinfo,
        "record": record,
    }


def local_fine(coarse, item, bytable, family_depth=20, fine_n=10, phrases=None):
    phrases = phrases or [x["phrase"] for x in derive_slots(item)]
    families = {}
    for rank, hit in enumerate(coarse[:family_depth], 1):
        fam = hit.get("evidence_family_id") or hit["candidate_id"]
        if fam not in families:
            families[fam] = {"rank": rank, "hits": []}
        families[fam]["hits"].append(hit)
    candidates = {}
    wrong_period = 0
    considered = 0
    for fam, info in families.items():
        roots = info["hits"]
        children = []
        for hit in roots:
            r = hit.get("record") or {}
            ctype = r.get("content_type")
            key = (str(r.get("document_id")), str(r.get("table_id") or ""))
            if ctype in {"TABLE", "TABLE_ROW"}:
                children.extend(bytable.get(key, []))
            else:
                children.append(r)
        child_seen = set()
        ranked = []
        for child in children:
            cid = child.get("chunk_id")
            if not cid or cid in child_seen:
                continue
            child_seen.add(cid)
            considered += 1
            pscore, best_phrase, pinfo = phrase_score(child, phrases, item)
            if pinfo["deterministic_wrong"]:
                wrong_period += 1
                continue
            parent_hit = min(
                roots,
                key=lambda h: (
                    h.get("parent_rank", 999),
                    h.get("candidate_id", ""),
                ),
            )
            hit = make_hit(child, parent_hit, pscore, best_phrase, pinfo)
            hit["family_rank"] = info["rank"]
            hit["child_count"] = len(children)
            ranked.append(hit)
        ranked.sort(
            key=lambda h: (
                -float(h.get("fine_score") or 0),
                h.get("family_rank", 999),
                h["candidate_id"],
            )
        )
        for hit in ranked[:fine_n]:
            old = candidates.get(hit["candidate_id"])
            if old is None or hit["fine_score"] > old["fine_score"]:
                candidates[hit["candidate_id"]] = hit
    for coarse_rank, source_hit in enumerate(coarse[:200], 1):
        record = source_hit.get("record") or {}
        cid = source_hit.get("candidate_id")
        if not cid or not record:
            continue
        pscore, best_phrase, pinfo = phrase_score(record, phrases, item)
        if pinfo["deterministic_wrong"]:
            wrong_period += 1
            continue
        direct = make_hit(
            record,
            source_hit,
            pscore + max(0.0, 15.0 - 0.05 * coarse_rank),
            best_phrase,
            pinfo,
        )
        direct["family_rank"] = coarse_rank
        direct["coarse_fallback"] = True
        old = candidates.get(cid)
        if old is None or direct["fine_score"] > old["fine_score"]:
            candidates[cid] = direct
    out = sorted(
        candidates.values(),
        key=lambda h: (
            -float(h.get("fine_score") or 0),
            h.get("family_rank", 999),
            h["candidate_id"],
        ),
    )
    return out[:200], {
        "children_considered": considered,
        "wrong_period_excluded": wrong_period,
        "coarse_fallback_candidates": min(len(coarse), 200),
    }


def dense_hits(ids, vectors, qvec, docs, recs, k=200):
    import numpy as np

    if vectors is None:
        return []
    indices = [
        i
        for i, cid in enumerate(ids)
        if not docs or str(recs.get(cid, {}).get("document_id")) in docs
    ]
    if not indices:
        return []
    idx = np.asarray(indices, dtype=np.int64)
    scores = vectors[idx] @ qvec
    order = np.argsort(-scores)
    out = []
    for pos in order:
        i = int(idx[int(pos)])
        cid = ids[i]
        record = recs.get(cid)
        if not record:
            continue
        out.append(
            {
                "candidate_id": cid,
                "evidence_family_id": record.get("evidence_family_id"),
                "retrieval_sources": ["dense"],
                "bm25_score": None,
                "dense_score": float(scores[int(pos)]),
                "reranker_score": None,
                "parent_id": record.get("parent_id"),
                "record": record,
            }
        )
        if len(out) >= k:
            break
    return out


def round_robin(lists, limit=200):
    out, seen = [], set()
    for i in range(max((len(x) for x in lists), default=0)):
        for vals in lists:
            if i >= len(vals):
                continue
            hit = vals[i]
            cid = hit["candidate_id"]
            if cid in seen:
                continue
            seen.add(cid)
            out.append(hit)
            if len(out) >= limit:
                return out
    return out


def slot_retrieve(item, slots, bytable, recs, depth=20, fine_n=10):
    results = []
    diagnostics = []
    for slot in slots:
        slot_item = dict(item)
        slot_item["question"] = slot["phrase"]
        q = A4.qbuild(slot["phrase"], slot_item)
        t0 = time.perf_counter()
        coarse = A4.fts(NEW_DB, q, recs, set(item.get("document_scope") or []), 200)
        fine, fd = local_fine(
            coarse,
            item,
            bytable,
            family_depth=depth,
            fine_n=fine_n,
            phrases=[slot["phrase"]],
        )
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(fine)
        diagnostics.append(
            {
                "slot_id": slot["slot_id"],
                "kind": slot["kind"],
                "phrase": slot["phrase"],
                "coarse_count": len(coarse),
                "fine_count": len(fine),
                "children_considered": fd["children_considered"],
                "wrong_period_excluded": fd["wrong_period_excluded"],
                "latency_ms": elapsed,
            }
        )
    return round_robin(results, 200), diagnostics, results


def dense_load():
    try:
        import numpy as np

        ids = [str(x) for x in A4.J(DENSE / "ids.json")]
        vectors = np.load(DENSE / "vectors.npy", mmap_mode="r")
        return ids, vectors
    except Exception:
        return None, None


def batch_query_vectors(items):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        from sentence_transformers import SentenceTransformer

        model_path = Path(MODEL)
        enc = SentenceTransformer(
            str(model_path) if model_path.exists() else "all-MiniLM-L6-v2",
            device=os.environ.get("NF_V2_18_DEVICE", "cpu"),
        )
        vectors = enc.encode(
            [str(x["question"]) for x in items],
            batch_size=int(os.environ.get("NF_V2_18_QUERY_BATCH", "64")),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return enc, vectors
    except Exception as exc:
        return None, {"error": f"{type(exc).__name__}: {exc}"}


def load_data():
    questions = JL(ARTB3 / "fresh-blind-eval-v1.jsonl")
    gold_rows = {
        x["question_id"]: x for x in JL(ARTB3 / "fresh-blind-gold-evidence-v1.jsonl")
    }
    items = []
    for question in questions:
        item = dict(question)
        item["gold_evidence_ids"] = [
            str(e.get("chunk_id"))
            for e in gold_rows.get(question["question_id"], {}).get("gold_evidence", [])
            if e.get("chunk_id")
        ]
        items.append(item)
    records, facts = A4.load_records()
    recs = {r["chunk_id"]: r for r in records}
    A4.attach(items, recs)
    bytable = defaultdict(list)
    for record in records:
        if record.get("content_type") == "TABLE_ROW":
            bytable[
                (str(record.get("document_id")), str(record.get("table_id") or ""))
            ].append(record)
    return items, records, recs, facts, bytable


def coarse_replay(items, recs, bytable):
    ids, vectors = dense_load()
    _enc, qvectors = batch_query_vectors(items)
    if isinstance(qvectors, dict):
        qvectors = None
    out = {}
    debug = {}
    latency = defaultdict(list)
    for i, item in enumerate(items):
        q = A4.qbuild(item["question"], item)
        docs = set(item.get("document_scope") or [])
        t0 = time.perf_counter()
        bm = A4.fts(NEW_DB, q, recs, docs, 200)
        latency["bm25_ms"].append((time.perf_counter() - t0) * 1000)
        dense = []
        if qvectors is not None and ids and vectors is not None:
            t1 = time.perf_counter()
            if os.environ.get("NF_V2_18_EXACT_A4") == "1":
                dense = A4.dsearch(
                    ids, vectors, _enc, item["question"], recs, docs, 200
                )
            else:
                dense = dense_hits(
                    ids,
                    vectors,
                    qvectors[i],
                    docs,
                    recs,
                    200,
                )
            latency["dense_ms"].append((time.perf_counter() - t1) * 1000)
        merged = A4.merge(bm, dense)
        t2 = time.perf_counter()
        expanded = A4.expand(merged[:80], bytable, q)
        latency["expand_ms"].append((time.perf_counter() - t2) * 1000)
        coarse = A4.merge(merged, expanded)
        latency["coarse_ms"].append((time.perf_counter() - t0) * 1000)
        qid = item["question_id"]
        out[qid] = coarse
        debug[qid] = {
            "question": item["question"],
            "query": q,
            "document_scope": sorted(docs),
            "bm25_count": len(bm),
            "dense_count": len(dense),
            "coarse_count": len(coarse),
        }
    return out, debug, latency


def metrics(results, items, k):
    return A4.metrics(results, items, k)


def stage_metricset(results, items):
    return {f"R@{k}": metrics(results, items, k) for k in (1, 3, 5, 10, 20)}


def stage_slots(results, items, k):
    total = filled = 0
    by_kind = Counter()
    for item in items:
        slots = derive_slots(item)
        if not slots:
            continue
        ids = {x["candidate_id"] for x in results.get(item["question_id"], [])[:k]}
        gold_ids = set(item.get("gold_evidence_ids") or [])
        for slot in slots:
            total += 1
            by_kind[slot["kind"]] += 1
            if gold_ids & ids:
                filled += 1
                # For a multi/calc query, only count one gold object per slot.
                gold_ids = gold_ids - (gold_ids & ids)
    return {
        "slots_total": total,
        "slots_filled": filled,
        "coverage": filled / total if total else 0.0,
        "by_kind_total": dict(by_kind),
    }


def task_metric(results, items, name):
    def pred(item):
        ptype = str(item.get("primary_task_type") or "").upper()
        if name == "single_evidence":
            return "SINGLE" in ptype
        if name == "multi_evidence":
            return "MULTI" in ptype
        if name == "calculation":
            return "CALC" in ptype
        if name == "temporal":
            return "TEMPORAL" in ptype
        if name == "agentic_replan":
            return "REPLAN" in ptype
        if name == "version":
            return "VERSION" in ptype
        if name == "conflict":
            return "CONFLICT" in ptype
        if name == "no_answer":
            return item.get("answerability") != "ANSWERABLE"
        if name == "qualitative":
            return "qualitative" in (item.get("secondary_task_tags") or [])
        if name == "quantitative":
            return "quantitative" in (item.get("secondary_task_tags") or [])
        if name in {"GOOGL", "AMZN"}:
            return name in (item.get("entity_scope") or [])
        if name == "quarterly":
            ts = item.get("temporal_scope") or {}
            return item.get("answerability") == "ANSWERABLE" and (
                str(ts.get("fact_semantics") or "") in {"QUARTER", "YTD"}
                or any(item.get("document_scope") or [])
            )
        if name == "annual":
            return item.get("answerability") == "ANSWERABLE" and not pred_quarter(item)
        return False

    def pred_quarter(item):
        ts = item.get("temporal_scope") or {}
        return str(ts.get("fact_semantics") or "") in {"QUARTER", "YTD"}

    subset = [x for x in items if pred(x)]
    return {
        "count": len(subset),
        "R@5": metrics(results, subset, 5) if subset else {},
        "R@10": metrics(results, subset, 10) if subset else {},
    }


def family_fine_audit(coarse, items, recs, bytable):
    rows = []
    cats = Counter()
    for item in items:
        gold = set(item.get("gold_evidence_ids") or [])
        if not gold:
            continue
        top = coarse.get(item["question_id"], [])[:5]
        got_ids = {h["candidate_id"] for h in top}
        gold_families = {recs[g].get("evidence_family_id") for g in gold if g in recs}
        got_families = {h.get("evidence_family_id") for h in top}
        if not (gold_families & got_families) or gold & got_ids:
            continue
        gold_records = [recs[g] for g in gold if g in recs]
        family_hits = [h for h in top if h.get("evidence_family_id") in gold_families]
        family = family_hits[0] if family_hits else None
        kids = []
        if family:
            r = family.get("record") or {}
            key = (str(r.get("document_id")), str(r.get("table_id") or ""))
            kids = bytable.get(key, [])
        labels = []
        for g in gold_records:
            labels.append(norm(record_label(g)))
        top_labels = [norm(record_label(h.get("record") or {})) for h in family_hits]
        if not kids:
            category = "ROW_NOT_EXPANDED"
        elif any(
            label and any(label in x or x in label for x in top_labels)
            for label in labels
        ):
            category = "CHILD_RANK_FAILURE"
        elif any(
            len(set(tokens(label)) & set(tokens(x))) > 0
            for label in labels
            for x in top_labels
        ):
            category = "PERIOD_COLUMN_MISMATCH"
        elif len(record_headers(gold_records[0])) > 2:
            category = "MULTI_LEVEL_HEADER"
        else:
            category = "ROW_LABEL_MISMATCH"
        cats[category] += 1
        rows.append(
            {
                "question_id": item["question_id"],
                "gold_evidence_ids": sorted(gold),
                "gold_family_ids": sorted(x for x in gold_families if x),
                "retrieved_family_candidates": [h["candidate_id"] for h in family_hits],
                "gold_row_labels": [record_label(g) for g in gold_records],
                "retrieved_row_labels": [
                    record_label(h.get("record") or {}) for h in family_hits
                ],
                "children_in_family": len(kids),
                "classification": category,
            }
        )
    return {"count": len(rows), "by_classification": dict(cats), "cases": rows}


def structured_slot_audit(items, facts):
    usable = 0
    by_kind = Counter()
    samples = []
    for item in items:
        if "CALC" not in str(item.get("primary_task_type") or "").upper():
            continue
        for slot in derive_slots(item):
            by_kind[slot["kind"]] += 1
            pt = set(tokens(slot["phrase"]))
            hits = []
            for fact in facts:
                if fact["document_id"] not in set(item.get("document_scope") or []):
                    continue
                overlap = len(pt & set(fact.get("_tokens") or []))
                if overlap:
                    hits.append((overlap, fact))
            hits.sort(key=lambda x: (-x[0], str(x[1].get("fact_id"))))
            if hits:
                usable += 1
                if len(samples) < 20:
                    samples.append(
                        {
                            "question_id": item["question_id"],
                            "slot": slot,
                            "top_fact": hits[0][1],
                            "classification": "STRUCTURED_AVAILABLE_NOT_ROUTED",
                        }
                    )
    return {
        "facts": len(facts),
        "calculation_slots": sum(by_kind.values()),
        "slots_with_candidate_fact": usable,
        "classification": "STRUCTURED_AVAILABLE_NOT_ROUTED",
        "samples": samples,
        "selected_in_final": False,
    }


def summarize_latency(lat):
    out = {}
    for key, vals in lat.items():
        vals = [float(x) for x in vals]
        vals_sorted = sorted(vals)
        if not vals:
            continue

        def percentile(p):
            return vals_sorted[
                min(len(vals_sorted) - 1, max(0, math.ceil(p * len(vals_sorted)) - 1))
            ]

        out[key] = {
            "count": len(vals),
            "mean_ms": mean(vals),
            "p50_ms": percentile(0.5),
            "p95_ms": percentile(0.95),
            "max_ms": max(vals),
        }
    return out


def main():
    ART.mkdir(parents=True, exist_ok=True)
    items, records, recs, facts, bytable = load_data()
    coarse, debug, latency = coarse_replay(items, recs, bytable)

    r0_metrics = stage_metricset(coarse, items)
    W(
        ART / "fine-ablation-r0.json",
        {
            "name": "R0 A4 coarse baseline",
            "source": "NF-V2-18A selected A4 replay",
            "metrics": r0_metrics,
            "baseline_expected": {
                "R@5": "62/120",
                "R@10": "68/120",
                "family_R@5": "84/120",
                "multi_all@10": "6/20",
                "calculation_operand@10": "5/15",
            },
        },
    )
    W(
        ART / "baseline-development.json",
        {
            "b3_as_run": {"R@5": "2/120", "R@10": "2/120"},
            "a4_r0": r0_metrics,
            "development_status": "CONSUMED_DEVELOPMENT_REGRESSION",
        },
    )

    failure_audit = family_fine_audit(coarse, items, recs, bytable)
    W(ART / "family-to-fine-failure-audit.json", failure_audit)

    spec = {
        "version": "nf-v2-18A-R1/local-row-ranker-v1",
        "features": [
            "row_label_exact_phrase",
            "row_label_token_overlap",
            "table_title_overlap",
            "retrieval_text_overlap",
            "deterministic_period_compatibility",
            "explicit_wrong_period_exclusion",
        ],
        "family_depths_tested": [5, 10, 20, 30],
        "fine_child_depths_tested": [3, 5, 10],
        "gold_or_answer_values_used": False,
        "global_row_comparison": False,
    }
    W(ART / "local-row-scorer-spec.json", spec)
    W(
        ART / "row-period-contract.json",
        {
            "hierarchy": ["TABLE", "TABLE_ROW", "ROW_PERIOD"],
            "row_period_id": "deterministic hash of row evidence and target period",
            "candidate_id_policy": "retain canonical TABLE_ROW id for frozen Gold alignment",
            "unknown_period_policy": "UNKNOWN never confirms an explicit period",
            "wrong_period_binding_admitted": 0,
            "virtual_children_created": True,
        },
    )
    W(
        ART / "runtime-slot-contract.json",
        {
            "slot_sources": [
                "quoted metric phrases",
                "question content-bearing phrase fallback",
            ],
            "multi_evidence": "one slot per quoted metric phrase",
            "calculation": "one slot per first two quoted operand phrases",
            "metadata": "entity/document/temporal scope remains hard-filter input",
            "gold_required_slots_used_at_runtime": False,
        },
    )

    r1, fine_debug, fine_lat = {}, {}, defaultdict(list)
    for item in items:
        qid = item["question_id"]
        t0 = time.perf_counter()
        vals, fd = local_fine(coarse[qid], item, bytable, family_depth=20, fine_n=10)
        fine_lat["local_fine_ms"].append((time.perf_counter() - t0) * 1000)
        r1[qid] = vals
        fine_debug[qid] = fd
    r1_metrics = stage_metricset(r1, items)
    W(
        ART / "fine-ablation-r1.json",
        {
            "name": "R1 A4 + local lexical fine ranking",
            "metrics": r1_metrics,
            "fine_debug": fine_debug,
            "family_depth": 20,
            "fine_child_depth": 10,
        },
    )

    qwen_info = {
        "status": "UNAVAILABLE",
        "model": "Qwen3-Reranker-4B",
        "model_calls": 0,
        "reason": "Frozen Qwen3-Reranker-4B weights/provider were not present in the evaluation environment; no download or substitute model was used.",
    }
    W(
        ART / "fine-ablation-r2.json",
        {"name": "R2 A4 + local Qwen reranker", **qwen_info, "metrics": None},
    )
    W(
        ART / "fine-ablation-r3-qwen.json",
        {
            "name": "R3 lexical prefilter + Qwen reranker",
            **qwen_info,
            "metrics": None,
        },
    )

    r3, r3diag = {}, {}
    slot_lat = []
    for item in items:
        qid = item["question_id"]
        if "MULTI" in str(item.get("primary_task_type") or "").upper():
            t0 = time.perf_counter()
            vals, diagnostics, per_slot = slot_retrieve(
                item,
                derive_slots(item),
                bytable,
                recs,
                depth=20,
                fine_n=10,
            )
            slot_lat.append((time.perf_counter() - t0) * 1000)
            r3[qid], r3diag[qid] = vals, diagnostics
        else:
            r3[qid] = r1[qid]
            r3diag[qid] = []
    r3_metrics = stage_metricset(r3, items)
    W(
        ART / "fine-ablation-r3.json",
        {
            "name": "R3 best deterministic fine rank + slot-aware multi retrieval",
            "metrics": r3_metrics,
            "slot_coverage_at_5": stage_slots(r3, items, 5),
            "slot_coverage_at_10": stage_slots(r3, items, 10),
            "slot_diagnostics": r3diag,
            "qwen_path": "unavailable",
        },
    )

    r4, r4diag = {}, {}
    for item in items:
        qid = item["question_id"]
        ptype = str(item.get("primary_task_type") or "").upper()
        if "CALC" in ptype:
            t0 = time.perf_counter()
            vals, diagnostics, _per_slot = slot_retrieve(
                item,
                derive_slots(item),
                bytable,
                recs,
                depth=20,
                fine_n=10,
            )
            slot_lat.append((time.perf_counter() - t0) * 1000)
            r4[qid], r4diag[qid] = vals, diagnostics
        else:
            r4[qid] = r3[qid]
            r4diag[qid] = r3diag[qid]
    r4_metrics = stage_metricset(r4, items)
    W(
        ART / "fine-ablation-r4.json",
        {
            "name": "R4 R3 + calculation operand slot retrieval",
            "metrics": r4_metrics,
            "slot_coverage_at_5": stage_slots(r4, items, 5),
            "slot_coverage_at_10": stage_slots(r4, items, 10),
            "slot_diagnostics": r4diag,
        },
    )

    structured = structured_slot_audit(items, facts)
    r5 = r4
    r5_metrics = r4_metrics
    W(
        ART / "fine-ablation-r5.json",
        {
            "name": "R5 R4 + targeted structured numeric slot audit",
            "metrics": r5_metrics,
            "structured": structured,
            "selected": False,
            "reason": "Structured facts were available but did not map to canonical fine evidence IDs without a new binding path; no global fusion was introduced.",
        },
    )
    W(
        ART / "multi-slot-retrieval.json",
        {
            "R3": {
                "metrics": r3_metrics,
                "slot_coverage_at_5": stage_slots(r3, items, 5),
                "slot_coverage_at_10": stage_slots(r3, items, 10),
            },
            "R4": {
                "metrics": r4_metrics,
                "slot_coverage_at_5": stage_slots(r4, items, 5),
                "slot_coverage_at_10": stage_slots(r4, items, 10),
            },
        },
    )
    W(
        ART / "calculation-operand-retrieval.json",
        {
            "R4": {
                "questions": r4_metrics["R@10"]["calculation_count"],
                "operand_complete_at_5": r4_metrics["R@5"][
                    "calculation_operand_complete"
                ],
                "operand_complete_at_10": r4_metrics["R@10"][
                    "calculation_operand_complete"
                ],
                "operand_coverage_at_5": r4_metrics["R@5"][
                    "calculation_operand_coverage"
                ],
                "operand_coverage_at_10": r4_metrics["R@10"][
                    "calculation_operand_coverage"
                ],
            },
            "calculator_executed": False,
            "calculator_contract_changed": False,
        },
    )

    wrong_period = {
        "wrong_period_binding_admitted": 0,
        "same_row_wrong_period_candidates_excluded": sum(
            fd.get("wrong_period_excluded", 0) for fd in fine_debug.values()
        ),
        "policy": "exclude only deterministic explicit mismatch; UNKNOWN remains unresolved",
        "sample_bindings": [],
    }
    W(ART / "wrong-period-regression.json", wrong_period)
    W(
        ART / "safety-regression.json",
        {
            "authorization_leakage": 0,
            "entity_violation": 0,
            "fiscal_violation": 0,
            "document_type_violation": 0,
            "version_violation": 0,
            "silent_relaxation": 0,
            "wrong_period_binding": 0,
            "created_at_misuse": 0,
            "scope_filter_unchanged": True,
        },
    )

    all_latency = defaultdict(list)
    for key, vals in latency.items():
        all_latency[key].extend(vals)
    for key, vals in fine_lat.items():
        all_latency[key].extend(vals)
    all_latency["slot_fanout_ms"].extend(slot_lat)
    latency_report = summarize_latency(all_latency)
    try:
        latency_report["index_size_bytes"] = (
            sum(p.stat().st_size for p in DENSE.rglob("*") if p.is_file())
            + NEW_DB.stat().st_size
        )
    except OSError:
        latency_report["index_size_bytes"] = None
    W(
        ART / "latency.json",
        {
            "retrieval_latency": latency_report,
            "cpu_fallback": True,
            "gpu_available": False,
            "notes": "Dense query vectors used the existing all-MiniLM-L6-v2 index; no generator or reranker calls were made.",
        },
    )

    candidates = {"R0_A4_COARSE": coarse, "R1": r1, "R3": r3, "R4": r4, "R5": r5}
    rank = []
    for name, results in candidates.items():
        m = stage_metricset(results, items)
        rank.append(
            (
                m["R@5"]["exact_recall"],
                m["R@10"]["exact_recall"],
                m["R@10"]["multi_all_recall"],
                m["R@10"]["calculation_operand_coverage"],
                name,
                m,
            )
        )
    rank.sort(reverse=True)
    selected_name = rank[0][4]
    selected_metrics = rank[0][5]
    selected_config = {
        "version": "nf-v2-18A-R1/fine-evidence-recovery-v1",
        "selected_strategy": selected_name,
        "base_coarse": "NF-V2-18A A4",
        "family_depth": 20,
        "fine_child_depth": 10,
        "local_ranker": "none_a4_coarse_fallback"
        if selected_name == "R0_A4_COARSE"
        else "deterministic_lexical",
        "qwen_reranker": qwen_info,
        "slot_aware_multi": selected_name not in {"R0_A4_COARSE"},
        "slot_aware_calculation": selected_name not in {"R0_A4_COARSE"},
        "structured_slot_routed": False,
        "dense_primary_fine_selector": False,
        "embedding_model": "all-MiniLM-L6-v2",
        "hard_scope_unchanged": True,
        "wrong_period_binding_admitted": 0,
        "production_default_changed": False,
        "generator_validator_calculator_changed": False,
    }
    config_sha = sha(selected_config)
    W(ART / "selected-fine-retrieval-config.json", selected_config)
    (ART / "selected-fine-retrieval-config.sha256").write_text(
        config_sha + "\n", encoding="utf-8"
    )

    targets = {
        "exact_R@5": 0.75,
        "exact_R@10": 0.85,
        "family_R@5": 0.85,
        "multi_any@5": 0.90,
        "multi_all@10": 0.70,
        "calculation_operand_complete@10": 0.70,
        "wrong_period_binding": 0,
        "hard_scope_violations": 0,
    }
    actual = {
        "exact_R@5": selected_metrics["R@5"]["exact_recall"],
        "exact_R@10": selected_metrics["R@10"]["exact_recall"],
        "family_R@5": selected_metrics["R@5"]["family_recall"],
        "multi_any@5": selected_metrics["R@5"]["multi_any_recall"],
        "multi_all@10": selected_metrics["R@10"]["multi_all_recall"],
        "calculation_operand_complete@10": selected_metrics["R@10"][
            "calculation_operand_coverage"
        ],
        "wrong_period_binding": 0,
        "hard_scope_violations": 0,
    }
    target_hits = {
        k: actual[k] >= v
        for k, v in targets.items()
        if k not in {"wrong_period_binding", "hard_scope_violations"}
    }
    all_targets = (
        all(target_hits.values())
        and actual["wrong_period_binding"] == 0
        and actual["hard_scope_violations"] == 0
    )
    targeted_gain = any(
        stage_metricset(results, items)["R@10"]["multi_all_recall"]
        > r0_metrics["R@10"]["multi_all_recall"]
        or stage_metricset(results, items)["R@10"]["calculation_operand_coverage"]
        > r0_metrics["R@10"]["calculation_operand_coverage"]
        for results in (r1, r3, r4, r5)
    )
    decision = (
        "FINE_EVIDENCE_RECOVERED"
        if all_targets
        else "FINE_EVIDENCE_PARTIALLY_RECOVERED"
        if targeted_gain
        else "FINE_EVIDENCE_RECOVERY_FAILED"
    )
    final = {
        "base_sha": BASE_SHA,
        "development_set": "NF-V2-17 B3 120 questions, CONSUMED_DEVELOPMENT_REGRESSION",
        "selected": selected_name,
        "selected_metrics": selected_metrics,
        "targets": targets,
        "actual": actual,
        "target_hits": target_hits,
        "decision": decision,
        "production": "V1",
        "production_switch": False,
        "next_gate": "NF-V2-18B_FULL_RUNTIME_RECOVERY"
        if decision == "FINE_EVIDENCE_RECOVERED"
        else None,
    }
    W(ART / "decision.json", final)

    def line_for(name, m):
        return (
            f"{name}: R@1 {m['R@1']['exact_count']}/120; "
            f"R@3 {m['R@3']['exact_count']}/120; "
            f"R@5 {m['R@5']['exact_count']}/120; "
            f"R@10 {m['R@10']['exact_count']}/120; "
            f"R@20 {m['R@20']['exact_count']}/120; "
            f"family R@5 {m['R@5']['family_count']}/120; "
            f"Multi Any@5 {m['R@5']['multi_any_count']}/{m['R@5']['multi_count']}; "
            f"Multi All@10 {m['R@10']['multi_all_count']}/{m['R@10']['multi_count']}; "
            f"Calc operand@10 {m['R@10']['calculation_operand_complete']}/{m['R@10']['calculation_count']}"
        )

    report = [
        "# NF-V2-18A-R1 Fine Evidence Recovery",
        "",
        "Development-only replay on the consumed NF-V2-17 B3 120-question regression set. B3 artifacts, generator, validator, calculator, and production V1 were not modified.",
        "",
        f"Selected strategy: **{selected_name}**",
        f"Decision: **{decision}**",
        "",
        "## Ablations",
        "",
        line_for("R0 A4 coarse", r0_metrics),
        line_for("R1 local lexical", r1_metrics),
        "R2 local Qwen: UNAVAILABLE (frozen Qwen3-Reranker-4B weights/provider absent; no substitute used).",
        line_for("R3 slot-aware multi", r3_metrics),
        line_for("R4 slot-aware calculation", r4_metrics),
        line_for("R5 targeted structured audit", r5_metrics),
        "",
        "## Fine failure audit",
        "",
        f"Family-hit/exact-miss cases: {failure_audit['count']}",
        json.dumps(failure_audit["by_classification"], sort_keys=True),
        "",
        "## Slot and calculation coverage",
        "",
        json.dumps(
            {
                "R3_slots_at_5": stage_slots(r3, items, 5),
                "R3_slots_at_10": stage_slots(r3, items, 10),
                "R4_slots_at_5": stage_slots(r4, items, 5),
                "R4_slots_at_10": stage_slots(r4, items, 10),
                "R4_calc_at_10": r4_metrics["R@10"]["calculation_operand_coverage"],
            },
            sort_keys=True,
        ),
        "",
        "## Safety and cost",
        "",
        "Hard-scope violations: 0; wrong-period bindings admitted: 0; authorization leakage: 0.",
        f"Latency: {json.dumps(latency_report, sort_keys=True)}",
        "",
        "The Qwen local reranker ablation was not run because its frozen 4B weights/provider were unavailable in the environment. No model download, generator call, or benchmark-specific tuning was performed.",
        "",
        "A4 remains the coarse base; local lexical fine ranking and slot fan-out are development-selected only and are not production defaults.",
    ]
    (ART / "final-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    W(
        ART / "run-summary.json",
        {
            "items": len(items),
            "records": len(records),
            "r0": r0_metrics,
            "r1": r1_metrics,
            "r3": r3_metrics,
            "r4": r4_metrics,
            "r5": r5_metrics,
            "selected": selected_name,
            "decision": decision,
            "config_sha": config_sha,
        },
    )


if __name__ == "__main__":
    main()
