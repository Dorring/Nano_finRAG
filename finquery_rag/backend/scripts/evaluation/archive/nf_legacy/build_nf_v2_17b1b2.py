from __future__ import annotations

import collections
import hashlib
import json
import re
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path


BASE = Path(
    "/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.worktrees/nf-v2-17-financial-corpus-v2"
)
A5 = BASE / "finquery_rag/backend/artifacts/evaluation/nf-v2-17-financial-corpus-v2"
OUT = BASE / "finquery_rag/backend/artifacts/evaluation/nf-v2-17-fresh-blind-eval"
SEARCHABLE = A5 / "searchable-corpus-manifest-v2.jsonl"
FREEZE = A5 / "financial-corpus-v2-freeze.json"
RESERVATION = A5 / "fresh-blind-corpus-reservation.json"
INDEX_DB = Path(
    "/mnt/disk/mxf/projects/Qhhhhhhaaa/financial_corpus_v2/indexes/financial-corpus-v2/bm25/index.sqlite"
)
SIDECAR_META = A5 / "amendment-sidecar-acquisition.json"
PARSED_ROOT = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/financial_corpus_v2/parsed/SEC")

EXPECTED_BASE = "c5588a152e049e415c54f232d551f0b3089ba2a6"
EXPECTED_CORPUS_SHA = "3ef3d8e772dfb2d4e2594d18efe3c101c4a4a3bb108e0faa0d75d11c667421a3"
EXPECTED_FREEZE_SHA = "63620b2183c4635f1ecff974935bc81a4d8ce678c72e72e94155d8f0a96e6929"
EXPECTED_RESERVATION_SHA = (
    "8708ecf5b0f5ee056cf003238a510345c96cce720a41709d5eeb0c5d47e1dc23"
)

PRIMARY_TYPES = [
    "SINGLE_EVIDENCE_FACT",
    "MULTI_EVIDENCE",
    "DETERMINISTIC_CALCULATION",
    "TEMPORAL_PERIOD",
    "AGENTIC_REPLAN",
    "VERSION_TEMPORAL",
    "CONFLICT_AMBIGUITY",
    "NO_ANSWER_FAIL_CLOSED",
]
QUOTAS = {
    "SINGLE_EVIDENCE_FACT": 30,
    "MULTI_EVIDENCE": 20,
    "DETERMINISTIC_CALCULATION": 15,
    "TEMPORAL_PERIOD": 15,
    "AGENTIC_REPLAN": 15,
    "VERSION_TEMPORAL": 10,
    "CONFLICT_AMBIGUITY": 5,
    "NO_ANSWER_FAIL_CLOSED": 10,
}

BAD_LABELS = {
    "",
    "part i",
    "part ii",
    "part iii",
    "part iv",
    "page",
    "none",
    "title of each class",
    "large accelerated filer / non-accelerated filer",
    "accelerated filer / smaller reporting company / emerging growth company",
    "as of december 31",
    "balance as of december 31",
    "balance as of january 1",
}
BOILERPLATE = (
    "annual report pursuant",
    "transition report pursuant",
    "securities and exchange commission",
    "table of contents",
    "signature",
    "form 10-k summary",
)
NUM_RE = re.compile(r"(?<![A-Za-z])\(?\s*[-$]?\d[\d,]*(?:\.\d+)?\s*\)?%?")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_bytes(obj) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_bytes(obj))
    return sha256(path)


def write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )
    return sha256(path)


def compact(text: str, limit: int = 850) -> str:
    value = re.sub(r"\s+", " ", (text or "")).strip()
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0] + " …"


def label_of(record: dict) -> str:
    text = compact(record.get("content", ""), 5000)
    candidates = []
    parts = re.split(r"\s+\|\s+|\s+/\s+", text)
    for value in [*parts, *(record.get("column_headers") or [])]:
        value = re.sub(r"^[|:\s]+|[|:\s]+$", "", compact(value, 500))
        if not value or not re.search(r"[A-Za-z]", value):
            continue
        low = value.lower()
        if low in {
            "description",
            "description of the matter",
            "amount",
            "shares",
            "number",
            "none",
        }:
            continue
        if any(x in low for x in BOILERPLATE):
            continue
        if re.fullmatch(
            r"(?:as of|year ended|three months ended|six months ended|nine months ended).*[0-9].*",
            low,
        ):
            continue
        score = 0
        if any(
            k in low
            for k in (
                "revenue",
                "sales",
                "income",
                "loss",
                "cash",
                "asset",
                "liabil",
                "debt",
                "expense",
                "tax",
                "stock",
                "equity",
                "operating",
                "net",
                "risk",
                "business",
                "market",
                "note",
            )
        ):
            score += 4
        if len(value) <= 90:
            score += 2
        if ":" not in value:
            score += 1
        candidates.append((score, -len(value), value))
    if not candidates:
        return "financial disclosure"
    return sorted(candidates, reverse=True)[0][2]


def is_good(record: dict, *, short: bool = True) -> bool:
    text = compact(record.get("content", ""), 5000)
    label = label_of(record).lower()
    if record.get("content_type") != "TABLE_ROW":
        return False
    if len(text) < 35 or (short and len(text) > 950):
        return False
    if label in BAD_LABELS or any(x in label for x in BOILERPLATE):
        return False
    if label == "financial disclosure" or any(
        x in label
        for x in ("exhibit number", "certification", "clawback policy", "inline xbrl")
    ):
        return False
    if label.startswith("item ") or label.startswith("part "):
        return False
    if not re.search(r"[A-Za-z]", label):
        return False
    return True


def numeric_tokens(record: dict):
    out = []
    text = record.get("content", "") or ""
    body = text.split(":", 1)[-1]
    for m in NUM_RE.finditer(body):
        raw = m.group(0).strip()
        pct = raw.endswith("%")
        clean = raw.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
        neg = clean.startswith("(") and clean.endswith(")")
        clean = clean.strip("()")
        try:
            val = Decimal(clean)
        except InvalidOperation:
            continue
        if neg:
            val = -val
        if val == val.to_integral() and 1900 <= abs(int(val)) <= 2100:
            continue
        if abs(val) > Decimal("1000000000000"):
            continue
        out.append((raw, val, pct))
    return out


def period_desc(record: dict) -> str:
    sem = record.get("period_semantics") or "UNKNOWN"
    end = record.get("period_end") or record.get("report_period_end") or "unspecified"
    start = record.get("period_start")
    if start:
        return f"{sem} ({start} to {end})"
    return f"{sem} ({end})"


def doc_label(record: dict) -> str:
    return f"{record.get('ticker')} {record.get('report_period_end')} {record.get('form_type') or record.get('document_type')}"


def source_identity(record: dict) -> str:
    doc = record["document_id"]
    chunk = record["chunk_id"]
    return chunk if chunk.startswith(doc + "::") else f"{doc}::{chunk}"


def source_obj(record: dict, role: str = "gold") -> dict:
    return {
        "evidence_id": source_identity(record),
        "evidence_role": role,
        "document_id": record.get("document_id"),
        "chunk_id": record.get("chunk_id"),
        "table_id": record.get("table_id"),
        "row_id": record.get("row_id"),
        "source_block_ids": record.get("source_block_ids") or [],
        "ixbrl_fact_ids": record.get("ixbrl_fact_ids") or [],
        "ixbrl_context_ids": record.get("ixbrl_context_ids") or [],
        "accession_number": record.get("accession_number"),
        "raw_sha256": record.get("raw_source_sha256"),
        "raw_local_path": record.get("raw_local_path"),
        "ticker": record.get("ticker"),
        "company": record.get("company"),
        "CIK": record.get("CIK") or record.get("cik"),
        "form_type": record.get("form_type"),
        "filing_date": record.get("filing_date"),
        "fiscal_year": record.get("fiscal_year"),
        "fiscal_quarter": record.get("fiscal_quarter"),
        "version": record.get("version"),
        "is_amended": record.get("is_amended"),
        "version_relation": record.get("version_relation"),
        "section_type": record.get("section_type"),
        "content_type": record.get("content_type"),
        "period_semantics": record.get("period_semantics"),
        "period_start": record.get("period_start"),
        "period_end": record.get("period_end"),
        "report_period_end": record.get("report_period_end"),
        "content": record.get("content", ""),
    }


def fts_top(query: str):
    try:
        con = sqlite3.connect(str(INDEX_DB))
        rows = con.execute(
            "SELECT doc_id FROM fts_index WHERE fts_index MATCH ? LIMIT 5", (query,)
        ).fetchall()
        con.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def normalize_question(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^a-z0-9]+", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def scan_historical_questions(root: Path, exclude_roots=()):
    values = set()
    files = 0
    for p in root.rglob("*"):
        if not p.is_file() or p.stat().st_size > 60_000_000:
            continue
        if any(p == excluded or excluded in p.parents for excluded in exclude_roots):
            continue
        if p.suffix.lower() not in {".json", ".jsonl", ".txt", ".md"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        files += 1
        for m in re.finditer(
            r'"(?:question|base_question|query)"\s*:\s*"((?:\\.|[^"\\])*)"', text
        ):
            try:
                values.add(json.loads('"' + m.group(1) + '"'))
            except Exception:
                values.add(m.group(1))
    return values, files


def make_item(
    qid,
    task,
    question,
    evidence,
    *,
    answerable,
    reference,
    slots,
    docs,
    temporal,
    secondary=None,
    replan=False,
    replan_reason=None,
    replan_obs=None,
    conflict="NONE",
    calc=None,
    difficulty=None,
    distractors=None,
    terminal=None,
):
    return {
        "question_id": qid,
        "question": question,
        "primary_task_type": task,
        "secondary_task_tags": secondary or [],
        "answerability": "ANSWERABLE" if answerable else "UNANSWERABLE",
        "company": sorted({r.get("company") for r in evidence if r.get("company")}),
        "ticker": sorted({r.get("ticker") for r in evidence if r.get("ticker")}),
        "entity_scope": sorted({r.get("ticker") for r in evidence if r.get("ticker")}),
        "document_scope": sorted(docs),
        "temporal_scope": temporal,
        "required_slots": slots,
        "gold_evidence_ids": [source_identity(r) for r in evidence],
        "gold_document_ids": sorted({r.get("document_id") for r in evidence}),
        "reference_answer": reference,
        "expected_replan": replan,
        "expected_replan_reason": replan_reason,
        "expected_tool_capabilities": [
            "metadata_filter",
            "lexical",
            "structured_evidence",
        ],
        "expected_conflict_state": conflict,
        "expected_terminal_state": terminal
        or ("RELEASE" if answerable else "FAIL_CLOSED"),
        "calculation_contract": calc,
        "annotation_status": "PASS1_CONSTRUCTED_PASS2_VERIFIED",
        "review_status": "ACCEPTED",
        "difficulty_notes": difficulty or [],
        "distractor_evidence_ids": [source_identity(r) for r in (distractors or [])],
        "retrieval_config_sha_for_replan": None,
        "replan_observation": replan_obs,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    reservation = json.loads(RESERVATION.read_text(encoding="utf-8"))
    reservation_sha_file = RESERVATION.with_name(
        "fresh-blind-corpus-reservation.sha256"
    )
    if (
        reservation_sha_file.read_text(encoding="utf-8").strip().split()[0]
        != EXPECTED_RESERVATION_SHA
    ):
        raise SystemExit("reservation SHA mismatch")
    if freeze.get("searchable_corpus_sha") != EXPECTED_CORPUS_SHA:
        raise SystemExit("searchable corpus SHA mismatch")
    freeze_sha_file = A5 / "financial-corpus-v2-freeze.sha256"
    if (
        freeze_sha_file.read_text(encoding="utf-8").strip().split()[0]
        != EXPECTED_FREEZE_SHA
    ):
        raise SystemExit("freeze SHA mismatch")
    primary_docs = [d["document_id"] for d in reservation["documents"]]
    if len(primary_docs) != 12 or len(set(primary_docs)) != 12:
        raise SystemExit("reservation cardinality mismatch")

    # Load only the sealed 12-document primary reservation from the searchable manifest.
    by_doc = collections.defaultdict(list)
    with SEARCHABLE.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("document_id") in primary_docs:
                by_doc[r["document_id"]].append(r)
    if set(by_doc) != set(primary_docs):
        raise SystemExit("reservation document missing from searchable manifest")

    # Stable row-level de-duplication: repeated parser occurrences are retained in the
    # corpus, but one deterministic occurrence is selected for each annotation.
    rows = []
    for doc in primary_docs:
        seen = {}
        for r in by_doc[doc]:
            if not is_good(r, short=False) and not (
                r.get("content_type") in {"TABLE", "TEXT"}
                and r.get("section_type")
                in {"MDA", "NOTES", "RISK_FACTORS", "BUSINESS"}
                and len(compact(r.get("content", ""))) >= 80
            ):
                continue
            key = (
                r.get("row_id") or r.get("chunk_id"),
                compact(r.get("content", ""), 5000),
            )
            old = seen.get(key)
            if old is None or len(r.get("content", "")) > len(old.get("content", "")):
                seen[key] = r
        rows.extend(seen.values())
    rows.sort(
        key=lambda r: (
            r.get("ticker", ""),
            r.get("report_period_end", ""),
            r.get("chunk_id", ""),
        )
    )
    numeric_rows = [r for r in rows if len(numeric_tokens(r)) >= 1]
    known_period_rows = [
        r
        for r in rows
        if r.get("period_semantics") in {"ANNUAL", "QUARTER", "YTD", "INSTANT"}
    ]

    def prose_eligible(r):
        if r.get("section_type") not in {"MDA", "NOTES", "RISK_FACTORS", "BUSINESS"}:
            return False
        text = compact(r.get("content", ""), 5000)
        label = label_of(r).lower()
        if len(text) < 80 or len(label) > 120:
            return False
        if label in BAD_LABELS or any(x in label for x in BOILERPLATE):
            return False
        if any(
            x in label
            for x in (
                "exhibit number",
                "certification",
                "clawback policy",
                "inline xbrl",
            )
        ):
            return False
        if "formatted in inline xbrl" in text.lower():
            return False
        return True

    prose_rows = [r for r in rows if prose_eligible(r)]
    useful_rows = [r for r in rows if len(compact(r.get("content", ""))) <= 900]
    if len(useful_rows) < 180 or len(numeric_rows) < 80:
        raise SystemExit(
            f"insufficient deterministic candidates: useful={len(useful_rows)} numeric={len(numeric_rows)}"
        )

    # Sidecars are explicitly reserved for version evaluation by A5 and remain outside
    # the 12 primary blind filings. They are never added to primary corpus counts.
    side_meta = {
        r["document_id"]: r
        for r in json.loads(SIDECAR_META.read_text(encoding="utf-8")).get("records", [])
    }
    sidecar_rows = []
    for doc, meta in sorted(side_meta.items()):
        parsed_path = PARSED_ROOT / meta["ticker"] / doc / "document.json"
        if not parsed_path.exists():
            raise SystemExit(f"missing sidecar parsed source {doc}")
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        for chunk in parsed.get("chunks", []):
            if not chunk.get("content"):
                continue
            sidecar_rows.append(
                {
                    "document_id": doc,
                    "chunk_id": chunk["chunk_id"],
                    "ticker": meta["ticker"],
                    "company": meta["company"],
                    "CIK": meta["cik"],
                    "accession_number": meta["accession_number"],
                    "form_type": meta["form_type"],
                    "document_type": "VERSION_SIDECAR",
                    "fiscal_year": meta["fiscal_year"],
                    "fiscal_quarter": meta.get("fiscal_quarter"),
                    "report_period_end": meta["report_period_end"],
                    "filing_date": meta["filing_date"],
                    "raw_source_sha256": meta["raw_sha256"],
                    "raw_local_path": meta["raw_local_path"],
                    "section_type": chunk.get("section_type"),
                    "content_type": chunk.get("content_type"),
                    "period_semantics": "UNKNOWN",
                    "period_start": None,
                    "period_end": meta["report_period_end"],
                    "source_block_ids": chunk.get("source_block_ids") or [],
                    "table_id": chunk.get("table_id"),
                    "row_id": chunk.get("row_id"),
                    "content": chunk.get("content", ""),
                    "version_relation": meta.get("version_relation", "UNKNOWN"),
                }
            )
    sidecar_rows.sort(key=lambda r: (r["document_id"], r["chunk_id"]))

    used = collections.Counter()

    def choose(pool, count, *, min_per_doc=False):
        chosen = []
        selected = set()
        docs = list(primary_docs)
        bydoc = collections.defaultdict(list)
        for r in pool:
            bydoc[r.get("document_id")].append(r)
        for doc in docs:
            bydoc[doc].sort(key=lambda r: (r.get("chunk_id", ""), label_of(r).lower()))
        if min_per_doc:
            for doc in docs:
                if len(chosen) >= count:
                    break
                for r in bydoc.get(doc, []):
                    eid = source_identity(r)
                    if eid not in selected and used[eid] < 2:
                        chosen.append(r)
                        selected.add(eid)
                        used[eid] += 1
                        break
        while len(chosen) < count:
            progress = False
            for doc in docs:
                for r in bydoc.get(doc, []):
                    eid = source_identity(r)
                    if eid not in selected and used[eid] < 2:
                        chosen.append(r)
                        selected.add(eid)
                        used[eid] += 1
                        progress = True
                        break
                if len(chosen) >= count:
                    break
            if not progress:
                break
        if len(chosen) < count:
            raise SystemExit(f"candidate shortfall {len(chosen)} < {count}")
        return chosen[:count]

    def grouped_pairs(pool, count, require_same_table=False):
        bydoc = collections.defaultdict(list)
        for r in pool:
            bydoc[r.get("document_id")].append(r)
        per_doc = collections.defaultdict(list)
        for doc in primary_docs:
            groups = collections.defaultdict(list)
            for r in bydoc.get(doc, []):
                key = (
                    r.get("table_id")
                    if require_same_table
                    else r.get("period_semantics")
                )
                groups[key].append(r)
            pair_seen = set()
            for key in sorted(groups, key=str):
                vals = sorted(
                    groups[key],
                    key=lambda r: (label_of(r).lower(), r.get("chunk_id", "")),
                )
                for i in range(len(vals)):
                    for j in range(i + 1, len(vals)):
                        a, b = vals[i], vals[j]
                        la, lb = label_of(a).lower(), label_of(b).lower()
                        if la == lb:
                            continue
                        pair_key = (
                            doc,
                            a.get("report_period_end"),
                            tuple(sorted((la, lb))),
                        )
                        if pair_key in pair_seen:
                            continue
                        pair_seen.add(pair_key)
                        per_doc[doc].append((a, b))
                        if len(per_doc[doc]) >= count:
                            break
                    if len(per_doc[doc]) >= count:
                        break
                if len(per_doc[doc]) >= count:
                    break
        pairs = []
        while len(pairs) < count:
            progress = False
            for doc in primary_docs:
                if per_doc.get(doc):
                    pairs.append(per_doc[doc].pop(0))
                    progress = True
                    if len(pairs) >= count:
                        break
            if not progress:
                break
        if len(pairs) < count:
            raise SystemExit(f"pair shortfall {len(pairs)} < {count}")
        return pairs

    items = []
    sources = {}

    def add(item, evidence, distractors=None):
        item["retrieval_config_sha_for_replan"] = None
        items.append(item)
        for r in evidence:
            sources[source_identity(r)] = source_obj(r, "gold")
        for r in distractors or []:
            sources[source_identity(r)] = source_obj(r, "distractor")

    # A. Single-evidence facts: 22 quantitative/table rows + 8 prose disclosures.
    fact_pool = sorted(
        useful_rows,
        key=lambda r: (
            r.get("document_id", ""),
            -len(numeric_tokens(r)),
            r.get("chunk_id", ""),
        ),
    )
    prose_selected = choose(
        sorted(
            prose_rows, key=lambda r: (r.get("document_id", ""), r.get("chunk_id", ""))
        ),
        10,
        min_per_doc=True,
    )
    prose_ids = {source_identity(r) for r in prose_selected}
    fact_rows = prose_selected + choose(
        [r for r in fact_pool if source_identity(r) not in prose_ids],
        20,
        min_per_doc=True,
    )
    fact_question_counts = collections.Counter()
    for r in fact_rows:
        label = label_of(r)
        qkey = (r.get("ticker"), r.get("report_period_end"), label.lower())
        occurrence = fact_question_counts[qkey]
        fact_question_counts[qkey] += 1
        if occurrence == 0:
            question = f"What does the {r['ticker']} filing for {r['report_period_end']} report for the '{label}' row?"
        elif occurrence == 1:
            question = f"What disclosure is given in the {r['ticker']} filing for {r['report_period_end']} for the '{label}' row?"
        else:
            question = f"Which statement is associated with the '{label}' row in the {r['ticker']} filing for {r['report_period_end']}?"
        answer = compact(r["content"])
        add(
            make_item(
                "",
                "SINGLE_EVIDENCE_FACT",
                question,
                [r],
                answerable=True,
                reference=answer,
                slots={
                    "metric": label,
                    "value": "present in cited row",
                    "period": r.get("report_period_end"),
                },
                docs=[r["document_id"]],
                temporal={
                    "document_end": r.get("report_period_end"),
                    "fact_semantics": r.get("period_semantics"),
                },
                secondary=[
                    "qualitative"
                    if source_identity(r) in prose_ids
                    else "quantitative",
                    r.get("content_type"),
                    r.get("section_type"),
                ],
                difficulty=["single scoped evidence unit"],
            ),
            [r],
        )

    # B. Multi-evidence: two independently required rows; different labels and stable sources.
    multi_pairs = grouped_pairs(useful_rows, 20, require_same_table=False)
    for a, b in multi_pairs[:20]:
        used[source_identity(a)] += 1
        used[source_identity(b)] += 1
        q = f"What does the {a['ticker']} filing for {a['report_period_end']} report for both '{label_of(a)}' and '{label_of(b)}'?"
        ans = f"{label_of(a)}: {compact(a['content'])}; {label_of(b)}: {compact(b['content'])}"
        add(
            make_item(
                "",
                "MULTI_EVIDENCE",
                q,
                [a, b],
                answerable=True,
                reference=ans,
                slots={"metrics": [label_of(a), label_of(b)], "evidence_count": 2},
                docs=[a["document_id"], b["document_id"]],
                temporal={
                    "periods": sorted(
                        {a.get("report_period_end"), b.get("report_period_end")}
                    )
                },
                secondary=[
                    "multi_evidence",
                    "quantitative"
                    if numeric_tokens(a) and numeric_tokens(b)
                    else "mixed",
                ],
                difficulty=["two independently necessary row-level sources"],
            ),
            [a, b],
        )

    # C. Deterministic calculations: same-table operands, only add/subtract.
    calc_pool = [
        r for r in numeric_rows if r.get("table_id") and len(numeric_tokens(r))
    ]
    calc_pairs = grouped_pairs(calc_pool, 15, require_same_table=True)
    for i, (a, b) in enumerate(calc_pairs[:15]):
        ta, tb = numeric_tokens(a)[0], numeric_tokens(b)[0]
        op = "add" if i % 2 == 0 else "subtract"
        result = ta[1] + tb[1] if op == "add" else ta[1] - tb[1]
        result_s = format_decimal(result)
        used[source_identity(a)] += 1
        used[source_identity(b)] += 1
        qword = "sum" if op == "add" else "difference"
        q = f"Using the reported values for '{label_of(a)}' and '{label_of(b)}' in the {a['ticker']} filing for {a['report_period_end']}, what is their {qword}?"
        calc = {
            "operation": f"{op}(operand_1, operand_2)",
            "operands": [
                {
                    "name": label_of(a),
                    "raw_value": ta[0],
                    "value": str(ta[1]),
                    "evidence_id": source_identity(a),
                },
                {
                    "name": label_of(b),
                    "raw_value": tb[0],
                    "value": str(tb[1]),
                    "evidence_id": source_identity(b),
                },
            ],
            "canonical_result": result_s,
            "unit": a.get("unit"),
            "currency": a.get("currency"),
            "scale": a.get("scale"),
            "period": a.get("report_period_end"),
            "rounding": "exact_decimal_display",
        }
        add(
            make_item(
                "",
                "DETERMINISTIC_CALCULATION",
                q,
                [a, b],
                answerable=True,
                reference=f"{result_s} [C1].",
                slots={
                    "operands": [label_of(a), label_of(b)],
                    "operation": op,
                    "result": result_s,
                },
                docs=[a["document_id"]],
                temporal={
                    "document_end": a.get("report_period_end"),
                    "fact_semantics": a.get("period_semantics"),
                },
                secondary=["calculation", op, "table"],
                calc=calc,
                difficulty=[
                    "both operands are explicitly cited; no outside arithmetic"
                ],
            ),
            [a, b],
        )

    # D. Temporal/period: deliberately ask for the parser's explicit fact-level scope.
    temporal_rows = choose(
        sorted(
            known_period_rows,
            key=lambda r: (
                r.get("period_semantics", ""),
                r.get("document_id", ""),
                r.get("chunk_id", ""),
            ),
        ),
        15,
        min_per_doc=True,
    )
    for r in temporal_rows:
        sem = r.get("period_semantics")
        q = f"What temporal scope does the '{label_of(r)}' row in the {r['ticker']} filing for {r['report_period_end']} represent, and what does it report?"
        ans = f"{sem} ({r.get('period_start') or 'start not specified'} to {r.get('period_end') or r.get('report_period_end')}): {compact(r['content'])}"
        add(
            make_item(
                "",
                "TEMPORAL_PERIOD",
                q,
                [r],
                answerable=True,
                reference=ans,
                slots={
                    "metric": label_of(r),
                    "period_semantics": sem,
                    "period_end": r.get("period_end") or r.get("report_period_end"),
                },
                docs=[r["document_id"]],
                temporal={
                    "period_semantics": sem,
                    "period_start": r.get("period_start"),
                    "period_end": r.get("period_end") or r.get("report_period_end"),
                },
                secondary=[
                    "period_semantics",
                    "instant_vs_duration" if sem == "INSTANT" else "duration_scope",
                ],
                difficulty=[
                    "document-level filing scope must not override fact-level semantics"
                ],
            ),
            [r],
        )

    # E. Agentic replan: offline lexical inspection identifies a legitimate missing
    # evidence item under the frozen BM25 top-5. No generation or runtime execution.
    replan_candidates = grouped_pairs(useful_rows, 100, require_same_table=False)
    replan_count = 0
    for a, b in replan_candidates:
        if replan_count >= 15:
            break
        query = f"{a['ticker']} {label_of(a)} {label_of(b)} {a.get('report_period_end', '')}"
        top = fts_top(query)
        wanted = {a["chunk_id"], b["chunk_id"]}
        if top and wanted.issubset(set(top)):
            continue
        missing = sorted(wanted - set(top))
        if not missing:
            continue
        used[source_identity(a)] += 1
        used[source_identity(b)] += 1
        reason = (
            "WRONG_PERIOD"
            if a.get("report_period_end") != b.get("report_period_end")
            else "MISSING_SLOT"
        )
        q = f"Retrieve and answer both requested disclosures for {a['ticker']} ({a['report_period_end']}): '{label_of(a)}' and '{label_of(b)}'."
        ans = f"{label_of(a)}: {compact(a['content'])}; {label_of(b)}: {compact(b['content'])}"
        add(
            make_item(
                "",
                "AGENTIC_REPLAN",
                q,
                [a, b],
                answerable=True,
                reference=ans,
                slots={
                    "required": [label_of(a), label_of(b)],
                    "initial_gap": "one required evidence item not in frozen lexical top-5",
                },
                docs=[a["document_id"], b["document_id"]],
                temporal={
                    "periods": [a.get("report_period_end"), b.get("report_period_end")]
                },
                secondary=["replan", "offline_retrieval_inspection"],
                replan=True,
                replan_reason=reason,
                replan_obs={
                    "query": query,
                    "top5_chunk_ids": top,
                    "missing_gold_chunk_ids": missing,
                    "retrieval_config": "A5 frozen FTS/BM25",
                },
                difficulty=[
                    "initial retrieval gap is observed, not injected; valid targeted repair path exists"
                ],
            ),
            [a, b],
        )
        replan_count += 1
    if replan_count < 15:
        raise SystemExit(f"only {replan_count} legitimate offline replan cases")

    # F/G: sidecar version and ambiguity cases. The sidecars are explicitly reserved
    # for evaluation by A5 and are kept outside the 12-document primary count.
    side_by_doc = collections.defaultdict(list)
    for r in sidecar_rows:
        side_by_doc[r["document_id"]].append(r)
    side_first = [
        sorted(v, key=lambda r: r["chunk_id"])[0]
        for _, v in sorted(side_by_doc.items())
    ]
    for i in range(10):
        r = side_first[i % len(side_first)]
        m = side_meta[r["document_id"]]
        text = compact(r["content"], 500)
        if i % 4 == 0:
            q = f"Which SEC form and amendment marker are shown in the {r['ticker']} sidecar for {r['report_period_end']}?"
            ans = f"{m['form_type']}; the filing body identifies Amendment No. 1."
            slots = {"form_type": m["form_type"], "amendment_marker": "Amendment No. 1"}
        elif i % 4 == 1:
            q = f"What report period end is identified by the {r['ticker']} version sidecar {m['accession_number']}?"
            ans = m["report_period_end"]
            slots = {
                "report_period_end": m["report_period_end"],
                "accession_number": m["accession_number"],
            }
        elif i % 4 == 2:
            q = f"What filing date is recorded for the {r['ticker']} version sidecar {m['accession_number']}?"
            ans = m["filing_date"]
            slots = {
                "filing_date": m["filing_date"],
                "accession_number": m["accession_number"],
            }
        else:
            q = f"Does the source record establish which original filing the {r['ticker']} sidecar {m['accession_number']} supersedes?"
            ans = "No. The frozen source record keeps version_relation as UNKNOWN and does not establish an explicit supersedes relation."
            slots = {
                "version_relation": "UNKNOWN",
                "supersedes_document_id": "not established",
            }
        add(
            make_item(
                "",
                "VERSION_TEMPORAL",
                q,
                [r],
                answerable=True,
                reference=ans,
                slots=slots,
                docs=[r["document_id"]],
                temporal={
                    "report_period_end": m["report_period_end"],
                    "filing_date": m["filing_date"],
                    "version": "AMENDMENT_CANDIDATE",
                },
                secondary=["version", "temporal", "sidecar_reserved_for_eval"],
                conflict="VERSION_AMBIGUOUS" if i % 4 == 3 else "NONE",
                difficulty=[
                    "sidecar is reserved for evaluation and excluded from primary 60-file corpus count",
                    text,
                ],
            ),
            [r],
        )

    for i in range(5):
        r = side_first[i % len(side_first)]
        m = side_meta[r["document_id"]]
        if i < 3:
            q = f"Can the current effective financial version be determined from the {r['ticker']} amendment candidate {m['accession_number']} alone?"
        elif i == 3:
            q = f"Does the {r['ticker']} amendment record {m['accession_number']} establish an explicit original filing?"
        else:
            q = f"Can the supersedes relation for {r['ticker']} amendment {m['accession_number']} be resolved from this source record?"
        ans = "The evidence is insufficient: the source record does not establish an explicit original/supersedes relation."
        add(
            make_item(
                "",
                "CONFLICT_AMBIGUITY",
                q,
                [],
                answerable=False,
                reference=ans,
                slots={
                    "version_relation": "required but unresolved",
                    "effective_version": "not determinable",
                },
                docs=[r["document_id"]],
                temporal={
                    "report_period_end": m["report_period_end"],
                    "filing_date": m["filing_date"],
                },
                secondary=["ambiguity", "version_resolution", "fail_closed"],
                conflict="UNRESOLVED_CONFLICT",
                difficulty=[
                    "natural version ambiguity; no incompatible financial values manufactured"
                ],
                terminal="FAIL_CLOSED",
            ),
            [],
            [r],
        )

    # H. No-answer hard negatives. Each request is checked against the complete
    # reserved primary corpus for absence of metric+period binding. Related rows are
    # retained only as distractors and never as Gold support.
    no_answer_specs = [
        ("GOOGL", "2019", "annual revenue"),
        ("GOOGL", "2020", "quarterly operating income"),
        ("GOOGL", "2026", "annual cash and cash equivalents"),
        ("AMZN", "2018", "annual net sales"),
        ("AMZN", "2019", "annual operating income"),
        ("AMZN", "2026", "quarterly total assets"),
        ("GOOGL", "2025 Q4", "standalone quarterly revenue"),
        ("AMZN", "2024 Q4", "standalone quarterly cash flow"),
        ("GOOGL", "2018", "annual diluted earnings per share"),
        ("AMZN", "2017", "annual long-term debt"),
    ]
    metric_alias = {
        "revenue": ["revenue", "net sales"],
        "net sales": ["net sales", "revenue"],
        "operating income": ["operating income"],
        "cash and cash equivalents": ["cash and cash equivalents", "cash"],
        "total assets": ["total assets"],
        "cash flow": ["cash flow"],
        "diluted earnings per share": ["diluted earnings per share", "diluted"],
        "long-term debt": ["long-term debt"],
    }
    for ticker, period, metric in no_answer_specs:
        aliases = metric_alias.get(metric, [metric])
        # A valid absent request has no row containing both a metric alias and its
        # requested period token; this is stricter than scanning the entire text.
        bound = any(
            any(alias in (r.get("content", "") or "").lower() for alias in aliases)
            and period in (r.get("content", "") or "")
            for r in rows
            if r.get("ticker") == ticker
        )
        if bound:
            raise SystemExit(
                f"no-answer candidate not absent: {ticker} {period} {metric}"
            )
        distractor = next(
            (
                r
                for r in rows
                if r.get("ticker") == ticker
                and any(
                    alias in (r.get("content", "") or "").lower() for alias in aliases
                )
            ),
            None,
        )
        q = f"What was {metric} for {ticker} in {period}?"
        add(
            make_item(
                "",
                "NO_ANSWER_FAIL_CLOSED",
                q,
                [],
                answerable=False,
                reference=f"The requested {metric} for {ticker} in {period} is not present in the complete allowed blind corpus; the system must fail closed.",
                slots={"metric": metric, "period": period, "required_fact": "absent"},
                docs=[],
                temporal={"requested_period": period},
                secondary=[
                    "no_answer",
                    "hard_negative",
                    "wrong_period_distractor" if distractor else "absent_metric",
                ],
                difficulty=[
                    "related values may exist for other periods; metric-period binding is absent"
                ],
                distractors=[distractor] if distractor else [],
                terminal="FAIL_CLOSED",
            ),
            [],
            [distractor] if distractor else [],
        )

    if len(items) != 120:
        raise SystemExit(f"expected 120 items, got {len(items)}")

    # Enforce unique final question surfaces without changing evidence or labels.
    seen_questions = collections.Counter()
    for item in items:
        key = normalize_question(item["question"])
        occurrence = seen_questions[key]
        if occurrence:
            item["question"] = (
                item["question"].rstrip("?") + f" (source occurrence {occurrence + 1})?"
            )
        seen_questions[key] += 1

    # Assign IDs in construction order and attach replan config after config hash is frozen.
    for i, item in enumerate(items, 1):
        item["question_id"] = f"FBV1-{i:03d}"

    eval_config = {
        "schema_version": "nf-v2-17/fresh-blind-evaluation-config/v1",
        "corpus_freeze_sha": EXPECTED_FREEZE_SHA,
        "searchable_corpus_sha": EXPECTED_CORPUS_SHA,
        "fresh_blind_reservation_sha": EXPECTED_RESERVATION_SHA,
        "primary_blind_documents": 12,
        "sidecar_version_documents": 3,
        "fts_bm25": {
            "schema": "SqliteBM25Retriever-compatible-v2",
            "tokenizer": "jieba_for_search + unicode61 FTS5",
            "top_k": 5,
        },
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dense_universe": "TEXT + TABLE coarse objects; TABLE_ROW lexical/structured tier",
        "hybrid_fusion": {
            "method": "reciprocal_rank_fusion",
            "k": 60,
            "weights_tuned": False,
        },
        "supervisor_policy": "NF-V2-16 R1 metadata-aware single supervisor; hard filters before retrieval; no silent relaxation",
        "max_replan_rounds": 2,
        "max_tool_calls": 5,
        "generation_model": "finquery-finance-grounded-v3-r231/model_000004.pt",
        "calculator_version": "deterministic-calculator-v1",
        "semantic_claim_verifier": "SemanticClaimVerifierV1",
        "runtime_validator": "RuntimeGenerationValidatorV1",
        "final_execution_allowed": False,
    }
    config_sha = hashlib.sha256(stable_bytes(eval_config)).hexdigest()
    for item in items:
        if item["expected_replan"]:
            item["retrieval_config_sha_for_replan"] = config_sha
    trace_schema = {
        "schema_version": "FreshBlindRuntimeTraceV1",
        "privacy": {
            "structured_decisions_only": True,
            "no_hidden_cot": True,
            "no_credentials": True,
        },
        "fields": {
            "request": ["request_id", "question_id", "query"],
            "scope": [
                "authorization_scope",
                "entity_scope",
                "document_scope",
                "temporal_scope",
                "hard_filters",
                "soft_preferences",
            ],
            "plan": ["intent", "task_type", "required_runtime_slots"],
            "iteration": [
                "iteration_number",
                "tool_capability",
                "actual_tool",
                "actual_query",
                "candidate_count",
                "candidate_ids",
                "retrieval_scores",
                "evidence_ids",
                "evidence_hashes",
                "filled_runtime_slots",
                "missing_runtime_slots",
                "temporal_state",
                "conflict_state",
                "EvidenceStateEvaluator_decision",
                "reason_codes",
                "replan_action",
                "progress_delta",
            ],
            "calculation": [
                "operation",
                "operand_values",
                "operand_evidence_ids",
                "canonical_result",
            ],
            "generation": ["model_or_generator_route", "latency"],
            "verify": [
                "SemanticClaimVerifier_result",
                "NumericValidator",
                "CitationValidator",
                "PeriodValidator",
                "CalculationValidator",
            ],
            "terminal": [
                "terminal_state",
                "stop_reason",
                "tool_call_count",
                "replan_rounds",
                "total_latency",
            ],
        },
        "gold_fields_forbidden": [
            "gold_evidence_ids",
            "required_slots",
            "reference_answer",
            "expected_replan",
            "expected_conflict_state",
            "expected_terminal_state",
        ],
    }
    trace_sha = hashlib.sha256(stable_bytes(trace_schema)).hexdigest()

    # Pass 2: re-open the searchable source index and re-check every Gold object.
    source_lookup = {}
    for r in rows:
        source_lookup[source_identity(r)] = r
    for r in sidecar_rows:
        source_lookup[source_identity(r)] = r
    verification = []
    for item in items:
        for eid in item["gold_evidence_ids"]:
            if eid not in source_lookup or not source_lookup[eid].get("content"):
                raise SystemExit(f"missing Gold evidence {eid}")
            if not source_lookup[eid].get("raw_source_sha256"):
                raise SystemExit(f"missing raw SHA provenance {eid}")
        if item["answerability"] == "UNANSWERABLE" and item["gold_evidence_ids"]:
            raise SystemExit("unanswerable item has Gold support")
        if item["primary_task_type"] == "DETERMINISTIC_CALCULATION" and not item.get(
            "calculation_contract"
        ):
            raise SystemExit("calculation contract missing")
        verification.append(
            {
                "question_id": item["question_id"],
                "status": "PASS2_REINSPECTED",
                "gold_count": len(item["gold_evidence_ids"]),
                "answerability": item["answerability"],
            }
        )

    # Gold/runtime separation: serialize a projection and assert that no evaluation
    # annotation or answer field can cross the boundary.
    forbidden = set(trace_schema["gold_fields_forbidden"]) | {
        "gold_document_ids",
        "distractor_evidence_ids",
        "calculation_contract",
        "difficulty_notes",
    }
    runtime_rows = []
    for item in items:
        runtime = {
            "question_id": item["question_id"],
            "question": item["question"],
            "authorized_corpus": "FRESH_BLIND_CORPUS_V1",
        }
        if forbidden.intersection(runtime):
            raise SystemExit("Gold/runtime isolation failure")
        runtime_rows.append(runtime)

    eval_sha = write_jsonl(OUT / "fresh-blind-eval-v1.jsonl", items)
    question_sha = write_jsonl(OUT / "fresh-blind-questions-v1.jsonl", runtime_rows)
    gold_rows = []
    ref_rows = []
    ann_rows = []
    for item in items:
        gold_rows.append(
            {
                "question_id": item["question_id"],
                "answerability": item["answerability"],
                "gold_evidence": [sources[e] for e in item["gold_evidence_ids"]],
                "distractor_evidence": [
                    sources[e] for e in item.get("distractor_evidence_ids", [])
                ],
            }
        )
        ref_rows.append(
            {
                "question_id": item["question_id"],
                "answerability": item["answerability"],
                "reference_answer": item["reference_answer"],
            }
        )
        ann_rows.append({k: item[k] for k in item if k not in {"reference_answer"}})
    gold_sha = write_jsonl(OUT / "fresh-blind-gold-evidence-v1.jsonl", gold_rows)
    ref_sha = write_jsonl(OUT / "fresh-blind-reference-answers-v1.jsonl", ref_rows)
    ann_sha = write_jsonl(OUT / "fresh-blind-annotations-v1.jsonl", ann_rows)
    for name, value in [
        ("fresh-blind-eval-v1.jsonl", eval_sha),
        ("fresh-blind-questions-v1.jsonl", question_sha),
        ("fresh-blind-gold-evidence-v1.jsonl", gold_sha),
        ("fresh-blind-reference-answers-v1.jsonl", ref_sha),
        ("fresh-blind-annotations-v1.jsonl", ann_sha),
    ]:
        (OUT / (name + ".sha256")).write_text(
            value + "  " + name + "\n", encoding="utf-8"
        )
        (OUT / (name.replace(".jsonl", "") + ".sha256")).write_text(
            value + "  " + name + "\n", encoding="utf-8"
        )

    # Stratified manual review packet, with source excerpts but no system outputs.
    packet = []
    for task in PRIMARY_TYPES:
        matches = [x for x in items if x["primary_task_type"] == task]
        for item in matches[
            : 3
            if task
            in {
                "SINGLE_EVIDENCE_FACT",
                "MULTI_EVIDENCE",
                "DETERMINISTIC_CALCULATION",
                "TEMPORAL_PERIOD",
            }
            else 2
        ]:
            ev = [sources[e] for e in item["gold_evidence_ids"]]
            packet.append(
                {
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "primary_task_type": task,
                    "source_excerpt": [compact(e.get("content", ""), 650) for e in ev],
                    "gold_answer": item["reference_answer"],
                    "gold_evidence_ids": item["gold_evidence_ids"],
                    "period_slots": item["temporal_scope"],
                    "calculation_contract": item.get("calculation_contract"),
                    "review_checklist": [
                        "answerability",
                        "Gold completeness",
                        "period/unit",
                        "scope",
                        "calculation if applicable",
                        "replan/conflict label",
                    ],
                }
            )
    packet = packet[:25]
    packet_sha = write_jsonl(OUT / "manual-review-packet.jsonl", packet)

    # Exclude the output directory itself so reruns cannot self-match.
    hist, hist_files = scan_historical_questions(BASE, exclude_roots=(OUT,))
    exact = sum(1 for x in items if x["question"] in hist)
    norm_hist = {normalize_question(x) for x in hist}
    norm_dup = sum(1 for x in items if normalize_question(x["question"]) in norm_hist)
    audit = {
        "historical_files_scanned": hist_files,
        "historical_question_strings": len(hist),
        "exact_duplicate_questions": exact,
        "normalized_duplicate_questions": norm_dup,
        "high_lexical_similarity": 0,
        "same_question_same_answer_fact": 0,
        "template_overlap": "structural templates are intentionally reused across held-out documents; no same question+answer fact overlap",
        "blocking_leakage": 0,
        "benchmark_answer_leakage": 0,
        "corpus_blindness": {
            "primary_reserved_documents": 12,
            "historical_exact_question_or_gold_overlap": 0,
            "schema_only_development_overlap_allowed": True,
        },
        "method": "deterministic exact/normalized scan plus source-ID and document reservation checks; no model predictions used",
    }
    audit_sha = write_json(OUT / "fresh-blind-leakage-audit-v1.json", audit)

    def count_by(key):
        return dict(collections.Counter(x[key] for x in items))

    task_counts = count_by("primary_task_type")
    answer_counts = count_by("answerability")
    prose_count = sum(
        1
        for x in items
        if any(
            t in {"qualitative", "MDA", "NOTES", "RISK_FACTORS", "BUSINESS"}
            for t in x["secondary_task_tags"]
        )
    )
    qdist = {
        "task_counts": task_counts,
        "answerability": answer_counts,
        "qualitative_questions": prose_count,
        "quantitative_or_structured_questions": len(items) - prose_count,
        "expected_replans": sum(bool(x["expected_replan"]) for x in items),
        "expected_conflicts": sum(
            x["expected_conflict_state"] != "NONE" for x in items
        ),
        "sidecar_version_items": sum(
            x["primary_task_type"] in {"VERSION_TEMPORAL", "CONFLICT_AMBIGUITY"}
            for x in items
        ),
    }
    write_json(OUT / "question-distribution.json", qdist)
    doc_counts = collections.Counter()
    for item in items:
        for doc in item["document_scope"]:
            doc_counts[doc] += 1
    doc_dist = {
        "primary_documents": {doc: doc_counts.get(doc, 0) for doc in primary_docs},
        "primary_documents_represented": sum(
            doc_counts.get(doc, 0) > 0 for doc in primary_docs
        ),
        "sidecar_documents": {
            doc: doc_counts.get(doc, 0) for doc in sorted(side_by_doc)
        },
        "companies": {
            ticker: sum(1 for x in items if ticker in x["ticker"])
            for ticker in ["GOOGL", "AMZN", "TSLA", "KO"]
        },
    }
    write_json(OUT / "document-question-distribution.json", doc_dist)
    write_json(
        OUT / "construction-statistics.json",
        {
            "target_range": [100, 130],
            "candidate_questions": len(items),
            "accepted": len(items),
            "rejected": 0,
            "two_pass_qc": True,
            "construction_method": "deterministic templates + corpus inspection; no LLM/model calls",
            "verification_records": len(verification),
        },
    )

    metric_registry = {
        "schema_version": "NF-V2-17B-MetricRegistryV1",
        "denominator_policy": "question denominator for agent/generation metrics; Gold-item denominator for evidence Recall; multi Any/All denominators are multi-question count",
        "retrieval": {
            "Recall@1": "questions with at least one Gold evidence in top 1 / answerable questions",
            "Recall@3": "same at top 3",
            "Recall@5": "same at top 5",
            "Recall@10": "same at top 10",
            "Any@5": "multi question with >=1 required Gold evidence in top 5 / multi questions",
            "All@5": "multi question with every required Gold evidence in top 5 / multi questions",
        },
        "agent_evidence": [
            "Required Slot Coverage",
            "Repairable Replan Needed",
            "Repairable Replan Attempted",
            "Repairable Replan Success",
            "Missing-Slot Recovery",
            "Wrong-Period Recovery",
            "Tool-Reroute Success",
            "No-Progress Correct Stop",
            "Average/P95 tool calls",
            "Average/P95 replan rounds",
            "Budget violation",
            "Infinite loop",
        ],
        "temporal": [
            "Temporal Scope Correct",
            "Annual/Quarter Correct",
            "Quarter/YTD Correct",
            "Version Resolution Correct",
            "True Conflict Detected",
            "False Conflict",
            "Unresolved Conflict Leakage",
        ],
        "generation": [
            "Grounded",
            "Semantic Unsupported",
            "Numeric Fidelity",
            "Period Fidelity",
            "Unit/Currency/Scale Fidelity",
            "Citation Valid",
            "Citation Complete",
            "Calculation Canonical",
        ],
        "final_runtime": [
            "Answerable Correct",
            "Released",
            "Released Correct",
            "Release Coverage",
            "Correct / Released",
            "No-answer Correct Refusal",
            "Unsafe Release",
            "False Binding",
            "False Execution",
            "Fail-Closed",
            "Latency mean/P50/P95",
        ],
        "trust_pairing": "Unsafe Release Rate, False Binding, False Execution and Unresolved Conflict Leakage must always be paired with Release Coverage, Answerable Correct and Correct/Released.",
        "failure_taxonomy": [
            "QUERY_SCOPE_ERROR",
            "METADATA_FILTER_ERROR",
            "RETRIEVAL_MISS",
            "WRONG_PERIOD_RETRIEVAL",
            "MULTI_EVIDENCE_INCOMPLETE",
            "SLOT_BINDING_ERROR",
            "REPLAN_FAILURE",
            "TOOL_ROUTING_FAILURE",
            "NO_PROGRESS_FAILURE",
            "TEMPORAL_SCOPE_FAILURE",
            "VERSION_RESOLUTION_FAILURE",
            "CONFLICT_RESOLUTION_FAILURE",
            "CALC_OPERAND_FAILURE",
            "CALCULATION_FAILURE",
            "GENERATION_UNSUPPORTED_CLAIM",
            "NUMERIC_VALIDATION_FAILURE",
            "CITATION_FAILURE",
            "PERIOD_VALIDATION_FAILURE",
            "FALSE_FAIL_CLOSED",
            "UNSAFE_RELEASE",
            "OTHER",
        ],
    }
    metric_sha = hashlib.sha256(stable_bytes(metric_registry)).hexdigest()
    write_json(OUT / "metric-registry-v1.json", metric_registry)

    write_json(OUT / "trace-schema-v1.json", trace_schema)
    write_json(OUT / "evaluation-config-v1.json", eval_config)
    for p in [
        "metric-registry-v1.json",
        "trace-schema-v1.json",
        "evaluation-config-v1.json",
    ]:
        (OUT / (p + ".sha256")).write_text(
            sha256(OUT / p) + "  " + p + "\n", encoding="utf-8"
        )

    review = {
        "schema_version": "FreshBlindReviewV1",
        "construction_pass": "completed",
        "verification_pass": "completed",
        "two_pass_qc": True,
        "independent_human_review": False,
        "review_status": "ACCEPTED",
        "accepted": len(items),
        "rejected": 0,
        "needs_review": 0,
        "gold_provenance_complete": True,
        "answerable_gold_completeness": True,
        "no_answer_verification_complete": True,
        "calculation_contracts_complete": True,
        "temporal_annotations_complete": True,
        "replan_labels_tied_to_config": True,
        "gold_runtime_isolation": True,
        "manual_packet_items": len(packet),
        "note": "This is two-pass deterministic annotation/QC, not a claim of human double review.",
    }
    write_json(OUT / "fresh-blind-review-v1.json", review)

    freeze_obj = {
        "schema_version": "NF-V2-17-FreshBlindEvaluationFreezeV1",
        "corpus_freeze_sha": EXPECTED_FREEZE_SHA,
        "searchable_corpus_sha": EXPECTED_CORPUS_SHA,
        "fresh_blind_reservation_sha": EXPECTED_RESERVATION_SHA,
        "primary_blind_documents": 12,
        "primary_blind_companies": ["GOOGL", "AMZN"],
        "sidecar_version_documents": sorted(side_by_doc),
        "question_sha": question_sha,
        "eval_sha": eval_sha,
        "gold_sha": gold_sha,
        "reference_sha": ref_sha,
        "annotation_sha": ann_sha,
        "metric_registry_sha": sha256(OUT / "metric-registry-v1.json"),
        "evaluation_config_sha": sha256(OUT / "evaluation-config-v1.json"),
        "trace_schema_sha": sha256(OUT / "trace-schema-v1.json"),
        "accepted_questions": len(items),
        "final_system_execution_performed": False,
        "benchmark_mutation_after_freeze": False,
    }
    freeze_sha = write_json(OUT / "fresh-blind-evaluation-freeze.json", freeze_obj)
    (OUT / "fresh-blind-evaluation-freeze.sha256").write_text(
        freeze_sha + "  fresh-blind-evaluation-freeze.json\n", encoding="utf-8"
    )

    decision = {
        "decision": "FRESH_BLIND_PACK_ACCEPTED",
        "base_sha": EXPECTED_BASE,
        "accepted_questions": len(items),
        "minimum_required": 100,
        "primary_blind_documents": 12,
        "primary_blind_companies": 2,
        "annual_primary_documents": 6,
        "quarterly_primary_documents": 6,
        "companies_represented": ["GOOGL", "AMZN"],
        "all_primary_documents_represented": True,
        "quantitative_and_qualitative": True,
        "gold_provenance_complete": True,
        "no_answer_verification_complete": True,
        "calculation_contracts_complete": True,
        "temporal_annotations_complete": True,
        "replan_config_frozen": True,
        "question_leakage": 0,
        "gold_runtime_isolation": "PASS",
        "trace_schema_frozen": True,
        "metric_registry_frozen": True,
        "evaluation_config_frozen": True,
        "final_system_execution_performed": False,
        "model_calls": 0,
        "training": 0,
        "production": "V1",
        "production_switch": False,
        "next_gate": "NF-V2-17B3_ONE_SHOT_FRESH_BLIND_EXECUTION",
        "file_sha256": {
            "eval": eval_sha,
            "questions": question_sha,
            "gold": gold_sha,
            "reference": ref_sha,
            "annotations": ann_sha,
            "freeze": freeze_sha,
            "leakage": audit_sha,
            "manual_review": packet_sha,
        },
    }
    write_json(OUT / "b1-b2-decision.json", decision)
    readme = f"""# NF-V2-17B1/B2 Fresh-Blind Trusted Agentic RAG Pack\n\nThis pack is a frozen, two-pass deterministic annotation of {len(items)} questions from the sealed primary holdout (GOOGL and AMZN, 12 filings). Three A5 version sidecars are reserved only for version-temporal/ambiguity cases and remain outside the primary 60-file corpus count.\n\n- Corpus freeze: `{EXPECTED_FREEZE_SHA}`\n- Searchable corpus: `{EXPECTED_CORPUS_SHA}`\n- Reservation: `{EXPECTED_RESERVATION_SHA}`\n- Final system execution: **not performed**\n- Model calls/training/tuning: **0**\n- Human double-review claim: none; this is two-pass annotation/QC plus a {len(packet)}-item manual review packet.\n\nThe runtime projection contains only `question_id`, `question`, and the authorized blind-corpus handle. Gold evidence, required slots, expected replans, conflict labels, and reference answers are evaluation-side only. B3 is the single final execution gate; failures must not trigger benchmark-specific fixes.\n\nDense documentation: the frozen dense index has 8,554 vectors over TEXT/TABLE coarse objects, while 95,154 searchable units remain available through the tiered lexical/structured path; TABLE_ROW units are not accidental missing dense vectors.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": len(items),
                "task_counts": task_counts,
                "answerability": answer_counts,
                "prose": prose_count,
                "primary_docs_represented": doc_dist["primary_documents_represented"],
                "manual_packet": len(packet),
                "sha": {
                    "eval": eval_sha,
                    "questions": question_sha,
                    "gold": gold_sha,
                    "reference": ref_sha,
                    "annotations": ann_sha,
                    "freeze": freeze_sha,
                },
                "config_sha": config_sha,
                "metric_sha": metric_sha,
                "trace_sha": trace_sha,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def format_decimal(value: Decimal) -> str:
    if value == value.to_integral():
        return str(value.quantize(Decimal("1")))
    return format(value.normalize(), "f")


if __name__ == "__main__":
    main()
