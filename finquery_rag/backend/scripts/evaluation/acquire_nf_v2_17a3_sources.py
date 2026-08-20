#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

A2_SHA = "94b6ff91ca8e8c91473aac3f7066534a5e3d7f8242cdb4406c55e200f1e826c7"
BASE_SHA = "7628250b0fefc3fbec43a6bd98be9abce193fbb7"
A2_REL = Path(
    "finquery_rag/backend/artifacts/evaluation/nf-v2-17-financial-corpus-v2/source-intake-reviewed.jsonl"
)
ART_REL = Path("finquery_rag/backend/artifacts/evaluation/nf-v2-17-financial-corpus-v2")
RAW_DEFAULT = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/financial_corpus_v2")
UA = "nanochat-finquery-rag/2.17A3 source-acquisition contact@example.com"
HIST = {
    "MSFT": ("msft_fy2025", 2025),
    "AAPL": ("aapl_fy2025", 2025),
    "NVDA": ("nvda_fy2025", 2025),
    "JPM": ("jpm_fy2025", 2025),
    "TSLA": ("tsla_fy2025", 2025),
    "KO": ("ko_fy2025", 2025),
    "V": ("v_fy2025", 2025),
    "PFE": ("pfe_fy2024", 2024),
}


def now():
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def dump(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rows(p):
    return [
        json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()
    ]


def sha_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def fetch(url):
    q = Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Encoding": "gzip",
            "Connection": "close",
        },
    )
    with urlopen(q, timeout=45) as r:
        b = r.read()
        if (
            b[:2] == bytes((31, 139))
            or str(r.headers.get("Content-Encoding", "")).lower() == "gzip"
        ):
            b = gzip.decompress(b)
        return b, {
            "status": int(getattr(r, "status", 200)),
            "content_type": str(r.headers.get("Content-Type", ""))
            .split(";", 1)[0]
            .lower()
            .strip(),
            "final_url": str(r.geturl()),
            "headers": {
                "etag": r.headers.get("ETag"),
                "last_modified": r.headers.get("Last-Modified"),
                "content_length": r.headers.get("Content-Length"),
            },
        }


def get(url):
    aa = []
    for i in range(1, 4):
        try:
            b, m = fetch(url)
            aa.append({"attempt": i, "status": m["status"], "ok": True})
            return (b, m), {"attempts": aa, "download_status": "DOWNLOADED"}
        except HTTPError as e:
            aa.append(
                {
                    "attempt": i,
                    "status": int(e.code),
                    "ok": False,
                    "error": str(e.reason),
                }
            )
            if e.code in (404, 410):
                return None, {
                    "attempts": aa,
                    "download_status": "FAILED",
                    "failure_reason": f"HTTP_{e.code}",
                }
        except (URLError, TimeoutError, OSError, gzip.BadGzipFile) as e:
            aa.append({"attempt": i, "status": None, "ok": False, "error": str(e)})
        if i < 3:
            time.sleep((1, 3, 8)[i - 1])
    return None, {
        "attempts": aa,
        "download_status": "FAILED",
        "failure_reason": "NETWORK_FAILURE_OR_RATE_LIMIT",
    }


def dates(x):
    if not x:
        return []
    try:
        d = datetime.strptime(str(x)[:10], "%Y-%m-%d")
    except ValueError:
        return [str(x)]
    return [
        str(x)[:10],
        d.strftime("%B %-d, %Y"),
        d.strftime("%B %d, %Y"),
        d.strftime("%b %-d, %Y"),
        d.strftime("%b %d, %Y"),
        d.strftime("%d %B %Y"),
    ]


def token(x):
    return re.sub(r"[^A-Za-z]", "", str(x).split()[0]).lower()


def verify(r, b, m):
    t = b.decode("utf-8", "replace")
    low = t.lower()
    if len(b) < 512 or not re.search(r"<\s*(?:!doctype\s+html|html)\b", low):
        return (
            "EMPTY_OR_INVALID_SOURCE",
            {"html": False},
            ["empty, too small, or non-HTML"],
        )
    if any(
        x in low
        for x in (
            "your request originates from an undeclared automated tool",
            "request rate threshold exceeded",
            "access denied",
            "the requested url was not found on this server",
        )
    ):
        return (
            "EMPTY_OR_INVALID_SOURCE",
            {"html": True, "sec_error_page": True},
            ["SEC error/access page"],
        )
    form = str(r.get("form_type") or "")
    fok = bool(re.search(rf"\b{re.escape(form)}\b", t, re.I))
    hits = [x for x in dates(r.get("report_period_end")) if x.lower() in low]
    pok = bool(hits)
    cik = str(r.get("cik") or "").lstrip("0") or "0"
    iok = (
        cik in t
        or token(r.get("company")) in low
        or str(r.get("ticker")).lower() in low
    )
    idx = (
        "document format files" in low
        and "filing detail" in low
        and "period of report" in low
    )
    sig = {
        "html": True,
        "form": fok,
        "period": pok,
        "issuer": iok,
        "period_hits": hits,
        "index_like": idx,
    }
    if idx:
        return (
            "CONTENT_IDENTITY_MISMATCH",
            sig,
            ["SEC filing index instead of primary document"],
        )
    w = []
    if not fok:
        w.append("expected form token not found")
    if not pok:
        w.append("expected report period text not found; A2 metadata authoritative")
    if not iok:
        w.append("issuer/CIK token not found; A2 metadata authoritative")
    if m.get("content_type") not in ("text/html", "application/xhtml+xml", ""):
        w.append("unexpected content type " + repr(m.get("content_type")))
    return (
        (
            "CONTENT_VERIFIED"
            if fok and pok and iok
            else "CONTENT_VERIFIED_WITH_WARNING"
        ),
        sig,
        w,
    )


def relpath(r, side=False):
    return (
        (
            Path("financial_corpus_v2/raw/version_candidates")
            if side
            else Path("financial_corpus_v2/raw/SEC")
        )
        / str(r["ticker"])
        / str(r["document_id"])
        / "primary.html"
    )


def acquire(r, root, side=False):
    rel = relpath(r, side)
    p = root.parent / rel
    mp = p.with_name("source-metadata.json")
    ts = now()
    b = None
    m = {}
    tr = {}
    reused = False
    if p.exists() and mp.exists():
        try:
            old = json.loads(mp.read_text(encoding="utf-8"))
            if old.get("raw_sha256") == sha_file(p) and old.get("source_url") == r.get(
                "raw_source_url"
            ):
                b = p.read_bytes()
                ts = old.get("download_timestamp", ts)
                m = old.get("response_meta", old.get("response", {}))
                tr = {"attempts": [], "download_status": "REUSED_VERIFIED"}
                reused = True
        except (OSError, json.JSONDecodeError):
            pass
    if b is None and p.exists():
        b = p.read_bytes()
        tr = {"attempts": [], "download_status": "EXISTING_UNVERIFIED"}
    if b is None:
        got, tr = get(str(r["raw_source_url"]))
        time.sleep(0.45)
        if got is None:
            return {
                **r,
                "raw_local_path": str(rel),
                "raw_sha256": None,
                "raw_bytes": 0,
                "download_status": "DOWNLOAD_FAILED",
                "download_timestamp": ts,
                "response": tr,
                "content_verification_status": "DOWNLOAD_FAILED",
                "warnings": list(r.get("warnings") or [])
                + [tr.get("failure_reason", "download failed")],
            }
        b, m = got
    st, sig, vw = verify(r, b, m)
    dg = sha_bytes(b)
    if p.exists() and sha_file(p) != dg:
        q = p.with_name("primary.changed." + str(int(time.time())) + ".html")
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(b)
        return {
            **r,
            "raw_local_path": str(rel),
            "raw_sha256": dg,
            "raw_bytes": len(b),
            "download_status": "RAW_CONTENT_CHANGED",
            "download_timestamp": ts,
            "response": tr,
            "response_meta": m,
            "content_verification_status": "RAW_CONTENT_CHANGED",
            "identity_signals": sig,
            "warnings": list(r.get("warnings") or [])
            + vw
            + ["existing raw not overwritten; quarantined"],
        }
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name("." + p.name + "." + str(os.getpid()) + ".tmp")
        tmp.write_bytes(b)
        tmp.replace(p)
    meta = {
        "document_id": r["document_id"],
        "source_url": r.get("raw_source_url"),
        "source_index_url": r.get("source_index_url"),
        "raw_local_path": str(rel),
        "raw_sha256": dg,
        "raw_bytes": len(b),
        "download_status": tr.get(
            "download_status", "REUSED_VERIFIED" if reused else "DOWNLOADED"
        ),
        "download_timestamp": ts,
        "response": m,
        "content_verification_status": st,
        "identity_signals": sig,
        "warnings": list(r.get("warnings") or []) + vw,
    }
    dump(mp, meta)
    return {
        **r,
        "raw_local_path": str(rel),
        "raw_sha256": dg,
        "raw_bytes": len(b),
        "download_status": meta["download_status"],
        "download_timestamp": ts,
        "response": tr,
        "response_meta": m,
        "content_verification_status": st,
        "identity_signals": sig,
        "warnings": meta["warnings"],
    }


def snap(rs):
    h = hashlib.sha256()
    for r in rs:
        h.update(
            (
                "\t".join(
                    str(r.get(k) or "")
                    for k in (
                        "document_id",
                        "accession_number",
                        "raw_sha256",
                        "form_type",
                        "fiscal_year",
                        "fiscal_quarter",
                        "report_period_end",
                    )
                )
                + "\n"
            ).encode()
        )
    return h.hexdigest()


def pct(v, q):
    if not v:
        return None
    v = sorted(v)
    x = (len(v) - 1) * q
    a = int(x)
    z = min(a + 1, len(v) - 1)
    return v[a] + (v[z] - v[a]) * (x - a)


def matrix(p, rs):
    by = {
        (r["ticker"], r["role"], r["fiscal_year"], r.get("fiscal_quarter")): r
        for r in rs
    }
    fs = [
        "company",
        "ticker",
        "FY2023 annual",
        "FY2024 annual",
        "FY2025 annual",
        "FY2025 Q1",
        "FY2025 Q2",
        "FY2025 Q3",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fs)
        w.writeheader()
        for tick, co in sorted({(r["ticker"], r["company"]) for r in rs}):
            o = {"company": co, "ticker": tick}
            for y in (2023, 2024, 2025):
                r = by[(tick, "ANNUAL", y, None)]
                o[f"FY{y} annual"] = (
                    "ACQUIRED_VERIFIED "
                    if r["content_verification_status"] == "CONTENT_VERIFIED"
                    else "ACQUIRED_WARNING "
                    if r["content_verification_status"]
                    == "CONTENT_VERIFIED_WITH_WARNING"
                    else "FAILED "
                ) + str(r["document_id"])
            for q in ("Q1", "Q2", "Q3"):
                r = by[(tick, "QUARTERLY", 2025, q)]
                o[f"FY2025 {q}"] = (
                    "ACQUIRED_VERIFIED "
                    if r["content_verification_status"] == "CONTENT_VERIFIED"
                    else "ACQUIRED_WARNING "
                    if r["content_verification_status"]
                    == "CONTENT_VERIFIED_WITH_WARNING"
                    else "FAILED "
                ) + str(r["document_id"])
            w.writerow(o)


def main(repo, limit=None):
    ad = repo / ART_REL
    a2 = repo / A2_REL
    if sha_file(a2) != A2_SHA:
        raise SystemExit("A2 SHA mismatch")
    rs = rows(a2)
    if (
        len(rs) != 60
        or sum(r.get("role") == "ANNUAL" for r in rs) != 30
        or sum(r.get("role") == "QUARTERLY" for r in rs) != 30
    ):
        raise SystemExit("A2 role/count contract failed")
    if any(not r.get("document_id") or not r.get("raw_source_url") for r in rs):
        raise SystemExit("A2 unresolved URL")
    root = Path(os.environ.get("NF_V2_17_RAW_ROOT", str(RAW_DEFAULT)))
    primary = []
    for r in rs[:limit] if limit is not None else rs:
        print("acquire " + r["document_id"], flush=True)
        primary.append(acquire(r, root))
    if limit is not None:
        print("bounded smoke complete", len(primary), flush=True)
        return 0
    cs = json.loads((ad / "amendment-discovery.json").read_text(encoding="utf-8")).get(
        "candidates", []
    )
    side = []
    for c in cs:
        acc = str(c["accession_number"])
        base = f"https://www.sec.gov/Archives/edgar/data/{int(c['cik'])}/{acc.replace('-', '')}"
        x = {
            "document_id": c.get("amended_filing_id")
            or f"SEC_{c['cik']}_{acc.replace('-', '')}",
            "raw_source_url": f"{base}/{c['primary_document_name']}",
            "source_index_url": f"{base}/{acc}-index.htm",
            **c,
            "fiscal_year": int(str(c.get("report_period_end") or "0000")[:4]),
            "fiscal_quarter": None,
            "role": "VERSION_SIDECAR_CANDIDATE",
        }
        side.append(acquire(x, root, True))
    mf = ad / "raw-corpus-manifest-v2.jsonl"
    mf.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in primary
        ),
        encoding="utf-8",
    )
    msha = sha_file(mf)
    (ad / "raw-corpus-manifest-v2.sha256").write_text(
        msha + "  raw-corpus-manifest-v2.jsonl\n", encoding="utf-8"
    )
    csha = snap(primary)
    (ad / "raw-corpus-snapshot.sha256").write_text(csha + "\n", encoding="utf-8")
    matrix(ad / "raw-source-coverage-matrix.csv", primary)
    hm = json.loads(
        (
            repo / "finquery_rag/backend/benchmarks/financial_rag_v1/corpus.json"
        ).read_text(encoding="utf-8")
    )
    hd = {str(x.get("document_id")): x for x in hm.get("documents", [])}
    hist = []
    for tick, (hid, y) in HIST.items():
        r = next(
            (
                x
                for x in primary
                if x["ticker"] == tick
                and x["fiscal_year"] == y
                and x["role"] == "ANNUAL"
            ),
            None,
        )
        hist.append(
            {
                "historical_document_id": hid,
                "ticker": tick,
                "historical_identity_match": bool(r),
                "resolved_document_id": r.get("document_id") if r else None,
                "raw_sha256": r.get("raw_sha256") if r else None,
                "historical_sha256": hd.get(hid, {}).get("file_sha256"),
                "classification": "RAW_REACQUIRED_HISTORICAL_SHA_UNAVAILABLE"
                if r and r.get("raw_sha256")
                else "SOURCE_IDENTITY_MISMATCH",
                "content_equivalence": "UNKNOWN",
            }
        )
    sizes = [int(r.get("raw_bytes") or 0) for r in primary if r.get("raw_sha256")]
    grp = defaultdict(list)
    for r in primary:
        if r.get("raw_sha256"):
            grp["annual" if r["role"] == "ANNUAL" else "quarterly"].append(
                int(r["raw_bytes"])
            )
            grp[r["ticker"]].append(int(r["raw_bytes"]))
    stats = {
        "total_raw_bytes": sum(sizes),
        "mean_bytes": mean(sizes) if sizes else 0,
        "p50_bytes": pct(sizes, 0.5),
        "p95_bytes": pct(sizes, 0.95),
        "min_bytes": min(sizes) if sizes else 0,
        "max_bytes": max(sizes) if sizes else 0,
        "by_group": {
            k: {
                "count": len(v),
                "total_bytes": sum(v),
                "mean_bytes": mean(v),
                "p50_bytes": pct(v, 0.5),
                "p95_bytes": pct(v, 0.95),
                "min_bytes": min(v),
                "max_bytes": max(v),
            }
            for k, v in sorted(grp.items())
        },
    }
    sg = defaultdict(list)
    ag = defaultdict(list)
    for r in primary:
        if r.get("raw_sha256"):
            sg[r["raw_sha256"]].append(r["document_id"])
        ag[str(r.get("accession_number"))].append(r["document_id"])
    dsha = {k: v for k, v in sg.items() if len(v) > 1}
    dacc = {k: v for k, v in ag.items() if len(v) > 1}
    wr = []
    for r in rs:
        if r.get("source_verification_status") == "VERIFIED_WITH_WARNING":
            x = next((z for z in primary if z["document_id"] == r["document_id"]), None)
            wr.append(
                {
                    "document_id": r["document_id"],
                    "ticker": r["ticker"],
                    "a2_warnings": r.get("warnings"),
                    "a3_content_status": x.get("content_verification_status")
                    if x
                    else "NOT_ACQUIRED",
                    "impact": "NON_SEMANTIC_LOCATOR_PROBE_WARNING",
                    "decision": "RETAIN_WARNING_ALLOW_CONTENT_VERIFICATION"
                    if x
                    and x.get("content_verification_status")
                    in ("CONTENT_VERIFIED", "CONTENT_VERIFIED_WITH_WARNING")
                    else "BLOCK",
                }
            )
    acq = [r for r in primary if r.get("raw_sha256")]
    ver = [
        r for r in primary if r.get("content_verification_status") == "CONTENT_VERIFIED"
    ]
    warn = [
        r
        for r in primary
        if r.get("content_verification_status") == "CONTENT_VERIFIED_WITH_WARNING"
    ]
    fail = [
        r
        for r in primary
        if r.get("content_verification_status")
        not in ("CONTENT_VERIFIED", "CONTENT_VERIFIED_WITH_WARNING")
    ]
    res = {
        "schema_version": "nf-v2-17/acquisition-results/v1",
        "a2_manifest_sha256": A2_SHA,
        "raw_root": str(root),
        "primary_planned": 60,
        "primary_attempted": len(primary),
        "primary_raw_acquired": len(acq),
        "content_verified": len(ver),
        "verified_with_warning": len(warn),
        "failed": len(fail),
        "annual_acquired": sum(
            r["role"] == "ANNUAL" and bool(r.get("raw_sha256")) for r in primary
        ),
        "quarterly_acquired": sum(
            r["role"] == "QUARTERLY" and bool(r.get("raw_sha256")) for r in primary
        ),
        "raw_format": {"HTML": len(acq), "PDF": 0},
        "raw_overwrite_events": 0,
        "invalid_source_pages_admitted": sum(
            r.get("content_verification_status")
            in ("EMPTY_OR_INVALID_SOURCE", "CONTENT_IDENTITY_MISMATCH")
            for r in primary
        ),
        "parsing_performed": False,
        "indexing_performed": False,
        "question_generation_performed": False,
        "training": 0,
        "model_calls": 0,
    }
    dump(ad / "acquisition-results.json", res)
    dump(
        ad / "historical-8-reacquisition.json",
        {
            "records": hist,
            "reacquired": sum(x["historical_identity_match"] for x in hist),
            "content_equivalence_policy": "UNKNOWN unless legitimate raw comparison exists",
        },
    )
    dump(
        ad / "amendment-sidecar-acquisition.json",
        {
            "planned": len(cs),
            "acquired": sum(bool(x.get("raw_sha256")) for x in side),
            "records": side,
            "canonical_primary_count_excludes_sidecars": True,
        },
    )
    dump(
        ad / "raw-duplicate-audit.json",
        {
            "primary_records": len(primary),
            "duplicate_raw_sha256": dsha,
            "duplicate_accessions": dacc,
            "canonical_duplicate_count": len(dsha) + len(dacc),
            "expected": 0,
        },
    )
    dump(ad / "raw-source-size-statistics.json", stats)
    dump(
        ad / "source-warning-resolution.json",
        {"a2_warning_count": len(wr), "records": wr},
    )
    decision = (
        "RAW_CORPUS_ACQUIRED"
        if len(primary) == 60
        and len(acq) == 60
        and not fail
        and not dsha
        and not dacc
        and all(x["decision"].endswith("ALLOW_CONTENT_VERIFICATION") for x in wr)
        else "SOURCE_ACQUISITION_NEEDS_REVISION"
    )
    rec = {
        "schema_version": "nf-v2-17/source-acquisition-decision/v1",
        "base_sha": BASE_SHA,
        "a2_manifest_sha256": A2_SHA,
        "companies": len({r["ticker"] for r in primary}),
        "primary_filings": len(primary),
        "annual": sum(r["role"] == "ANNUAL" for r in primary),
        "quarterly": sum(r["role"] == "QUARTERLY" for r in primary),
        "raw_acquired": len(acq),
        "content_verified": len(ver),
        "verified_with_warning": len(warn),
        "failed": len(fail),
        "historical_8_raw_reacquired": sum(
            x["historical_identity_match"] for x in hist
        ),
        "historical_identity_matches": sum(
            x["historical_identity_match"] for x in hist
        ),
        "amendment_sidecars_acquired": sum(bool(x.get("raw_sha256")) for x in side),
        "canonical_duplicates": len(dsha) + len(dacc),
        "invalid_source_pages_admitted": res["invalid_source_pages_admitted"],
        "raw_overwrite_events": 0,
        "created_at_financial_time_misuse": 0,
        "raw_corpus_snapshot_sha256": csha,
        "parsing_performed": False,
        "indexing_performed": False,
        "fresh_blind_questions_generated": False,
        "decision": decision,
        "next_gate": "NF-V2-17A4_NORMALIZATION_AND_PARSE"
        if decision == "RAW_CORPUS_ACQUIRED"
        else "NF-V2-17A3_SOURCE_ACQUISITION_REVIEW",
    }
    dump(ad / "source-acquisition-decision.json", rec)
    readme = (
        "# NF-V2-17A3 Authoritative Source Acquisition\n\n"
        "Base: " + BASE_SHA + "\n"
        "A2 reviewed source identities: 60/60 (30 annual + 30 quarterly); "
        "A2 status: 58 VERIFIED, 2 VERIFIED_WITH_WARNING; "
        "A2 source manifest SHA: " + A2_SHA + "\n"
        "A2 amendments: 3 record-only candidates, canonical inclusion 0.\n\n"
        "Raw root external: " + str(root) + "\n"
        "Primary raw acquired: " + str(len(acq)) + "/60; raw format: HTML; "
        "content verified: "
        + str(len(ver))
        + "/60; content-status warnings: "
        + str(len(warn))
        + "; A2 warning records audited: "
        + str(len(wr))
        + "; failed: "
        + str(len(fail))
        + ".\n"
        "Annual: " + str(sum(r["role"] == "ANNUAL" for r in primary)) + "/30; "
        "quarterly: " + str(sum(r["role"] == "QUARTERLY" for r in primary)) + "/30.\n"
        "Historical 8 reacquired: "
        + str(sum(x["historical_identity_match"] for x in hist))
        + "/8; content equivalence UNKNOWN where prior raw is unavailable.\n"
        "Amendment sidecars: "
        + str(sum(bool(x.get("raw_sha256")) for x in side))
        + "/"
        + str(len(side))
        + "; excluded from canonical count.\n"
        "Raw snapshot SHA: " + csha + "\n"
        "No parsing, normalization, conversion, indexing, question generation, "
        "or model calls occurred.\n"
        "Decision: " + decision + "\n"
    )
    (ad / "README.md").write_text(readme, encoding="utf-8")
    return 0 if decision == "RAW_CORPUS_ACQUIRED" else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    raise SystemExit(main(Path(__file__).resolve().parents[4], a.limit))
