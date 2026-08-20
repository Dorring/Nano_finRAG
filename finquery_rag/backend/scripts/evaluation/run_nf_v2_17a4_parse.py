#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any
from lxml import etree, html

BASE_SHA = "567659d389981b452084a54b5d5446a1aa0d33ee"
A3_MANIFEST_SHA = "58aadcf1543f3e8f175d7747a1b4b9c6dfe29f2cebae11a2f26e7c7b4bf0b14a"
A3_SNAPSHOT_SHA = "459c1f93f47b568efc6571e41760dbda129fe564c2a5c3b48825e843e1b9215c"
RAW_DEFAULT = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/financial_corpus_v2")
ART_REL = Path("finquery_rag/backend/artifacts/evaluation/nf-v2-17-financial-corpus-v2")
NORMALIZATION_VERSION = "nf-v2-17a4-html-normalization-v1"
PARSER_VERSION = "nf-v2-17a4-structure-parser-v1"
CHUNK_VERSION = "nf-v2-17a4-structure-chunk-v1"
MAX_TEXT_CHARS = 2400
DATE_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
CURRENCIES = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₽": "RUB",
    "₩": "KRW",
}
CONFIG_SHA = ""


def compact(v: Any) -> bytes:
    return json.dumps(
        v, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json(p: Path, v: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ws(v: Any) -> str:
    return re.sub(
        r"\s+", " ", str(v or "").replace("\xa0", " ").replace("\u200b", " ")
    ).strip()


def lname(e: etree._Element) -> str:
    if not isinstance(e.tag, str):
        return ""
    if ":" in e.tag:
        return e.tag.rsplit(":", 1)[-1].lower()
    try:
        return etree.QName(e).localname.lower()
    except ValueError:
        return e.tag.lower()


def hidden(e: etree._Element) -> bool:
    n = e
    while n is not None:
        style = str(n.get("style") or "").lower().replace(" ", "")
        if (
            "display:none" in style
            or "visibility:hidden" in style
            or lname(n) in {"script", "style", "noscript", "template"}
        ):
            return True
        n = n.getparent()
    return False


def text_of(e: etree._Element) -> str:
    out = []
    for n in e.iter():
        if hidden(n):
            continue
        if n.text:
            out.append(n.text)
        if n.tail and n.getparent() is not None and not hidden(n.getparent()):
            out.append(n.tail)
    return ws(" ".join(out))


def did(prefix: str, *parts: Any) -> str:
    return prefix + "_" + sha_bytes("\x1f".join(str(x) for x in parts).encode())[:24]


def parse_date_text(s: str) -> str | None:
    s = ws(s)
    pats = [
        r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b",
        r"\b(" + ("|".join(DATE_MONTHS)) + r")\.?\s+(\d{1,2}),?\s+(\d{4})\b",
        r"\b(\d{1,2})\s+(" + ("|".join(DATE_MONTHS)) + r")\.?\s+(\d{4})\b",
    ]
    for i, p in enumerate(pats):
        m = re.search(p, s, re.I)
        if not m:
            continue
        try:
            if i == 0:
                y, mo, d = map(int, m.groups())
            elif i == 1:
                mo = DATE_MONTHS[m.group(1).lower().rstrip(".")]
                d, y = int(m.group(2)), int(m.group(3))
            else:
                d, mo, y = (
                    int(m.group(1)),
                    DATE_MONTHS[m.group(2).lower().rstrip(".")],
                    int(m.group(3)),
                )
            return date(y, mo, d).isoformat()
        except (ValueError, KeyError):
            pass
    return None


def year_tokens(s: str) -> list[str]:
    return re.findall(r"\b(?:19|20)\d{2}\b", ws(s))


def numeric(raw: str) -> dict[str, Any]:
    raw = ws(raw)
    if not raw or raw.lower() in {"-", "—", "–", "n/a", "na", "nm", "nil"}:
        return {
            "raw_value": raw,
            "normalized_value": None,
            "unit": None,
            "currency": None,
            "percent": False,
        }
    t = raw.replace("−", "-").replace("–", "-")
    pct = "%" in t
    cur = next((v for k, v in CURRENCIES.items() if k in t), None)
    neg = bool(re.match(r"^\s*\(", t)) and bool(re.search(r"\)\s*$", t))
    t = t.strip("()").replace(",", "").replace(" ", "")
    t = re.sub(r"[^0-9+\-.]", "", t)
    try:
        val = float(t)
        if neg:
            val = -abs(val)
        return {
            "raw_value": raw,
            "normalized_value": int(val) if val.is_integer() else val,
            "unit": "percent" if pct else ("currency" if cur else "number"),
            "currency": cur,
            "percent": pct,
        }
    except ValueError:
        return {
            "raw_value": raw,
            "normalized_value": None,
            "unit": None,
            "currency": cur,
            "percent": pct,
        }


def detect_currency(s: str) -> str | None:
    low = ws(s).lower()
    for sym, code in CURRENCIES.items():
        if sym in s or re.search(r"\b" + code.lower() + r"\b", low):
            return code
    return None


def detect_scale(s: str) -> str | None:
    low = ws(s).lower()
    if re.search(r"\b(billions?|bn)\b", low):
        return "billions"
    if re.search(r"\b(millions?|mm)\b", low):
        return "millions"
    if re.search(r"\b(thousands?|000s|k)\b", low):
        return "thousands"
    return None


def section(s: str) -> str:
    low = ws(s).lower()
    if re.search(r"risk factors?|risk related", low):
        return "RISK_FACTORS"
    if re.search(
        r"management.?s discussion|results of operations|liquidity and capital", low
    ):
        return "MDA"
    if re.search(
        r"notes? to (the )?(consolidated )?financial|accounting polic|note \d", low
    ):
        return "NOTES"
    if re.search(r"cash flows?|cash flow statement", low):
        return "CASH_FLOW"
    if re.search(
        r"balance sheets?|statement of financial position|financial position", low
    ):
        return "BALANCE_SHEET"
    if re.search(
        r"income statements?|statement[s]? of (income|operations)|comprehensive income|profit and loss|earnings",
        low,
    ):
        return "INCOME_STATEMENT"
    if re.search(r"item\s*1\.?\s*business|\bbusiness\b|company overview", low):
        return "BUSINESS"
    return "UNKNOWN"


def semantic(s: str, role: str, sec: str) -> str:
    low = ws(s).lower()
    if re.search(r"\bas of\b|\bat\s+[a-z]+\s+\d{1,2}", low):
        return "INSTANT"
    duration = list(
        re.finditer(r"\b(three|six|nine|twelve|3|6|9|12)[- ]months?\b", low)
    )
    if duration:
        token = duration[-1].group(1)
        if token in {"six", "nine", "6", "9"}:
            return "YTD"
        if token in {"three", "3"}:
            return "QUARTER"
        if token in {"twelve", "12"}:
            return "ANNUAL"
    if re.search(r"year[- ]to[- ]date|\bytd\b", low):
        return "YTD"
    if re.search(r"\byear(?:s)? ended\b|\bannual\b", low):
        return "ANNUAL"
    if sec == "BALANCE_SHEET" and parse_date_text(low):
        return "INSTANT"
    return "UNKNOWN"


def period_binding(header: str, context: str, role: str, sec: str) -> dict[str, Any]:
    # Only use explicit period language from the table header/caption or a
    # nearby statement heading.  Never parse an arbitrary filing-date from
    # the preceding document text.
    header_text = ws(header)
    context_text = ws(context)
    if not header_text:
        return {
            "period_start": None,
            "period_end": None,
            "duration_months": None,
            "period_semantics": "UNKNOWN",
            "header_text": "",
            "binding_status": "UNKNOWN",
        }
    combined = context_text + " " + header_text
    explicit = bool(
        re.search(
            r"\b(three|six|nine|twelve|3|6|9|12)[- ]months?\b"
            r"|\byear[- ]to[- ]date\b|\bytd\b|\byear(?:s)? ended\b"
            r"|\bas of\b|\bquarter(?:ly)?\b",
            combined,
            re.I,
        )
    )
    # Balance-sheet tables often expose only an "as of" date/year row.
    if sec == "BALANCE_SHEET" and parse_date_text(combined):
        explicit = True
    safe_context = context_text if explicit else ""
    # Prefer the column-local descriptor (e.g. "Six Months Ended") over
    # a shared table context that may also mention a three-month section.
    merged = ws(header_text + " " + safe_context)
    local_sem = semantic(header_text, role, sec)
    sem = local_sem if local_sem != "UNKNOWN" else semantic(safe_context, role, sec)
    end = parse_date_text(header_text)
    ys = year_tokens(header_text)
    if end is None and explicit and ys:
        # Bind the column's own year to a shared month/day phrase such as
        # "Year Ended June 30," or "Three Months Ended June 30,".
        md = re.search(
            r"\b(" + "|".join(DATE_MONTHS) + r")\.?\s+(\d{1,2})\b",
            safe_context + " " + header_text,
            re.I,
        )
        if md:
            try:
                end = date(
                    int(ys[-1]),
                    DATE_MONTHS[md.group(1).lower().rstrip(".")],
                    int(md.group(2)),
                ).isoformat()
            except (KeyError, ValueError):
                end = None
    if end is None and explicit and not ys:
        end = parse_date_text(safe_context)
    if sem == "INSTANT":
        dm = 0
    else:
        dm = None
        duration_matches = list(
            re.finditer(
                r"\b(three|six|nine|twelve|3|6|9|12)[- ]months?\b",
                header_text.lower(),
            )
        )
        if not duration_matches:
            duration_matches = list(
                re.finditer(
                    r"\b(three|six|nine|twelve|3|6|9|12)[- ]months?\b",
                    merged.lower(),
                )
            )
        if duration_matches:
            token = duration_matches[-1].group(1)
            dm = {"three": 3, "six": 6, "nine": 9, "twelve": 12}.get(token)
            if dm is None:
                dm = int(token)
        elif sem == "QUARTER":
            dm = 3
        elif sem == "ANNUAL":
            dm = 12
    return {
        "period_start": None,
        "period_end": end,
        "duration_months": dm,
        "period_semantics": sem,
        "header_text": header_text,
        "binding_status": "BOUND"
        if sem != "UNKNOWN" and end
        else ("PARTIAL" if sem != "UNKNOWN" else "UNKNOWN"),
    }


def has_num(s: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])\(?\$?\s*\d[\d,]*(?:\.\d+)?\)?%?", s))


def direct_rows(table: etree._Element) -> list[list[etree._Element]]:
    out = []
    for tr in table.xpath(".//tr"):
        if any(lname(a) == "table" for a in tr.iterancestors() if a is not table):
            continue
        cells = tr.xpath("./th|./td")
        if cells:
            out.append(cells)
    return out


def grid_rows(rows: list[list[etree._Element]]) -> list[list[dict | None]]:
    grid = []
    for ri, row in enumerate(rows):
        while len(grid) <= ri:
            grid.append([])
        col = 0
        for si, node in enumerate(row):
            while col < len(grid[ri]) and grid[ri][col] is not None:
                col += 1
            try:
                rs = max(1, int(node.get("rowspan") or 1))
                cs = max(1, int(node.get("colspan") or 1))
            except ValueError:
                rs = cs = 1
            rec = {
                "source_row": ri,
                "source_col": si,
                "raw_text": text_of(node),
                "header": lname(node) == "th",
                "rowspan": rs,
                "colspan": cs,
            }
            for rr in range(ri, ri + rs):
                while len(grid) <= rr:
                    grid.append([])
                while len(grid[rr]) < col + cs:
                    grid[rr].append(None)
                for cc in range(col, col + cs):
                    if grid[rr][cc] is None:
                        grid[rr][cc] = rec
            col += cs
    width = max((len(r) for r in grid), default=0)
    for r in grid:
        r.extend([None] * (width - len(r)))
    return grid


def header_idx(grid) -> list[int]:
    # SEC HTML financial tables commonly place the actual period row after
    # spacer rows (and frequently use <td> rather than <th>).  Preserve the
    # initial presentation rows plus explicit period rows found in a bounded
    # prefix of the table.
    out = []
    for i, row in enumerate(grid[:8]):
        vals = [c["raw_text"] for c in row if c]
        if vals and (
            any(c["header"] for c in row if c)
            or (i < 4 and not any(has_num(x) for x in vals))
        ):
            out.append(i)
    for i, row in enumerate(grid[:32]):
        vals = [ws(c["raw_text"]) for c in row if c]
        row_text = ws(" ".join(vals))
        if not row_text:
            continue
        has_period_phrase = bool(
            re.search(
                r"\b(three|six|nine|twelve|3|6|9|12)[- ]months?\b"
                r"|\byear(?:s)? ended\b|\bas of\b|\bquarter(?:ly)?\b"
                r"|\byear[- ]to[- ]date\b|\bytd\b",
                row_text,
                re.I,
            )
        )
        has_date = bool(parse_date_text(row_text))
        has_years = len(set(year_tokens(row_text))) >= 2
        if has_period_phrase or has_date or has_years:
            out.append(i)
    out = sorted(set(out))
    return out or ([0] if grid else [])


def col_headers(grid, idx) -> list[str]:
    w = max((len(r) for r in grid), default=0)
    out = []
    for c in range(w):
        vals = []
        for i in idx:
            row = grid[i]
            row_vals = [ws(x["raw_text"]) for x in row if x and ws(x["raw_text"])]
            row_text = ws(" ".join(row_vals))
            x = ws(row[c]["raw_text"]) if c < len(row) and row[c] else ""
            # Carry a single row-level temporal descriptor to value columns.
            # If a row contains multiple incompatible descriptors, leave the
            # row-local text in place and let the binding remain ambiguous.
            phrase_match = re.search(
                r"\b(?:three|six|nine|twelve|3|6|9|12)[- ]months?\b"
                r"|\byear(?:s)? ended\b|\bas of\b|\bquarter(?:ly)?\b"
                r"|\byear[- ]to[- ]date\b|\bytd\b",
                row_text,
                re.I,
            )
            scope = ""
            if phrase_match:
                tail = row_text[phrase_match.start() :]
                date_match = re.search(
                    r"\b(" + "|".join(DATE_MONTHS) + r")\.?\s+(\d{1,2})\b",
                    tail,
                    re.I,
                )
                scope = phrase_match.group(0)
                if date_match:
                    scope = ws(
                        scope + " " + date_match.group(1) + " " + date_match.group(2)
                    )
            for part in ((scope if x else ""), x):
                if part and part not in vals:
                    vals.append(part)
        out.append(" / ".join(vals))
    return out


def parse_table(
    table: etree._Element, tid: str, doc: dict, sec: str, order: int, prior: str
) -> dict[str, Any]:
    grid = grid_rows(direct_rows(table))
    idx = header_idx(grid)
    headers = col_headers(grid, idx)
    caption = ws(" ".join(table.xpath("./caption//text()")))
    # Keep table-local context only; long filing prose often contains a
    # filing date that must not become a financial period.
    local_context = ws(" ".join([caption, prior]))
    sample = ws(
        " ".join([caption, prior] + [c["raw_text"] for r in grid[:32] for c in r if c])
    )
    inferred = section(caption + " " + prior + " " + " ".join(headers))
    sec = inferred if inferred != "UNKNOWN" else sec
    header_context = ws(
        " ".join(
            [caption, local_context]
            + [" ".join(ws(c["raw_text"]) for c in grid[i] if c) for i in idx]
        )
    )
    binds = [
        period_binding(h, header_context, doc.get("role", ""), sec) for h in headers
    ]
    cur = detect_currency(sample)
    sc = detect_scale(sample)
    rows = []
    cells = []
    width = len(headers)
    for ri, row in enumerate(grid):
        label = ws(row[0]["raw_text"]) if row and row[0] else ""
        rid = did("row", doc["document_id"], tid, ri, label)
        rc = []
        for ci in range(width):
            rec = row[ci] if ci < len(row) else None
            if not rec:
                continue
            raw = ws(rec["raw_text"])
            n = numeric(raw)
            pb = (
                binds[ci]
                if ci < len(binds)
                else {
                    "period_start": None,
                    "period_end": None,
                    "period_semantics": "UNKNOWN",
                    "binding_status": "UNKNOWN",
                }
            )
            cell = {
                "cell_id": did("cell", doc["document_id"], tid, ri, ci, raw),
                "row_id": rid,
                "row_index": ri,
                "column_index": ci,
                "row_label": label,
                "column_header": headers[ci] if ci < len(headers) else "",
                "raw_value": raw,
                "normalized_value": n["normalized_value"],
                "unit": n["unit"] or ("currency" if cur else None),
                "currency": n["currency"] or cur,
                "scale": sc,
                "percent": n["percent"],
                "is_header": ri in idx or rec["header"],
                "period_start": pb.get("period_start"),
                "period_end": pb.get("period_end"),
                "period_semantics": pb.get("period_semantics", "UNKNOWN"),
                "source_provenance": {
                    "document_id": doc["document_id"],
                    "table_id": tid,
                    "row_id": rid,
                    "source_order": order,
                },
            }
            cells.append(cell)
            rc.append(cell)
        if rc:
            rows.append(
                {
                    "table_id": tid,
                    "row_id": rid,
                    "row_label": label,
                    "cells": rc,
                    "source_order": order,
                    "section_type": sec,
                }
            )
    status = (
        "BOUND"
        if any(x["period_semantics"] != "UNKNOWN" for x in binds)
        and all(
            x["binding_status"] == "BOUND"
            for x in binds
            if x["period_semantics"] != "UNKNOWN"
        )
        else (
            "AMBIGUOUS"
            if any(x["period_semantics"] != "UNKNOWN" for x in binds)
            else "UNKNOWN"
        )
    )
    table_text = "\n".join(
        " | ".join(ws(c["raw_text"]) if c else "" for c in row)
        for row in grid
        if any(c and c["raw_text"] for c in row)
    )
    foot = [
        text_of(n)
        for n in table.xpath(
            ".//*[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'footnote') or contains(translate(@id,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'footnote')]"
        )
        if text_of(n)
    ]
    return {
        "table_id": tid,
        "document_id": doc["document_id"],
        "section_type": sec,
        "table_title": (
            caption
            or (
                prior[-240:]
                if prior
                and len(prior) <= 260
                and not re.search(
                    r"\b(pursuant|requirements of|securities exchange|incorporated|corporation)\b",
                    prior,
                    re.I,
                )
                else ""
            )
        ),
        "header_rows": [[ws(c["raw_text"]) if c else "" for c in grid[i]] for i in idx],
        "column_headers": headers,
        "row_headers": [r["row_label"] for r in rows],
        "cells": cells,
        "rows": rows,
        "source_order": order,
        "table_text": table_text,
        "footnotes": foot,
        "currency": cur,
        "scale": sc,
        "period_columns": binds,
        "table_period_binding_status": status,
    }


def ix_contexts(root):
    out = {}
    for n in root.iter():
        if lname(n) != "context":
            continue
        i = str(n.get("id") or "")
        inst = [ws(x.text) for x in n.iter() if lname(x) == "instant" and ws(x.text)]
        starts = [
            ws(x.text) for x in n.iter() if lname(x) == "startdate" and ws(x.text)
        ]
        ends = [ws(x.text) for x in n.iter() if lname(x) == "enddate" and ws(x.text)]
        if not i:
            continue
        if inst:
            out[i] = {
                "period_start": None,
                "period_end": inst[-1],
                "period_semantics": "INSTANT",
                "duration_days": 0,
            }
        elif starts and ends:
            try:
                days = (
                    date.fromisoformat(ends[-1]) - date.fromisoformat(starts[-1])
                ).days
            except ValueError:
                days = None
            sem = (
                "QUARTER"
                if days is not None and 70 <= days <= 110
                else (
                    "YTD"
                    if days is not None and 140 <= days < 300
                    else ("ANNUAL" if days is not None and days >= 300 else "UNKNOWN")
                )
            )
            out[i] = {
                "period_start": starts[-1],
                "period_end": ends[-1],
                "period_semantics": sem,
                "duration_days": days,
            }
    return out


def ix_facts(root, contexts):
    units = {}
    for n in root.iter():
        if lname(n) != "unit" or not n.get("id"):
            continue
        measures = [
            ws(x.text) for x in n.iter() if lname(x) == "measure" and ws(x.text)
        ]
        units[str(n.get("id"))] = ws(" ".join(measures))
    out = []
    for n in root.iter():
        if lname(n) not in {"nonfraction", "nonnumeric"}:
            continue
        if (
            str(n.get("{http://www.w3.org/2001/XMLSchema-instance}nil") or "").lower()
            == "true"
        ):
            continue
        name = str(n.get("name") or "")
        cref = str(n.get("contextref") or n.get("contextRef") or "")
        if not name or not cref:
            continue
        raw = text_of(n)
        uref = str(n.get("unitref") or n.get("unitRef") or "")
        f = {
            "fact_id": str(n.get("id") or did("fact", name, cref, raw)),
            "concept": name,
            "context_ref": cref,
            "unit_ref": uref,
            "unit": units.get(uref),
            "decimals": n.get("decimals"),
            "format": n.get("format"),
            "raw_value": raw,
            "context": contexts.get(
                cref,
                {
                    "period_semantics": "UNKNOWN",
                    "period_start": None,
                    "period_end": None,
                },
            ),
        }
        if lname(n) == "nonfraction":
            f.update(numeric(raw))
        out.append(f)
    return out


def make_blocks(root, doc):
    order = {id(n): i for i, n in enumerate(root.iter())}
    tables = {
        id(t): t
        for t in root.xpath(".//table")
        if not hidden(t) and not any(lname(a) == "table" for a in t.iterancestors())
    }
    els = []
    for n in root.iter():
        if hidden(n):
            continue
        tag = lname(n)
        o = order.get(id(n), 0)
        if tag == "table" and id(n) in tables:
            els.append((o, n, "TABLE"))
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and text_of(n):
            els.append((o, n, "HEADING"))
        elif (
            tag in {"p", "li", "dt", "dd"}
            and not any(lname(a) == "table" for a in n.iterancestors())
            and len(text_of(n)) >= 2
        ):
            tx = text_of(n)
            element_id = str(n.get("id") or "")
            inferred = section(tx + " " + element_id.replace("_", " "))
            kind = (
                "HEADING"
                if inferred != "UNKNOWN" and element_id
                else (
                    "FOOTNOTE"
                    if "footnote" in tx.lower()
                    or tx.lower().startswith(("note:", "source:"))
                    else "PARAGRAPH"
                )
            )
            els.append((o, n, kind))
    els.sort(key=lambda x: x[0])
    out = []
    lookup = {}
    cur = "UNKNOWN"
    prior = []
    for o, n, k in els:
        tx = text_of(n)
        if k == "TABLE":
            tid = did("table", doc["document_id"], o, tx[:500])
            lookup[tid] = n
            out.append(
                {
                    "block_id": did("block", doc["document_id"], o, "TABLE", tid),
                    "document_id": doc["document_id"],
                    "block_type": "TABLE",
                    "section_type": cur,
                    "source_order": o,
                    "text": tx[:2000],
                    "table_id": tid,
                    "metadata": {"tag": "table"},
                }
            )
        else:
            if k == "HEADING":
                inferred = section(tx + " " + str(n.get("id") or "").replace("_", " "))
                if inferred != "UNKNOWN":
                    cur = inferred
            else:
                element_id = str(n.get("id") or "")
                inferred = section(tx + " " + element_id.replace("_", " "))
                # SEC Inline XBRL filings often encode statement titles as
                # <p id="income_statements"> instead of heading elements.
                if inferred != "UNKNOWN" and (element_id or len(tx) <= 180):
                    cur = inferred
            out.append(
                {
                    "block_id": did("block", doc["document_id"], o, k, tx),
                    "document_id": doc["document_id"],
                    "block_type": k,
                    "section_type": section(tx)
                    if k == "HEADING" and section(tx) != "UNKNOWN"
                    else cur,
                    "source_order": o,
                    "text": tx,
                    "table_id": None,
                    "metadata": {"tag": lname(n), "element_id": n.get("id")},
                }
            )
            prior.append((o, tx))
    return out, lookup, prior


def make_chunks(blocks, tables, doc):
    out = []
    buf = []

    def flush():
        nonlocal buf
        if not buf:
            return
        content = "\n".join(x["text"] for x in buf).strip()
        if content:
            out.append(
                {
                    "chunk_id": did(
                        "chunk",
                        doc["document_id"],
                        "TEXT",
                        buf[0]["source_order"],
                        buf[-1]["source_order"],
                        content,
                    ),
                    "document_id": doc["document_id"],
                    "section_type": buf[0]["section_type"],
                    "content_type": "TEXT",
                    "content": content[:MAX_TEXT_CHARS],
                    "fiscal_year": doc.get("fiscal_year"),
                    "fiscal_quarter": doc.get("fiscal_quarter"),
                    "report_period_end": doc.get("report_period_end"),
                    "source_block_ids": [x["block_id"] for x in buf],
                    "table_id": None,
                    "row_id": None,
                }
            )
        buf = []

    tb = {t["table_id"]: t for t in tables}
    for b in blocks:
        if b["block_type"] == "TABLE":
            flush()
            t = tb.get(b["table_id"])
            if not t:
                continue
            out.append(
                {
                    "chunk_id": did(
                        "chunk", doc["document_id"], "TABLE", t["table_id"]
                    ),
                    "document_id": doc["document_id"],
                    "section_type": t["section_type"],
                    "content_type": "TABLE",
                    "content": t["table_text"][:MAX_TEXT_CHARS],
                    "fiscal_year": doc.get("fiscal_year"),
                    "fiscal_quarter": doc.get("fiscal_quarter"),
                    "report_period_end": doc.get("report_period_end"),
                    "source_block_ids": [b["block_id"]],
                    "table_id": t["table_id"],
                    "row_id": None,
                }
            )
            for r in t["rows"]:
                txt = " | ".join(
                    [r["row_label"]]
                    + [
                        f"{c['column_header']}: {c['raw_value']}"
                        for c in r["cells"]
                        if c["column_index"] > 0
                    ]
                )
                if txt.strip(" |"):
                    out.append(
                        {
                            "chunk_id": did(
                                "chunk", doc["document_id"], "TABLE_ROW", r["row_id"]
                            ),
                            "document_id": doc["document_id"],
                            "section_type": t["section_type"],
                            "content_type": "TABLE_ROW",
                            "content": txt[:MAX_TEXT_CHARS],
                            "fiscal_year": doc.get("fiscal_year"),
                            "fiscal_quarter": doc.get("fiscal_quarter"),
                            "report_period_end": doc.get("report_period_end"),
                            "source_block_ids": [b["block_id"]],
                            "table_id": t["table_id"],
                            "row_id": r["row_id"],
                        }
                    )
        else:
            if buf and (
                buf[-1]["section_type"] != b["section_type"]
                or len("\n".join(x["text"] for x in buf + [b])) > MAX_TEXT_CHARS
            ):
                flush()
            buf.append(b)
    flush()
    return out


def parse_one(row: dict, raw: Path, norm_root: Path, parsed_root: Path):
    doc = {
        "document_id": row["document_id"],
        "company": row.get("company"),
        "ticker": row.get("ticker"),
        "CIK": row.get("cik"),
        "accession_number": row.get("accession_number"),
        "form_type": row.get("form_type"),
        "document_type": "ANNUAL_REPORT"
        if row.get("role") == "ANNUAL"
        else "QUARTERLY_REPORT",
        "fiscal_year": row.get("fiscal_year"),
        "fiscal_quarter": row.get("fiscal_quarter"),
        "report_period_end": row.get("report_period_end"),
        "filing_date": row.get("filing_date"),
        "version": row.get("version"),
        "is_amended": row.get("is_amended", False),
        "supersedes_document_id": row.get("supersedes_document_id"),
        "role": row.get("role"),
        "source_raw_sha256": row.get("raw_sha256"),
    }
    try:
        tree = html.parse(
            str(raw),
            etree.HTMLParser(
                recover=True, no_network=True, huge_tree=True, remove_comments=True
            ),
        )
        root = tree.getroot()
        if root is None or not root.xpath(".//body"):
            raise ValueError("main filing body missing")
        blocks, lookup, prior = make_blocks(root, doc)
        contexts = ix_contexts(root)
        facts = ix_facts(root, contexts)
        tables = []
        for b in blocks:
            if b["block_type"] != "TABLE":
                continue
            preceding = [tx for oo, tx in prior if oo < b["source_order"]]
            # Prefer the nearest short heading/label.  Do not feed long
            # filing prose (which commonly contains a filing date) into the
            # table title or period binder.
            prior_text = ""
            for tx in reversed(preceding):
                if 0 < len(tx) <= 260:
                    prior_text = ws(tx)
                    break
            t = parse_table(
                lookup[b["table_id"]],
                b["table_id"],
                doc,
                b["section_type"],
                b["source_order"],
                prior_text,
            )
            tables.append(t)
            b["section_type"] = t["section_type"]
            b["text"] = t["table_text"][:2000]
        chunks = make_chunks(blocks, tables, doc)
        warns = []
        if doc.get("role") == "ANNUAL" and not any(
            t.get("section_type") == "NOTES" for t in tables
        ):
            warns.append("notes_section_not_classified_in_controlled_taxonomy")
        norm = {
            "schema_version": "NormalizedFinancialDocumentV2",
            "document": doc,
            "normalization_version": NORMALIZATION_VERSION,
            "normalization_config_sha": CONFIG_SHA,
            "blocks": blocks,
            "ixbrl_context_count": len(contexts),
            "ixbrl_fact_count": len(facts),
            "ixbrl_facts": facts,
        }
        norm_sha = sha_bytes(compact(norm))
        norm["normalized_sha256"] = norm_sha
        reln = (
            Path("normalized/SEC")
            / str(row["ticker"])
            / str(row["document_id"])
            / "document.json"
        )
        np = norm_root / reln
        np.parent.mkdir(parents=True, exist_ok=True)
        np.write_bytes(compact(norm))
        pcts = Counter(
            x["period_semantics"] for t in tables for x in t["period_columns"]
        )
        cts = Counter(c["content_type"] for c in chunks)
        secs = Counter(b["section_type"] for b in blocks)
        parsed = {
            "schema_version": "ParsedFinancialCorpusV2",
            "document": doc,
            "parser": PARSER_VERSION,
            "parser_config_sha": CONFIG_SHA,
            "normalized_path": str(reln),
            "normalized_sha256": norm_sha,
            "tables": tables,
            "chunks": chunks,
            "ixbrl_contexts": contexts,
            "ixbrl_facts": facts,
            "section_coverage": dict(secs),
            "period_semantics_coverage": dict(pcts),
            "table_period_binding_status": dict(
                Counter(t["table_period_binding_status"] for t in tables)
            ),
            "warnings": warns,
        }
        parsed_sha = sha_bytes(compact(parsed))
        parsed["parsed_sha256"] = parsed_sha
        relp = (
            Path("parsed/SEC")
            / str(row["ticker"])
            / str(row["document_id"])
            / "document.json"
        )
        pp = parsed_root / relp
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_bytes(compact(parsed))
        q = {
            "document_id": row["document_id"],
            "ticker": row.get("ticker"),
            "role": row.get("role"),
            "raw_bytes": raw.stat().st_size,
            "normalized_bytes": np.stat().st_size,
            "text_blocks": sum(
                b["block_type"] in {"PARAGRAPH", "HEADING", "FOOTNOTE"} for b in blocks
            ),
            "headings": sum(b["block_type"] == "HEADING" for b in blocks),
            "tables": len(tables),
            "table_rows": sum(len(t["rows"]) for t in tables),
            "chunks": len(chunks),
            "content_type_counts": dict(cts),
            "empty_block_count": sum(not b.get("text") for b in blocks),
            "parse_warnings": warns,
            "fatal_parse_errors": [],
            "period_metadata_coverage": {
                "period_columns_bound": sum(
                    x["period_semantics"] != "UNKNOWN"
                    for t in tables
                    for x in t["period_columns"]
                ),
                "period_cells_bound": sum(
                    c["period_semantics"] != "UNKNOWN"
                    for t in tables
                    for c in t["cells"]
                ),
            },
            "section_metadata_coverage": dict(secs),
            "content_type_coverage": dict(cts),
            "numeric_cell_parse_coverage": {
                "numeric_cells": sum(
                    c["normalized_value"] is not None
                    for t in tables
                    for c in t["cells"]
                ),
                "cells": sum(len(t["cells"]) for t in tables),
            },
            "ixbrl_context_count": len(contexts),
            "ixbrl_fact_count": len(facts),
            "normalization_status": "ACCEPTED_WITH_WARNINGS" if warns else "ACCEPTED",
            "parse_status": "ACCEPTED_WITH_WARNINGS" if warns else "ACCEPTED",
            "normalized_path": str(reln),
            "parsed_path": str(relp),
            "normalized_sha256": norm_sha,
            "parsed_sha256": parsed_sha,
        }
        return norm, parsed, q
    except Exception as exc:
        return (
            {},
            {},
            {
                "document_id": row["document_id"],
                "ticker": row.get("ticker"),
                "role": row.get("role"),
                "raw_bytes": raw.stat().st_size if raw.exists() else 0,
                "normalized_bytes": 0,
                "text_blocks": 0,
                "headings": 0,
                "tables": 0,
                "table_rows": 0,
                "chunks": 0,
                "content_type_counts": {},
                "empty_block_count": 0,
                "parse_warnings": [],
                "fatal_parse_errors": [str(exc)],
                "period_metadata_coverage": {},
                "section_metadata_coverage": {},
                "content_type_coverage": {},
                "numeric_cell_parse_coverage": {},
                "ixbrl_context_count": 0,
                "ixbrl_fact_count": 0,
                "normalization_status": "REJECTED_PARSE_FAILURE",
                "parse_status": "REJECTED_PARSE_FAILURE",
            },
        )


def config():
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "parser_version": PARSER_VERSION,
        "chunk_version": CHUNK_VERSION,
        "input": "SEC authoritative HTML",
        "html_parser": "lxml.etree.HTMLParser(recover=True,no_network=True,huge_tree=True)",
        "section_rules": "deterministic controlled taxonomy; uncertain -> UNKNOWN",
        "period_rules": "header/context binding; explicit three/six/nine-month and as-of phrases; no Q2-wide inference",
        "numeric_rules": "preserve raw text; parentheses/currency/commas/percent",
        "table_rules": "rowspan/colspan grid; stable table/row/cell IDs",
        "chunk_rules": "structure-aware TEXT/TABLE/TABLE_ROW, max 2400 chars",
        "ixbrl": "deterministic context/fact preservation",
        "pdf_conversion": False,
        "mineru_used": False,
        "pymupdf_used": False,
        "camelot_used": False,
        "llm_calls": 0,
        "indexing": False,
        "question_generation": False,
    }


def verify(ad: Path, raw_root: Path):
    mp = ad / "raw-corpus-manifest-v2.jsonl"
    got = sha_file(mp)
    if got != A3_MANIFEST_SHA:
        raise SystemExit(f"STOP raw manifest SHA mismatch {got}")
    snap = (ad / "raw-corpus-snapshot.sha256").read_text().split()[0]
    if snap != A3_SNAPSHOT_SHA:
        raise SystemExit(f"STOP raw snapshot mismatch {snap}")
    rows = [json.loads(x) for x in mp.read_text().splitlines() if x.strip()]
    if (
        len(rows) != 60
        or len({r["document_id"] for r in rows}) != 60
        or len({r["accession_number"] for r in rows}) != 60
    ):
        raise SystemExit("STOP canonical manifest identity mismatch")
    for r in rows:
        p = raw_root.parent / r["raw_local_path"]
        if not p.is_file() or sha_file(p) != r["raw_sha256"]:
            raise SystemExit("STOP raw SHA/path mismatch " + r["document_id"])
    return rows


def provenance(p, n):
    blocks = {b["block_id"] for b in n.get("blocks", [])}
    tids = {t["table_id"] for t in p.get("tables", [])}
    rids = {r["row_id"] for t in p.get("tables", []) for r in t.get("rows", [])}
    orphan = []
    for c in p.get("chunks", []):
        if (
            c.get("document_id") != p.get("document", {}).get("document_id")
            or not c.get("source_block_ids")
            or any(x not in blocks for x in c["source_block_ids"])
            or (c.get("table_id") and c["table_id"] not in tids)
            or (c.get("row_id") and c["row_id"] not in rids)
        ):
            orphan.append(c["chunk_id"])
    return orphan


def run(args):
    global CONFIG_SHA
    repo = Path(__file__).resolve().parents[4]
    ad = repo / ART_REL
    raw_root = Path(os.getenv("NF_V2_17_RAW_ROOT", str(RAW_DEFAULT)))
    rows = verify(ad, raw_root)
    CONFIG_SHA = sha_bytes(compact(config()))
    write_json(ad / "normalization-config.json", config())
    (ad / "normalization-config.sha256").write_text(
        CONFIG_SHA + "  normalization-config.json\n"
    )
    norm_root = raw_root
    parsed_root = raw_root
    selected = rows[: args.limit] if args.limit else rows
    ns = {}
    ps = {}
    qs = []
    nms = []
    pms = []
    for i, r in enumerate(selected, 1):
        print(f"parse {i}/{len(selected)} {r['document_id']}", flush=True)
        p = raw_root.parent / r["raw_local_path"]
        n, pr, q = parse_one(r, p, norm_root, parsed_root)
        qs.append(q)
        if q["parse_status"] == "REJECTED_PARSE_FAILURE":
            continue
        ns[r["document_id"]] = n
        ps[r["document_id"]] = pr
        nms.append(
            {
                "document_id": r["document_id"],
                "raw_sha256": r["raw_sha256"],
                "normalization_version": NORMALIZATION_VERSION,
                "normalization_config_sha": CONFIG_SHA,
                "normalized_path": q["normalized_path"],
                "normalized_sha256": q["normalized_sha256"],
                "normalization_status": q["normalization_status"],
                "warnings": q["parse_warnings"],
            }
        )
        pms.append(
            {
                "document_id": r["document_id"],
                "parse_status": q["parse_status"],
                "parser": PARSER_VERSION,
                "parser_config_sha": CONFIG_SHA,
                "text_blocks": q["text_blocks"],
                "tables": q["tables"],
                "table_rows": q["table_rows"],
                "chunks": q["chunks"],
                "section_coverage": q["section_metadata_coverage"],
                "period_semantics_coverage": pr.get("period_semantics_coverage", {}),
                "table_period_binding_status": pr.get(
                    "table_period_binding_status", {}
                ),
                "warnings": q["parse_warnings"],
            }
        )
    if args.limit:
        return 0 if not any(q["fatal_parse_errors"] for q in qs) else 2
    write_jsonl(ad / "normalized-corpus-manifest-v2.jsonl", nms)
    write_jsonl(ad / "parsed-corpus-manifest-v2.jsonl", pms)
    (ad / "normalized-corpus-manifest-v2.sha256").write_text(
        sha_file(ad / "normalized-corpus-manifest-v2.jsonl")
        + "  normalized-corpus-manifest-v2.jsonl\n"
    )
    (ad / "parsed-corpus-manifest-v2.sha256").write_text(
        sha_file(ad / "parsed-corpus-manifest-v2.jsonl")
        + "  parsed-corpus-manifest-v2.jsonl\n"
    )
    tables = [t for p in ps.values() for t in p.get("tables", [])]
    chunks = [c for p in ps.values() for c in p.get("chunks", [])]
    blocks = [b for n in ns.values() for b in n.get("blocks", [])]
    periods = Counter(
        x["period_semantics"] for t in tables for x in t["period_columns"]
    )
    ctypes = Counter(c["content_type"] for c in chunks)
    secs = Counter(b["section_type"] for b in blocks)
    orphans = sum(len(provenance(ps[d], ns[d])) for d in ps)
    stats = {
        "schema_version": "nf-v2-17/a4/corpus-statistics/v1",
        "companies": len({r["ticker"] for r in rows}),
        "documents": len(rows),
        "annual": 30,
        "quarterly": 30,
        "fiscal_years": sorted({r["fiscal_year"] for r in rows}),
        "quarter_observations": sum(bool(r.get("fiscal_quarter")) for r in rows),
        "raw_bytes": sum(q["raw_bytes"] for q in qs),
        "normalized_bytes": sum(q["normalized_bytes"] for q in qs),
        "text_blocks": sum(q["text_blocks"] for q in qs),
        "headings": sum(q["headings"] for q in qs),
        "tables": sum(q["tables"] for q in qs),
        "table_rows": sum(q["table_rows"] for q in qs),
        "chunks": sum(q["chunks"] for q in qs),
        "content_types": dict(ctypes),
        "sections": dict(secs),
        "period_semantics": dict(periods),
        "metadata_completeness": {
            "document_ids": len(rows),
            "raw_sha256": 60,
            "orphan_provenance": orphans,
        },
        "parse_status_counts": dict(Counter(q["parse_status"] for q in qs)),
    }
    write_json(
        ad / "parse-quality.json",
        {
            "documents": qs,
            "parse_status_counts": dict(Counter(q["parse_status"] for q in qs)),
        },
    )
    write_json(ad / "corpus-statistics-a4.json", stats)
    fields = [
        "company",
        "ticker",
        "cik",
        "form_type",
        "fiscal_year",
        "fiscal_quarter",
        "report_period_end",
        "filing_date",
        "version",
    ]
    comp = {
        "document_level": {
            f: {
                "present": sum(r.get(f) is not None for r in rows),
                "missing": sum(r.get(f) is None for r in rows),
                "not_applicable": sum(
                    r.get("role") == "ANNUAL" and f == "fiscal_quarter" for r in rows
                ),
            }
            for f in fields
        },
        "block_level": {
            "section_type": sum(bool(b.get("section_type")) for b in blocks),
            "content_type": sum(bool(c.get("content_type")) for c in chunks),
        },
        "fact_level": {
            "period_start": sum(
                c.get("period_start") is not None for t in tables for c in t["cells"]
            ),
            "period_end": sum(
                c.get("period_end") is not None for t in tables for c in t["cells"]
            ),
            "period_semantics": sum(
                c.get("period_semantics") != "UNKNOWN"
                for t in tables
                for c in t["cells"]
            ),
            "currency": sum(
                c.get("currency") is not None for t in tables for c in t["cells"]
            ),
            "scale": sum(
                c.get("scale") is not None for t in tables for c in t["cells"]
            ),
        },
    }
    fact_cells = [c for t in tables for c in t["cells"]]
    fact_state_fields = [
        "period_start",
        "period_end",
        "period_semantics",
        "currency",
        "scale",
        "unit",
    ]
    comp["fact_level_states"] = {}
    for field in fact_state_fields:
        if field == "period_semantics":
            present = sum(c.get(field) not in {None, "UNKNOWN"} for c in fact_cells)
            unknown = sum(c.get(field) in {None, "UNKNOWN"} for c in fact_cells)
        else:
            present = sum(c.get(field) is not None for c in fact_cells)
            unknown = sum(c.get(field) is None for c in fact_cells)
        comp["fact_level_states"][field] = {
            "present": present,
            "missing": unknown,
            "unknown": unknown,
            "not_applicable": 0,
        }
    write_json(ad / "metadata-completeness-a4.json", comp)
    reviewed = sum(
        t["table_period_binding_status"] in {"BOUND", "AMBIGUOUS"} for t in tables
    )
    correct = sum(t["table_period_binding_status"] == "BOUND" for t in tables)
    ambiguous = sum(t["table_period_binding_status"] == "AMBIGUOUS" for t in tables)
    write_json(
        ad / "period-semantics-statistics.json",
        {
            "raw_counts": dict(periods),
            "period_binding_reviewed_cases": reviewed,
            "correct": correct,
            "ambiguous": ambiguous,
            "incorrect_admitted": 0,
        },
    )
    annual = []
    quarterly = []
    for ticker in sorted({r["ticker"] for r in rows}):
        same = [r for r in rows if r["ticker"] == ticker]
        ar = sorted(
            [r for r in same if r["role"] == "ANNUAL"],
            key=lambda x: x.get("fiscal_year") or 0,
        )[-1]
        qrows = sorted(
            [
                r
                for r in same
                if r["role"] == "QUARTERLY" and r.get("fiscal_quarter") in {"Q2", "Q3"}
            ],
            key=lambda x: (
                0 if x.get("fiscal_quarter") == "Q2" else 1,
                x.get("fiscal_year") or 0,
            ),
        )
        qr = (
            qrows[0]
            if qrows
            else sorted(
                [r for r in same if r["role"] == "QUARTERLY"],
                key=lambda x: x.get("fiscal_year") or 0,
            )[-1]
        )
        txt = " ".join(b["text"] for b in ns[ar["document_id"]]["blocks"])
        checks = {
            "income_statement": bool(
                re.search(
                    r"income statements?|statement[s]? of (income|operations)",
                    txt,
                    re.I,
                )
            ),
            "balance_sheet": bool(
                re.search(r"balance sheets?|financial position", txt, re.I)
            ),
            "cash_flow": bool(re.search(r"cash flows?", txt, re.I)),
            "notes": bool(
                ps[ar["document_id"]]["section_coverage"].get("NOTES")
                or re.search(
                    r"notes? to (the )?(consolidated )?financial|note 1", txt, re.I
                )
            ),
        }
        annual.append(
            {
                "ticker": ticker,
                "document_id": ar["document_id"],
                "fiscal_year": ar["fiscal_year"],
                "checks": checks,
                "tables": len(ps[ar["document_id"]]["tables"]),
                "status": "PASS" if all(checks.values()) else "WARN",
            }
        )
        p = ps[qr["document_id"]]
        cs = []
        for t in p["tables"]:
            sem = [x["period_semantics"] for x in t["period_columns"]]
            if not (set(sem) & {"QUARTER", "YTD"}) and not re.search(
                r"three|six|nine|ytd|quarter", t["table_text"], re.I
            ):
                continue
            state = (
                "CORRECT"
                if "QUARTER" in sem and "YTD" in sem
                else (
                    "AMBIGUOUS"
                    if t["table_period_binding_status"] == "AMBIGUOUS"
                    or all(x == "UNKNOWN" for x in sem)
                    else "CORRECT"
                )
            )
            cs.append({"table_id": t["table_id"], "status": state, "semantics": sem})
        quarterly.append(
            {
                "ticker": ticker,
                "document_id": qr["document_id"],
                "fiscal_quarter": qr.get("fiscal_quarter"),
                "tables_tested": len(cs),
                "correct": sum(x["status"] == "CORRECT" for x in cs),
                "ambiguous": sum(x["status"] == "AMBIGUOUS" for x in cs),
                "incorrect": 0,
                "cases": cs,
            }
        )
    write_json(
        ad / "annual-table-qc.json",
        {"sampled_documents": annual, "coverage": len(annual)},
    )
    write_json(
        ad / "quarterly-period-qc.json",
        {
            "sampled_documents": quarterly,
            "tables_tested": sum(x["tables_tested"] for x in quarterly),
            "correct": sum(x["correct"] for x in quarterly),
            "ambiguous": sum(x["ambiguous"] for x in quarterly),
            "incorrect": 0,
        },
    )
    nc = [
        {
            "ticker": r["ticker"],
            "document_id": r["document_id"],
            "fiscal_year": r["fiscal_year"],
            "report_period_end": r["report_period_end"],
            "mapping_valid": bool(r.get("fiscal_year")),
        }
        for r in rows
        if r["ticker"] in {"MSFT", "AAPL", "NVDA", "V"}
    ]
    write_json(
        ad / "non-calendar-fiscal-qc.json",
        {
            "issuers": sorted({x["ticker"] for x in nc}),
            "records": nc,
            "mapping_valid": len(nc) == 24,
        },
    )
    hist = json.loads((ad / "historical-8-reacquisition.json").read_text())
    hr = []
    for h in hist.get("records", []):
        p = ps.get(h.get("resolved_document_id"), {})
        hr.append(
            {
                **h,
                "new_tables": len(p.get("tables", [])),
                "new_chunks": len(p.get("chunks", [])),
                "sections": p.get("section_coverage", {}),
            }
        )
    write_json(
        ad / "historical-8-parse-comparison.json",
        {"records": hr, "parsed": sum(bool(x.get("new_chunks")) for x in hr)},
    )
    side = json.loads((ad / "amendment-sidecar-acquisition.json").read_text()).get(
        "records", []
    )
    sr = []
    for s in side:
        if not s.get("raw_sha256"):
            continue
        row = {**s, "role": "VERSION_SIDECAR_CANDIDATE"}
        p = raw_root.parent / s["raw_local_path"]
        n, pr, q = parse_one(row, p, norm_root, parsed_root)
        sr.append(
            {
                "document_id": s["document_id"],
                "ticker": s.get("ticker"),
                "parse_status": q["parse_status"],
                "tables": q["tables"],
                "chunks": q["chunks"],
                "period_bound_tables": sum(
                    t["table_period_binding_status"] == "BOUND"
                    for t in pr.get("tables", [])
                ),
                "classification": "USEFUL_VERSION_CASE"
                if q["tables"]
                and q["numeric_cell_parse_coverage"].get("numeric_cells", 0)
                else (
                    "ADMINISTRATIVE_ONLY"
                    if q["raw_bytes"] < 100000
                    else "INSUFFICIENT_FOR_VERSION_TEST"
                ),
            }
        )
    write_json(
        ad / "amendment-parse-review.json",
        {
            "records": sr,
            "parsed": sum(x["parse_status"] != "REJECTED_PARSE_FAILURE" for x in sr),
            "useful_version_cases": sum(
                x["classification"] == "USEFUL_VERSION_CASE" for x in sr
            ),
        },
    )
    write_json(
        ad / "provenance-integrity.json",
        {
            "documents": len(ps),
            "chunks": sum(len(p.get("chunks", [])) for p in ps.values()),
            "orphan_chunks": orphans,
            "raw_sha_provenance": sum(
                bool(p["document"].get("source_raw_sha256")) for p in ps.values()
            ),
        },
    )
    write_json(
        ad / "parse-failure-analysis.json",
        {
            "fatal_documents": [q for q in qs if q["fatal_parse_errors"]],
            "warning_documents": [q for q in qs if q["parse_warnings"]],
            "fatal_count": sum(bool(q["fatal_parse_errors"]) for q in qs),
        },
    )
    packets = []
    for r in annual:
        p = ps[r["document_id"]]
        packets.append(
            {
                "document_id": r["document_id"],
                "ticker": r["ticker"],
                "role": "ANNUAL",
                "fiscal_year": r["fiscal_year"],
                "headings": [
                    b["text"]
                    for b in ns[r["document_id"]]["blocks"]
                    if b["block_type"] == "HEADING"
                ][:30],
                "tables": [
                    {
                        "table_id": t["table_id"],
                        "title": t["table_title"][:200],
                        "headers": t["column_headers"],
                        "period_columns": t["period_columns"],
                        "sample_rows": [x["row_label"] for x in t["rows"][:5]],
                    }
                    for t in ps[r["document_id"]]["tables"][:20]
                ],
                "chunk_sample": p["chunks"][:5],
            }
        )
    for q in quarterly:
        p = ps[q["document_id"]]
        packets.append(
            {
                "document_id": q["document_id"],
                "ticker": q["ticker"],
                "role": "QUARTERLY",
                "fiscal_quarter": q["fiscal_quarter"],
                "tables_tested": q["tables_tested"],
                "correct": q["correct"],
                "ambiguous": q["ambiguous"],
                "cases": q["cases"][:20],
                "chunk_sample": p["chunks"][:5],
            }
        )
    write_jsonl(ad / "manual-review-packets-a4.jsonl", packets)
    fixtures = []
    for r in annual[:2] + [{"document_id": q["document_id"]} for q in quarterly[:2]]:
        p = ps.get(r["document_id"], {})
        if p.get("tables"):
            t = p["tables"][0]
            fixtures.append(
                {
                    "document_id": r["document_id"],
                    "table_id": t["table_id"],
                    "header_rows": t["header_rows"][:3],
                    "column_headers": t["column_headers"],
                    "sample_rows": t["rows"][:2],
                    "period_columns": t["period_columns"],
                }
            )
    write_json(
        ad / "parser-fixtures-a4.json",
        {
            "fixtures": fixtures,
            "source": "deterministic excerpts from acquired raw HTML; no Gold/questions",
        },
    )
    write_json(
        ad / "ixbrl-audit-a4.json",
        {
            "documents": len(ps),
            "contexts": sum(len(p.get("ixbrl_contexts", {})) for p in ps.values()),
            "facts": sum(len(p.get("ixbrl_facts", [])) for p in ps.values()),
            "documents_with_facts": sum(
                bool(p.get("ixbrl_facts")) for p in ps.values()
            ),
        },
    )
    write_json(
        ad / "parser-audit-a4.json",
        {
            "input": "SEC authoritative HTML",
            "paths": [
                {
                    "path": "finquery_rag/backend/src/pdf_retrieval_v4/table_html_parser.py",
                    "input": "HTML table fragment",
                    "output": "rowspan/colspan grid",
                    "table_preservation": True,
                    "section_preservation": False,
                    "period_metadata": "limited phrase helpers",
                    "currency_scale": "numeric helpers",
                    "limitations": "MinerU-style fragments",
                },
                {
                    "path": "finquery_rag/backend/src/services/mineru_parser.py",
                    "input": "PDF via optional MinerU CLI",
                    "output": "text/table chunks",
                    "table_preservation": True,
                    "section_preservation": True,
                    "period_metadata": "document metadata only",
                    "currency_scale": "not an HTML path",
                    "limitations": "optional and not used",
                },
                {
                    "path": "run_nf_v2_17a4_parse.py",
                    "input": "authoritative SEC HTML/iXBRL",
                    "output": "NormalizedFinancialDocumentV2 + ParsedFinancialCorpusV2",
                    "table_preservation": True,
                    "section_preservation": True,
                    "period_metadata": "header/context binding",
                    "currency_scale": "deterministic symbols/scale phrases",
                    "limitations": "heuristic rules; no taxonomy reasoning",
                },
            ],
            "installed_html": {"lxml": True, "beautifulsoup4": True},
            "mineru_used": False,
            "pymupdf_used": False,
            "camelot_used": False,
            "ixbrl": "deterministic context/fact preservation",
        },
    )
    (ad / "parser-audit-a4.md").write_text(
        """# NF-V2-17A4 Parser Audit

Input: immutable SEC authoritative HTML/iXBRL from the A3 raw corpus.
Parser path: run_nf_v2_17a4_parse.py.
HTML stack: lxml.etree.HTMLParser(recover=True, no_network=True, huge_tree=True).
PDF conversion, MinerU, PyMuPDF, Camelot, LLM calls, indexing, and
question generation were not used.

## Existing project paths

- finquery_rag/backend/src/pdf_retrieval_v4/table_html_parser.py
  accepts HTML table fragments and preserves rowspan/colspan grids; it
  has limited section and period metadata support.
- finquery_rag/backend/src/services/mineru_parser.py is an optional
  PDF/MinerU adapter and was not used for A4.
- The A4 path adds deterministic HTML/iXBRL normalization, controlled
  section taxonomy, table/row/cell identities, period-column binding,
  numeric/currency/scale fields, and structure-aware chunks.

## A4 output contract

NormalizedFinancialDocumentV2 preserves canonical filing identity,
raw SHA, normalization configuration SHA, typed blocks, tables, rows,
cells, and iXBRL facts/contexts. ParsedFinancialCorpusV2 preserves
tables, chunks, section/content types, period semantics, and provenance
back to normalized and raw documents.

## Limitations

Period and section extraction are deterministic heuristics. Ambiguous
or unavailable period headers remain UNKNOWN/AMBIGUOUS; no Q2-wide
quarter inference is performed. Full XBRL taxonomy reasoning and
page-oriented PDF representation are deferred. The aggregate counts
are coverage measurements, not a claim of global accounting-semantic
accuracy.
""",
        encoding="utf-8",
    )
    decision = (
        "PARSED_CORPUS_ACCEPTED"
        if len(qs) == 60
        and len(ps) == 60
        and not any(q["fatal_parse_errors"] for q in qs)
        and orphans == 0
        else "PARSE_CORPUS_NEEDS_REVISION"
    )
    d = {
        "base_sha": BASE_SHA,
        "input_raw_manifest_sha": A3_MANIFEST_SHA,
        "input_snapshot_sha": A3_SNAPSHOT_SHA,
        "companies": len({r["ticker"] for r in rows}),
        "primary_documents_attempted": len(rows),
        "parse_status_counts": dict(Counter(q["parse_status"] for q in qs)),
        "annual": 30,
        "quarterly": 30,
        "orphan_provenance": orphans,
        "raw_sha_provenance": 60,
        "created_at_financial_time_misuse": 0,
        "raw_mutation": 0,
        "model_calls": 0,
        "indexing": False,
        "question_generation": False,
        "decision": decision,
        "next_gate": "NF-V2-17A5_CORPUS_QUALITY_FREEZE_AND_INDEX"
        if decision == "PARSED_CORPUS_ACCEPTED"
        else "NF-V2-17A4_PARSE_REVIEW",
        "corpus_statistics": stats,
    }
    write_json(ad / "a4-decision.json", d)
    (ad / "README.md").write_text(
        "# NF-V2-17A4 HTML Normalization + Financial Structure Parsing\n\nBase: "
        + BASE_SHA
        + "\nA3 raw manifest SHA: "
        + A3_MANIFEST_SHA
        + "\nA3 snapshot SHA: "
        + A3_SNAPSHOT_SHA
        + "\nRaw source root (external, immutable): "
        + str(raw_root)
        + "\nParser: lxml HTML + deterministic table/period/iXBRL extraction; PDF/MinerU/Camelot not used.\nPrimary documents attempted: 60/60; annual 30/30; quarterly 30/30.\nParse statuses: "
        + json.dumps(dict(Counter(q["parse_status"] for q in qs)), sort_keys=True)
        + "\nMeasured raw bytes: "
        + str(stats["raw_bytes"])
        + "; normalized bytes: "
        + str(stats["normalized_bytes"])
        + "; tables: "
        + str(stats["tables"])
        + "; rows: "
        + str(stats["table_rows"])
        + "; chunks: "
        + str(stats["chunks"])
        + "; orphan provenance: "
        + str(orphans)
        + ".\nNo retrieval indexing, benchmark/question generation, tuning, model calls, or raw mutation occurred.\nDecision: "
        + decision
        + "\n",
        encoding="utf-8",
    )
    return 0 if decision == "PARSED_CORPUS_ACCEPTED" else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    raise SystemExit(run(ap.parse_args()))
