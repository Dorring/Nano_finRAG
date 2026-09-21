# VERIFY-01 — Reconcile the SEC HTML corpus file inventory

**Audience:** an independent agent/engineer, with access to the 4090 benchmark host.
**Deliverable:** a filled `corpus-inventory.json` plus a short verdict on three
orphaned normalized documents.
**Estimated effort:** small. This is counting and metadata reading, not analysis.

---

## 1. What was claimed, and what is on disk

The corpus `financial_corpus_v2` was reported as migrated and verified as:

> 63 raw SEC EDGAR HTML filings, 126 files (HTML + metadata), 63 normalized
> `document.json`, all extracted without loss.

A recount on 2026-09-20 found:

| | Reported | Counted on disk |
|---|---|---|
| raw filings | 63 | **60** |
| raw files total | 126 | **120** |
| normalized `document.json` | 63 | **63** |

The raw layout is perfectly uniform — **10 companies × 6 filing directories = 60** —
so the raw side looks intentional rather than truncated. The number 63 belongs to the
**normalized** side only.

**The discrepancy is 3 normalized documents with no raw counterpart.** Determine
whether that is correct and intended, or whether raw material is missing.

---

## 2. Paths

```
/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC/<TICKER>/<ACCESSION>/{primary.html, source-metadata.json}
/disk/qh/nano-finrag/data/financial_corpus_v2/normalized/SEC/<TICKER>/<ACCESSION>/document.json

/disk/qh/nano-finrag/data/raw_sec_html/          -> symlink to the raw/SEC directory
/disk/qh/nano-finrag/data/financial_corpus_v2/   -> also at /disk/qh/financial_corpus_v2
```

Companies: `AAPL AMZN GOOGL JPM KO MSFT NVDA PFE TSLA V`

---

## 3. The three orphans

Present in `normalized/SEC/`, absent from `raw/SEC/`:

| ticker | accession | form | filing date | iXBRL facts | blocks |
|---|---|---|---|---|---|
| KO | `SEC_21344_000002134424000019` | **10-Q/A** | 2024-05-30 | 91 | 6 |
| TSLA | `SEC_1318605_000110465925042659` | **10-K/A** | 2025-04-30 | 37 | 391 |
| TSLA | `SEC_1318605_000110465926053166` | **10-K/A** | 2026-04-30 | 37 | 386 |

All three are SEC **amendments** (`/A`) and carry very few iXBRL facts — for scale, a
matched KO 10-Q (`SEC_21344_000002134425000061`) carries 1,762 facts across 60 blocks.

Full lists for the two affected companies:

```
KO   normalized: 7 dirs   raw: 6 dirs   (+ SEC_21344_000002134424000019)
TSLA normalized: 8 dirs   raw: 6 dirs   (+ SEC_1318605_000110465925042659,
                                         + SEC_1318605_000110465926053166)
all other 8 companies: 6 and 6
```

---

## 4. Questions to answer

For **each** of the three orphans:

1. Does a source document for it exist anywhere on the host (or in the archive it came
   from)? Search broadly — it may sit outside `financial_corpus_v2`.
2. If not, is it **present in SEC EDGAR**? The `source-metadata.json` of a sibling
   filing carries `accession_number`, `CIK` and the original URL; the accession number
   is enough to query EDGAR directly. Report whether the primary document for that
   accession is retrievable.
3. Was the normalization run for these three from a source that was never intended to
   be part of `raw/` (e.g. a separate normalization-only pass)? Check for build
   manifests, logs, or a normalization config referenced by
   `normalization_config_sha` in each `document.json`.
4. Hence: **intended, or missing?**

**Also independently recount**, without trusting §1:

- number of `raw/SEC` filing directories, per company and total
- number of files under `raw/SEC`, by extension
- number of `normalized/SEC` `document.json`, per company and total
- for every raw filing, confirm `primary.html` and `source-metadata.json` are both
  present and non-empty
- confirm the symlinks `data/raw_sec_html` and `/disk/qh/financial_corpus_v2` resolve
  as described

Optionally spot-check integrity: for a handful of filings, compare the `sha256` in
`source-metadata.json` against the actual `primary.html`.

---

## 5. Hard constraints

- **Read-only.** Do not re-run normalization, do not regenerate `document.json`, do not
  modify `financial_corpus_v2`. Report only.
- Do not start the application, tests, or any runtime framework. The GPU is expected
  to be idle and should stay that way.
- This corpus is **not** the source of the current benchmark's fact store, which is
  built from 8 separate 10-K PDFs under `data/raw_pdfs/`. Do not conflate the two, and
  do not attempt to reconcile them.

---

## 6. Output

Write to:

```
/disk/qh/nano-finrag/artifacts/evaluation/p1-6-0d-source-truth/corpus-inventory.json
```

```json
{
  "verifier": "<name/handle>",
  "generated_at": "<ISO-8601>",
  "recount": {
    "raw_filing_dirs_total": 60,
    "raw_dirs_per_company": {"AAPL": 6, "...": 6},
    "raw_files_by_extension": {"html": 60, "json": 60},
    "normalized_document_json_total": 63,
    "normalized_per_company": {"KO": 7, "TSLA": 8, "...": 6},
    "raw_filings_missing_a_companion_file": []
  },
  "orphans": [
    {
      "ticker": "KO",
      "accession": "SEC_21344_000002134424000019",
      "form": "10-Q/A",
      "source_found_on_host": false,
      "source_paths_checked": ["..."],
      "present_in_edgar": true,
      "edgar_url": "https://www.sec.gov/Archives/edgar/data/21344/...",
      "verdict": "MISSING_RAW",
      "note": "..."
    }
  ],
  "overall": "RAW_MATERIAL_MISSING | INTENDED_AMENDMENT_ONLY | UNCLEAR"
}
```

`verdict` per orphan ∈ `{"MISSING_RAW", "INTENDED_NO_RAW", "UNRESOLVED"}`.

`overall` is the headline: is the corpus consistent, and if not, what is missing.

---

## 7. Why this matters (context for your judgement)

If raw material is genuinely missing, then three normalized documents have no
reproducible source — the normalization cannot be re-derived or audited, and any
future phase that consumes this corpus inherits three unexplained objects. If the
amendments were deliberately normalized without archiving their raw HTML, that should
be recorded as a corpus property so it is not rediscovered later as a defect.

Either answer is fine. An unexplained gap is not.
