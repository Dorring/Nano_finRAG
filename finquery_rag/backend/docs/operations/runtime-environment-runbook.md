# FinQuery runtime environment runbook

Last verified: 2026-08-04. This operator reference records service locations and
environment identities only. Do not commit runtime state, credentials, logs,
model caches, databases, or generated evaluation artifacts.

## Server and repository

| Item | Location |
| --- | --- |
| SSH | ssh mxf@10.157.195.124 |
| Repository root | /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat |
| Backend | finquery_rag/backend |
| Frontend | finquery_rag/frontend |
| Runtime scratch directory | .runtime/ |
| Real evaluation outputs | .claude/finquery_real_eval/ |

Never commit .runtime/, .claude/finquery_real_eval/, persistent indexes,
databases, logs, tokens, model caches, or old environment backups.

Before commands that create temporary data, use the project disk:

~~~
export TMPDIR=/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.runtime/finquery_tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p "$TMPDIR"
~~~

## Current local services

All services bind to 127.0.0.1.

| Service | Port | Current role | Do not |
| --- | ---: | --- | --- |
| RAG production backend | 18002 | Main FastAPI backend | Stop or restart casually |
| RAG experiment backend | 18004 | Isolated experiment service | Reuse without checking branch/config |
| SFT OpenAI-compatible adapter | 18001 | Current model adapter | Assume model identity |
| Frontend Vite server | 18003 | Local development UI | Treat as production |

Verify service state without a health route:

~~~
pgrep -af "uvicorn src.main:app"
pgrep -af "chat_openai_compat"
ss -ltnp | grep -E "18001|18002|18003|18004"
curl -i http://127.0.0.1:18002/docs
curl -sS http://127.0.0.1:18001/v1/models
~~~

The model adapter last reported finquery-finance-v2-lr010-150. Its observed
launch uses backend .venv, source sft, step 150, port 18001, and
CUDA_VISIBLE_DEVICES=3. Query /v1/models before recording a model identity in
an evaluation artifact.

The frontend is normally started from finquery_rag/frontend:

~~~
npm run dev -- --host 127.0.0.1 --port 18003
~~~

A workstation tunnel may expose local services without public binding:

~~~
ssh -N -L 18003:127.0.0.1:18003 -L 18002:127.0.0.1:18002 mxf@10.157.195.124
~~~

## RAG backend environment

Production-native backend environment:

~~~
/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend/.venv
~~~

Use it explicitly:

~~~
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend
./.venv/bin/python --version
./.venv/bin/python -c "import pymupdf; print(pymupdf.__version__)"
~~~

Default parser chain:

~~~
PyMuPDF/native text
-> Camelot stream then lattice fallback
-> structured chunks, table rows/cells, parent-child metadata
-> optional table enhancement
-> current index
~~~

PARSER_BACKEND=native is the default. MinerU is optional. A Camelot failure must
not block native text ingestion; malformed table output must be skipped rather
than indexed as false structure.

For offline models:

~~~
export HF_HOME=/mnt/disk/mxf/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export EMBEDDING_MODEL_NAME=/mnt/disk/mxf/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
~~~

Do not restart production 18002 for an experiment. Use a separate port only
after recording its branch and effective environment.

## MinerU fallback environment

MinerU remains isolated from the RAG .venv:

~~~
/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.runtime/mineru-venv-cu126
~~~

| Field | Last verified value |
| --- | --- |
| MinerU | 3.4.4 |
| Python | 3.12.11 |
| Torch | 2.7.1+cu126 |
| CUDA runtime | 12.6 |
| GPU visibility | 8 NVIDIA RTX A6000 devices |
| CLI | .runtime/mineru-venv-cu126/bin/mineru |

Verify before use:

~~~
MINERU_ENV=/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.runtime/mineru-venv-cu126
"$MINERU_ENV/bin/mineru" --version
"$MINERU_ENV/bin/python" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
~~~

Do not install MinerU, Torch, or CUDA into the backend .venv. Use MinerU only
after native PyMuPDF/Camelot Shadow evaluation is insufficient. First run a
single-PDF smoke in a fresh project-disk temp directory. Record selected
CUDA_VISIBLE_DEVICES, parser version, backend/method, cache location, and
nvidia-smi state. Do not share GPU 3 with the SFT adapter without an explicit
capacity check.

## Runtime and evaluation boundaries

Financial PDFs, BM25, Chroma, document registry, and real evaluation artifacts
are runtime state. Shadow code may read them but must not commit, overwrite, or
write production document records, BM25, Chroma, or the current index.

For candidate replay, read Chroma first, then rag_bm25.db chunk_store. A
BM25-only table row is not a lost candidate merely because Chroma lacks it.
