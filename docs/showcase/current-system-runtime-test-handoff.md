# 当前 FinQuery 系统全链路运行测试交接文档

> 交给 CC 执行。本文档以当前仓库代码为准，目标是做一次可复现的系统运行/集成回归，不是重新训练模型，也不是重新跑历史 benchmark。
>
> 最后核对时间：2026-09-08。执行前先确认 Git SHA；如果 SHA 不同，以实际 checkout 的代码和配置为准，并在报告中记录差异。

## 0. 先给结论：当前版本与启动前提

当前工作区基线：

```text
Repository       = E:\nanochat (local)
Branch           = feat/context-trust-runtime-trace
HEAD             = d133cbaa93fafce1de640c81a16ca0a171b05340
Production API   = /query + /query/stream
Runtime          = Trusted Financial Runtime V2
Default runtime  = v2 (代码默认值；以 online.env 最终值为准)
Default context  = on  (代码默认值；以 online.env 最终值为准)
V1               = explicit rollback mode
Shadow           = optional diagnostic mode
Runtime V3       = 不存在；V3 只出现在历史模型/检索实验命名中
```

当前代码已经有完整 V2 生产入口：

```text
/query /query/stream
        ↓
QueryLifecycleService
        ↓
FinancialRuntimeRouter
        ↓
TrustedFinancialRuntimeV2
        ↓
Supervisor → SupervisorPlan → bounded runtime
        ↓
R4 Retrieval → Semantic Binder → bounded recovery
        ↓
Calculator / Deterministic Renderer / Financial Specialist
        ↓
Validator → Repair Once → Release / Fail-Closed
```

但是 V2 不是“只 checkout 就一定能启动”的单文件程序。`FINANCIAL_RUNTIME_MODE=v2` 时，`/readyz` 会强制检查真实 Builder、R4 四路索引、结构化 fact store、Provider 和 Specialist checkpoint。缺一项都应保持未就绪，不能静默降级到 V1。

**当前服务器实测的部署缺口（先验证，不要假设已经解决）：**

```text
config/deployment/online.env 当前缺少：
  FINANCIAL_RUNTIME_MODE
  MULTITURN_CONTEXT_MODE
  TRUSTED_V2_RUNTIME_BUILDER
  TRUSTED_V2_R4_INDEX_DIR
  TRUSTED_V2_FACT_STORE_PATH
  TRUSTED_V2_SPECIALIST_CHECKPOINT
  V2_SUPERVISOR_API_KEY
  V2_BINDER_API_KEY

当前服务器未找到可直接满足 Builder 合约的：
  candidate-metadata.sqlite + 四路 dense/BM25 index

因此在补齐这些资源以前：
  /readyz 预期返回 503
  不得把服务启动失败解释为代码回归
```

如果只是验证旧 V1 链路，可临时使用 `FINANCIAL_RUNTIME_MODE=v1`；这不等于 V2 已经部署成功。正式 V2 测试必须使用真实资源并以 `/readyz=200` 为前提。

## 1. 机器、目录与服务端口

### 1.1 本地开发机

```text
仓库根目录：E:\nanochat
后端：      E:\nanochat\finquery_rag\backend
前端：      E:\nanochat\finquery_rag\frontend
部署模板：  E:\nanochat\config\deployment\online.env.example
启动脚本：  E:\nanochat\scripts\deploy\
运行文档：  E:\nanochat\docs\deployment\
```

### 1.2 当前服务器

```text
SSH：       mxf@10.110.16.5
项目根：    /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
后端 venv： /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend/.venv
MinerU：    /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.runtime/mineru-venv-cu126
HF cache：  /mnt/disk/mxf/.cache/huggingface
NanoChat：  /home/mxf/.cache/nanochat
```

### 1.3 端口与 tmux session

所有服务默认只绑定 loopback，不直接暴露公网：

| 组件 | 地址 | tmux session | 作用 |
|---|---|---|---|
| NanoChat OpenAI-compatible model server | `127.0.0.1:18001` | `nano-finance-model` | `/health`、`/v1/models`、`/v1/chat/completions` |
| FinQuery backend | `127.0.0.1:18002` | `nano-finance-backend` | FastAPI、V2 runtime、会话、上传、SSE |
| React/Vite frontend | `127.0.0.1:18003` | `nano-finance-frontend` | Web UI |

启动顺序固定为：

```text
model → backend → frontend
```

停止顺序反过来。当前部署脚本使用 tmux 和 `runtime/phase7/{logs,pids,status}`，不是 Docker、systemd 或 root 服务。

截至本文档核对时，服务器上的三个 tmux session 和 18001/18002/18003 端口均未运行；先执行 health/status 检查，不要假设已有旧进程可复用。

## 2. 代码同步与安全约定

### 2.1 正常同步

在服务器执行：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
git fetch origin
git checkout feat/context-trust-runtime-trace
git pull --ff-only origin feat/context-trust-runtime-trace
git rev-parse --short HEAD
git status --short
```

期望 HEAD 为当前交接基线的短 SHA：

```text
d133cba
```

如果服务器有本地未提交修改或 untracked 文件，先保存清单，不要使用 `git reset --hard`、`git clean -fd` 或覆盖 `online.env`。已有服务器 untracked 分析文件属于现场资料。

### 2.2 GitHub 连接不可用时

如果服务器不能访问 GitHub，不要改变业务代码绕过网络。可以在可联网的本地导出 bundle，再通过 SSH 复制：

```powershell
cd E:\nanochat
git bundle create C:\Temp\nanochat-tv2.bundle feat/context-trust-runtime-trace
scp C:\Temp\nanochat-tv2.bundle mxf@10.110.16.5:/tmp/
```

服务器：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
git bundle verify /tmp/nanochat-tv2.bundle
git fetch /tmp/nanochat-tv2.bundle feat/context-trust-runtime-trace:refs/remotes/origin/feat/context-trust-runtime-trace
git checkout feat/context-trust-runtime-trace
```

只有确认没有要保留的 tracked 修改后，才允许将 worktree 对齐到远端 ref；不要在现场直接强制覆盖。

## 3. Canonical 运行环境

### 3.1 后端 Python/依赖

后端 `finquery_rag/backend/pyproject.toml` 要求 Python `>=3.12`，CI 使用 Python 3.12 和 `uv.lock`。根目录 `.python-version=3.10` 是 NanoChat 根项目设置，不适用于 FinQuery backend。

服务器已核对：

```text
Python  = 3.12.2
uvicorn = 0.38.0
uv      = /home/mxf/.local/bin/uv
conda   = /mnt/disk/mxf/anaconda3
conda env = nano
```

检查：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend
.venv/bin/python --version
.venv/bin/python -m pip --version
/home/mxf/.local/bin/uv --version
```

安装/同步（会访问包源；已有锁定环境时只做核对）：

```bash
/home/mxf/.local/bin/uv sync --locked
```

不要为了 Python 3.10 写 `StrEnum` 兼容补丁；当前 backend canonical runtime 是 Python 3.12。

### 3.2 系统库

CI 安装：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ghostscript libgl1 poppler-utils
```

如果服务器没有 sudo 权限，记录缺失项，不要把系统库复制进 Python venv。

### 3.3 GPU

服务器有 8 张 NVIDIA RTX A6000（每张约 49140 MiB）。先看占用：

```bash
nvidia-smi
```

建议当前 NanoChat model 使用 `CUDA_VISIBLE_DEVICES=4`，MinerU 使用独立的 `MINERU_CUDA_VISIBLE_DEVICES=1`；这只是当前服务器已有配置，若 GPU 已被其他任务占用，先重新选择并记录，不能同时让两个重模型抢同一张卡。

### 3.4 前端 Node

服务器已核对：

```text
node = v24.18.0
npm  = 11.16.0
```

检查：

```bash
node --version
npm --version
```

## 4. 生产配置 `online.env`

### 4.1 创建与权限

服务器首次配置：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
cp config/deployment/online.env.example config/deployment/online.env
chmod 600 config/deployment/online.env
```

`online.env` 含 API key、JWT secret 等敏感值，禁止提交 Git、禁止贴到日志或交给 CC 输出。启动脚本会优先读取 `config/deployment/online.env`，不存在时才读取 example。

编辑：

```bash
${EDITOR:-vi} config/deployment/online.env
```

### 4.2 当前服务器可复用的非秘密路径值

下面是此前在该服务器观察到的值，使用前仍需 `test -e` 验证：

```ini
MODEL_HOST=127.0.0.1
MODEL_PORT=18001
MODEL_NAME=finquery-finance-v2-lr010-150
MODEL_SOURCE=sft
MODEL_TAG=d24_finance_v2_lr010
MODEL_STEP=150
MODEL_TEMPERATURE=0
MODEL_MAX_TOKENS=512
CUDA_VISIBLE_DEVICES=4
MODEL_PYTHON=/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend/.venv/bin/python

BACKEND_HOST=127.0.0.1
BACKEND_PORT=18002
BACKEND_WORKERS=1
BACKEND_RELOAD=false
BACKEND_VENV_PATH=/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend/.venv

FRONTEND_HOST=127.0.0.1
FRONTEND_PORT=18003
VITE_API_URL=/api
VITE_API_PROXY_TARGET=http://127.0.0.1:18002

CONDA_ENV_NAME=nano
```

NanoChat model loader 默认从以下模式寻找权重：

```text
${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}/chatsft_checkpoints/${MODEL_TAG}/model_$(printf '%06d' ${MODEL_STEP}).pt
```

当前服务器此前存在：

```text
/home/mxf/.cache/nanochat/chatsft_checkpoints/
  d24_finance_v2_lr010/model_000150.pt
```

这个模型服务是 OpenAI-compatible 兼容服务/legacy 连接，不等于 V2 Specialist。V2 Specialist 是下面单独的 checkpoint。

### 4.3 V2 正式必填配置

默认 V2 使用以下结构；真实 key 只能由部署负责人写入 `online.env`，本文档不保存 secret：

```ini
FINANCIAL_RUNTIME_MODE=v2
MULTITURN_CONTEXT_MODE=on
# FINANCIAL_RUNTIME_ADAPTER_ENABLED=true  # code default; do not set false for V2

TRUSTED_V2_RUNTIME_BUILDER=src.runtime.trusted_v2_production:build_trusted_v2_runtime_for_request

TRUSTED_V2_R4_INDEX_DIR=/ABSOLUTE/PATH/TO/current-four-lane-r4-index
TRUSTED_V2_FACT_STORE_PATH=/ABSOLUTE/PATH/TO/financial-facts.jsonl.gz

V2_SUPERVISOR_PROVIDER=bailian
V2_SUPERVISOR_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
V2_SUPERVISOR_API_KEY=<deployment-secret>
V2_SUPERVISOR_MODEL=qwen-plus

V2_BINDER_PROVIDER=bailian
V2_BINDER_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
V2_BINDER_API_KEY=<deployment-secret>
V2_BINDER_MODEL=qwen-plus

TRUSTED_V2_SPECIALIST_CHECKPOINT=/ABSOLUTE/PATH/TO/model_000156.pt
TRUSTED_V2_SPECIALIST_DEVICE=cuda:0
```

`bailian` 是当前模板/Builder 的首选 Provider。代码也支持 `api`，但只有在已验证 API-compatible endpoint、模型和 timeout 后才可使用。Supervisor/Binder 的远程 Provider 不能被本地 18001 model server 自动替代，除非明确设置并测试了 `provider=api` 合约。

可选受控预算（没有特殊需求时沿用代码默认）：

```ini
V2_MAX_REPLANS=2
V2_MAX_TOOL_CALLS=5
V2_MAX_SAME_TOOL_RETRIES=1
V2_MAX_IDENTICAL_QUERY_RETRIES=0
```

### 4.4 V2 Specialist checkpoint

模型文件不是代码仓库的一部分。服务器此前已存在：

```text
/home/mxf/.cache/nanochat/chatsft_checkpoints/
  d24_grounded_specialist_v3_lr5e6/model_000156.pt
```

注意：目录名里的 `v3` 是 Specialist 训练/蒸馏版本，不是生产 Runtime V3。启动前检查：

```bash
SPECIALIST=/home/mxf/.cache/nanochat/chatsft_checkpoints/d24_grounded_specialist_v3_lr5e6/model_000156.pt
test -r "$SPECIALIST" && ls -lh "$SPECIALIST"
```

可选记录 hash（大文件会耗时）：

```bash
sha256sum "$SPECIALIST"
```

### 4.5 Legacy/V1 回滚与数据库路径

V1 是显式回滚/兼容模式，建议保留路径：

```ini
LLM_API_BASE_URL=http://127.0.0.1:18001/v1
LLM_API_KEY=not-needed-for-local
LLM_MODEL_NAME=finquery-finance-v2-lr010-150

SECRET_KEY=<long-random-deployment-secret>
ALLOWED_ORIGINS=http://127.0.0.1:18003,http://localhost:18003

DATABASE_URL=sqlite:///./runtime/phase7/finquery.db
CHROMA_PATH=./finquery_rag/backend/chroma_db
BM25_DB_PATH=./finquery_rag/backend/rag_bm25.db
DOCUMENT_REGISTRY_DB_PATH=./finquery_rag/backend/document_registry.db
SESSIONS_DB_PATH=./finquery_rag/backend/sessions.db
TRACE_DB_PATH=./finquery_rag/backend/trace_log.db
```

V2 官方路径不会为了查询再构造 legacy `RAGEngine`；但当前部署脚本和 V1 rollback 仍会检查/使用 model server 与 legacy 存储，因此不要在未核对模式前删除这些配置。

### 4.6 MinerU 配置

当前服务器已有独立 MinerU 环境：

```ini
PARSER_BACKEND=mineru
MINERU_COMMAND=/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.runtime/mineru-venv-cu126/bin/mineru
MINERU_BACKEND=pipeline
MINERU_TIMEOUT_SECONDS=3600
MINERU_METHOD=auto
MINERU_FORCE_CPU=false
MINERU_CUDA_VISIBLE_DEVICES=1
```

也可以使用：

```ini
PARSER_BACKEND=native
```

其中 `native` 使用 PyMuPDF + Camelot；`mineru` 是明确的 MinerU `*_content_list.json` 解析路径；`auto` 只有满足代码中的自动路由开关和 API 配置时才会按低文本 PDF 选择 MinerU。不要把 MinerU/Torch/CUDA 安装到 backend `.venv`。

此前服务器 `online.env` 曾出现重复的 `MINERU_*` 行。Shell 会采用最后一行，但交接时应清理为单一配置，避免 CC 误判实际生效值。

### 4.7 Reranker/Embedding 配置

当前默认是依赖较少的 heuristic reranker：

```ini
RAG_RERANKER=heuristic
# RAG_RERANKER_MODEL=
# RAG_CANDIDATE_MULTIPLIER=4
# EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

代码读取优先级：

```text
RERANKER_PROVIDER 或 RAG_RERANKER
RERANKER_MODEL_NAME 或 RAG_RERANKER_MODEL
```

如果设置 `cross-encoder`，必须提供可离线读取的本地 `RAG_RERANKER_MODEL`。当前 TV2-03 审计确认正式 R4 policy root 不包含独立 Qwen reranker；`src/pdf_retrieval_v4/qwen3_reranker*.py` 主要是历史/评测代码，不要在交接报告中称当前生产 R4 是 Qwen reranker-powered。

### 4.8 本次 CC 测试的 Provider/GPU 覆盖配置

本节是对上面历史观测值的当前任务覆盖，不代表已经写入服务器。当前要求是：

```text
V2 Supervisor/Binder Provider = DeepSeek OpenAI-compatible API
Local model/Specialist/MinerU = physical GPU 1
V1 baseline                  = 保持可回滚、先不修改
```

建议的 V2 配置目标：

```ini
V2_SUPERVISOR_PROVIDER=api
V2_SUPERVISOR_BASE_URL=https://api.deepseek.com
V2_SUPERVISOR_MODEL=deepseek-v4-flash
V2_SUPERVISOR_API_KEY=<由部署负责人写入服务器，不要提交>

V2_BINDER_PROVIDER=api
V2_BINDER_BASE_URL=https://api.deepseek.com
V2_BINDER_MODEL=deepseek-v4-flash
V2_BINDER_API_KEY=<由部署负责人写入服务器，不要提交>

# CUDA_VISIBLE_DEVICES=1 将物理 GPU 1 映射为进程内 cuda:0。
CUDA_VISIBLE_DEVICES=1
TRUSTED_V2_SPECIALIST_DEVICE=cuda:0
MINERU_CUDA_VISIBLE_DEVICES=1
```

DeepSeek 调用使用 OpenAI-compatible `https://api.deepseek.com`，Supervisor 和 Binder 的响应必须使用 JSON mode，并在本地继续执行冻结的严格 schema 校验；不能以答案非空判断成功，也不能回退到 Bailian/Qwen 或 V1。当前代码的 `_build_supervisor()` 已支持 `provider=api`，但 `_build_binder()` 仍只接受 `bailian`，所以 CC 必须先实现通用 OpenAI-compatible Binder provider，再做 live key smoke。远程 DeepSeek API 不占服务器 GPU；BM25/dense serving 默认走 CPU，GPU1 限制主要作用于 NanoChat、Specialist 和 MinerU。

## 5. V2 真实资源合约（最容易出错的部分）

### 5.1 R4 四路索引

`TRUSTED_V2_R4_INDEX_DIR` 不是任意 Chroma/BM25 目录。当前 Builder 要求至少存在：

```text
<index-dir>/candidate-metadata.sqlite
<index-dir>/candidate_raw_bm25/bm25/index.sqlite
<index-dir>/candidate_structured_bm25/bm25/index.sqlite
<index-dir>/candidate_raw_dense/dense/ids.json
<index-dir>/candidate_raw_dense/dense/vectors.npy
<index-dir>/candidate_structured_dense/dense/ids.json
<index-dir>/candidate_structured_dense/dense/vectors.npy
```

`candidate-metadata.sqlite` 的 `view_metadata` 表需要有：

```text
lane, view_id, candidate_key, view_type,
retrieval_text, document_id, metadata_json
```

并且 row count > 0、四个 lane 都存在。索引应以只读方式使用。

检查命令：

```bash
R4=/ABSOLUTE/PATH/TO/current-four-lane-r4-index
test -f "$R4/candidate-metadata.sqlite"
test -f "$R4/candidate_raw_bm25/bm25/index.sqlite"
test -f "$R4/candidate_structured_bm25/bm25/index.sqlite"
test -f "$R4/candidate_raw_dense/dense/ids.json"
test -f "$R4/candidate_raw_dense/dense/vectors.npy"
test -f "$R4/candidate_structured_dense/dense/ids.json"
test -f "$R4/candidate_structured_dense/dense/vectors.npy"
```

代码/构建路径（证明能力已实现，但不代表索引已部署）：

```text
finquery_rag/backend/src/pdf_retrieval_v4/candidate_view_index.py
finquery_rag/backend/src/pdf_retrieval_v4/candidate_direct_retriever.py
finquery_rag/backend/src/runtime/trusted_v2_r4.py
finquery_rag/backend/scripts/evaluation/build_pdf_v4_candidate_views.py
finquery_rag/backend/scripts/evaluation/build_pdf_v4_candidate_indexes.py
```

本地目前可见的：

```text
finquery_rag/backend/artifacts/evaluation/pdf-retrieval-v4-gate-08-r3/control-indexes/
```

只有 `candidate_structured_bm25`、`candidate_structured_dense` 和 metadata（464 views），缺少 `candidate_raw_bm25`、`candidate_raw_dense`，属于评测 control index，不能直接填入 `TRUSTED_V2_R4_INDEX_DIR`。`finquery_rag/backend/rag_bm25.db` 是 V1/legacy 存储，也不能替代 V2 四路 CandidateView index。CC 必须用 `inspect_r4_index()` 和 `/readyz` 验证真实服务器路径，找不到完整四路时保持 V2 未就绪并报告 blocker。

旧的 `financial_corpus_v2/indexes/...` dense/BM25 布局没有 `candidate-metadata.sqlite` 和四 lane 元数据，不能直接填入 V2 配置。当前服务器的 `.runtime/structured-fact-v2/native-facts.jsonl` 首行看起来是 SEC facts，但此前检查显示并不满足完整 `evidence_id/citation_id/provenance_complete` 合约，不能未经验证直接当 V2 fact store。

### 5.2 Fact store

`TRUSTED_V2_FACT_STORE_PATH` 可为 JSON、JSONL 或 gzip JSONL。每条记录至少应能提供：

```text
candidate_key / candidate_id
evidence_id 或 fact_id
结构化 citation_id（或单元素 citation_ids）
physical source identity（source_id / physical_source_id / document_id / cell_id）
provenance_complete=true
```

Binder 使用的字段可能是：

```text
normalized_metric/raw_metric
normalized_period/raw_period
normalized_scale/raw_scale
parsed_numeric_value/raw_value
pdf_page
physical_source_id/document_id
evidence_text/content/raw_content/row_label
```

验证只允许从结构化对象产生 provenance；禁止从回答文本正则解析 evidence/citation/calculation ID。

### 5.3 Provider

检查但不要打印 key：

```bash
for k in V2_SUPERVISOR_PROVIDER V2_SUPERVISOR_BASE_URL V2_SUPERVISOR_MODEL \
         V2_BINDER_PROVIDER V2_BINDER_BASE_URL V2_BINDER_MODEL \
         V2_SUPERVISOR_API_KEY V2_BINDER_API_KEY; do
  if grep -q "^${k}=." config/deployment/online.env; then
    echo "$k=SET"
  else
    echo "$k=MISSING"
  fi
done
```

真实 Provider 网络不可用时，应该是 preflight/执行失败并记录原因，不要把错误自动改成 `FAIL_CLOSED` 或静默转 V1。

## 6. 代码级检查（不调用生产模型）

### 6.1 编译与 Ruff

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend
.venv/bin/python -m compileall -q src tests scripts
.venv/bin/python -m ruff check src tests scripts
```

### 6.2 运行当前 CI 回归

V1/Conversation 回归：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend
FINANCIAL_RUNTIME_MODE=v1 MULTITURN_CONTEXT_MODE=off \
  .venv/bin/python -m pytest -q \
  --ignore=tests/evaluation \
  --ignore=tests/pdf_retrieval_v4 \
  --ignore=tests/rag_v2/test_nf_v2_02_top20_financial_fact.py
```

V2 路由/生产激活/评测 harness 回归（需要测试依赖，但不是完整 readiness benchmark）：

```bash
FINANCIAL_RUNTIME_MODE=v2 MULTITURN_CONTEXT_MODE=on \
  .venv/bin/python -m pytest -q \
  tests/test_tv2_08_production_integration.py \
  tests/test_runtime_router_shadow.py \
  tests/test_runtime_router_api_shadow.py \
  tests/test_tv2_07_readiness.py \
  tests/test_tv2_07_r1_readiness.py \
  tests/evaluation/test_deterministic_runtime.py
```

已有 CI smoke：

```bash
FINQUERY_EVAL_ARTIFACT_DIR=/tmp/finquery_eval_artifacts \
  .venv/bin/python scripts/ci_eval_gate.py

FINQUERY_PREFLIGHT_ARTIFACT_DIR=/tmp/finquery_preflight_artifacts \
  .venv/bin/python scripts/ci_preflight_smoke.py
```

这些命令不等于真实 V2 资源可用性；真实部署仍必须通过下面的 `/readyz`。

### 6.3 V2 配置 preflight

在 backend 目录加载部署环境，然后只输出结构化状态，不输出 secret：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
set -a
. config/deployment/online.env
set +a
cd finquery_rag/backend
PYTHONPATH=. .venv/bin/python -c \
 'import json; from src.runtime import validate_trusted_v2_production_configuration as f; print(json.dumps(f(), ensure_ascii=False, indent=2, default=str))'
```

若函数返回缺失项，先修配置/资源，再启动服务；不要用 fake builder 或把旧索引改名冒充四路索引。

## 7. 启动、健康检查、停止

### 7.1 一键启动

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
bash scripts/deploy/start_all.sh
```

脚本行为：

```text
load_env.sh
→ model
→ 等待 model /health
→ backend
→ 等待 backend /healthz
→ 等待 backend /readyz
→ frontend
```

V2 模式下 `/readyz` 失败应使启动失败；这是安全行为，不要改成“服务先起来再每个请求报错”。

### 7.2 单服务手工启动（定位问题时使用）

正常测试优先使用 `start_all.sh`，只有需要隔离故障时才单独启动。三个脚本都会读取同一份 `config/deployment/online.env`，不要给某个服务另写一套配置。

模型服务：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
bash scripts/deploy/start_model.sh
curl -fsS http://127.0.0.1:18001/health
curl -fsS http://127.0.0.1:18001/v1/models
```

模型服务的实现入口是：

```text
scripts/chat_openai_compat.py
```

它根据 `MODEL_SOURCE`、`MODEL_TAG`、`MODEL_STEP` 从 NanoChat cache 加载 checkpoint，并暴露 OpenAI-compatible API。不要把这个服务误认为 V2 Supervisor/Binder Provider 或 V2 Specialist；三者职责不同。

可做一个最小兼容接口检查（不代表金融答案正确）：

```bash
curl -fsS -X POST http://127.0.0.1:18001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"finquery-finance-v2-lr010-150","messages":[{"role":"user","content":"ping"}],"temperature":0,"max_tokens":8}'
```

后端服务：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
bash scripts/deploy/start_backend.sh
curl -fsS http://127.0.0.1:18002/healthz
curl -i  http://127.0.0.1:18002/readyz
```

如果不用脚本直接运行，必须从项目根目录加载 `online.env`，并让 `PYTHONPATH` 指向 backend：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
set -a; . config/deployment/online.env; set +a
cd finquery_rag/backend
PYTHONPATH=. .venv/bin/python -m uvicorn src.main:app \
  --host "${BACKEND_HOST:-127.0.0.1}" \
  --port "${BACKEND_PORT:-18002}" \
  --workers "${BACKEND_WORKERS:-1}"
```

前端服务：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
bash scripts/deploy/start_frontend.sh
curl -fsS http://127.0.0.1:18003/
```

前端直接启动方式：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/frontend
npm run dev -- --host 127.0.0.1 --port 18003
```

### 7.3 一键部署脚本的真实顺序

### 7.4 状态、健康、冒烟

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
bash scripts/deploy/status.sh
PYTHON=/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend/.venv/bin/python
"$PYTHON" scripts/deploy/healthcheck.py
"$PYTHON" scripts/deploy/smoke_test.py
"$PYTHON" scripts/deploy/collect_deployment_report.py
```

预期产物目录：

```text
runtime/phase7/logs/{model,backend,frontend}.log
artifacts/deployment/phase7/health-report.json
artifacts/deployment/phase7/smoke-report.json
artifacts/deployment/phase7/deployment-report.json
```

实时日志：

```bash
tail -f runtime/phase7/logs/model.log
tail -f runtime/phase7/logs/backend.log
tail -f runtime/phase7/logs/frontend.log
```

如果需要交互排错：

```bash
tmux attach -t nano-finance-model
tmux attach -t nano-finance-backend
tmux attach -t nano-finance-frontend
```

退出 tmux 不要按 Ctrl-C；使用 `Ctrl-b d` detach。

停止/重启：

```bash
bash scripts/deploy/stop_all.sh
bash scripts/deploy/restart_all.sh
```

### 7.5 直接检查端口

```bash
curl -fsS http://127.0.0.1:18001/health
curl -fsS http://127.0.0.1:18001/v1/models
curl -fsS http://127.0.0.1:18002/healthz
curl -i  http://127.0.0.1:18002/readyz
curl -fsS http://127.0.0.1:18003/
```

只有 `/readyz` 返回 200 才能把 V2 称为“可运行”；`/healthz=200` 只代表进程存活。

## 8. 手工 API 端到端测试

以下示例在服务器本机执行。密码和邮箱只作为临时 smoke 测试值，使用唯一邮箱，不要在共享环境复用；测试后按需要清理账户/会话。

```bash
BACKEND=http://127.0.0.1:18002
PYTHON=/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend/.venv/bin/python
EMAIL="cc-smoke-$(date +%s)@example.com"
PASSWORD='cc-smoke-password-change-me'

TOKEN=$(curl -fsS -X POST "$BACKEND/register" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}" \
  | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

curl -fsS "$BACKEND/readyz"
```

### 8.1 Direct fact

```bash
curl -fsS -X POST "$BACKEND/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: cc-fact-001' \
  -d '{"question":"贵州茅台2023年营业收入是多少?"}'
```

检查返回结构中至少有：

```text
status
answer / clarification
runtime_version=V2
release_status
evidence_ids / citation_ids（正式答案应来自结构化结果）
request_id / session_id
```

### 8.2 Calculation

```bash
curl -fsS -X POST "$BACKEND/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: cc-calc-001' \
  -d '{"question":"贵州茅台2023年毛利率是多少?"}'
```

检查 `calculation_ids`、operand lineage、period/unit/currency/scale，以及 `release_status`。不能只凭答案文本判断计算成功。

### 8.3 无答案/Fail-Closed

```bash
curl -fsS -X POST "$BACKEND/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: cc-noanswer-001' \
  -d '{"question":"今天天气怎么样?"}'
```

确认：

```text
status=FAIL_CLOSED 或当前 Contract 对应的安全拒答状态
release_status=NOT_RELEASED
不调用 V1 fallback
不产生虚假 evidence/citation/calculation provenance
```

### 8.4 多轮上下文

复用同一个 `session_id`（从上一响应取出；如果 API 没有返回则在请求体显式指定 UUID）：

```bash
SESSION_ID="cc-session-$(date +%s)"

curl -fsS -X POST "$BACKEND/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: cc-mt-001' \
  -d "{\"session_id\":\"$SESSION_ID\",\"question\":\"贵州茅台2024年营业收入是多少?\"}"

curl -fsS -X POST "$BACKEND/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: cc-mt-002' \
  -d "{\"session_id\":\"$SESSION_ID\",\"question\":\"上一年呢？\"}"
```

第二轮必须由 Conversation Layer 形成 `standalone_query` 后进入 V2；V2 不应自行读取完整 raw history。若第一轮产生 clarification，第三轮使用同一 session 回答候选指标，检查 `pending_clarification` 被清除且金融执行只在 clarification resolved 后发生。

### 8.5 Stream

当前 `/query/stream` 不是 token-level streaming，而是完整执行、验证、最后发送一次 validated final-response SSE：

```bash
curl -N -fsS -X POST "$BACKEND/query/stream" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: cc-stream-001' \
  -d '{"question":"贵州茅台2023年营业收入是多少?"}'
```

检查：

```text
V2 完成 Validator 后才有 SSE 输出
clarification/FAIL_CLOSED 也只发送最终用户可见结果
不会把 Specialist token 在验证前发出
```

### 8.6 幂等

重复发送同一个 `X-Request-ID`，检查：

```text
不重复推进 turn_count
不重复写 assistant turn
不重复写 conversation provenance
不在 Session 中产生两次当前用户轮
```

不要把“状态提交幂等”误解成“可以缓存并跳过金融 Runtime”；以当前 API Contract 为准记录重复请求行为。

## 9. 前端与 SSH 隧道

### 9.1 SSH 隧道

在本地另开终端：

```bash
ssh -N \
  -L 18003:127.0.0.1:18003 \
  -L 18002:127.0.0.1:18002 \
  -L 18001:127.0.0.1:18001 \
  mxf@10.110.16.5
```

浏览器打开：

```text
http://127.0.0.1:18003/
```

`VITE_API_URL=/api` + `VITE_API_PROXY_TARGET=http://127.0.0.1:18002` 时，前端通过 Vite proxy 访问后端。不要直接沿用 `frontend/.env.production` 中历史的公网 IP，除非部署负责人明确更新并验证。

### 9.2 前端单独检查

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/frontend
npm install --no-audit --no-fund
npm run lint
npm run build
```

当前部署脚本用的是：

```bash
npm run dev -- --host 127.0.0.1 --port 18003
```

`npm run build` 是静态构建检查，不会替代部署脚本的 Vite dev server。

## 10. MinerU 全链路验证

### 10.1 环境检查

```bash
MINERU_ENV=/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.runtime/mineru-venv-cu126
"$MINERU_ENV/bin/mineru" --version
"$MINERU_ENV/bin/python" -c \
 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
nvidia-smi
```

上次核对的环境约为 MinerU 3.4.4、Python 3.12.11、Torch 2.7.1+cu126、CUDA runtime 12.6；以本次命令输出为准。

### 10.2 单 PDF 解析 smoke

不要一上来重建全量 corpus。先用一份小 PDF：

```bash
MINERU_ENV=/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.runtime/mineru-venv-cu126
INPUT=/ABSOLUTE/PATH/TO/sample.pdf
OUT=/tmp/cc-mineru-smoke-$(date +%s)
mkdir -p "$OUT"

"$MINERU_ENV/bin/mineru" \
  -p "$INPUT" \
  -o "$OUT" \
  -b pipeline \
  -m auto

find "$OUT" -type f -name '*_content_list.json' -print
```

确认 `*_content_list.json`、表格/页码/结构化文本可读后，再决定是否上传/重建索引。显式 `PARSER_BACKEND=mineru` 失败时应显示失败，不要静默混用 native 结果。

### 10.3 通过 API 上传

认证后：

```bash
curl -fsS -X POST "$BACKEND/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@/ABSOLUTE/PATH/TO/sample.pdf'
```

切换 parser 后，已有文档不会自动重解析；需要对受影响 PDF 重新上传以重建 dense/sparse 索引。解析 lineage 应能看到 `native-layout-v2` 或 `mineru-content-list-v2`，splitter 为 `page-boundary-section-v2`。

## 11. Reranker/Embedding 运行检查

查看最终解析配置但不触发模型下载：

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
set -a; . config/deployment/online.env; set +a
cd finquery_rag/backend
PYTHONPATH=. .venv/bin/python -c \
 'import json; from src.services.retrieval_config import build_retrieval_model_config; print(json.dumps(build_retrieval_model_config(), ensure_ascii=False, default=str, indent=2))'
```

基线检查项：

```text
RAG_RERANKER=heuristic       → 无额外 cross-encoder 依赖
RAG_RERANKER=cross-encoder   → RAG_RERANKER_MODEL 必须是本地可读路径
EMBEDDING_MODEL_NAME         → 离线环境必须是已缓存/本地路径
```

不要在本次系统测试中切换 reranker、top-k、validator threshold 或 budget；那会改变运行基线。

## 12. CC 必测场景矩阵

### 12.1 V2 官方路径（必须真实 Builder）

```text
[ ] /readyz=200
[ ] direct fact → ANSWER/RELEASED 或有证据的安全拒答
[ ] multi-evidence fact → 所有 RequiredSlot 绑定
[ ] calculation → Binder-admitted operands → Calculator → calculation_id
[ ] qualitative → Specialist → ClaimVerifier → Validator → release/fail-closed
[ ] missing evidence → bounded recovery → FAIL_CLOSED
[ ] wrong period / wrong row → 不得错误 RELEASE
[ ] repair-once → repair_count <= 1
[ ] /query/stream → 验证后一次最终 SSE
[ ] V2 不调用 V1 fallback
[ ] Session/DialogueState 写入的是官方 V2 结果
[ ] answer text 不产生 evidence/operand
```

### 12.2 多轮与安全边界

```text
[ ] 显式问题 → 上下文省略问题 → standalone_query 正确
[ ] ambiguous metric → clarification，Financial Runtime calls=0
[ ] clarification follow-up → pending 清除后才执行 V2
[ ] topic switch → 取消旧 clarification
[ ] Assistant 历史中的假数字不进入 Evidence/Operand
[ ] 同一 X-Request-ID 不重复推进状态
[ ] clear session 后不继承旧 pending/provenance
[ ] 不同 user/session 互相隔离
```

### 12.3 Runtime 模式矩阵

```text
FINANCIAL_RUNTIME_MODE=v2
  V2 official；V1 calls=0；V2 error 不 fallback

FINANCIAL_RUNTIME_MODE=v1
  V1 official；V2 calls=0（需要 legacy 资源）

FINANCIAL_RUNTIME_MODE=shadow
  V1 official + V2 observation；V2 不写 Session/Conversation
```

当前默认应为：

```text
FINANCIAL_RUNTIME_MODE=v2
MULTITURN_CONTEXT_MODE=on
```

如果为了诊断暂时切换 v1/off，报告中必须明确这是 rollback/test 配置，不得称为默认链路。

## 13. 运行报告与证据要求

CC 返回以下信息即可，不要粘贴 secret、完整 prompt、模型隐藏思维链：

```text
1. git branch + full HEAD SHA
2. Python/uv/Node/npm/Torch/MinerU 版本
3. GPU 与 CUDA 可用性
4. online.env 的非秘密 key presence（SET/MISSING，不打印值）
5. V2 preflight JSON（脱敏）
6. /healthz、/readyz、model/frontend health 结果
7. model/backend/frontend 启动日志尾部
8. 直接 fact、calculation、fail-closed、multi-turn、stream 结果摘要
9. runtime_version、status、release_status、route、trace_id
10. evidence_ids/citation_ids/calculation_ids 是否结构化透传
11. X-Request-ID 重放结果与 assistant commit 次数
12. MinerU 单 PDF smoke 结果（若执行）
13. reranker/embedding 最终解析配置（不下载模型）
14. 失败项、错误阶段、下一步缺失资源
```

推荐保存：

```text
artifacts/deployment/phase7/health-report.json
artifacts/deployment/phase7/smoke-report.json
artifacts/deployment/phase7/deployment-report.json
artifacts/cc/current-system-test/<timestamp>/environment.json
artifacts/cc/current-system-test/<timestamp>/v2-preflight.json
artifacts/cc/current-system-test/<timestamp>/api-results.jsonl
```

报告中不要保存：

```text
API keys
SECRET_KEY
完整 Authorization header
模型 Chain-of-Thought
未脱敏用户数据
```

## 14. 常见故障判断

| 现象 | 先检查 | 正确处理 |
|---|---|---|
| `/readyz=503` 且缺 builder | `TRUSTED_V2_RUNTIME_BUILDER` | 配置真实 callable，不改成 fake |
| `/readyz=503` 且缺 R4 文件 | 四 lane + `candidate-metadata.sqlite` | 提供兼容索引，不把旧 Chroma/BM25 改名 |
| fact store contract error | evidence/citation/source/provenance 字段 | 修复 fact store 或停在未就绪 |
| Provider timeout | base URL、DNS、key、模型权限 | 记录 provider error；不转 V1 |
| Specialist load error | checkpoint、device、dtype、权限 | 修复模型路径/GPU；不硬编码开发机路径 |
| MinerU import/torch error | 独立 MinerU venv 与 CUDA | 修复 MinerU 环境，不污染 backend venv |
| backend 导入 PyMuPDF/Chroma 失败 | 当前模式是否 v2 | V2 不应构造 legacy engine；V1/rollback 需补依赖 |
| frontend 打开但 API 失败 | `VITE_API_URL`、proxy、隧道 | 优先使用 `/api` proxy，不用历史公网 IP |
| stream 提前吐 token | 检查启动的代码/路由 | 当前安全语义必须先验证后一次 SSE |
| V2 失败却得到 V1 答案 | 检查 Router/fallback | 这是严重错误，V2 internal fallback 必须为 0 |

## 15. 安全边界与不要做的事

```text
❌ 不把 V2 失败静默降级到 V1
❌ 不把旧索引/Gold evidence 当成 V2 生产资源
❌ 不从 assistant 文本解析 evidence/citation/calculation
❌ 不在 Validator 前发送 token
❌ 不把 raw conversation history 直接交给 V2 作为事实
❌ 不把模型摘要或 assistant 文本当 Operand
❌ 不修改 Supervisor/R4/Binder/Calculator/Validator 算法来迎合测试
❌ 不边跑边调 top-k、budget、threshold、prompt
❌ 不运行未授权的全量 benchmark/训练/GPU 长任务
❌ 不把历史训练目录名 v3 说成 Runtime V3
❌ 不提交 online.env、API key、SECRET_KEY、真实用户数据
❌ 不执行 git reset --hard / git clean -fd 清理现场
```

## 16. 最终回传模板

CC 完成后按以下格式回传：

```text
TEST_DATE:
REPO:
BRANCH:
HEAD:

ENVIRONMENT:
  python:
  uv:
  node:
  npm:
  torch/cuda:
  mineru:
  gpu:

CONFIG:
  FINANCIAL_RUNTIME_MODE: v2/v1/shadow
  MULTITURN_CONTEXT_MODE: on/off/shadow
  V2_BUILDER: SET/MISSING
  R4_INDEX: PASS/MISSING/INCOMPATIBLE
  FACT_STORE: PASS/MISSING/INVALID
  SUPERVISOR_PROVIDER: SET/MISSING/UNREACHABLE
  BINDER_PROVIDER: SET/MISSING/UNREACHABLE
  SPECIALIST_CHECKPOINT: PASS/MISSING/LOAD_ERROR
  RERANKER: heuristic/cross-encoder + model path status
  MINERU: native/mineru/auto + smoke status

HEALTH:
  model /health:
  backend /healthz:
  backend /readyz:
  frontend /:

TESTS:
  compileall:
  ruff:
  backend regression:
  TV2 focused tests:
  eval/preflight smoke:
  API fact:
  API calculation:
  API fail-closed:
  API multi-turn:
  API stream:
  idempotency:
  frontend:
  MinerU:

SAFETY:
  V2 internal V1 fallback:
  assistant-text-to-evidence:
  validator-before-release:
  session/provenance writes:

BLOCKERS:
  ...

ARTIFACTS:
  ...
```

本次测试的成功定义是：**当前 checkout 的完整调用链、环境、配置和安全边界被真实验证，并清楚列出 V2 仍缺少的部署资源**。如果 R4 index/fact store/provider/checkpoint 尚未提供，结论应写成“代码与启动编排已验证，V2 运行资源未就绪”，而不是伪造 V2 已上线。
