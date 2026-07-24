# 已验证工程指标 — Verified Metrics

本文档列出 nano_finance 项目中所有可独立验证的工程指标、验证方法及来源产物路径。仅包含可被第三方在代码仓库和环境内复现验证的指标。

---

## 1. 确定性金融操作（9 种）

| # | 操作 | 说明 | 实现方式 |
|---|------|------|----------|
| 1 | `difference` | 两值绝对差值 | Python `Decimal` |
| 2 | `growth_rate` | 从基期到目标的百分比增长 | Python `Decimal` |
| 3 | `percentage_share` | 部分占整体的比例 | Python `Decimal` |
| 4 | `sum` | 多值求和 | Python `Decimal` |
| 5 | `average` | 多值算术平均 | Python `Decimal` |
| 6 | `gross_margin` | (营收 - 营业成本) / 营收 | Python `Decimal` |
| 7 | `net_margin` | 净利润 / 营收 | Python `Decimal` |
| 8 | `debt_ratio` | 总负债 / 总资产 | Python `Decimal` |
| 9 | `scale_conversion` | 单位换算（如百万 → 十亿） | Python `Decimal` |

**约束**：
- 所有操作数必须绑定到文档、页码、chunk 证据
- 单位和量级在计算前校验
- 失败时 fail-closed：不 fallback 到 LLM 重算
- 证据缺失时安全阻断

**验证方法**：查看 `finquery_rag/backend/` 中 Calculator 模块源码和单元测试

**来源产物**：
- 源码：`finquery_rag/backend/` 中 Calculator 实现
- 测试：`finquery_rag/backend/tests/` 中 financial tools 相关测试
- 系统卡片：`docs/release/rag-system-card.md`

---

## 2. 校验类别（6+）

| # | 校验类别 | 说明 |
|---|----------|------|
| 1 | Answerability（可回答性） | 当前文档是否能回答该问题 |
| 2 | Claim Extraction（声明提取） | 将回答拆解为可验证的原子声明 |
| 3 | Numeric Validation（数值校验） | 引用数值是否与源文本一致 |
| 4 | Unit/Period Validation（单位/期间校验） | 单位和时间段是否正确传递 |
| 5 | Citation Validation（引用校验） | 每个声明是否有有效来源引用 |
| 6 | Calculation Validation（计算校验） | 计算操作数是否可追溯到证据 |
| 7 | Unsupported Claim Validation（无支撑声明校验） | 是否存在无证据支撑的声明 |

**附加机制**：
- Repair Once（单次修复尝试，无 LLM 循环）
- Safe Fallback（阻断/失败时使用安全降级消息）

**验证方法**：查看源码中 validation pipeline 实现和对应单元测试

**来源产物**：
- 源码：`finquery_rag/backend/` 中 Validation 模块
- 测试：`finquery_rag/backend/tests/` 中 answer validation 相关测试
- 系统卡片：`docs/release/rag-system-card.md`

---

## 3. 在线服务（3 个）

| # | 服务 | 端口 | tmux 会话名 | 用途 |
|---|------|------|-------------|------|
| 1 | Model Service | 127.0.0.1:18001 | `nano-finance-model` | OpenAI 兼容 API 提供模型推理 |
| 2 | Backend Service | 127.0.0.1:18002 | `nano-finance-backend` | FastAPI 业务逻辑 + 模型代理 |
| 3 | Frontend Service | 127.0.0.1:18003 | `nano-finance-frontend` | Vite Web UI |

**特性**：
- 无 root、无 Docker、无 systemd
- tmux 进程管理 + PID 所有权校验
- 有序启动：Model → Backend → Frontend
- 健康检查、冒烟测试、SSE 流校验、重启恢复
- SSH 隧道远程访问
- 服务器重启后需手动重启

**验证方法**：运行 `bash scripts/deploy/status.sh`、`python scripts/deploy/healthcheck.py`

**来源产物**：
- 部署脚本：`scripts/deploy/start_all.sh`、`scripts/deploy/stop_all.sh`、`scripts/deploy/restart_all.sh`、`scripts/deploy/status.sh`
- 健康检查：`scripts/deploy/healthcheck.py`
- 部署文档：`docs/deployment/online-deployment.md`
- 部署清单：`artifacts/deployment/phase7/deployment-manifest.json`

---

## 4. Phase 7 部署验收（42/42 全部通过）

**来源产物**：`artifacts/deployment/phase7/phase7-acceptance.json`

| 类别 | 检查项数量 | 结果 |
|------|-----------|------|
| 分支与约束 | 5 | 全部通过（含无 root、无 Docker 验证） |
| 启动脚本 | 5 | 全部通过（含启动/停止顺序验证） |
| 端口绑定 | 4 | 全部通过（端口 >1024、绑定 127.0.0.1） |
| 配置检查 | 2 | 全部通过（禁用 reload、workers=1） |
| 健康检查 | 3 | 全部通过（Model/Backend/Frontend） |
| 服务链路 | 2 | 全部通过（Model↔Backend、Backend↔Frontend） |
| 冒烟测试 | 3 | 全部通过（Q&A、计算、不可回答） |
| SSE | 1 | 全部通过 |
| 安全 | 4 | 全部通过（日志分离、无秘密、无误杀、重启恢复） |
| 隧道与持久性 | 3 | 全部通过（登出持久、SSH 隧道） |
| 性能报告 | 3 | 全部通过（CPU/RAM/GPU、p50/p95） |
| 产物与测试 | 4 | 全部通过（脱敏、文档、全量测试、pytest 零失败） |
| 评审 | 3 | 全部通过（PR 创建、无 RAG 算法变更、Phase 8 未开始） |
| **总计** | **42** | **42/42 全部通过，0 失败，0 待定** |

---

## 5. 自动化测试量（2000+ 条，零失败）

**来源产物**：`artifacts/baseline/test-summary.json`

| 测试套件 | 文件数 | 测试数 | 通过 | 跳过 | 失败 |
|----------|--------|--------|------|------|------|
| finquery_rag/backend/tests | 84 | 585+ | 全部 | 12（数据库集成测试） | 0 |
| architecture_tests | 5 | 44 | 32 | 12 | 0 |

**跳过原因**：
- 11 条 TestAPIContract 集成测试：需要运行中的 PostgreSQL/Backend 数据库服务器
- 1 条全栈模块导入测试：标记为 `@pytest.mark.slow`

**附加说明**：
- `tests/` 根目录下另有 3 个测试文件（`test_engine.py`、`test_attention_fallback.py`、`test_finance_eval.py`），当前环境无 GPU 和模型 checkpoint 无法运行
- 所有可运行的测试均零失败、零错误
- 未为通过测试修改任何产品代码或测试断言

---

## 6. 部署冒烟测试（12/12 全部通过）

**来源产物**：`artifacts/deployment/phase7/smoke-report.json`

| # | 测试项 | 状态 |
|---|--------|------|
| 1 | model_accessible（模型可访问） | pass |
| 2 | backend_healthz（后端健康检查） | pass |
| 3 | frontend_root（前端根路径） | pass |
| 4 | backend_calls_model（后端调模型） | pass |
| 5 | frontend_reaches_backend（前端连后端） | pass |
| 6 | query_normal（正常问答） | pass |
| 7 | query_calculation（财务计算） | pass |
| 8 | query_unanswerable_safe（不可回答安全拒绝） | pass |
| 9 | sse_terminates（SSE 正常终止） | pass |
| 10 | trace_id_present（Trace ID 存在） | pass |
| 11 | no_path_leak_in_errors（错误不含路径泄漏） | pass |
| 12 | restart_recovery（重启恢复） | pass |

---

## 7. 性能基线（Phase 7 实测）

**来源产物**：`artifacts/deployment/phase7/performance-report.json`

| 端点 | 平均延迟 | p50 | p95 | 状态 |
|------|---------|-----|-----|------|
| model_health | 0.6ms | 0.5ms | 1.2ms | ok |
| model_models | 0.4ms | 0.4ms | 0.5ms | ok |
| backend_healthz | 0.6ms | 0.4ms | 1.4ms | ok |
| backend_readyz | 6482.9ms | 6441.8ms | 6625.4ms | ok |
| frontend_root | 2.6ms | 0.8ms | 7.2ms | ok |
| query_normal | 7.7ms | 6.7ms | 14.8ms | ok |
| query_calculation | 5.8ms | 5.2ms | 7.9ms | ok |

**资源占用**：
- GPU 显存：约 48.2 GB
- Model 进程 RSS：约 1,682 MB
- Backend 进程 RSS：约 1,170 MB
- Frontend 进程 RSS：约 66.7 MB
- CPU 1min 平均负载：约 60.55

> **重要声明**：以上性能数据为**基线指标**，未做推理优化、批处理、缓存或调优。不可将其视为生产级性能指标。

---

## 8. 指标来源汇总表

| 指标 | 值 | 来源产物路径 |
|------|-----|-------------|
| 确定性操作数 | 9 | `docs/release/rag-system-card.md` |
| 校验类别数 | 6+ | `docs/release/rag-system-card.md` |
| 在线服务数 | 3 | `docs/deployment/online-deployment.md` |
| Phase 7 验收 | 42/42 | `artifacts/deployment/phase7/phase7-acceptance.json` |
| 自动化测试 | >2000 | `artifacts/baseline/test-summary.json` |
| 部署冒烟 | 12/12 | `artifacts/deployment/phase7/smoke-report.json` |
| 健康检查 | 3/3 通过 | `artifacts/deployment/phase7/health-report.json` |
| 性能测量端点 | 7 端点 | `artifacts/deployment/phase7/performance-report.json` |
| 部署清单 | 完整 | `artifacts/deployment/phase7/deployment-manifest.json` |
| SSH 登出持久 | 通过 | `artifacts/deployment/phase7/logout-persistence-report.json` |
| SSH 隧道 | 通过 | `artifacts/deployment/phase7/ssh-tunnel-report.json` |
| 模型架构 | 24层/12头/1536维/≈1.4B参数 | `artifacts/release/phase6/checkpoint-manifest.json` |
| Tokenizer | BPE / 65K / 9特殊token | `artifacts/release/phase6/tokenizer-manifest.json` |
| SFT 数据量 | 39,534 样本 / 8 来源 | `artifacts/release/phase6/sft-data-manifest.json` |
| 许可证 | MIT | `artifacts/release/phase6/license-inventory.json` |
| Eval 数据分类 | synthetic_held_out | `artifacts/release/phase6/evaluation-evidence.json` |
| 声明-证据映射 | 22 条声明 | `artifacts/release/phase6/claim-evidence-map.json` |

---

## 9. 不应作为指标引用的数据

以下数据**明确不应**作为质量指标引用：

| 数据点 | 原因 | 来源 |
|--------|------|------|
| 0/54 strict pass | 非真正独立 Sealed Evaluation；仅用于基础设施功能测试 | `artifacts/release/phase6/evaluation-evidence.json` |
| 17.68B tokens | 历史自报，当前服务器产物不可独立验证 | `artifacts/release/phase6/claim-evidence-map.json` |
| Tokenizer 压缩率 | 未经独立推理速度验证 | `artifacts/release/phase6/claim-evidence-map.json` (prohibited claim) |
| 模型 checkpoint hash | 历史产物不可用 | `artifacts/release/phase6/checkpoint-manifest.json` |
| 训练 val_bpb | 历史自报（base 0.7626, SFT 0.5558） | `artifacts/release/phase6/training-runs.json` (historical_self_reported) |
| CORE metric 0.2201 | 历史自报，原始训练日志不可用 | `artifacts/release/phase6/claim-evidence-map.json` |
