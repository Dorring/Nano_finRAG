<div align="center">

# 🚀 NanoFinance: Trusted Financial Agentic RAG

**融合金融领域模型、Agentic 调度编排与确定性可信计算的金融多轮问答系统**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%20CUDA%2012.6-ee4c2c.svg)](https://pytorch.org/)
[![Model Size](https://img.shields.io/badge/Model-2.08B%20Specialist-blueviolet.svg)](finquery_rag/backend/docs/showcase/nano-finance-interview-evidence.md)
[![Holdout Accuracy](https://img.shields.io/badge/ORCL%20Holdout-99.80%25-success.svg)](finquery_rag/backend/docs/showcase/nano-finance-interview-evidence.md)
[![Retrieval Latency](https://img.shields.io/badge/Retrieval%20P95-31.2ms-brightgreen.svg)](finquery_rag/backend/docs/showcase/nano-finance-interview-evidence.md)

[**[系统架构]**](#-系统全景架构) • [**[核心指标]**](#-核心评测与量化效果) • [**[关键技术]**](#-四大核心技术模块) • [**[多轮评测]**](#-140-case-多轮上下文基准) • [**[快速上手]**](#-快速上手)

</div>

---

## 📌 项目介绍

在上市公司财报分析与专业金融问答场景中，通用大模型与传统 RAG 面临三大核心工程挑战：
1. **复杂表格检索噪声高**：财报行列交叉、附注交叉披露导致单查询召回容易丢失多槽位关键证据；
2. **数值计算幻觉频发**：纯生成式模型擅自篡改数字、量纲（如把 `$49.8M` 篡改成 `$49.8%`）或执行不可靠的浮点心算；
3. **多轮对话上下文稀释与污染**：历史会话中的未核验数字被错误当作后续计算事实，长会话导致 Token 与延迟线性爆炸。

**NanoFinance** 构建了一套**端到端分层解耦的金融智能体推理体系**：通过在 17.7B Token 混合语料上预训练的 **2.08B 本地金融专家模型**，结合 **Agentic RAG 调度编排**、**多轮上下文自适应降噪**、**9 类确定性金融计算工具**与**零释放权限外部强校验链**，实现数值计算与自然语言生成的彻底解耦，保障财报问答的绝对事实准确性与安全可信。

---

## 📊 核心评测与量化效果

> 本项目所有指标均来自严格对齐的基准测试集、未见过的上市公司盲测集（Company-Held-Out）及端到端自动化回归套件。

| 核心维度 | 评测基准 / 范围 | 对照基线 (Baseline) | NanoFinance 最终效果 | 核心增益 / 性能优势 |
| :--- | :--- | :--- | :--- | :--- |
| **金融领域模型能力** | 金融多任务宏观评测 (Macro QA) | Qwen3.5-2B: `7.86%` | **NanoFinance 2.08B: 19.78%** | **+11.92 pp** (性能提升 2.5x) |
| **证据约束生成能力** | ORCL 500 题盲测集 (Company-Held-Out) | 原始通用模型: ~12% | **Step-156 专家: 99.80% (499/500)** | **释放正确率 100%**，实质错误 0 |
| **生成器训练效果** | 严格对齐 68-Packet 消费回归 | 原始 Financial SFT: `11.76%` | **Step-156 专家: 76.47% (52/68)** | **+64.71 pp** (净拯救 44 题，0 回归) |
| **检索第一阶段召回** | T²-RAGBench 财报复杂检索集 | 基础 BM25 检索: `78.1%` | **R4 组合检索: 88.6% (Recall@5)** | **Recall@10 达 100.0%**，多槽位召回 100% |
| **多轮独立意图重构** | 140-Case 多轮金融问答压力基准 | 基础拼接: 易受旧主题污染 | **独立问题重构率: 97.86% (137/140)** | 歧义澄清率 **100%**，信任违规 0 |
| **可信计算与执行安全** | 全流程端到端执行回归 | 传统 Agent 心算易错 | **False Binding = 0, False Execution = 0** | **未核验证据零泄露**，异常 Fail-Closed |
| **工程端到端时延** | 真实生产环境压测 (RTX A6000) | 传统多 Agent 级联: >5s | **检索 P95 31.2ms / P50 15.6ms** | 模型生成 P50 1.89s，0 额外 LLM 重写开销 |

---

## 🏗️ 系统全景架构

NanoFinance 确立了严格清晰的**三层职责分离架构**：
- **Layer 1** 负责 *“理解用户在多轮中到底想问什么”*（将省略句恢复为独立金融请求）；
- **Layer 2** 负责 *“拆解金融任务并受约束地调度能力与工具”*；
- **Layer 3** 负责 *“依据可信证据与数学引擎，严格决定答案能否释放”*。

```
                       ┌──────────────────────────────────────────────┐
                       │        User Turn (多轮对话输入 / Session)      │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Conversation Context Layer (多轮会话管理与意图重构)                                     │
│  ├─ 动态相关性过滤 (Relevance Filter): Query + State + Recent Turns 多维信号评分动态降噪          │
│  ├─ 三层分层记忆管理 (L1 原始近期轮次 / L2 结构化语义状态 / L3 话题摘要压缩): 500 轮 Token 恒定收敛   │
│  ├─ 上下文预算控制 (Application Context Budget): 严格 application-level 预算裁剪，防注意力稀释   │
│  ├─ 指代与相对时期消解: 自动还原实体继承、指标继承、相对时间 (如 "上一年" -> "FY2023")            │
│  ├─ 显式优先准则 (Explicit Query Override): 用户当前输入 100% 优先，历史仅补全缺失槽位          │
│  ├─ 主动歧义澄清门禁 (Ambiguity Clarification Gate): 多指标歧义主动返回选项交互，严禁模型盲猜     │
│  └─ 自包含快速旁路 (Fast-Path Bypass): 单轮与自包含请求直接直通，0 额外模型开销                  │
└─────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                      │
                                      ▼ [重构为语义完备的 Standalone Financial Query]
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Agentic Orchestration (Financial RAG Supervisor 金融调度中枢)                         │
│  ├─ 受约束请求解析 (SupervisorPlan): 确定性提取 Query 语义坐标与意图                             │
│  ├─ 能力与工具分流 (GeneratorRoutingPolicy):                                                   │
│  │   ├── STRUCTURED_SINGLE  ──► 确定性结构化渲染器 (Deterministic Renderer)                     │
│  │   ├── CALCULATION        ──► 9 类确定性金融计算工具 / C1 算术核验 (Deterministic Calculator)    │
│  │   └── QUALITATIVE/MULTI  ──► 2.08B 本地金融专家模型 (Local Financial Specialist)            │
│  └─ 有界状态机控制: 依据证据完整性与异常原因码 (Reason Codes) 触发有界定向补检与安全退出           │
└─────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│  Layer 3 — Trusted Financial Execution (可信证据检索、绑定、计算与零释放权限校验链)                 │
│  ├─ R4 组合检索架构:                                                                           │
│  │   ├── 结构化文档元数据检索 (Structured Sidecar Search) ──► 财报表头/版本信息 100% 召回        │
│  │   └── 分槽感知检索 (Slot-Aware Retrieval & Dedup Merge) ──► 解决跨表/附注第二证据挤压缺失      │
│  ├─ 语义证据绑定器 (Semantic Evidence Binder): 强制执行 RequiredSlot 语义对齐与证据约束         │
│  ├─ 2.08B 本地金融专家模型: 基于 FinancialGenerationViewV1 结构化证据契约受约束生成               │
│  ├─ 确定性计算引擎: 浮点精确计算同比增长、毛利率变动、EPS、复合增长率等，与生成彻底解耦           │
│  ├─ 外部零释放权限强校验链 (Runtime Validator Chain):                                          │
│  │   ├── 语义断言验证 (SemanticClaimVerifier) ──► 严禁无源事实                                  │
│  │   ├── 数值与量纲校验 (Numeric / UnitCurrencyScaleValidator) ──► 拦截 "$49.8% M" 等畸变       │
│  │   ├── 期间与引用校验 (Period / CitationValidator) ──► 校验财报时期与事实溯源                 │
│  │   └── 算术一致性校验 (C1Validator) ──► 强制校验生成内容与计算工具输出完全一致                │
│  └─ 确定性安全门禁 (Fail-Closed Release Gate): 校验链拥有一票否决权，模型自身 0 释放权限        │
└─────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
      【 Verified Trusted Answer 】               【 Safe Fail-Closed 】
      (全链路强核验证据与确定性答案)               (证据不足/校验失败时安全拒答)
```

---

## 💡 四大核心技术模块

### 1. 金融领域模型端到端预训练与证据约束生成对齐 (Financial LLM & Grounded SFT)
- **领域持续预训练 (CPT)**：基于 SEC 10-K/10-Q 财报、金融研报与结构化披露等 **17.7B Token 金融混合语料库**，完成 2.08B 参数模型的金融领域知识注入。
- **多任务指令微调 (SFT)**：构建 **4 万条高质量金融多任务指令数据集**，涵盖财报指标提取、跨期对比、比率分析与风险因素归纳，金融多任务评测达到 **19.78%**（较同参数通用模型 Qwen3.5-2B **绝对提升 +11.92pp**）。
- **可信证据约束蒸馏 (Grounded V3 Distillation)**：构建 **1.6 万条**高精度 Teacher 蒸馏样本，混合 20% 领域通用回放数据（80/20 Mixture），通过仅对 Response 计算 Loss 的严谨微调，强化模型在多证据综合、事实引用溯源与计算结果保持上的能力。
- **词典序模型选择 (Lexicographic Selection)**：制定 `Safety Invariants (0 违规) -> Financial Macro Retention (≥18%) -> Strict Correct (盲测最高)` 的多级筛选准则，选出最优 Step-156 Checkpoint。在未参与训练的 **ORCL 500 题全新盲测集**上达到 **99.80% Strict Correct (499/500)**。

---

### 2. 多轮会话管理与 Agentic RAG 编排 (Multi-turn Context & Supervisor)
- **三层分层上下文管理机制**：坚持 `Model Context Capacity != Conversation Memory Strategy` 原则。即便外层模型具备大窗口，依然通过 **L1（最近 4 轮原始交互）+ L2（结构化语义状态）+ L3（高阶话题摘要压缩）** 进行分层管理，并在 **500 轮长会话压力测试**下保证有效上下文稳定收敛在 ~500 tokens，杜绝语义稀释与上下文污染。
- **动态相关性降噪 (Context Relevance Filter)**：综合考虑当前 Query、DialogueState 实体、活动主题、显式引用与时效衰减，对历史轮次进行多维评分与动态淘汰，精准隔离闲聊与过时主题。
- **强信息优先级与显式覆盖**：确立 $\text{当前显式输入} > \text{明确引用轮次} > \text{结构化状态} > \text{历史摘要}$ 的优先级准则（`EXPLICIT_QUERY_OVERRIDE`），历史仅用于补全省略槽位，绝不反向污染用户当前指定的实体。
- **Financial RAG Supervisor 与有界状态机**：Supervisor 接收重构后的独立金融请求，根据 `SupervisorPlan` 进行确定性意图分流与工具调度；依据异常原因码（如 `FIRST_STAGE_MISS`、`MULTI_SLOT_INCOMPLETE`）触发有界定向补检，达成高效闭环。

---

### 3. 可信推理与确定性计算解耦 (Trusted Inference & Deterministic Execution)
- **结构化证据契约 (`FinancialGenerationViewV1`)**：定义包含文档元数据、时间跨度语义、表格单元格溯源标识（Cell ID/Table ID）的权威视图契约，统一检索与生成的交互接口。
- **9 类确定性金融计算工具**：针对财务分析中的**同比增长率 (YoY)、环比增长率 (QoQ)、结构占比、毛利率变动 (Margin Delta)、每股收益 (EPS)、复合年均增长率 (CAGR)、营运资金变动**等，通过确定性 Python 数学引擎精确计算，彻底杜绝大模型浮点运算幻觉。
- **不可信会话信任边界 (`CONVERSATION_CONTEXT_NOT_EVIDENCE`)**：确立不可动摇的安全准则——历史会话中的 Assistant 回答即便包含数字，也绝不转换为权威证据或直接填入计算工具操作数；所有数值必须重新通过检索与证据绑定器从财报源头提取。

---

### 4. 外部零释放权限强校验链与 Fail-Closed 门禁 (Runtime Validator Chain)
- **解耦生成与准入权限**：2.08B 本地金融专家模型**仅承担受约束的自然语言组织与观点陈述，模型自身具备 0 释放权限**。
- **多维度强校验流水线**：
  1. `SemanticClaimVerifier`：校验生成内容中的每一个金融事实断言是否均存在对应的已核验证据；
  2. `Numeric & UnitCurrencyScaleValidator`：严格比对生成文本中的数值与量纲（拦截如 `$49.8% million`、`百万元 vs 十亿元` 等常见单位混乱）；
  3. `PeriodValidator & CitationValidator`：校验财务时期与引用标签的严格一致性，杜绝伪造引用（Phantom Citations）；
  4. `C1Validator`：强制比对模型陈述的计算结果与确定性计算工具输出，差异为 0 方可通过。
- **确定性 Fail-Closed 机制**：一旦校验链中任何一项规则未通过，系统立即拦截并执行 Fail-Closed 安全拒答，确保进入生产环境的答案**实质性错误释放数为 0**。

---

## 🧪 140-Case 多轮上下文基准评测

为了全面验证 Conversation Context Layer 在复杂多轮财务交互中的稳定性，构建了覆盖 11 个维度的 **140-Case Multi-Turn Context Benchmark**：

```text
======================================================================
140-Case Multi-Turn Context Extension Benchmark Results
======================================================================
  [01] 实体继承 (Entity inheritance)            : 12 / 15 ( 80.00%) | 违规: 0
  [02] 指标继承 (Metric inheritance)            : 15 / 15 (100.00%) | 违规: 0
  [03] 时期继承 (Period inheritance)            : 15 / 15 (100.00%) | 违规: 0
  [04] 相对时期消解 (Relative-time resolution)  : 10 / 10 (100.00%) | 违规: 0
  [05] 代词指代消解 (Pronoun/reference)         : 10 / 10 (100.00%) | 违规: 0
  [06] 跨轮多步计算 (Cross-turn calculation)    : 15 / 15 (100.00%) | 违规: 0
  [07] 主题切换降噪 (Topic switch / noise)      : 15 / 15 (100.00%) | 违规: 0
  [08] 歧义主动澄清 (Ambiguity clarification)   : 15 / 15 (100.00%) | 违规: 0 (15/15 触发澄清)
  [09] 长上下文压力 (Long-context stress)       : 10 / 10 (100.00%) | 违规: 0
  [10] 信任边界对抗 (Trust-boundary adversarial): 10 / 10 (100.00%) | 违规: 0 (
  [11] 独立性保持 (Standalone preservation)     : 10 / 10 (100.00%) | 违规: 0
----------------------------------------------------------------------
  总体独立意图重构准确率: 137 / 140 (97.86%) | 信任边界违规: 0 | 状态污染: 0
======================================================================
```

---

## 📂 项目工程目录结构

```text
finquery_rag/backend/
├── docs/showcase/                              # 权威证据封存与设计展示文档
│   ├── nano-finance-interview-evidence.md     # 面试证据链、标准量化话术与局限性声明
│   ├── multiturn-context-baseline-audit.md    # M0 阶段基准架构审计文档
│   ├── multiturn-context-design.md            # 多轮上下文层完整架构设计规范
│   └── multiturn-context-evaluation.md        # 140 题多轮压力基准评测报告
├── src/
│   ├── conversation/                          # Layer 1: 多轮会话管理与上下文解释层
│   │   ├── contracts.py                       # 对话轮次、语义状态、意图重构数据契约
│   │   ├── bailian_client.py                  # 阿里云百炼客户端封装 (Thinking=False, 指数退避重试)
│   │   ├── resolver.py                        # ContextualQueryResolver (Fast Path 与重构逻辑)
│   │   ├── relevance_filter.py                # 动态降噪过滤 (多维信号评分)
│   │   ├── context_budget.py                  # 三层分层记忆 (L1/L2/L3) 与 Token 预算管理器
│   │   ├── store.py                           # 线程安全会话隔离存储 (InMemoryConversationStore)
│   │   └── service.py                         # ConversationContextManager 核心编排服务
│   ├── generation/                            # Layer 3: 本地金融专家生成器与校验链
│   │   ├── local_specialist_generator.py      # Step-156 模型加载、显存动态控制与推理服务
│   │   ├── generator_routing_policy.py        # 渲染器 / 计算器 / 专家模型分流路由策略
│   │   └── runtime_validator_chain.py         # 语义、数值、量纲、期间、引用、C1 强校验流水线
│   └── retrieval/                             # Layer 3: 结构化与分槽检索模块
│       └── metadata_scope.py                  # 财报元数据过滤与检索范围规划
├── rag_v2/runtime/                            # Layer 2: Financial RAG Supervisor 调度中枢
│   ├── contracts.py                           # TrustedRAGQueryV2, TrustedRAGResponseV2 契约
│   └── runtime.py                             # TrustedRAGRuntimeV2 核心执行引擎
├── scripts/
│   ├── evaluation/                            # 评测基准运行脚本
│   │   └── run_multiturn_context_eval.py      # 140-Case 多轮上下文基准评测套件
│   └── runtime/                               # 生产与回归验证脚本
│       ├── run_nf_v2_21_runtime_integration.py# 运行时集成与显存/时延压测脚本
│       ├── run_nf_v2_22_shadow_verification.py# Shadow 模式验证与 31 项 Fail-Closed 对账
│       └── run_nf_v2_23_retrieval_final_mile.py# R4 检索最终一公里恢复与 120 题回归套件
└── tests/conversation/                        # 单元测试与对抗测试套件
    ├── test_conversation_contracts.py         # 契约实例化与 ReasonCode 测试
    ├── test_resolver.py                       # 快速旁路、继承与歧义门禁测试
    ├── test_relevance_and_budget.py           # 动态降噪与 500 轮 Token 预算收敛测试
    ├── test_trust_boundary.py                 # 历史虚假数字零传播对抗测试
    ├── test_standalone_preservation.py        # 长历史后独立问题防污染测试
    └── test_session_isolation.py              # 多 Session 状态物理隔离测试
```

---

## ⚡ 快速上手

### 1. 环境依赖配置
```bash
# 克隆仓库
git clone https://github.com/Dorring/Nano_finRAG.git
cd Nano_finRAG/finquery_rag/backend

# 安装依赖
pip install torch transformers pydantic tiktoken
```

### 2. 环境变量配置
系统所有核心开关均已参数化配置，无任何硬编码常量：
```bash
# 多轮会话扩展配置
export MULTITURN_CONTEXT_ENABLED=true
export BAILIAN_API_KEY="your-bailian-api-key"
export BAILIAN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export BAILIAN_CONTEXT_MODEL="qwen3.6-flash"
export BAILIAN_CONTEXT_THINKING=false

# 内存预算配置 (Application-Level Budget)
export CONTEXT_RECENT_TURNS=4
export CONTEXT_SUMMARY_TRIGGER_TURNS=8
export CONTEXT_TARGET_TOKENS=4096
export CONTEXT_MAX_TOKENS=8192
export CONTEXT_RESOLVER_MAX_OUTPUT_TOKENS=512
```

### 3. 运行全量测试与基准评测
```bash
# 1. 运行多轮会话模块全量单元测试与对抗测试
python -m unittest discover -s tests/conversation -p 'test_*.py'

# 2. 运行 140-Case 多轮金融上下文压力基准
python scripts/evaluation/run_multiturn_context_eval.py

# 3. 运行 120 题端到端单轮财务问答回归
python scripts/runtime/run_nf_v2_23_retrieval_final_mile.py
```

---

## 📄 许可证 (License)

本项目采用 [Apache License 2.0](LICENSE) 许可证开源。
