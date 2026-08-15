# nano_finance

*金融垂类大模型训练、有据可依的RAG、确定性计算、校验与无Root在线部署*

[English version](README.md)

---

## 项目概览

nano_finance 是一个端到端的金融领域语言模型与 RAG 系统，基于原生
[NanoChat](https://github.com/karpathy/nanochat) 训练栈构建。

项目覆盖分词器适配、领域预训练、监督微调、混合检索、确定性金融计算、有据可依的答案校验、评估治理以及无 Root 在线部署——全部设计为可在无 root 权限、无 Docker、无系统级工具的大学服务器上运行。

---

## 解决的核心问题

| 问题 | 解决方案 |
|---|---|
| 英文原生分词器对中文金融文本效率低下 | 自定义 65K 词表大小的 Byte-Level BPE 分词器 |
| 通用大模型缺乏金融领域知识 | 领域自适应预训练 + 金融 SFT |
| 普通 RAG 产生检索错误、数字幻觉和单位错误 | 混合检索（Dense + BM25 + RRF + Reranker）配合层次化上下文 |
| LLM 自由形式计算缺乏可审计性 | 基于 Decimal 的确定性金融计算器，操作数绑定证据来源 |
| 答案可能缺乏来源或包含错误引用 | 有据可依的校验流水线，覆盖 6+ 个校验类别 |
| 大学服务器：无 root、无 Docker、端口受限 | 基于 tmux 的三服务无 Root 部署，配合 SSH 隧道访问 |

---

## 核心能力

| 能力 | 描述 |
|---|---|
| **领域分词器** | Byte-Level BPE，65K 词表，中英混合 + 金融语料 |
| **金融预训练与 SFT** | 基础预训练 → 领域自适应 → 仅对助手部分计算损失的监督微调 |
| **混合 RAG** | Dense 检索 + BM25 + RRF 融合 + Reranker + 层次化上下文 |
| **确定性金融计算器** | 9 种金融运算，Decimal 精度，操作数绑定证据，单位/量纲校验 |
| **有据可依的校验** | 可回答性、论断提取、数值/单位/时期/引用/计算校验、修复一次、安全回退 |
| **无 Root 三服务部署** | 模型（18001）→ 后端（18002）→ 前端（18003），tmux，SSH 隧道，健康/冒烟/重启验证 |

---

## 系统架构

```mermaid
flowchart LR
    A[用户 / Web 界面] --> B[FastAPI 后端]
    B --> C[查询处理]
    C --> D[Dense 检索]
    C --> E[BM25 检索]
    D --> F[RRF 融合]
    E --> F
    F --> G[Reranker]
    G --> H[上下文构建器]

    H --> I{意图识别}
    I -->|计算| J[确定性计算器]
    I -->|文档问答| K[金融 LLM]

    J --> L[有据可依的校验]
    K --> L
    L --> M[答案 / 安全回退]
```

### 训练流水线

```text
原始语料 → 金融分词器 → 基础预训练 → 领域预训练 → SFT → 模型服务 → RAG 应用
```

架构将**模型能力**、**系统编排**、**确定性计算**、**检索**、**校验**和**在线部署**分离为独立、可审计的层次。确定性计算器是系统组件，而非模型原生的工具调用。

---

## 核心能力详解

### 1. 分词器与训练流水线

- 自定义 Byte-Level BPE 分词器，65K 词表大小
- 中英混合通用语料 + 中文金融语料
- 基础预训练 → 领域自适应 → 监督微调
- SFT 阶段仅对助手部分计算损失

> 标记为*历史自述*的训练数据目前无法进行独立验证。

### 2. 混合 RAG

- **Dense 检索**：通过 ChromaDB 进行语义向量搜索
- **BM25**：稀疏词汇检索，用于关键词匹配
- **RRF 融合**：对稠密和稀疏结果进行倒数排序融合
- **Reranker**：对融合后的候选结果进行 Cross-encoder 重排序
- **层次化上下文**：文档范围控制、页面级分块、来源归属
- **上下文充分性**：自动检测上下文不足并安全拒绝

### 3. 确定性金融计算

使用 Python `Decimal` 实现的九种确定性运算：

| 运算 | 描述 |
|---|---|
| `difference` | 两个值之间的绝对差值 |
| `growth_rate` | 从基期到目标值的百分比增长率 |
| `percentage_share` | 某部分占整体的比例 |
| `sum` | 多个值的求和 |
| `average` | 多个值的算术平均 |
| `gross_margin` | （营收 - 营业成本）/ 营收 |
| `net_margin` | 净利润 / 营收 |
| `debt_ratio` | 总负债 / 总资产 |
| `scale_conversion` | 单位转换（如百万转十亿） |

核心保障：
- 所有操作数必须绑定到文档、页面和块级别的证据
- 单位和量纲在计算前进行校验
- 失败时故障关闭：不回退到 LLM 重新计算
- 证据缺失时安全阻断

### 4. 有据可依的校验

校验流水线对每个答案检查以下方面：

- **可回答性**：该问题是否能从现有文档中回答？
- **论断提取**：将答案分解为可验证的论断
- **数值校验**：引用的数字是否与源文本一致？
- **单位/时期校验**：单位和时间周期是否正确传递？
- **引用校验**：每个论断是否有有效的来源引用？
- **计算校验**：计算操作数是否可追溯到证据？
- **无据论断校验**：是否存在无证据支持的论断？
- **修复一次**：单次确定性修复尝试（无 LLM 循环）
- **安全回退**：被阻断或失败的答案使用安全回退消息

系统通过确定性校验和故障关闭的响应处理来减少无依据的回复。

### 5. 在线部署

三个服务作为用户空间进程在 tmux 下运行：

| 服务 | 端口 | 会话名称 |
|---|---|---|
| 模型服务 | 127.0.0.1:18001 | `nano-finance-model` |
| 后端服务 | 127.0.0.1:18002 | `nano-finance-backend` |
| 前端服务 | 127.0.0.1:18003 | `nano-finance-frontend` |

特性：
- 无 root、无 Docker、无 systemd
- 基于 tmux 的进程管理，带 PID 属主验证
- 有序启动：模型 → 后端 → 前端
- 健康检查、冒烟测试、SSE 流验证、重启恢复
- SSH 隧道用于远程访问
- 服务器重启后需手动重启

---

## 已验证的工程指标

仅呈现可独立验证的结果：

| 指标 | 数值 | 含义 |
|---|---|---|
| 确定性金融运算 | 9 | 覆盖差值、增长率、利润率比率和单位转换 |
| 校验类别 | 6+ | 数值、单位/时期、引用、计算、无据论断、可回答性 |
| 在线服务数 | 3 | 模型、后端、前端 |
| 阶段七部署验收 | 42/42 | 完整的三服务链路验证 |
| 自动化测试 | 2,000+ | 全部通过，零失败 |
| 部署冒烟测试 | 12/12 | 健康、问答、计算、SSE、重启 |

详细指标及其来源见
[docs/showcase/verified-metrics.md](docs/showcase/verified-metrics.md)。

> 以下内容**明确不**作为质量声明使用：合成留出集 0/54、无法验证的 17.68B tokens、未复现的分词器压缩率、未验证的检查点哈希、失败的实验 BPB，或任何"生产级精度"声明。

---

## 演示

五个演示场景记录在
[docs/showcase/demo-guide.md](docs/showcase/demo-guide.md) 中：

1. **财报问答**：上传文档，提问，查看带页码的来源
2. **确定性金融计算**：增长率与利润率计算，操作数可追溯
3. **单位/时期歧义**：系统阻断歧义计算而非猜测
4. **无法回答的问题**：证据缺失时安全拒绝
5. **在线三服务状态**：模型/后端/前端就绪状态及 SSH 隧道访问

截图可在 [assets/demo/](assets/demo/) 中查看。

---

## 快速开始

### 本地开发

后端和前端依赖的项目特定配置见 `finquery_rag/` 目录。

### 在线部署（大学服务器）

```bash
# 启动全部三个服务
bash scripts/deploy/start_all.sh

# 检查服务状态
bash scripts/deploy/status.sh

# 运行健康检查
python scripts/deploy/healthcheck.py
```

### SSH 隧道访问

```bash
ssh -N \
  -L 18003:127.0.0.1:18003 \
  -L 18002:127.0.0.1:18002 \
  <user>@<server>
```

然后在浏览器中打开 `http://127.0.0.1:18003`。

---

## 项目时间线

| 阶段 | 重点 |
|---|---|
| 阶段一 | 检索完整性 |
| 阶段二 | RAG 编排 |
| 阶段三 | 金融计算流水线 |
| 阶段四 | 有据可依与校验 |
| 阶段五 | 评估基础设施 |
| 阶段六 | 发布证据分类 |
| 阶段七 | 无 Root 在线部署 |

详细文档：

- [docs/architecture/](docs/architecture/)
- [docs/deployment/](docs/deployment/)
- [docs/release/](docs/release/)
- [docs/showcase/](docs/showcase/)

---

## 文档索引

| 文档 | 描述 |
|---|---|
| [docs/showcase/demo-guide.md](docs/showcase/demo-guide.md) | 逐步演示指南 |
| [docs/showcase/demo-script.md](docs/showcase/demo-script.md) | 演示脚本（用于展示） |
| [docs/showcase/verified-metrics.md](docs/showcase/verified-metrics.md) | 已验证的工程指标 |
| [docs/showcase/interview-guide.md](docs/showcase/interview-guide.md) | 面试准备指南 |
| [docs/showcase/resume-evidence.md](docs/showcase/resume-evidence.md) | 简历级项目证据 |
| [docs/showcase/known-claims.md](docs/showcase/known-claims.md) | 不应做出的声明 |
| [docs/deployment/online-deployment.md](docs/deployment/online-deployment.md) | 部署指南 |
| [docs/deployment/ssh-tunnel.md](docs/deployment/ssh-tunnel.md) | SSH 隧道设置 |
| [docs/deployment/troubleshooting.md](docs/deployment/troubleshooting.md) | 故障排查指南 |
| [docs/release/model-card.md](docs/release/model-card.md) | 模型卡 |
| [docs/release/rag-system-card.md](docs/release/rag-system-card.md) | RAG 系统卡 |

---

## 已知局限

- 模型检查点验证依赖历史记录（目前无法进行独立验证）
- 服务器重启后系统需要手动重启
- 无公网访问；远程访问需通过 SSH 隧道
- 无自动扩缩容或多机部署
- 计算器仅限于文档中列出的九种运算
- 校验为尽力而为；无法保证消除所有错误

参见 [docs/deployment/known-limitations.md](docs/deployment/known-limitations.md)
和 [docs/release/limitations-and-risks.md](docs/release/limitations-and-risks.md)

---

## 上游项目与致谢

nano_finance 基于 Andrej Karpathy 的
[NanoChat](https://github.com/karpathy/nanochat) 构建——这是一个在单 GPU 节点上训练 LLM 的实验性工具集，覆盖分词、预训练、微调、评估、推理和聊天界面。

原始 nanochat 训练栈提供了基础训练基础设施、GPT 模型架构、分词器框架和聊天界面基础，nano_finance 在此基础上扩展了金融领域能力。

额外致谢：

- [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) —— 预训练优化思路
- [HuggingFace](https://hf-mirror.com/) —— 数据集（FineWeb、SmolTalk）
- [ChromaDB](https://www.trychroma.com/) —— 向量存储
- [FastAPI](https://fastapi.tiangolo.com/) —— 后端框架

---

## 许可证

MIT

## NF-V2-11 ??????

???????? Supervisor ???

Financial RAG Supervisor -> ?? / ???? -> ???????? -> ??? Calculator -> ???? Financial Specialist -> ??? Validator -> ?? / fallback / fail-closed?

Financial Specialist ???????????????? Supervisor?????????Calculator ?????????Grounding ????????????????? V1?PROJECT_FREEZE_V1_PRODUCTION??

Grounded Financial SFT ??????????? grounded???? canonical calculation ???47/64?52/64?7/11??? Multi ? 5/5??????? oracle/trusted evidence ???????? E2E ?????? E2E ???? 4/64 ? answerable ???????/grounded ? 3/64?8 ? no-answer ???????false execution ? false binding ?? 0?68/72 ??? fail-closed??? 1 ????????????? V2 ??????

???????????? finquery_rag/backend/artifacts/final-project-freeze/????????????? known-limitations.md?R1 + LoRA/DPO ???????????????????????

## NF-V2-15 Claim-Verifier 收口

最终 V2 候选链路在 `RuntimeGenerationValidatorV1` 前加入已验证的后生成
`SemanticClaimVerifierV1`。在同一份冻结 72 题回放中，保留原先 3 个正确释放，
拦截 1 个历史单位幻觉，结果为 3 个释放且 3 个正确、语义不安全最终释放为 0。
这只是组件/回放证据，不是 fresh-blind E2E 精度；覆盖仍然有限，Production 继续保持 V1。
被拒绝的 R1 + LoRA/DPO 实验只作为历史研究证据，不进入运行时。
