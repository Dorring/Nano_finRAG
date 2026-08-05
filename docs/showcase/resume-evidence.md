# 简历素材 — Resume Evidence

本文档提供 nano_finance 项目在简历中可以使用的定位描述、可引用指标和明确禁止声明的清单。所有数据均来自 `README.md` 和 `artifacts/` 目录下可验证的产物。

---

## 项目一句话定位

> 基于 NanoChat 训练框架，从零构建了一个端到端的中文金融领域大模型（约 1.4B 参数）与 RAG 检索增强系统，包含自定义 tokenizer、混合检索、确定性金融计算、接地校验（fail-closed）和无 root 在线部署。

---

## 可选变体（根据简历空间调整）

**简短版（1 行）**：
> 从 tokenizer 到部署，全栈构建了一个中文金融领域大模型 + 确定性计算 RAG 系统（1.4B 参数 / 9 种金融操作 / 3 服务上线）。

**详细版（3-4 行，适合项目描述区）**：
> 基于 Andrej Karpathy 的 NanoChat 训练框架，从 tokenizer 训练开始，完成了中文金融领域大模型的全流程构建：Byte-Level BPE tokenizer（65K 词汇表）、领域预训练 + SFT（assistant-only loss）、混合检索（Dense + BM25 + RRF + Reranker）、确定性 Decimal 金融计算器（9 种操作）、接地校验管道（6+ 校验类别，fail-closed 设计）、以及无 root 三服务在线部署（Model/Backend/Frontend，tmux + SSH 隧道）。

---

## 可引用的工程指标

以下所有指标均有源产物可验证：

| 指标 | 数值 | 来源 |
|------|------|------|
| 金融确定性操作 | 9 种（差值、增长率、占比、求和、均值、毛利率、净利率、负债率、单位换算） | `docs/release/rag-system-card.md` |
| 校验类别 | 6+（Answerability + Claim + Numeric + Unit/Period + Citation + Calculation + Unsupported Claim） | `docs/release/rag-system-card.md` |
| 在线服务 | 3 个（Model :18001 / Backend :18002 / Frontend :18003） | `docs/deployment/online-deployment.md` |
| 部署验收 | Phase 7 验收 42/42 全部通过，0 失败 | `artifacts/deployment/phase7/phase7-acceptance.json` |
| 自动化测试 | >2000 条，零失败 | `artifacts/baseline/test-summary.json` |
| 冒烟测试 | 12/12 全部通过 | `artifacts/deployment/phase7/smoke-report.json` |
| 模型参数 | ≈1.4B（24 层 / 12 头 / 1536 维） | `artifacts/release/phase6/checkpoint-manifest.json` |
| Tokenizer | Byte-Level BPE / 65K 词汇表 / 中英 + 金融语料 | `artifacts/release/phase6/tokenizer-manifest.json` |
| SFT 数据 | 39,534 样本 / 8 数据源 | `artifacts/release/phase6/sft-data-manifest.json` |
| 混合检索 | Dense (ChromaDB) + BM25 + RRF + Reranker | `docs/release/rag-system-card.md` |
| 部署方式 | tmux + shell 脚本 / 无 root / 无 Docker / 无 systemd / SSH 隧道 | `docs/deployment/online-deployment.md` |
| 许可证 | MIT | `artifacts/release/phase6/license-inventory.json` |

---

## 可引用的技术标签

以下标签可在简历的技能/技术栈部分使用：

| 标签 | 上下文 |
|------|--------|
| LLM 训练 | Tokenizer 训练、预训练、SFT、assistant-only loss |
| RAG | 混合检索、RRF 融合、Reranker、接地校验、Answerability |
| 确定性计算 | Python Decimal、操作数证据绑定、fail-closed |
| 部署 | 无 root 运维、tmux、SSH 隧道、健康检查 |
| 测试 | pytest、冒烟测试、验收测试、评估隔离 |
| 评测 | Blind Scoring、Ablation Study、Calibration Set、RC Freeze |

---

## 明确不声称的内容

以下内容**绝不能**出现在简历中：

| ❌ 不要声称 | ✅ 原因 |
|-------------|---------|
| 模型达到生产级金融精度 | 0/54 synthetic-held-out 不是质量指标；被 claim-evidence-map 列为 prohibited |
| 系统消除了幻觉 | Validation 是 best-effort，不能完全消除；被列为 prohibited claim |
| 支持原生 Function Calling | 模型未训练 Tool Use；Calculator 是系统编排，非模型能力 |
| Phase 5 证明强泛化能力 | Phase 5 是基础设施功能测试，不是质量评估 |
| Tokenizer 带来 X 倍推理加速 | 压缩率 ≠ 推理速度，未经实测验证 |
| 训练数据量 17.68B tokens | 历史自报数据，当前产物不可独立验证 |
| 特定 val_bpb 值作为性能指标 | base BPB 0.7626 和 SFT BPB 0.5558 均为 historical_self_reported |

---

## 面试中的"诚实叙述"模板

当被问到项目成果时，建议使用以下结构：

1. **说清楚做了什么**：9 种金融操作、6+ 校验类别、3 服务部署、2000+ 测试、42/42 验收
2. **说清楚设计理由**：为什么不用 LLM 算数、为什么融合 Dense 和 BM25、为什么 Repair Once
3. **主动说出局限性**：模型不可复现验证、Calculator 仅 9 种操作、Validation 不能消除幻觉、部署无自动扩展
4. **区分系统和模型**：Calculator 是系统组件，不是模型能力；Validation 是系统校验，不是模型不会犯错

这种"主动谈局限"的叙述方式在面试中通常比纯吹牛更有效。

---

## 相关文档

- [verified-metrics.md](verified-metrics.md) — 所有已验证指标及源产物路径
- [demo-guide.md](demo-guide.md) — 演示场景操作指南
- [interview-guide.md](interview-guide.md) — 面试 Q&A
- [known-claims.md](known-claims.md) — 不应做的声明

---

## NF-OPT-17：独立金融 Hard-negative 评测

可用于简历的严谨表述：

> 构建80题、160条可追溯金融Hard Negative的跨公司隔离评测集，完成Issuer-disjoint Reranker诊断；通用Reranker Zero-shot Top-1达到87.5%、Pairwise达到93.75%，并以预注册门禁控制训练与生产切换，避免测试集过拟合。

证据边界：

- 80题来自Alphabet、Amazon、Meta和Netflix四份独立SEC 10-K，不是冻结72题Benchmark。
- Pairwise 93.75%是三候选开发任务指标，不是生产Final Recall@5。
- 唯一有效训练仅更新Classification Head；Netflix留出集净提升为0，因此停止训练。
- 当前生产严格Final Source Recall@5继续冻结为`13/80`，没有切换Reranker。
- 不得声称“金融Reranker准确率93.75%”或“微调后获得提升”。

来源：

- `artifacts/evaluation/nf-opt-17-gate-e/zero-shot-development-results.json`
- `artifacts/evaluation/nf-opt-17-gate-f/issuer-disjoint-training-result.json`
- `artifacts/evaluation/nf-opt-17-gate-f/nf-opt-17-gate-f-acceptance.json`

---

## Structured Financial Fact Retrieval V2

可用于简历的严谨表述：

> 构建PDF混合检索与原生Inline XBRL结构化Fact双口径评测：PDF严格Candidate Identity第一阶段Dense Source Recall@200为65%；在由Costco、Home Depot和PepsiCo三家全新10-K组成的模板约束留出集上，Structured Fact Recall@5达到120/120，并实现双期间Complete Pair Recall@5 30/30。

必须同时保留以下边界：

- `120/120`仅适用于原生Inline XBRL、固定Question模板和精确Document/Concept/Period检索。
- 它不覆盖PDF叙述、附注语义、开放式问题或同义Metric改写，不能写成“金融RAG Recall@5=100%”。
- 旧PDF严格Final Source Recall@5仍为`13/80`，没有被V2指标替换。
- 留出集包含100题：60条单事实、30条双期间、10条拒答；共120条Gold Fact。
- 第一次本地留出执行因读取预解析Case Slot而作废；最终结果来自Question-only修正版，修正过程已单独留档。

来源：

- `artifacts/evaluation/structured-fact-v2-benchmark/holdout-retrieval-results.json`
- `artifacts/evaluation/structured-fact-v2-benchmark/holdout-acceptance.json`
- `artifacts/evaluation/structured-fact-v2-benchmark/evaluation-semantic-closure.json`
