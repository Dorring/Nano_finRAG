# 已知不应做的声明 — Known Claims That Should NOT Be Made

本文档列出 nano_finance 项目中所有**明确不应做**的声明，每一条附带解释。此清单同时服务于面试准备、演示脚本编辑和简历写作——确保在对外沟通中不会做出未经证据支撑的过度断言。

所有条目来自 `artifacts/release/phase6/claim-evidence-map.json` 中的 `prohibited` 和 `unverified` 分类，以及 `README.md` 中的警告说明。

---

## 禁止声明（Prohibited Claims）

以下声明被明确标记为 `prohibited`，在所有对外沟通中（README、论文、简历、面试、演示）均**不可**使用：

### 1. "模型达到生产级金融精度"

**为什么不能声称**：

- 0/54 synthetic-held-out strict pass 仅用于基础设施功能测试，不是质量评估。
- 评估数据分类为 `synthetic_held_out`，不是真正独立的 Sealed Evaluation。
- `evaluation-evidence.json` 中明确标记：`is_quality_metric: false`、`not_for_quality_estimation: true`、`not_for_model_comparison: true`、`not_for_resume: true`。
- `claim-evidence-map.json` 将此声明列为 `prohibited`。

**正确的叙述方式**：
> "系统实现了端到端的功能验证（synthetic-held-out），但不作为模型的精度评估指标。模型的金融精度尚无独立基准评测数据支撑。"

---

### 2. "系统消除了幻觉"

**为什么不能声称**：

- Validation 是规则/校验系统，仅能检查数值、引用和计算来源，不能验证深层自然语言事实、逻辑一致性和推理正确性。
- 系统是 fail-closed 设计，能减少不受支撑的回答，但不能保证完全消除幻觉。
- `limitations-and-risks.md` 中明确声明：**"不声称完全消除幻觉"**。
- `claim-evidence-map.json` 将此声明列为 `prohibited`。

**正确的叙述方式**：
> "通过三层校验（Answerability → Grounding → Validation）和 fail-closed 设计，系统减少了不受支撑的回答，但不能承诺零幻觉。"

---

### 3. "模型支持原生 Function Calling"

**为什么不能声称**：

- 模型 SFT 训练使用标准 (instruction, response) 格式和 assistant-only loss，训练数据中不包含 Tool Use 格式。
- RAG 系统中的 Calculator 调用由 FastAPI 后端系统编排，不是模型自主决定调用的。
- `model-card.md` 中明确声明未训练原生 Function Calling。
- `claim-evidence-map.json` 将此声明列为 `prohibited`。

**正确的叙述方式**：
> "系统的工具调用（Calculator）由后端路由逻辑编排，不是模型的原生 Function Calling 能力。模型不具备自主决定调用工具的能力。"

---

### 4. "Phase 5 证明强泛化能力"

**为什么不能声称**：

- Phase 5 数据分类为 `synthetic_held_out`，评估目的是 `infrastructure_functional_test`。
- 0/54 strict pass 结果仅用于验证评估基础设施的功能正确性。
- `evaluation-evidence.json` 中明确标记：`is_real_sealed_evaluation: false`。
- `claim-evidence-map.json` 将此声明列为 `prohibited`。

**正确的叙述方式**：
> "Phase 5 建立了评估基础设施（盲评、隔离、消融实验），但评估数据不是真正独立的 Sealed Evaluation，结果不代表模型泛化能力。"

---

### 5. "Tokenizer 带来 X 倍推理加速"

**为什么不能声称**：

- Tokenizer 的压缩率（中文文本用更少 token 表示）与实际的推理速度提升不是简单线性关系。
- 推理速度受模型架构、显存带宽、批处理策略等多种因素影响，未经独立实验验证的提速声明不具有科学可信度。
- `claim-evidence-map.json` 将此声明列为 `prohibited`。

**正确的叙述方式**：
> "Tokenizer 针对中文金融语料进行了优化，提高了编码效率，但具体的推理性能影响需在相同硬件条件下独立测量。"

---

## 不可验证声明（Unverified Claims）

以下声明当前缺乏独立可验证的证据，在对外沟通中**不应**用作确定事实：

### 6. 预训练完成于 step 28000，val_bpb 0.7626

**为什么不能声称**：

- Checkpoint 内容哈希当前不可用（标记为 `historical_unavailable`）。
- `training-runs.json` 为历史自报数据，不可被第三方在现有服务器产物上验证。
- `checkpoint-manifest.json` 中 base checkpoints 的 `d24_final_mixdata` 为空数组。

---

### 7. SFT V2 lr010 checkpoint step 150，val_bpb 0.5558

**为什么不能声称**：

- Checkpoint 内容哈希当前不可用（标记为 `historical_unavailable`）。
- Step 150 是一个失败实验的烟雾检查点（smoke checkpoint），不是发布模型。
- `training-evidence.json` 中 `sft_checkpoints` 下所有条目均为空数组。

---

### 8. SFT checkpoint 派生自 step 28000 的 base checkpoint

**为什么不能声称**：

- 模型血缘使用的是 identity_digest（基于路径名的字符串 SHA256），而非 checkpoint 的实际权重内容哈希。
- 父子 checkpoint 之间的权重内容 lineage 未被独立验证。
- `model-lineage.json` 标记为 `historical_self_reported`。

---

### 9. 历史预训练 token 总量为 17.68B

**为什么不能声称**：

- 数据来源在当前服务器上不再完整可用。
- 无法被独立第三方复现验证。
- `claim-evidence-map.json` 标记为 `historical_self_reported`。
- `README.md` 明确将此数据列为"不用于质量声明"。

---

### 10. CORE metric 0.2201（step 7060）

**为什么不能声称**：

- 原始训练日志不再可用。
- 无法在当前环境中复现此指标的计算。
- `claim-evidence-map.json` 标记为 `historical_self_reported`。

---

## 汇总清单

| # | 声明内容 | 分类 | 应替换为 |
|---|----------|------|----------|
| 1 | 模型达到生产级金融精度 | ❌ Prohibited | "系统实现端到端功能验证，不作为精度评估指标" |
| 2 | 系统消除了幻觉 | ❌ Prohibited | "系统通过 fail-closed 设计减少不受支撑的回答" |
| 3 | 模型支持原生 Function Calling | ❌ Prohibited | "Calculator 由系统编排，非模型原生能力" |
| 4 | Phase 5 证明强泛化能力 | ❌ Prohibited | "Phase 5 建立了评估基础设施" |
| 5 | Tokenizer 带来 X 倍推理加速 | ❌ Prohibited | "Tokenizer 提高了中文金融文本的编码效率" |
| 6 | val_bpb 0.7626 / 0.5558 | ⚠️ Unverified | 不引用具体值；如需引用需注明"历史自报，不可验证" |
| 7 | 模型血缘（base → SFT） | ⚠️ Unverified | 不声称权重内容的 lineage 已验证 |
| 8 | 预训练 17.68B tokens | ⚠️ Unverified | 不引用具体数字 |
| 9 | CORE metric 0.2201 | ⚠️ Unverified | 不引用 |

---

## 正确可用的声明示例

| ✅ 应该这样说 | 来源 |
|---------------|------|
| 9 种确定性金融操作（Decimal 实现） | 代码 + system card |
| 6+ 类校验（fail-closed 设计） | 代码 + system card |
| Dense + BM25 + RRF + Reranker 混合检索 | 代码 + system card |
| Phase 7 部署验收 42/42 | phase7-acceptance.json |
| 自动化测试 >2000 条，零失败 | test-summary.json |
| 冒烟测试 12/12 全部通过 | smoke-report.json |
| 三服务无 root 部署（tmux + SSH 隧道） | 部署文档 + 脚本 |
| Byte-Level BPE tokenizer，65K 词汇表 | tokenizer-manifest.json |
| SFT 数据 39,534 样本，8 数据源 | sft-data-manifest.json |
| 模型约 1.4B 参数（24 层 / 12 头） | checkpoint-manifest.json |
| Phase 5 synthetic_held_out 非真正独立 Sealed Eval | evaluation-evidence.json |

---

## 相关文档

- [verified-metrics.md](verified-metrics.md) — 可验证指标及源产物路径
- [resume-evidence.md](resume-evidence.md) — 简历可用素材及禁止声明
- [interview-guide.md](interview-guide.md) — 面试问答
- 源数据：`artifacts/release/phase6/claim-evidence-map.json`
