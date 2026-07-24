# 面试问答指南 — Interview Guide

本文档包含 nano_finance 项目面试中最可能被问到的 10 个问题，每条提供"30 秒简短回答"和"2 分钟详细展开"两个版本。所有回答中的数字和事实均来自项目 README 和已验证的产物。

---

## Q1: 为什么重训 tokenizer？

### 30 秒回答

> 原生 NanoChat 的 tokenizer 是为英文训练的，对中文和金融术语的编码效率很低。我们重新训练了一个 Byte-Level BPE tokenizer（65K 词汇表），混合了中英文通用语料和中文金融语料，使得模型在中文金融文本上的编码效率显著提升。

### 2 分钟展开

> NanoChat 上游的 tokenizer 基于英文语料训练，在处理中文文本时会出现严重的 token 碎片化——一个汉字可能被拆成多个 token，金融专用词汇（如"资产负债率"、"扣非净利润"）更是如此。这会浪费上下文窗口、降低推理效率，并影响模型对金融术语的语义理解。
>
> 我们的方案是：使用 Byte-Level BPE 算法，从头训练一个 65K 词汇表的 tokenizer。训练语料混合了中英文通用文本和中文金融语料，确保 tokenizer 在通用文本和金融领域文本上都有良好的覆盖率。
>
> Tokenizer 设有 9 个特殊 token（不包含 pad），其 SHA256 已在模型血缘中记录，可在 `artifacts/release/phase6/tokenizer-manifest.json` 中验证。
>
> **必须注意**：我们不应该声称 tokenizer 带来了"X 倍推理加速"——压缩率提高不等于实际推理速度提升，这是一个 prohibited claim。Tokenizer 的主要价值在于提高编码效率和语义一致性，而非直接的速度指标。

---

## Q2: 为什么不让 LLM 直接做财务计算？

### 30 秒回答

> LLM 是概率模型，做数学运算本质上是"模式匹配"而非"计算"。一个 1.4B 参数的模型连对数的乘法都可能出错。在金融场景中，一个百分点的误差可能导致严重后果。我们用 Python Decimal 实现的确定性 Calculator 替代 LLM 做数值计算。

### 2 分钟展开

> 这个问题涉及到我们对 LLM 数学能力的根本立场。已经有大量研究表明，语言模型在数值计算上不可靠——特别是对于小模型。我们约 1.4B 参数的模型，其数学推理能力更加有限。
>
> 我们的选择是：**彻底分离"生成"和"计算"**。模型负责理解问题、从文档中定位信息、生成自然语言回答。但数值计算由一个独立的、确定性的 Calculator 模块完成——使用 Python 的 `Decimal` 类型，保证精度（不走浮点数）。
>
> Calculator 目前支持 9 种金融操作：差值（difference）、增长率（growth_rate）、占比（percentage_share）、求和（sum）、平均值（average）、毛利率（gross_margin）、净利率（net_margin）、负债率（debt_ratio）和单位换算（scale_conversion）。
>
> 更重要的是：Calculator 的输入操作数必须绑定到文档证据（文档名、页码、chunk ID）。如果操作数来源不可追溯，Calculator 会直接 fail——不会有"模型猜一个数再算"的 fallback 路径。这是一个 **fail-closed** 设计，即：宁可拒绝回答，也不给一个看起来合理但无法验证的数字。
>
> 这也意味着：超出 9 种操作的计算需求，系统目前无法满足。我们在文档中明确声明了这个限制。

---

## Q3: 如何确保操作数来自文档？

### 30 秒回答

> Calculator 接收的所有操作数都必须附带证据绑定——包括文档名、页码和 chunk ID。Validation 层的 Calculation Validation 会检查每一个操作数是否能追溯到原始文档中的具体位置。无法追溯的操作数会被阻断。

### 2 分钟展开

> 操作数的证据绑定是贯穿整个 RAG pipeline 的约束，不是一个事后检查。
>
> 整个链路如下：
> 1. **检索阶段**：Dense + BM25 双路检索返回的每个 chunk 都带有文档 ID 和页码元数据。
> 2. **上下文构建**：Context Builder 在组装 prompt 时保留这些元数据，并要求模型在回答中引用。
> 3. **计算阶段**：Calculator 要求每个操作数都标注来源——数字 X 来自文档 Y 的第 Z 页——否则拒绝执行。
> 4. **校验阶段**：Calculation Validation 逐操作数验证：这个数字确实在文档的那个位置吗？提取正确吗？单位和量级标注正确吗？
>
> 如果任何一环断裂——比如文档该位置的数字确实被 OCR 错误识别了——Validation 会触发 Repair Once（单次修复尝试），将错误反馈给模型要求重新提取。如果修复也失败了，系统进入 Safe Fallback，返回类似"无法验证计算操作数来源"的安全降级消息。
>
> 这种多层证据链的设计确保了**可审计性**：任何一条数值结论，都可以从最终回答追回到原始 PDF 的具体页码。

---

## Q4: 为什么融合 Dense 和 BM25？

### 30 秒回答

> Dense 检索擅长语义匹配（"营业收入"能匹配到"销售额"），但对精确关键词（如公司名"阿里巴巴"、股票代码"000001"）会漏掉或漂移。BM25 稀疏检索擅长精确关键词匹配，但不理解语义。两者通过 RRF（Reciprocal Rank Fusion）融合后各自取长补短，再经 Reranker 重排序。

### 2 分钟展开

> 这是信息检索中的一个经典取舍：
>
> **Dense 检索**（ChromaDB 向量检索）将文本编码为语义向量，用余弦相似度匹配。优点是可以找到语义相近但用词不同的内容——"营收增长"能匹配到"销售额同比上升"。缺点是对精确词汇——特别是专有名词、数字、代码——可能产生语义漂移，找回来的不是同一个公司或同一年。
>
> **BM25 检索**基于词频-逆文档频率的稀疏表示，精确匹配关键词。优点是专有名词匹配非常精确——"贵州茅台"就是"贵州茅台"，不会漂移到"泸州老窖"。缺点是完全不理解语义——"营收"和"销售额"对它来说就是不同的字符串。
>
> 在金融领域，这两种需求都很强烈：有时你需要精确匹配一个财务科目的名称（BM25 更优），有时你需要理解"盈利能力"和毛利率、净利率之间的关系（Dense 更优）。单独使用任何一种都会导致召回不全或精确度下降。
>
> 融合方案是 **RRF（Reciprocal Rank Fusion）**：将两路检索的排序结果按倒数排名加权合并，然后用 Cross-encoder Reranker 对合并后的候选列表重排序。每路检索都贡献 top-k，RRF 确保两路的强信号都被保留，Reranker 做最终的相关性判断。

---

## Q5: Validation 和 prompt 约束有什么不同？

### 30 秒回答

> Prompt 约束是**软约束**——在 prompt 中写"请提供引用来源"，模型可以无视。Validation 是**硬阻断**——在模型生成回答后，用规则系统逐条检查：每个声明有来源吗？数字对吗？单位对吗？不通过就阻断回答。Prompt 是建议，Validation 是门禁。

### 2 分钟展开

> 这是很多人容易混淆的概念。
>
> **Prompt 约束**（如 system prompt 中的"请在回答中引用来源"）本质上是给模型的一个建议。模型可能遵守，也可能不遵守——特别是在长文本生成中，模型经常"忘记"引用要求，或者在回答后半段开始自由发挥。
>
> **Validation** 是后置的规则校验系统。不管模型生成了什么，Validation 都会执行：提取声明（Claim Extraction）、检查每个声明是否有有效引用（Citation Validation）、验证引用中的数字是否与源文本一致（Numeric&nbsp;Validation）、确认单位和时间期间是否被正确传递（Unit/Period&nbsp;Validation）、验证 Calculator 的操作数来源（Calculation Validation）、检测是否存在无任何证据支撑的声明（Unsupported Claim Validation）。
>
> Validation 的另一个关键特征是 **fail-closed** 设计：只要有一条校验不通过，回答就不会被返回给用户。系统最多尝试一次修复（Repair Once），如果修复后仍不通过，就返回 Safe Fallback 消息。
>
> 但必须坦率说明：**Validation 不能验证所有自然语言事实**。它主要检查数值、引用和计算，对于纯自然语言的逻辑一致性、推理正确性等深层语义问题，Validation 是无能为力的。这就是为什么我们从不声称"消除幻觉"——那在技术上是做不到的。

---

## Q6: 为什么只有一次修复尝试？

### 30 秒回答

> 修复循环是危险的——如果用 LLM 来修复 LLM 的错误，可能越修越错，陷入无限循环。一次修复是平衡"给系统一个改正机会"和"避免失控"的折衷。一次修不好就 Safe Fallback，这是 fail-closed 原则的体现。

### 2 分钟展开

> 在 Grounding/Validation 系统中，一个常见的设计诱惑是："校验不通过就反馈给 LLM 重生成，再校验，直到通过"。这个思路看起来很吸引人——自我纠错、迭代优化——但实践中非常危险。
>
> 有多个层面的风险：
>
> 1. **无限循环风险**：如果 LLM 无法满足校验要求（比如文档中根本没有那个信息，但 LLM 坚持认为有），每次重生成都会产生新的失败变体，永远不会通过。
> 2. **质量退化**：多次重生成后，LLM 倾向于生成更"安全"（也更模糊）的表述来满足校验，实际信息量反而下降。
> 3. **延迟不可控**：每次循环都要等待 LLM 推理（model service latency 在 readyz 端点的 p50 约 6.4 秒），多次循环的延迟叠加无法接受。
> 4. **收敛性无保证**：没有理论保证 LLM 的重生成会收敛到通过校验的状态——实际上可能发散。
>
> 我们的选择是 **Repair Once**：校验不通过时，将具体错误信息（不是整个 prompt）反馈给模型，要求修复指定的问题。一次修复后再次校验——通过就返回，不通过就 Safe Fallback。这体现了 fail-closed 原则：**不确定的时候，宁可少说，不可说错。**

---

## Q7: 为什么不用原生 Function Calling？

### 30 秒回答

> 因为我们的模型**没有训练过**原生 Function Calling。模型是用 assistant-only loss 做 SFT 的，训练数据中没有 Tool Use 格式。RAG 系统中的 Calculator 调用是由 FastAPI 后端的系统编排完成的——先走检索 → 意图判断 → 如果需要计算就调 Calculator → 再调模型生成回答。这是系统级编排，不是模型原生的 Function Calling 能力。

### 2 分钟展开

> 这是一个需要明确声明的问题，因为在 AI 圈里"Function Calling"已经成为一个热门标签——很多项目会声称"我们的模型支持 Function Calling"来显得更高级。
>
> **我们明确不声称支持原生 Function Calling。** 理由：
>
> 1. SFT 训练数据是标准的 (instruction, response) 格式，使用 assistant-only loss，不包含 `<tool_call>` 等 Tool Use 格式。
> 2. 模型架构和训练方式决定了它不具备"自主决定何时调用什么工具"的能力。
> 3. RAG 系统里的 Calculator 调用、检索调用等，全部由 FastAPI 后端的路由逻辑编排：后端接收用户 query → 调检索 → 根据意图判断是否走 Calculator → 将计算结果喂给模型 → 模型基于结果生成回答。模型在整个过程中只是"接收输入、生成输出"的角色。
>
> 这种设计有其优势：工具调用的逻辑完全在 Python 代码中，可调试、可测试、可审计。模型不会"幻觉"出工具调用——它根本没有这个能力。
>
> 声明"支持原生 Function Calling"是一个 **prohibited claim**，已在 `artifacts/release/phase6/claim-evidence-map.json` 中明确标注为禁止声明。

---

## Q8: 如何防止 eval 标签泄漏到生产代码？

### 30 秒回答

> 评估基础设施使用了严格的隔离机制：evaluation query label isolation（评估问题标签隔离）、blind scoring isolation（盲评打分隔离）、calibration set separated（校准集分离）、RC freeze verified（评测配置冻结验证）。评测数据和评测代码不会与生产服务的数据流混合。

### 2 分钟展开

> 这个问题的本质是：如何在评估自己的模型时，确保评测结果没有被"污染"？
>
> 我们的 Phase 5 评测基础设施实施了以下隔离机制（记录在 `artifacts/release/phase6/evaluation-evidence.json` 中）：
>
> 1. **Evaluation Query Label Isolation**：评估问题的 ground truth 标签与训练/检索数据物理分离，不存在于任何可以被生产服务读取的路径中。
> 2. **Blind Scoring Isolation**：打分脚本不依赖模型可以访问的任何数据——它使用独立的答案文件和标签文件，在模型推理完成后离线进行。
> 3. **Calibration Set Separated**：校准集（用于 ablation study 的阈值调校）与训练集、评估集三方分离，不存在数据交叉。
> 4. **RC Freeze Verified**：评估配置（retrieval config）在评估期间被冻结，记录了 8 个配置哈希值以确保没有事后篡改。
>
> 另外需要强调的是：Phase 5 的评估数据分类是 `synthetic_held_out`——**不是真正独立的 Sealed Evaluation**。0/54 的 strict pass 结果仅用于基础设施功能验证，不代表模型质量。这一点在 `evaluation-evidence.json` 中有明确的 `is_real_sealed_evaluation: false` 声明和 `is_quality_metric: false` 标记。

---

## Q9: 如何无 root 部署三个服务？

### 30 秒回答

> 三个服务（Model、Backend、Frontend）运行在三个独立的 tmux 会话中，由 shell 脚本管理生命周期。所有端口绑定到 127.0.0.1（防止公网暴露），端口号 >1024（不需要特权端口）。远程访问通过 SSH 隧道。没有使用 Docker、systemd 或任何需要 root 的服务管理工具。

### 2 分钟展开

> 在标准的创业公司或大厂环境中，你会用 Kubernetes + Docker + Nginx 来做服务部署。但在大学服务器上，你没有 root 权限、不能装 Docker、不能用 systemd、不能绑 80/443 端口。必须重新思考整个部署方案。
>
> 我们的方案：
>
> **进程管理**：三个服务分别跑在三个命名 tmux 会话中——`nano-finance-model`、`nano-finance-backend`、`nano-finance-frontend`。tmux 保证服务在 SSH 断开后继续运行。启动/停止/重启由 shell 脚本管理，脚本使用 PID 文件来验证进程所有权，不会误杀无关进程。
>
> **网络**：所有端口 >1024（18001/18002/18003），绑定到 127.0.0.1。这意味着没有公网 IP 可以访问这些服务。远程访问通过 SSH 隧道实现：`ssh -N -L 18003:127.0.0.1:18003 -L 18002:127.0.0.1:18002 <user>@<server>`。SSH 本身提供加密，弥补了应用层没有 TLS 的不足。
>
> **启动顺序**：Model → Backend → Frontend（因为 Backend 依赖 Model API，Frontend 依赖 Backend API）。停止顺序相反。
>
> **验收**：Phase 7 的 42/42 验收项全部通过，12/12 冒烟测试全部通过。涵盖了从端口绑定、健康检查、SSE 流、日志分离（无秘密泄漏）、登出持久性、重启恢复到性能基线测量。
>
> 局限性也必须说明：没有自动重启（服务器重启后需手动 `start_all.sh`）、没有外部监控、没有负载均衡、单 worker 部署。

---

## Q10: 当前系统最大的局限是什么？

### 30 秒回答

> 最大局限有三方面：一是模型 checkpoint 的历史 provenance 不可独立验证（训练产物历史不可用），这使模型产出的可信度打了折扣；二是部署层面无自动重启和水平扩展能力；三是 Calculator 仅支持 9 种操作，Validation 无法验证所有自然语言事实。

### 2 分钟展开

> 如果要挑一个最根本的局限，我认为是**模型可复现性**的问题。
>
> 项目中的模型 checkpoint 是在历史训练运行中产生的，但训练日志、checkpoint 文件哈希等用于验证模型 provenance 的关键产物在当前的服务器上不可用（标记为 `historical_unavailable`）。这意味着：
> - 我们无法向第三方独立证明"这个模型的权重确实是从声称的训练数据和流程中产生的"
> - 基础训练和 SFT 的训练指标（val_bpb 0.7626 和 0.5558）是历史自报数据，无法在当前环境中复现验证
> - 模型的 lineage（从 base 到 SFT 的派生关系）使用的是 identity_digest（基于路径名的字符串哈希），而非 checkpoint 内容哈希
>
> 这在实际工程中是一个严肃的问题——如果你要对这个模型的输出做审计，你连"这个模型是怎么来的"都证明不了。
>
> 其他重要局限包括：
> - 部署基础设施极简（无自动重启、无水平扩展、无外部监控），不能用于生产级服务
> - Calculator 仅覆盖 9 种操作，超出范围的计算需求无法支持
> - Validation 无法验证深层语义事实，不能声称消除幻觉
> - 模型约 1.4B 参数，复杂推理能力有限
> - 训练数据有时效性，不包含最新市场信息和法规变更
> - 无实时数据接入
>
> 详见 `docs/release/limitations-and-risks.md` 和 `docs/deployment/known-limitations.md`。
