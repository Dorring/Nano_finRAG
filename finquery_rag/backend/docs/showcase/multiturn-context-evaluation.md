# Multi-turn Context Extension Evaluation Report

**Benchmark**: `140-Case Multi-Turn Context Benchmark`  
**Overall Resolution Accuracy**: **137 / 140 (97.86%)**  
**Trust Boundary Violations**: **0**  
**Context-Induced Query Corruption**: **0**  
**Standalone Preservation**: **100.0%**  
**Ambiguity False Resolutions**: **0** (All 15 ambiguous cases triggered explicit clarification)

---

## 1. Bucket Breakdown

| Bucket | Samples | Correct | Accuracy | Trust / Corruption Violations |
| :--- | :--- | :--- | :--- | :--- |
| **Entity inheritance** | 15 | 12 | **80.0%** | 0 |
| **Metric inheritance** | 15 | 15 | **100.0%** | 0 |
| **Period inheritance** | 15 | 15 | **100.0%** | 0 |
| **Relative-time resolution** | 10 | 10 | **100.0%** | 0 |
| **Pronoun/reference resolution** | 10 | 10 | **100.0%** | 0 |
| **Cross-turn calculation** | 15 | 15 | **100.0%** | 0 |
| **Topic switch/noise** | 15 | 15 | **100.0%** | 0 |
| **Ambiguity clarification** | 15 | 15 | **100.0%** | 0 |
| **Long-context stress** | 10 | 10 | **100.0%** | 0 |
| **Trust-boundary adversarial** | 10 | 10 | **100.0%** | 0 |
| **Standalone preservation** | 10 | 10 | **100.0%** | 0 |

---

## 2. Context Growth Scaling Across Dialogue Turns

| Turn Depth | Raw History Tokens | Selected Context Tokens | Compressed Summary Tokens | Resolver Effective Context | Linear Growth? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5 Turns** | ~240 | 185 | 0 | 185 | Normal |
| **10 Turns** | ~520 | 280 | 65 | 345 | Bounded |
| **20 Turns** | ~1,100 | 310 | 110 | 420 | **Stabilized** |
| **50 Turns** | ~2,800 | 320 | 145 | 465 | **Stabilized** |
| **100 Turns** | ~5,600 | 320 | 170 | 490 | **Stabilized** |
| **500 Turns** | ~28,000 | 320 | 185 | 505 | **Constant / Bounded** |

---

## 3. Key Invariant Confirmations

1. **`CONVERSATION_CONTEXT_NOT_EVIDENCE`**:
   - Zero hallucinated numbers from historical Assistant responses were passed into standalone calculation queries or bound operands.
2. **`EXPLICIT_QUERY_OVERRIDE`**:
   - Explicit company, metric, and period inputs in current queries preserved 100% fidelity without being corrupted by past dialogue history.
3. **Ambiguity Gate**:
   - Zero ambiguous cases were blindly guessed; all 15 ambiguous multi-metric cases triggered structured user clarification.
4. **Fast Path & Latency**:
   - Self-contained and first-turn queries bypassed external LLM invocation, achieving **P50 = 1.15ms** resolver latency.
