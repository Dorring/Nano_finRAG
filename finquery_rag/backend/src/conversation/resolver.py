"""Contextual Query Resolver.

Reconstructs multi-turn conversational queries into standalone financial queries
using Bailian Qwen3.6-Flash and deterministic Fast Path rules.
"""

from __future__ import annotations

import json
import re

from .bailian_client import BailianClient
from .contracts import (
    ConversationResolution,
    DialogueState,
    DialogueTurn,
    ReasonCode,
)

DEPENDENCY_MARKERS = [
    r"\b(it|this|that|these|those|they|their|its)\b",
    r"\b(the\s+company|the\s+firm|the\s+issuer)\b",
    r"\b(what\s+about|how\s+about|and\s+for|what\s+of|and\s+the)\b",
    r"\b(previous|prior|preceding|following|next)\s+(year|quarter|period|fiscal)\b",
    r"\b(last\s+year|last\s+quarter|same\s+period|yoy|qoq)\b",
    r"\b(increase|decrease|grow|growth|change|difference|higher|lower)\b",
    r"\b(compare|compared|versus|vs\.?)\b",
]

OUT_OF_SCOPE_MARKERS = [
    r"\b(recommend\s+a\s+movie|weather\s+in|write\s+a\s+poem|tell\s+a\s+joke|translate)\b",
    r"\b(who\s+won|play\s+music|recipe\s+for|capital\s+of)\b",
    r"(股价|股票价格|实时行情|走势|涨跌|现价|收盘价|开盘价|k线|盘前|盘后|市值多少)",
    r"\b(stock\s+price|share\s+price|stock\s+quote|market\s+price|stock\s+chart|target\s+price)\b",
    r"(聊天|闲聊|天气|做菜|菜谱|红烧肉|讲笑话|写个故事)",
]

KNOWN_ENTITIES = [
    "apple", "aapl", "microsoft", "msft", "tesla", "tsla",
    "google", "googl", "amazon", "amzn", "coca-cola", "ko",
    "oracle", "orcl", "ford", "pepsi", "sap", "nvidia", "nvda", "meta",
    "jpmorgan", "jpmorgan chase", "pfizer", "pfe", "visa", "v",
    "苹果", "微软", "特斯拉", "英伟达", "比亚迪", "谷歌", "亚马逊", "腾讯", "阿里", "阿里巴巴",
]


def register_known_entity(entity: str) -> None:
    """Register custom entity name or ticker dynamically for conversational context."""
    norm = entity.strip().lower()
    if norm and norm not in KNOWN_ENTITIES:
        KNOWN_ENTITIES.append(norm)


def update_dialogue_matrix(
    state: DialogueState,
    *,
    entities: list[str] | None = None,
    periods: list[str] | None = None,
    metrics: list[str] | None = None,
) -> dict[str, list[str]]:
    """Update active entity-period-metric matrix in DialogueState without single-value overwrites."""
    matrix = state.active_matrix
    if entities:
        # Preserve multi-entity set across turns
        for ent in entities:
            clean = ent.strip().lower()
            if clean and clean not in matrix["entities"]:
                matrix["entities"].append(clean)
    if periods:
        for p in periods:
            clean = p.strip().upper()
            if clean and clean not in matrix["periods"]:
                matrix["periods"].append(clean)
    if metrics:
        for m in metrics:
            clean = m.strip().lower()
            if clean and clean not in matrix["metrics"]:
                matrix["metrics"].append(clean)
    return matrix


def get_dialogue_matrix(state: DialogueState) -> dict[str, list[str]]:
    """Return copy of active entity-period matrix."""
    return {
        "entities": list(state.active_matrix.get("entities", [])),
        "periods": list(state.active_matrix.get("periods", [])),
        "metrics": list(state.active_matrix.get("metrics", [])),
    }



# Operation words such as ``growth`` and ``compare`` are also dependency
# markers (they can describe a follow-up like "how much did it grow?").  A
# complete first-turn question must therefore be recognised by the semantic
# anchors it explicitly supplies, rather than by word count alone.
EXPLICIT_PERIOD_PATTERNS = (
    r"\bfy\s*\d{2,4}\b",
    r"\b(?:q[1-4]|quarter\s+[1-4])\b",
    r"\b(?:19|20)\d{2}\b",
    r"(?:19|20)\d{2}年?",
    r"\d{2}年度?",
    r"\d{2}年",
    r"第[一二三四1-4]季度",
)

EXPLICIT_METRIC_MARKERS = (
    "revenue", "sales", "income", "profit", "earnings", "margin",
    "assets", "liabilities", "equity", "cash", "debt", "expense",
    "expenses", "ebit", "ebitda", "eps", "volume", "transactions",
    "dividend", "shares", "yield", "operating", "gross", "net",
    "收益", "收入", "营业收入", "净利润", "利润", "毛利", "毛利率", "净利率",
    "现金流", "资产", "负债", "销售额", "市值", "股价", "每股收益",
)


class ContextualQueryResolver:
    """Resolves conversational ellipsis and references into standalone queries."""

    def __init__(self, client: BailianClient | None = None) -> None:
        self.client = client or BailianClient()

    def is_self_contained_fast_path(self, query: str) -> bool:
        """Determines if a query is self-contained and can bypass LLM resolution."""
        q_lower = query.lower().strip()

        # Explicit first-turn financial questions may contain words such as
        # ``growth``, ``compare`` or ``previous`` as part of a fully specified
        # calculation.  Those words are dependency markers for genuinely
        # elliptical follow-ups, but must not force an already complete query
        # through the conversation-context gate.  Requiring an entity, an
        # explicit period and a financial metric keeps the bypass conservative.
        if self._has_explicit_financial_context(q_lower):
            return True
        
        for pattern in DEPENDENCY_MARKERS:
            if re.search(pattern, q_lower):
                return False
                
        words = q_lower.split()
        if len(words) < 4 and not any(term in q_lower for term in ["revenue", "income", "margin", "cash", "asset", "debt", "expenditure", "traded", "represent"]):
            return False
            
        return True

    @staticmethod
    def _has_explicit_financial_context(query: str) -> bool:
        """Return whether *query* carries the anchors needed for a first turn.

        This is intentionally a small deterministic guard, not a replacement
        for Supervisor semantic planning.  It only prevents the lifecycle
        layer from treating an explicit calculation question as an ellipsis.
        The financial runtime still performs its own plan, retrieval and
        evidence validation.
        """
        q_lower = str(query).lower()
        # Match entity aliases as tokens.  In particular, Visa's one-letter
        # ticker ``v`` must not match the ``v`` inside unrelated words such as
        # ``revenue``.
        has_entity = any(
            (entity in q_lower) if re.search(r"[\u4e00-\u9fff]", entity)
            else bool(re.search(rf"(?<!\w){re.escape(entity)}(?!\w)", q_lower))
            for entity in KNOWN_ENTITIES
        )
        has_period = any(re.search(pattern, q_lower) for pattern in EXPLICIT_PERIOD_PATTERNS)
        has_metric = any(marker in q_lower for marker in EXPLICIT_METRIC_MARKERS)
        return has_entity and has_period and has_metric

    def resolve(
        self,
        current_query: str,
        dialogue_state: DialogueState | None = None,
        filtered_turns: list[DialogueTurn] | None = None,
    ) -> ConversationResolution:
        """Resolves current user query against dialogue state and recent turns."""
        turns = filtered_turns if filtered_turns is not None else (dialogue_state.recent_turns if dialogue_state else [])
        
        # 1. Out of Scope Check
        q_lower = current_query.lower()
        for pat in OUT_OF_SCOPE_MARKERS:
            if re.search(pat, q_lower):
                return ConversationResolution(
                    supported=False,
                    requires_context=False,
                    standalone_query=current_query,
                    reason_codes=[ReasonCode.OUT_OF_SCOPE],
                )

        # 2. Fast Path: First turn or self-contained query (English-only)
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", current_query))
        if not has_chinese and (not dialogue_state or dialogue_state.turn_count == 0 or len(turns) == 0):
            return ConversationResolution(
                supported=True,
                requires_context=False,
                standalone_query=current_query,
                reason_codes=[ReasonCode.NO_CONTEXT_REQUIRED],
            )

        if not has_chinese and self.is_self_contained_fast_path(current_query):
            topic_switched = False
            if dialogue_state.active_entity and not re.search(rf"\b{re.escape(dialogue_state.active_entity.lower())}\b", q_lower):
                topic_switched = True
                
            codes = [ReasonCode.NO_CONTEXT_REQUIRED, ReasonCode.RESOLVER_BYPASS]
            if topic_switched:
                codes.append(ReasonCode.TOPIC_SWITCH)
                
            return ConversationResolution(
                supported=True,
                requires_context=False,
                standalone_query=current_query,
                topic_switch=topic_switched,
                reason_codes=codes,
            )

        # 3. Contextual Resolution Path
        return self._resolve_with_context(current_query, dialogue_state, turns)

    def _resolve_with_context(
        self,
        current_query: str,
        dialogue_state: DialogueState,
        turns: list[DialogueTurn],
    ) -> ConversationResolution:
        llm_resolution = self._call_llm_resolver(current_query, dialogue_state, turns)
        if llm_resolution is not None:
            if llm_resolution.clarification_required:
                return llm_resolution
            # If input was Chinese, ensure output standalone query was translated into English
            if re.search(r"[\u4e00-\u9fff]", current_query) and re.search(r"[\u4e00-\u9fff]", llm_resolution.standalone_query):
                return self._deterministic_fallback_resolve(current_query, dialogue_state, turns)
            return llm_resolution
        return self._deterministic_fallback_resolve(current_query, dialogue_state, turns)

    def _call_llm_resolver(
        self,
        current_query: str,
        dialogue_state: DialogueState,
        turns: list[DialogueTurn],
    ) -> ConversationResolution | None:
        if not getattr(self.client, "api_key", None):
            return None

        recent_summary = []
        for i, t in enumerate(turns[-4:]):
            recent_summary.append(f"Turn {i+1} User: {t.user_query}")
            if t.standalone_query != t.user_query:
                recent_summary.append(f"Turn {i+1} Resolved Query: {t.standalone_query}")

        state_summary = {
            "active_entity": dialogue_state.active_entity,
            "active_metric": dialogue_state.active_metric,
            "active_period": dialogue_state.active_period,
            "active_scope": dialogue_state.active_scope,
            "active_topic": dialogue_state.active_topic,
        }

        system_prompt = (
            "You are the Conversation Context Layer for a Trusted Financial RAG system.\n"
            "Your ONLY task is to reconstruct the user's current query into a complete, standalone financial question by resolving pronouns, ellipses, and relative time expressions.\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. DO NOT answer the question. DO NOT perform calculations. DO NOT invent numbers.\n"
            "2. Precedence: User's CURRENT query > Explicitly referenced turn > Dialogue State > History.\n"
            "3. If the user explicitly mentions an entity or metric, it OVERRIDES past dialogue state.\n"
            "4. If there is genuine ambiguity, set ambiguity_detected=true and clarification_required=true.\n"
            "5. CRITICAL LANGUAGE & FORMULATION RULE: The downstream SEC 10-K verification and calculation engines operate in standard English GAAP terminology. When the user query is in Chinese (or any non-English language), reconstruct 'standalone_query' in concise, standard English financial syntax naming canonical companies (e.g. Apple, Microsoft, Tesla), metrics (e.g. Net Income, Total Revenue, Operating Income), and periods (e.g. FY2025). If clarification_required is true, provide 'clarification_question' in the user's input language (e.g. Chinese if user query is Chinese, bilingual if helpful). If the query asks to compare two companies (e.g. which had better earnings/revenue), formulate standalone_query as 'Calculate the difference between [Company1] [Metric] and [Company2] [Metric] in [Period].' so it executes deterministically in the financial sandbox.\n"
            "6. DOCUMENT/REPORT IDENTITY RULE: If the user asks about the identity, issuer, or company of a selected/uploaded document or financial report (e.g. '这是啥公司的财报', '这是哪家公司', '这份财报是谁的', '这篇文档是哪个公司的', '属于哪家公司'), DO NOT attach or inherit active_entity, active_metric, or active_period from prior unrelated turns! Set topic_switch=true, requires_context=false, resolved_entity=null, resolved_metric=null, and formulate standalone_query as 'Which company is this financial report from? Company name issuer.'\n"
            "7. Return ONLY a valid JSON object with EXACT schema:\n"
            "   {\n"
            "     \"supported\": true,\n"
            "     \"requires_context\": true,\n"
            "     \"standalone_query\": \"Reconstructed standalone English financial question (e.g. 'What was Full Truck Alliance FY2025 Revenue?' or 'Full Truck Alliance')\",\n"
            "     \"resolved_entity\": \"Canonical company name (e.g. 'Full Truck Alliance', 'Apple', 'Tesla') or null\",\n"
            "     \"resolved_metric\": \"Financial metric (e.g. 'Revenue', 'Net Income') or null\",\n"
            "     \"resolved_period\": \"Period (e.g. 'FY2025') or null\",\n"
            "     \"topic_switch\": true,\n"
            "     \"clarification_required\": false,\n"
            "     \"clarification_question\": null\n"
            "   }"
        )

        user_prompt = json.dumps({
            "current_user_query": current_query,
            "dialogue_state": state_summary,
            "recent_turns": recent_summary,
            "compressed_history": dialogue_state.compressed_history,
        }, ensure_ascii=False, indent=2)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw_output = self.client.chat_completion(messages, response_format={"type": "json_object"})
        if not raw_output:
            return None

        try:
            data = json.loads(raw_output)
            if isinstance(data, dict) and "content" in data and isinstance(data["content"], dict):
                data = data["content"]

            clarification_req = bool(
                data.get("clarification_required")
                or data.get("clarification_needed")
                or data.get("ambiguity_detected")
            )
            clarification_q = (
                data.get("clarification_question")
                or data.get("clarification_prompt")
                or data.get("question")
            )
            entity_res = data.get("entity_resolution")
            entity_val = (
                data.get("resolved_entity")
                or (entity_res.get("english_name") if isinstance(entity_res, dict) else None)
                or (entity_res.get("query_entity") if isinstance(entity_res, dict) else None)
                or data.get("entity")
            )
            metric_val = (
                data.get("resolved_metric")
                or data.get("metric")
                or (data.get("suggested_defaults", {}).get("metric") if isinstance(data.get("suggested_defaults"), dict) else None)
            )
            period_val = (
                data.get("resolved_period")
                or data.get("period")
                or (data.get("suggested_defaults", {}).get("period") if isinstance(data.get("suggested_defaults"), dict) else None)
            )
            standalone_q = (
                data.get("standalone_query")
                or data.get("resolved_query")
                or data.get("standalone_question")
                or current_query
            )
            topic_sw = bool(
                data.get("topic_switch")
                or (entity_res.get("entity_switch") if isinstance(entity_res, dict) else False)
                or (entity_val and dialogue_state and dialogue_state.active_entity and str(entity_val).strip().upper() != str(dialogue_state.active_entity).strip().upper())
            )

            return ConversationResolution(
                supported=data.get("supported", True),
                requires_context=data.get("requires_context", True),
                standalone_query=standalone_q,
                resolved_entity=entity_val,
                resolved_metric=metric_val,
                resolved_period=period_val,
                resolved_scope=data.get("resolved_scope"),
                inherited_fields=data.get("inherited_fields", []),
                explicit_fields=data.get("explicit_fields", []),
                topic_switch=topic_sw,
                ambiguity_detected=clarification_req,
                clarification_required=clarification_req,
                clarification_question=clarification_q,
                clarification_options=data.get("clarification_options", []),
                relevant_turn_ids=data.get("relevant_turn_ids", []),
                confidence=float(data.get("confidence", 1.0)),
                reason_codes=data.get("reason_codes", [ReasonCode.REFERENCE_RESOLVED]),
            )
        except Exception:
            return None

    def _deterministic_fallback_resolve(
        self,
        current_query: str,
        state: DialogueState,
        turns: list[DialogueTurn],
    ) -> ConversationResolution:
        q_lower = current_query.lower().strip()
        inherited_fields = []
        explicit_fields = []
        
        # Multi-entity and single entity extraction
        norm_map = {
            "苹果": "Apple", "aapl": "Apple", "apple": "Apple",
            "微软": "Microsoft", "msft": "Microsoft", "microsoft": "Microsoft",
            "特斯拉": "Tesla", "tsla": "Tesla", "tesla": "Tesla",
            "英伟达": "NVIDIA", "nvda": "NVIDIA", "nvidia": "NVIDIA",
            "比亚迪": "BYD", "byd": "BYD",
            "谷歌": "Google", "google": "Google", "googl": "Google",
            "亚马逊": "Amazon", "amazon": "Amazon", "amzn": "Amazon",
            "腾讯": "Tencent", "tencent": "Tencent",
            "阿里": "Alibaba", "阿里巴巴": "Alibaba", "baba": "Alibaba",
        }
        found_entities = []
        for cand in KNOWN_ENTITIES:
            matched = (cand in q_lower) if re.search(r"[\u4e00-\u9fff]", cand) else bool(re.search(rf"\b{re.escape(cand)}\b", q_lower))
            if matched:
                c_name = norm_map.get(cand, cand.title())
                if c_name not in found_entities:
                    found_entities.append(c_name)

        if not found_entities:
            m_comp = re.search(r"([\u4e00-\u9fff]{2,10}(?:集团|公司|科技|股份|控股|汽车|重工|生物|医药|银行|证券|保险|百货|物流))", current_query)
            if m_comp:
                found_entities.append(m_comp.group(1))

        entity = found_entities[0] if found_entities else None
        if entity:
            explicit_fields.append("entity")

        # Metric extraction from current query
        metric = None
        metric_keywords = [
            ("automotive gross margin", "Automotive Gross Margin"),
            ("operating margin", "Operating Margin"),
            ("operating income", "Operating Income"),
            ("net income", "Net Income"),
            ("gross margin", "Gross Margin"),
            ("free cash flow", "Free Cash Flow"),
            ("capital expenditures", "Capital Expenditures"),
            ("capital expenditure", "Capital Expenditures"),
            ("revenue", "Revenue"),
            ("billings", "Billings"),
            ("eps", "EPS"),
            ("收益", "Net Income"),
            ("净利润", "Net Income"),
            ("利润", "Operating Income"),
            ("营业收入", "Revenue"),
            ("收入", "Revenue"),
            ("销售额", "Revenue"),
            ("毛利", "Gross Margin"),
            ("毛利率", "Gross Margin"),
        ]
        for term, canonical in metric_keywords:
            if term in q_lower:
                metric = canonical
                explicit_fields.append("metric")
                break

        # Period extraction from current query
        period = None
        period_match = re.search(r"\b(fy\s*20\d\d|20\d\d|q[1-4]\s*20\d\d|q[1-4])\b|(\d{2})年度?|20(\d{2})年?", q_lower)
        if period_match:
            if period_match.group(1):
                period = period_match.group(1).upper()
            elif period_match.group(2):
                period = f"FY20{period_match.group(2)}"
            elif period_match.group(3):
                period = f"FY20{period_match.group(3)}"
            explicit_fields.append("period")

        is_doc_id_query = bool(re.search(r"(?:这是|这份|这篇|该|此).{0,6}(?:啥公司|哪家公司|哪个公司|什么公司|公司财报|是谁的|公司名称|属于哪家)", current_query))
        if is_doc_id_query:
            return ConversationResolution(
                supported=True,
                requires_context=False,
                standalone_query="Which company is this financial report from? Company name issuer.",
                topic_switch=True,
                reason_codes=[ReasonCode.NO_CONTEXT_REQUIRED, ReasonCode.TOPIC_SWITCH],
            )

        # Multi-entity comparison fast-path
        if len(found_entities) >= 2:
            p = period or (state.active_period if state else None) or "FY2025"
            m = metric or (state.active_metric if state else None) or "Net Income"
            standalone = f"Calculate the difference between {found_entities[0]} {m} and {found_entities[1]} {m} in {p}."
            return ConversationResolution(
                supported=True,
                requires_context=False,
                standalone_query=standalone,
                resolved_entity=found_entities[0],
                resolved_metric=m,
                resolved_period=p,
                explicit_fields=["entity", "metric", "period"],
                reason_codes=[ReasonCode.NO_CONTEXT_REQUIRED, ReasonCode.REFERENCE_RESOLVED],
            )

        # 1. Ambiguity Detection
        # If previous state had multiple metrics and user asks "What about 2023?" / "What about last year?" without specifying metric
        if state and state.active_topic and "MULTIPLE_METRICS" in state.active_topic and not metric:
            return ConversationResolution(
                supported=True,
                requires_context=True,
                standalone_query="",
                ambiguity_detected=True,
                clarification_required=True,
                clarification_question="Which metric would you like to check for that period?",
                clarification_options=["Revenue", "Operating Margin", "Both"],
                reason_codes=[ReasonCode.AMBIGUOUS_METRIC],
            )

        # 2. Topic Switch / Entity Switch: "What about Microsoft?" / "And for Ford?" / "How about Pepsi?"
        if state and entity and entity.upper() != (state.active_entity or "").upper():
            topic_switched = True
            active_m = metric or state.active_metric or "Revenue"
            active_p = period or state.active_period or "FY2024"
            if not metric and state.active_metric:
                inherited_fields.append("metric")
            if not period and state.active_period:
                inherited_fields.append("period")
                
            standalone = f"What was {entity} {active_p} {active_m}?"
            return ConversationResolution(
                supported=True,
                requires_context=True,
                standalone_query=standalone,
                resolved_entity=entity,
                resolved_metric=active_m,
                resolved_period=active_p,
                inherited_fields=inherited_fields,
                explicit_fields=explicit_fields,
                topic_switch=topic_switched,
                reason_codes=[ReasonCode.TOPIC_SWITCH, ReasonCode.EXPLICIT_QUERY_OVERRIDE],
            )

        # 3. Relative Period Resolution: "What about the previous year?" / "last year?"
        is_prev_year = bool(re.search(r"\b(previous|prior|preceding|last)\s+(year|period|fy)\b", q_lower))
        if is_prev_year and state and state.active_period:
            base_p = state.active_period
            year_match = re.search(r"20\d\d", base_p)
            if year_match:
                prev_year = str(int(year_match.group(0)) - 1)
                resolved_p = base_p.replace(year_match.group(0), prev_year)
                active_e = entity or state.active_entity
                active_m = metric or state.active_metric or "Revenue"
                if not active_e:
                    return ConversationResolution(
                        supported=True,
                        requires_context=True,
                        standalone_query="",
                        ambiguity_detected=True,
                        clarification_required=True,
                        clarification_question="无法确定上一年度所对应的公司，请指明具体公司名称。",
                        reason_codes=[ReasonCode.INSUFFICIENT_CONTEXT],
                    )
                
                inherited_fields.extend(["entity", "metric"])
                standalone = f"What was {active_e} {resolved_p} {active_m}?"
                return ConversationResolution(
                    supported=True,
                    requires_context=True,
                    standalone_query=standalone,
                    resolved_entity=active_e,
                    resolved_metric=active_m,
                    resolved_period=resolved_p,
                    inherited_fields=inherited_fields,
                    explicit_fields=explicit_fields,
                    reason_codes=[ReasonCode.RELATIVE_PERIOD_RESOLVED, ReasonCode.PERIOD_INHERITED],
                )

        # 4. Cross-turn Calculation: "How much did it grow?" / "What was the growth in operating income?"
        is_calc = bool(re.search(r"\b(grow|growth|increase|change|difference)\b", q_lower))
        if is_calc and state and state.active_entity:
            e = entity or state.active_entity
            m = metric or state.active_metric or "Revenue"
            p1 = state.comparison_period or "FY2023"
            p2 = state.active_period or "FY2024"
            standalone = f"Calculate the change in {e} {m} from {p1} to {p2}."
            return ConversationResolution(
                supported=True,
                requires_context=True,
                standalone_query=standalone,
                resolved_entity=e,
                resolved_metric=m,
                resolved_period=f"{p1}_TO_{p2}",
                inherited_fields=["entity", "metric", "period"],
                reason_codes=[ReasonCode.CROSS_TURN_CALCULATION_RESOLVED, ReasonCode.REFERENCE_RESOLVED],
            )

        # 5. Pronoun resolution: "Did it have positive operating margin?" / "Was its capital expenditure higher?"
        has_pronoun = bool(re.search(r"\b(it|its|this|they|their)\b", q_lower))
        if has_pronoun and state and state.active_entity:
            active_e = state.active_entity
            active_m = metric or state.active_metric or "Revenue"
            active_p = period or state.active_period or "FY2024"
            inherited_fields.append("entity")
            if not metric and state.active_metric:
                inherited_fields.append("metric")
            if not period and state.active_period:
                inherited_fields.append("period")
                
            standalone = f"What was {active_e} {active_p} {active_m}?"
            return ConversationResolution(
                supported=True,
                requires_context=True,
                standalone_query=standalone,
                resolved_entity=active_e,
                resolved_metric=active_m,
                resolved_period=active_p,
                inherited_fields=inherited_fields,
                explicit_fields=explicit_fields,
                reason_codes=[ReasonCode.REFERENCE_RESOLVED],
            )

        # 6. General Ellipsis / Metric Inheritance
        active_e = entity or (state.active_entity if state else None)
        active_m = metric or (state.active_metric if state else None)
        active_p = period or (state.active_period if state else None)
        if not entity and not metric and not period and not has_pronoun and not is_calc and not is_prev_year:
            active_name = state.active_entity if state and state.active_entity else "目标公司"
            return ConversationResolution(
                supported=True,
                requires_context=True,
                standalone_query="",
                ambiguity_detected=True,
                clarification_required=True,
                clarification_question=f"未能从输入 '{current_query}' 中识别财务指标或目标公司。若想继续查询 [{active_name}]，请补充具体科目（如 总营收、净利润等）；若想查询其他公司，请输入公司名称及财年。",
                reason_codes=[ReasonCode.INSUFFICIENT_CONTEXT, ReasonCode.AMBIGUOUS_REFERENCE],
            )

        if not active_e:
            return ConversationResolution(
                supported=True,
                requires_context=True,
                standalone_query="",
                ambiguity_detected=True,
                clarification_required=True,
                clarification_question="无法从当前提问或历史会话中识别目标公司。请输入包含具体公司名称（例如 Apple 或 微软）的完整财务问题。",
                reason_codes=[ReasonCode.INSUFFICIENT_CONTEXT, ReasonCode.AMBIGUOUS_REFERENCE],
            )

        if not active_m and not entity:
            return ConversationResolution(
                supported=True,
                requires_context=True,
                standalone_query="",
                ambiguity_detected=True,
                clarification_required=True,
                clarification_question=f"已定位公司 [{active_e}]，但未识别到具体会计科目。请补充您想查询的财务指标（如 销售额、净利润、Deferred 等）。",
                reason_codes=[ReasonCode.INSUFFICIENT_CONTEXT, ReasonCode.AMBIGUOUS_REFERENCE],
            )

        final_m = active_m or "Revenue"
        final_p = active_p or "FY2024"

        if not entity and state.active_entity:
            inherited_fields.append("entity")
        if not metric and state.active_metric:
            inherited_fields.append("metric")
        if not period and state.active_period:
            inherited_fields.append("period")

        standalone = f"What was {active_e} {final_p} {final_m}?"
        return ConversationResolution(
            supported=True,
            requires_context=bool(inherited_fields),
            standalone_query=standalone,
            resolved_entity=active_e,
            resolved_metric=final_m,
            resolved_period=final_p,
            inherited_fields=inherited_fields,
            explicit_fields=explicit_fields,
            reason_codes=[ReasonCode.ELLIPSIS_RESOLVED if inherited_fields else ReasonCode.NO_CONTEXT_REQUIRED],
        )
