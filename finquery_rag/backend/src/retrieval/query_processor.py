"""Query processing: expansion, classification, and rewriting.

Extracted from RAGEngine to isolate query-related logic.
All string patterns, regexes, and judgment conditions are preserved exactly.
"""
import asyncio
import re



class QueryProcessor:
    """Handles query expansion, classification, and conversational rewriting."""

    def __init__(self, memory_profile_context_fn=None) -> None:
        self._build_memory_profile_context = memory_profile_context_fn


    def expand(self, query: str) -> str:
        """Add lightweight retrieval terms for common finance PDF questions.

        Only generic financial terminology expansions remain. Works for unknown documents.
        """
        if not query:
            return query
        expansions = []
        lowered = query.lower()
        if self._has_cjk(query):
            if any(term in query for term in ("\u6807\u9898", "\u9898\u76ee", "\u8bba\u6587\u540d")):
                expansions.append("paper title title of this paper")
            if "\u4f5c\u8005" in query:
                expansions.append("paper authors author affiliation")
            if "\u6458\u8981" in query:
                expansions.append("abstract summary")
            if any(term in query for term in ("\u4e3b\u8981", "\u8d21\u732e", "\u7814\u7a76", "\u89e3\u51b3")):
                expansions.append("main contribution problem method approach")
            if any(term in query for term in ("\u9875", "\u51e0\u9875", "\u591a\u5c11\u9875")):
                expansions.append("number of pages page count")
        if "title" in lowered and "paper title" not in lowered:
            expansions.append("paper title")
        if "reporting period" in lowered:
            expansions.append("year ended reporting period fiscal year")
        if "total revenue" in lowered:
            expansions.append("total revenue revenue")
        if "net assets" in lowered:
            expansions.append("net assets statement of financial position")
        if "cash and cash equivalents" in lowered or "cash equivalents" in lowered:
            expansions.append("cash and cash equivalents current assets")
        if "credit facilities" in lowered:
            expansions.append("credit facilities revolving credit facility term loan")
        if "gross margin" in lowered:
            expansions.append("gross margin gross profit revenue")
        if "operating activities" in lowered or "operating cash flow" in lowered:
            expansions.append("net cash operating activities cash flows")
        if not expansions:
            return query
        return f"{query}\n" + "\n".join(dict.fromkeys(expansions))

    def is_front_matter_query(self, query: str) -> bool:
        """Check if the query is about document front matter (title, author, abstract)."""
        normalized = (query or "").lower()
        markers = (
            "title", "author", "abstract", "paper name", "paper title",
            "\u6807\u9898", "\u9898\u76ee", "\u8bba\u6587\u540d", "\u4f5c\u8005", "\u6458\u8981", "\u8fd9\u7bc7\u8bba\u6587",
        )
        return any(marker in normalized for marker in markers) or self.is_document_overview_query(query)

    @staticmethod
    def is_document_overview_query(query: str) -> bool:
        """Identify broad questions best answered from a document cover or title page."""
        normalized = (query or "").lower()
        patterns = (
            r"\bwhat\s+(?:topic|subject)\b",
            r"\bwhat\s+(?:does|is)\b.*\b(?:cover|about)\b",
            r"\bmain\s+(?:topic|subject)\b",
            r"\bwhat\s+is\s+this\s+document\s+about\b",
            r"(?:这篇|该|这个).{0,8}(?:讲什么|关于什么|主题)",
        )
        return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)

    def is_title_query(self, query: str) -> bool:
        """Check if the query is specifically about a document title."""
        normalized = (query or "").lower()
        return any(marker in normalized for marker in (
            "title", "paper title", "name of this paper",
            "\u6807\u9898", "\u9898\u76ee", "\u8bba\u6587\u540d",
        ))

    def is_numeric_query(self, query: str) -> bool:
        """Check if the query is a numeric financial question."""
        normalized = (query or "").lower()
        if "which documents mention" in normalized:
            return False
        numeric_markers = (
            "how much", "how many", "amount",
            "revenue", "cash", "equivalents", "margin", "growth", "rate",
            "percent", "percentage", "assets", "liabilities", "income",
            "expense", "profit", "loss", "budget", "net assets", "year-over-year",
            "credit facilities", "revolving credit facility", "term loan", "yoy", "$", "%",
        )
        cjk_markers = (
            "\u591a\u5c11", "\u91d1\u989d", "\u6536\u5165", "\u8425\u6536", "\u73b0\u91d1",
            "\u5229\u6da6", "\u589e\u957f", "\u6bd4\u7387", "\u767e\u5206\u6bd4",
        )
        return any(marker in normalized for marker in numeric_markers) or any(marker in query for marker in cjk_markers)

    def should_try_deterministic_numeric_answer(self, query: str, chunks: list) -> bool:
        """Attempt evidence extraction for any numeric question with evidence.

        The extractor itself returns ``None`` when it cannot find a
        query-aligned value. This keeps the router generic rather than
        maintaining a brittle per-metric allow-list, while retaining the
        normal LLM/refusal fallback when no deterministic answer is found.
        """
        return bool(chunks) and self.is_numeric_query(query)

    def should_try_deterministic_factual_answer(self, query: str) -> bool:
        """Check if we should attempt a deterministic factual answer."""
        normalized = (query or "").lower()
        factual_markers = (
            "what is the title", "title and reporting period",
            "which organization", "prepared",
            "according to",
            "list two criteria", "criteria that make an item current",
        )
        return any(marker in normalized for marker in factual_markers)

    def should_generate_with_low_confidence(
        self, query: str, chunks: list, *, numeric_rrf_floor: float, numeric_dense_floor: float
    ) -> bool:
        """Allow numeric finance QA to proceed when evidence exists but scores are under-calibrated."""
        if not chunks or not self.is_numeric_query(query):
            return False
        scores = [float(chunk.get("score", 0) or 0) for chunk in chunks]
        best_score = max(scores) if scores else 0.0
        if best_score <= 0:
            return False
        if best_score < 0.05:
            return best_score >= numeric_rrf_floor
        return best_score >= numeric_dense_floor

    def looks_like_followup_question(self, question: str) -> bool:
        """Return True only for questions that likely need conversation context."""
        normalized = (question or "").strip().lower()
        if not normalized:
            return False

        followup_markers = (
            "it", "its", "they", "them", "that", "this", "those", "these",
            "above", "previous", "same", "there", "what about", "how about",
            "\u7ee7\u7eed", "\u8fd9\u4e2a", "\u90a3\u4e2a", "\u4e0a\u8ff0", "\u524d\u9762", "\u4e0a\u4e00", "\u5b83", "\u4ed6\u4eec", "\u8fd9\u4e9b", "\u90a3\u4e9b",
        )
        standalone_markers = (
            "title", "paper", "document", "pdf", "\u8bba\u6587", "\u6587\u6863", "\u6807\u9898", "\u4f5c\u8005", "\u9875", "\u591a\u5c11",
        )

        has_followup = any(marker in normalized for marker in followup_markers)
        has_standalone = any(marker in normalized for marker in standalone_markers)
        return has_followup and not has_standalone

    def is_valid_rewritten_query(self, original: str, rewritten: str) -> bool:
        """Reject LLM rewrite artifacts that would poison retrieval."""
        if not rewritten:
            return False
        candidate = rewritten.strip()
        if len(candidate) < 5 or len(candidate) > max(200, len(original) * 4):
            return False
        if "\n" in candidate:
            return False
        artifact_patterns = (
            r"\bUser\s*:",
            r"\bAssistant\s*:",
            r"\[[^\]]+\.pdf\s*,\s*p\d+\]",
            r"Context\s*:",
            r"Answer\s*:",
        )
        if any(re.search(pattern, candidate, flags=re.IGNORECASE) for pattern in artifact_patterns):
            return False
        return True

    async def rewrite(
        self,
        question: str,
        conversation_history: list,
        memory_profile: dict | None = None,
        *,
        llm_client=None,
        model_name: str = "nanochat",
    ) -> str:
        """Rewrite only true follow-up questions. Bad rewrites are more harmful than
        no rewrite because retrieval uses the rewritten text directly.
        """
        if not conversation_history or len(conversation_history) < 2:
            return question
        if not self.looks_like_followup_question(question):
            return question

        recent = conversation_history[-4:]
        history_parts = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = (msg.get("content") or "")[:160]
            history_parts.append(f"{role}: {content}")
        history_text = "\n".join(history_parts)
        memory_text = (
            self._build_memory_profile_context(memory_profile)
            if self._build_memory_profile_context
            else ""
        )
        memory_block = (
            "User preference memory for query planning only; do not treat as document facts:\n"
            f"{memory_text}\n\n"
            if memory_text
            else ""
        )

        rewrite_prompt = (
            "Rewrite the current follow-up question into one standalone search query.\n"
            "Use the conversation only to resolve pronouns or omitted subjects.\n"
            "Use preference memory only to resolve language, company, period, unit, or metric ambiguity.\n"
            "Do not include role labels, citations, page markers, or prior answers.\n\n"
            f"{memory_block}"
            f"Conversation:\n{history_text}\n\n"
            f"Current question: {question}\n"
            "Standalone search query:"
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    temperature=0,
                    max_tokens=100,
                )
            )
            rewritten = response.choices[0].message.content
            if self.is_valid_rewritten_query(question, rewritten):
                return rewritten.strip()
            return question
        except Exception:
            return question

    @staticmethod
    def _has_cjk(text: str) -> bool:
        """Detect CJK characters in text."""
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))
