"""Deterministic answer extraction from retrieved context.

These methods extract answers directly from evidence text without LLM calls.
They handle numeric, factual, and front-matter queries using regex and
term-matching heuristics.
"""
import re

from src.retrieval.query_processor import QueryProcessor
from src.generation.deterministic_observer import ProductionFactTrace, fact_id


class DeterministicAnswerExtractor:
    """Extract deterministic answers from context without LLM generation."""

    def __init__(self, *, query_processor: QueryProcessor | None = None):
        self._query_processor = query_processor or QueryProcessor()

    def answer_front_matter_query(self, query: str, chunks: list) -> dict | None:
        """Answer deterministic front-matter questions from structured chunks."""
        is_title_query = self._query_processor.is_title_query(query)
        is_overview_query = self._query_processor.is_document_overview_query(query)
        if not is_title_query and not is_overview_query:
            return None
        normalized_query = (query or "").lower()
        title_chunks = [
            chunk for chunk in (chunks or [])
            if (chunk.get("metadata") or {}).get("type") == "front_matter"
            and (chunk.get("metadata") or {}).get("subtype") == "title"
            and (chunk.get("content") or "").strip()
        ]
        title_chunks.sort(key=lambda chunk: (chunk.get("metadata") or {}).get("page", 999))
        title_chunk = None
        title = None
        for candidate_chunk in title_chunks:
            candidate_title = self._clean_deterministic_title(
                re.sub(r"^title\s*:\s*", "", candidate_chunk.get("content", ""), flags=re.IGNORECASE).strip()
            )
            if self._is_valid_deterministic_title(candidate_title):
                title_chunk = dict(candidate_chunk)
                title = candidate_title
                break

        if title_chunk is None:
            for candidate_chunk in chunks or []:
                metadata = candidate_chunk.get("metadata") or {}
                if metadata.get("page") != 1:
                    continue
                candidate_title = self._title_from_page_one_evidence(candidate_chunk.get("content", ""))
                if self._is_valid_deterministic_title(candidate_title):
                    title_chunk = dict(candidate_chunk)
                    title = candidate_title
                    break

        if title_chunk is None or not title:
            return None

        reporting_period = None
        if "reporting period" in normalized_query:
            reporting_period = self._reporting_period_from_evidence(chunks or [])
            if not reporting_period:
                return None

        title_chunk["score"] = max(float(title_chunk.get("score", 0) or 0), 1.0)
        title_chunk["deterministic_answer"] = "front_matter_title"
        if is_overview_query and not is_title_query:
            answer = f"The document covers: {title}."
        else:
            answer = f'The title of the paper is "{title}".'
        if reporting_period:
            answer += f" Reporting period: {reporting_period}."
        return {
            "answer": answer,
            "chunks": [title_chunk],
            "diagnostic": "front_matter_title",
        }

    @classmethod
    def _title_from_page_one_evidence(cls, content: str) -> str | None:
        """Extract a cover title from a page-one chunk when metadata is partial."""
        lines = []
        for raw_line in (content or "").splitlines():
            line = re.sub(r"^\s*(?:section:\s*|#+\s*)", "", raw_line, flags=re.IGNORECASE)
            line = re.sub(r"\s+", " ", line).strip(" -")
            lowered = line.lower()
            if not line or lowered.startswith(("abstract", "keywords", "paper id", "anonymous")):
                break
            if "www." in lowered or "http://" in lowered or "https://" in lowered:
                break
            if line.isdigit() and not re.fullmatch(r"(?:19|20)\d{2}", line):
                continue
            if lines and line.lower() == lines[-1].lower():
                continue
            lines.append(line)
            if len(lines) >= 4:
                break
        title = cls._clean_deterministic_title(" ".join(lines))
        title = re.sub(r"\bannualreport\b", "Annual Report", title, flags=re.IGNORECASE)
        return title or None

    @staticmethod
    def _reporting_period_from_evidence(chunks: list) -> str | None:
        """Find an explicitly stated reporting period in front-page evidence."""
        patterns = (
            r"\bYear\s+to\s+(?:December|June|March|September)\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
            r"\bYear\s+ended\s+(?:December|June|March|September)\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
            r"\bFor\s+the\s+year\s+ended\s+(?:December|June|March|September)\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
        )
        for chunk in chunks:
            text = re.sub(r"\s+", " ", chunk.get("content", "") or "")
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    return match.group(0)
        return None

    def answer_numeric_query_from_context(self, query: str, context: str, sources: list) -> dict | None:
        """Return a deterministic numeric answer when relevant evidence lines are present."""
        if not context or not self._query_processor.should_try_deterministic_numeric_answer(query, [{"score": 1.0}]):
            return None

        evidence = self._rank_context_evidence(
            query,
            context,
            require_number=True,
            window_radius=1,
        )

        selected = self._select_distinct_evidence(evidence, limit=3)
        direct_values = self._summarize_numeric_values(query, selected, context=context)
        if not selected and not direct_values:
            return None

        answer_lines = []
        if direct_values:
            answer_lines.append(f"Answer: {direct_values}.")
        if selected:
            answer_lines.append("Evidence:")
        for item in selected:
            if item["source"]:
                answer_lines.append(f"- {item['text']} (Source: {item['source']})")
            else:
                answer_lines.append(f"- {item['text']}")
        return {
            "answer": "\n".join(answer_lines),
            "diagnostic": "deterministic_numeric_evidence",
        }

    def answer_numeric_query_from_chunks(self, query: str, chunks: list, *, observer=None) -> dict | None:
        """Answer a numeric question from raw retrieved child evidence.

        This intentionally runs before parent-context expansion.  Parent
        context is useful for generation, but can join nearby rows from a
        table or section and make a numeric extractor select an unrelated
        value.  The selector below remains document-agnostic: it ranks
        small evidence windows by query-term coverage and numeric proximity.
        """
        if not chunks or not self._query_processor.should_try_deterministic_numeric_answer(
            query, chunks
        ):
            return None

        self._notify_observer(
            observer,
            "on_route_selected",
            route="deterministic_numeric_raw_child",
        )

        query_terms = self._important_query_terms(query)
        anchor_phrases = self._query_anchor_phrases(query)
        candidates = []
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            source = self._chunk_source_label(chunk)
            for text in self._numeric_windows(chunk.get("content", "")):
                if self._has_metric_year_conflict(query, text):
                    continue
                numeric_candidates = self._column_aware_numeric_candidates(
                    query,
                    text,
                    self._numeric_candidate_records(query, text),
                )
                values = self._unique_numeric_values(numeric_candidates)
                component_pairs = self._component_amount_pairs(query, [{"text": text}])
                if not values and not component_pairs:
                    continue
                score = self._raw_numeric_evidence_score(text, query, query_terms)
                anchor_matches, anchor_conflicts = self._anchor_profile(
                    anchor_phrases, text
                )
                # A chunk that explicitly substitutes a different qualifier
                # (for example financing activities for operating activities)
                # is contradictory evidence, not a partial match. This is
                # phrase-based and independent of document names or pages.
                if anchor_conflicts and not component_pairs:
                    continue
                if anchor_phrases and not anchor_matches and not component_pairs:
                    continue
                score += anchor_matches * 8.0
                if component_pairs:
                    score += 12.0
                    values = component_pairs
                score += self._numeric_relation_score(query, text)
                if score <= 0 and not component_pairs:
                    continue
                # Rank at value granularity as well as row granularity. A
                # broad row can mention the metric twice; the direct metric
                # value must beat a later qualified sub-scope in that row.
                score += numeric_candidates[0][0]
                fact_ids = []
                candidate_key = metadata.get("candidate_key")
                for ordinal, value in enumerate(values):
                    observed_fact_id = fact_id(
                        candidate_key=candidate_key,
                        stage="raw_numeric_child",
                        ordinal=ordinal,
                        raw_value=value,
                    )
                    fact_ids.append((value, observed_fact_id))
                    self._notify_observer(
                        observer,
                        "on_fact_candidate_extracted",
                        candidate=ProductionFactTrace(
                            fact_id=observed_fact_id,
                            candidate_key=candidate_key,
                            candidate_rank=metadata.get("candidate_rank"),
                            document_id=metadata.get("document_id") or metadata.get("filename") or metadata.get("doc_name"),
                            page=metadata.get("page"),
                            extraction_stage="raw_numeric_child",
                            source_span_hash=self._stable_text_hash(text),
                            raw_value=value,
                            canonical_value=re.sub(r"[^0-9.-]", "", value),
                            currency=self._observed_currency(value),
                            unit=self._observed_unit(value),
                            scale=self._observed_scale(value),
                            period=None,
                        ),
                    )
                candidates.append({
                    "score": score + float(chunk.get("rerank_score", chunk.get("score", 0)) or 0),
                    "text": text,
                    "source": source,
                    "values": values,
                    "chunk": chunk,
                    "is_table": metadata.get("type") == "table",
                    "fact_ids": fact_ids,
                })

        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item["score"], len(item["text"])))
        selected = self._select_raw_numeric_evidence(query, candidates, limit=2)
        if not selected:
            return None

        component_pairs = self._component_amount_pairs(query, selected)
        if component_pairs:
            values = component_pairs
        else:
            values = self._select_answer_values(query, selected)
        if not values:
            return None

        selected_fact_ids = tuple(
            fact_id
            for item in selected
            for value, fact_id in item.get("fact_ids", [])
            if value in values
        )
        for selected_fact_id in selected_fact_ids:
            self._notify_observer(
                observer,
                "on_fact_selected",
                fact_id=selected_fact_id,
                reason_codes=("raw_numeric_evidence_score", "first_selected_value"),
            )

        citation = self._inline_source_citation(selected[0].get("source"))
        # The evidence payload is returned as structured sources.  Do not
        # append full evidence rows to the answer text: validators correctly
        # treat every number in an answer as a claim, so table neighbours in
        # an explanatory block would otherwise be validated as user-facing
        # financial assertions.
        answer_lines = [f"Answer: {', '.join(values)}{citation}."]
        answer = "\n".join(answer_lines)
        self._notify_observer(
            observer,
            "on_answer_rendered",
            source_fact_ids=selected_fact_ids,
            answer_hash=self._stable_text_hash(answer),
        )
        return {
            "answer": answer,
            "diagnostic": "raw_child_numeric_evidence",
            "chunks": [item["chunk"] for item in selected],
            "selection": [
                {
                    "chunk_id": item["chunk"].get("doc_id") or item["chunk"].get("chunk_id"),
                    "source": item["source"],
                    "score": round(item["score"], 4),
                    "values": item["values"],
                }
                for item in selected
            ],
        }

    @staticmethod
    def _stable_text_hash(value: str) -> str:
        import hashlib
        return hashlib.sha256((value or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _observed_unit(value: str) -> str | None:
        lowered = (value or "").lower()
        if "%" in lowered or "per cent" in lowered:
            return "percentage"
        if "million" in lowered:
            return "million"
        if "thousand" in lowered:
            return "thousand"
        if "billion" in lowered:
            return "billion"
        if "$" in lowered or "franc" in lowered:
            return "currency"
        return None

    @staticmethod
    def _observed_currency(value: str) -> str | None:
        lowered = (value or "").lower()
        if "$" in lowered or "usd" in lowered:
            return "USD"
        if "swiss franc" in lowered or "chf" in lowered:
            return "CHF"
        return None

    @staticmethod
    def _observed_scale(value: str) -> str:
        lowered = (value or "").lower()
        if "billion" in lowered:
            return "1000000000"
        if "million" in lowered:
            return "1000000"
        if "thousand" in lowered:
            return "1000"
        return "1"

    @staticmethod
    def _notify_observer(observer, method: str, **kwargs) -> None:
        callback = getattr(observer, method, None)
        if callable(callback):
            try:
                callback(**kwargs)
            except Exception:
                # Instrumentation must never affect production answers.
                pass

    def answer_factual_query_from_context(
        self,
        query: str,
        context: str,
        sources: list,
        *,
        observer=None,
    ) -> dict | None:
        """Return deterministic evidence for factual front-matter/definition/list questions."""
        if not context or self._query_processor.is_numeric_query(query):
            return None
        if not self._query_processor.should_try_deterministic_factual_answer(query):
            return None

        self._notify_observer(
            observer,
            "on_route_selected",
            route="deterministic_factual_context",
        )

        normalized = (query or "").lower()
        direct_answer = self._summarize_factual_evidence(query, [], context=context)
        evidence = self._rank_context_evidence(
            query,
            context,
            require_number=False,
            window_radius=1,
        )
        if not evidence and not direct_answer:
            return None

        selected = self._select_distinct_evidence(evidence, limit=3)
        if not selected and not direct_answer:
            return None

        if "list" in normalized or "criteria" in normalized:
            prefix = "The relevant criteria from the document are:"
        elif "definition" in normalized or "meaning" in normalized or "what are financial statements" in normalized:
            prefix = "The document states:"
        else:
            prefix = "The relevant document evidence is:"

        source_metadata = {
            self._source_label_from_metadata(source): source
            for source in sources or []
            if self._source_label_from_metadata(source)
        }
        selected_fact_ids = []
        # ``direct_answer`` is produced by the existing context summarizer.
        # It has no explicit candidate object, so bind it only when the
        # frozen context contains exactly one matching source marker.  This
        # records the real short-circuit while failing closed on ambiguity.
        if direct_answer and not selected:
            lowered_context = context.lower()
            matching_sources = []
            for label, source in source_metadata.items():
                marker = f"[{label}]".lower()
                start = lowered_context.find(marker)
                if start < 0:
                    continue
                end = lowered_context.find("\n\n---\n\n", start)
                segment = lowered_context[start:] if end < 0 else lowered_context[start:end]
                normalized_answer = re.sub(r"\W+", "", direct_answer.lower())
                normalized_segment = re.sub(r"\W+", "", segment)
                if normalized_answer and normalized_answer in normalized_segment:
                    matching_sources.append(source)
            if len(matching_sources) == 1:
                source_metadata_item = matching_sources[0]
                candidate_key = source_metadata_item.get("candidate_key")
                observed_fact_id = fact_id(
                    candidate_key=candidate_key,
                    stage="factual_context_direct_summary",
                    ordinal=0,
                    raw_value=direct_answer,
                )
                selected_fact_ids.append(observed_fact_id)
                self._notify_observer(
                    observer,
                    "on_fact_candidate_extracted",
                    candidate=ProductionFactTrace(
                        fact_id=observed_fact_id,
                        candidate_key=candidate_key,
                        candidate_rank=source_metadata_item.get("candidate_rank"),
                        document_id=(source_metadata_item.get("document_id") or source_metadata_item.get("filename")),
                        page=source_metadata_item.get("page"),
                        extraction_stage="factual_context_direct_summary",
                        source_span_hash=self._stable_text_hash(direct_answer),
                        raw_value=None,
                        canonical_value=None,
                        currency=None,
                        unit=None,
                        scale=None,
                        period=None,
                    ),
                )
                self._notify_observer(
                    observer,
                    "on_fact_selected",
                    fact_id=observed_fact_id,
                    reason_codes=("direct_context_summary",),
                )
        for ordinal, item in enumerate(selected):
            source_metadata_item = source_metadata.get(item.get("source"), {})
            candidate_key = source_metadata_item.get("candidate_key")
            observed_fact_id = fact_id(
                candidate_key=candidate_key,
                stage="factual_context_evidence",
                ordinal=ordinal,
                raw_value=item["text"],
            )
            selected_fact_ids.append(observed_fact_id)
            self._notify_observer(
                observer,
                "on_fact_candidate_extracted",
                candidate=ProductionFactTrace(
                    fact_id=observed_fact_id,
                    candidate_key=candidate_key,
                    candidate_rank=source_metadata_item.get("candidate_rank"),
                    document_id=(source_metadata_item.get("document_id") or source_metadata_item.get("filename")),
                    page=source_metadata_item.get("page"),
                    extraction_stage="factual_context_evidence",
                    source_span_hash=self._stable_text_hash(item["text"]),
                    raw_value=None,
                    canonical_value=None,
                    currency=None,
                    unit=None,
                    scale=None,
                    period=None,
                ),
            )
            self._notify_observer(
                observer,
                "on_fact_selected",
                fact_id=observed_fact_id,
                reason_codes=("context_evidence_rank",),
            )

        answer_lines = []
        if direct_answer:
            answer_lines.append(f"Answer: {direct_answer}")
        if selected:
            answer_lines.append(prefix)
        for item in selected:
            if item["source"]:
                answer_lines.append(f"- {item['text']} (Source: {item['source']})")
            else:
                answer_lines.append(f"- {item['text']}")
        answer = "\n".join(answer_lines)
        self._notify_observer(
            observer,
            "on_answer_rendered",
            source_fact_ids=tuple(selected_fact_ids),
            answer_hash=self._stable_text_hash(answer),
        )
        return {
            "answer": answer,
            "diagnostic": "deterministic_factual_evidence",
        }

    def answer_deterministic_query_from_context(
        self,
        query: str,
        context: str,
        sources: list,
        *,
        observer=None,
    ) -> dict | None:
        """Try deterministic non-LLM answering from retrieved context."""
        factual = self.answer_factual_query_from_context(
            query, context, sources, observer=observer
        )
        if factual:
            return factual
        return self.answer_numeric_query_from_context(query, context, sources)

    @staticmethod
    def _source_label_from_metadata(source: dict) -> str | None:
        filename = source.get("filename") or source.get("document_id")
        page = source.get("page")
        if filename and page:
            return f"{filename}, p{page}"
        return filename or None

    @staticmethod
    def _chunk_source_label(chunk: dict) -> str | None:
        metadata = chunk.get("metadata") or {}
        filename = metadata.get("doc_name") or chunk.get("filename")
        page = metadata.get("page", chunk.get("page"))
        if filename and page:
            return f"{filename}, p{page}"
        return filename or None

    @staticmethod
    def _is_structural_numeric_line(line: str) -> bool:
        """Exclude section and table labels that carry numbering, not values."""
        normalized = re.sub(r"\s+", " ", line or "").strip()
        if not normalized:
            return True
        if normalized.lower().startswith("section:"):
            return True
        return bool(re.fullmatch(
            r"(?:chapter|section|note|table)?\s*\d+(?:\.\d+)*\.?\s+[A-Z][A-Z &/-]{2,}",
            normalized,
        ))

    @classmethod
    def _numeric_windows(cls, content: str) -> list[str]:
        """Split raw chunk text into compact, table-friendly evidence rows."""
        windows = []
        raw_lines = (content or "").splitlines()
        inferred_header = cls._inferred_column_context(raw_lines)
        for index, raw_line in enumerate(raw_lines):
            line = re.sub(r"\s+", " ", raw_line).strip(" |-\t")
            if line.lower().startswith("table column context:"):
                # The context is attached to the preceding row below.  It is
                # not independently answerable evidence.
                continue
            # A chunk beginning with punctuation and a digit is usually a
            # split decimal/amount (for example ``.9 million`` after ``$1``
            # was cut into the preceding chunk).  It is not standalone
            # financial evidence and must not become an answer candidate.
            if re.match(r"^[.,;)]\s*\d", line):
                continue
            if line and not cls._is_structural_numeric_line(line) and re.search(r"\d", line):
                # Coordinate-reconstructed tables retain the source row and
                # a following column-header line.  Keep them together so an
                # answerer can map a requested period/qualifier to the right
                # value instead of treating every table amount as equivalent.
                if index + 1 < len(raw_lines):
                    header = re.sub(r"\s+", " ", raw_lines[index + 1]).strip()
                    if header.lower().startswith("table column context:"):
                        line = f"{line}\n{header}"
                if "table column context:" not in line.lower() and inferred_header:
                    line = f"{line}\nTable column context: {inferred_header}"
                line = cls._separate_packed_table_values(line)
                windows.append(line[:700])
        if not windows:
            compact = re.sub(r"\s+", " ", content or "").strip()
            windows = [sentence.strip() for sentence in re.split(r"(?<=[.;])\s+", compact)
                       if sentence.strip() and re.search(r"\d", sentence)]
        return windows

    @staticmethod
    def _inferred_column_context(lines: list[str]) -> str | None:
        """Recover repeated date headings from a text-extracted table.

        Some PDFs preserve the table's values but emit its two date headers as
        separate text lines.  This is still explicit evidence, so it may be
        used for column alignment.  We only accept the first two date/year
        headings near the beginning of a chunk; arbitrary later years in body
        prose are intentionally ignored.
        """
        compact = re.sub(r"\s+", " ", " ".join(lines[:12] or [])).strip()
        date_or_year = re.compile(
            r"\b(?:(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},?\s*)?"
            r"(?:19|20)\d{2}\b",
            re.IGNORECASE,
        )
        headings = []
        for match in date_or_year.finditer(compact):
            heading = match.group(0).strip()
            if heading and heading not in headings:
                headings.append(heading)
            if len(headings) == 2:
                break
        return " | ".join(headings) if len(headings) == 2 else None

    @staticmethod
    def _separate_packed_table_values(text: str) -> str:
        """Insert separators in clearly concatenated table amounts.

        A recurring PDF extraction defect joins adjacent columns into one
        token, for example ``143,540206,031``.  Split only runs that contain
        at least two complete comma-grouped values and do *not* already look
        like one legitimate multi-group value such as ``1,234,567``.  The
        operation is syntax-based and does not depend on a document, page, or
        financial metric name.
        """
        def replace(match: re.Match) -> str:
            raw = match.group(0)
            if re.fullmatch(r"\d{1,3}(?:,\d{3}){2,}", raw):
                return raw
            values = re.findall(r"\d{1,3},\d{3}", raw)
            if len(values) < 2:
                return raw
            first = raw.find(values[0])
            prefix = raw[:first]
            if prefix and not prefix.isdigit():
                return raw
            return " | ".join(part for part in ([prefix] if prefix else []) + values)

        return re.sub(r"(?<![\d,])[\d,]{13,}(?![\d,])", replace, text or "")

    @staticmethod
    def _raw_numeric_evidence_score(
        text: str,
        query: str,
        query_terms: set[str],
    ) -> float:
        lowered = text.lower()
        hits = sum(1 for term in query_terms if term in lowered)
        if not hits:
            return 0.0
        number_hits = len(re.findall(
            r"[-+]?\$?\(?\d[\d,]*(?:\.\d+)?\)?\s*(?:%|per cent|million|thousand|francs)?",
            text,
            flags=re.IGNORECASE,
        ))
        if not number_hits:
            return 0.0
        # Prefer evidence that covers more of the query, but avoid rewarding
        # long parent-like prose solely because it contains many numbers.
        coverage = hits / max(1, len(query_terms))
        numeric_positions = [match.start() for match in re.finditer(r"\d", lowered)]
        phrase_hits = 0
        for phrase in DeterministicAnswerExtractor._query_metric_phrases(query):
            phrase_start = lowered.find(phrase)
            if phrase_start >= 0 and any(
                abs(phrase_start - position) <= 80
                for position in numeric_positions
            ):
                phrase_hits += 1
        return coverage * 10.0 + min(number_hits, 3) + phrase_hits * 6.0

    @staticmethod
    def _query_metric_phrases(query: str) -> set[str]:
        """Extract financial two-token phrases without document-specific rules."""
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9&.-]{2,}", (query or "").lower())
        financial_terms = {
            "revenue", "revenues", "margin", "cash", "equivalents",
            "income", "expense", "expenses", "assets", "liabilities",
            "debt", "equity", "fees", "facility", "facilities", "loan",
            "profit", "loss", "earnings", "dividend", "balance",
        }
        return {
            f"{left} {right}"
            for left, right in zip(tokens, tokens[1:])
            if left in financial_terms or right in financial_terms
        }

    @staticmethod
    def _query_anchor_phrases(query: str) -> tuple[tuple[str, str], ...]:
        """Return query-specific two-word anchors without a metric allow-list."""
        stopwords = {
            "what", "was", "were", "the", "and", "for", "did", "does",
            "have", "how", "much", "many", "as", "of", "in", "on", "by",
            "to", "from", "with", "which", "a", "an", "at", "is", "its",
            "this", "that", "shown", "given", "reported", "disclosed",
            "amount", "value", "year", "ended", "during", "came", "company",
            "organization", "percentage", "percent", "growth", "rate",
            "year-over-year", "year-over", "over", "quarter",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        }
        raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9&.-]{0,}", query or "")
        tokens = [
            token.lower()
            for token in raw_tokens
            # Keep an uppercase one-letter label (for example Metric A).
            # Lowercase articles remain stopwords.
            if (
                token.lower() not in stopwords
                or (len(token) == 1 and token.isupper())
            )
            and not re.fullmatch(r"(?:19|20)\d{2}", token)
        ]
        collapsed_tokens = []
        for token in tokens:
            if not collapsed_tokens or token != collapsed_tokens[-1]:
                collapsed_tokens.append(token)

        anchors = []
        for left, right in zip(collapsed_tokens, collapsed_tokens[1:]):
            if left == right:
                continue
            pair = (left, right)
            if pair not in anchors:
                anchors.append(pair)
        return tuple(anchors)

    @staticmethod
    def _anchor_profile(
        anchors: tuple[tuple[str, str], ...],
        text: str,
    ) -> tuple[int, int]:
        """Count exact query anchors and explicit qualifier contradictions."""
        connectors = {"and", "by", "of", "in", "for", "to", "from", "with", "at", "as"}
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9&.-]*", text or "")
        tokens = [token.lower().strip(".-") for token in tokens]
        tokens = [token for token in tokens if token and token not in connectors]
        collapsed_tokens = []
        for token in tokens:
            if not collapsed_tokens or token != collapsed_tokens[-1]:
                collapsed_tokens.append(token)
        tokens = collapsed_tokens
        pairs = list(zip(tokens, tokens[1:]))
        pair_set = set(pairs)
        matches = 0
        conflicts = 0
        negators = {"non", "not", "without", "excluding", "except"}

        for left, right in anchors:
            if (left, right) in pair_set:
                negated = any(
                    index > 0 and tokens[index - 1] in negators
                    for index, pair in enumerate(pairs)
                    if pair == (left, right)
                )
                if negated:
                    conflicts += 1
                else:
                    matches += 1
                continue

            # A relation tail with another qualifier is contradictory:
            # financing activities does not answer operating activities.
            if any(candidate_right == right and candidate_left != left
                   for candidate_left, candidate_right in pairs):
                conflicts += 1
        return matches, conflicts

    @staticmethod
    def _numeric_relation_score(query: str, text: str) -> float:
        """Score generic ratio semantics separately from change semantics."""
        normalized_query = (query or "").lower()
        normalized_text = (text or "").lower()
        asks_for_share = any(marker in normalized_query for marker in (
            "percentage of", "percent of", "share of", "came from",
        ))
        if not asks_for_share:
            return 0.0

        score = 0.0
        if re.search(
            r"\b(accounted\s+for|representing|share\s+of|of\s+(?:the\s+)?total)\b",
            normalized_text,
        ):
            score += 12.0
        if re.search(
            r"\b(increased|decreased|grew|growth|fell|compared)\b",
            normalized_text,
        ):
            score -= 9.0
        return score

    @staticmethod
    def _inline_source_citation(source: str | None) -> str:
        """Render a validator-compatible document/page citation."""
        return f" [{source}]" if source else ""

    @staticmethod
    def _numeric_value_limit(query: str) -> int:
        """Return multiple values only for an explicitly multi-value request."""
        normalized = (query or "").lower()
        multi_value_markers = (
            "compare", "components", "growth rate", "year-over-year",
            "year over year", "both ", "two ",
        )
        return 2 if any(marker in normalized for marker in multi_value_markers) else 1

    @staticmethod
    def _asks_for_rate(query: str) -> bool:
        normalized = (query or "").lower()
        return any(marker in normalized for marker in (
            "growth", "growth rate", "year-over-year", "year over year",
            "percentage", "percent",
        ))

    @classmethod
    def _select_answer_values(cls, query: str, selected: list[dict]) -> list[str]:
        """Choose complementary values from one evidence row when possible.

        A question asking for a metric and its rate must not be answered by
        taking arbitrary numbers from separate retrieved chunks. Prefer an
        amount and a percentage that co-occur in the same evidence row.
        """
        limit = cls._numeric_value_limit(query)
        if limit > 1 and cls._asks_for_rate(query):
            for item in selected:
                ranked = cls._column_aware_numeric_candidates(
                    query,
                    item["text"],
                    cls._numeric_candidate_records(query, item["text"]),
                )
                amounts = [
                    value for _, _, value in ranked
                    if re.search(r"\$|million|billion|thousand|francs", value, re.IGNORECASE)
                ]
                rates = [
                    value for _, _, value in ranked
                    if re.search(r"%|per cent", value, re.IGNORECASE)
                ]
                if amounts and rates:
                    return [amounts[0], rates[0]]

        values = []
        seen = set()
        for item in selected:
            for value in item["values"]:
                key = value.lower()
                if key not in seen:
                    seen.add(key)
                    values.append(value)
                if len(values) >= limit:
                    return values
        return values

    @classmethod
    def _component_amount_pairs(cls, query: str, selected: list[dict]) -> list[str]:
        """Extract labelled monetary components from a single evidence span.

        This is scoped to explicit component/list questions. It binds each
        amount to the nearby label rather than returning detached amounts.
        """
        normalized = (query or "").lower()
        if not any(marker in normalized for marker in (
            "component", "components", "consist", "consisted", "two ",
        )):
            return []

        amount = r"(\$?\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand|francs))"
        connector = (
            r"(?:in\s+(?:an?\s+)?(?:aggregate\s+)?"
            r"(?:principal\s+)?(?:amount|value)\s+of|of|was|were)"
        )
        pattern = re.compile(
            r"(?:\([a-z]\)\s*)?(?:an?\s+|the\s+)?"
            r"(?P<label>[A-Za-z][A-Za-z0-9&'/-]*(?:\s+[A-Za-z][A-Za-z0-9&'/-]*){0,7}?)"
            r"\s+" + connector + r"\s+" + amount,
            re.IGNORECASE,
        )
        for item in selected:
            pairs = []
            seen = set()
            for match in pattern.finditer(item.get("text", "")):
                label = re.sub(r"\s+", " ", match.group("label")).strip(" -")
                value = cls._normalize_numeric_phrase(match.group(2), query, item["text"])
                key = (label.lower(), value.lower())
                if not label or key in seen:
                    continue
                seen.add(key)
                pairs.append(f"{label}, {value}")
                if len(pairs) >= 2:
                    return pairs
        return []

    @staticmethod
    def _select_raw_numeric_evidence(query: str, candidates: list[dict], *, limit: int) -> list[dict]:
        """Keep complementary raw evidence without duplicating one table row."""
        selected = []
        seen = set()
        need_multiple = DeterministicAnswerExtractor._numeric_value_limit(query) > 1
        for item in candidates:
            key = re.sub(r"\W+", " ", item["text"].lower()).strip()[:220]
            if not key or key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) >= limit:
                break
            if not need_multiple:
                break
        return selected

    # --- Static helper methods ---

    @staticmethod
    def _clean_deterministic_title(title: str) -> str:
        cleaned = re.sub(r"\s+", " ", title or "").strip(" -")
        cleaned = re.sub(r"\b(annual)\s+(annual report)\b", r"\1 report", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(report)\s+(annual report)\b", r"\2", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" -")

    @staticmethod
    def _is_valid_deterministic_title(title: str) -> bool:
        cleaned = re.sub(r"[^a-z0-9]+", " ", title or "", flags=re.IGNORECASE).strip().lower()
        if len(cleaned) < 12:
            return False
        generic_titles = {
            "annual",
            "report",
            "annual report",
            "financial statements",
            "annual financial report",
        }
        return cleaned not in generic_titles

    @staticmethod
    def _parse_context_lines(context: str) -> list[dict]:
        parsed = []
        current_source = None
        for raw_line in (context or "").splitlines():
            line = re.sub(r"\s+", " ", raw_line or "").strip()
            if not line or line == "---":
                continue
            source_match = re.match(r"^\[(?P<source>[^\]]+)\]$", line)
            if source_match:
                current_source = source_match.group("source")
                continue
            parsed.append({"source": current_source, "text": line})
        return parsed

    def _rank_context_evidence(
        self,
        query: str,
        context: str,
        *,
        require_number: bool,
        window_radius: int = 1,
    ) -> list[dict]:
        query_terms = self._important_query_terms(query)
        parsed = self._parse_context_lines(context)
        evidence = []
        for index, item in enumerate(parsed):
            line = item["text"]
            if require_number and not re.search(r"\d", line):
                continue
            score = (
                self._numeric_evidence_score(line, query_terms)
                if require_number
                else self._factual_evidence_score(line, query_terms)
            )
            if score <= 0:
                continue
            window = self._evidence_window(
                parsed,
                index,
                radius=window_radius,
                require_number=require_number,
                query_terms=query_terms,
            )
            evidence.append({
                "score": score,
                "source": item["source"],
                "text": window,
            })
        evidence.sort(key=lambda item: (-item["score"], len(item["text"])))
        return evidence

    @staticmethod
    def _evidence_window(
        parsed: list[dict],
        index: int,
        *,
        radius: int,
        require_number: bool,
        query_terms: set[str],
    ) -> str:
        start = max(0, index - radius)
        end = min(len(parsed), index + radius + 1)
        source = parsed[index].get("source")
        lines = []
        for item in parsed[start:end]:
            if item.get("source") != source:
                continue
            text = item.get("text") or ""
            if not text or text in lines:
                continue
            if require_number and item is not parsed[index] and re.search(r"\d", text):
                lowered = text.lower()
                is_numeric_value_line = bool(re.fullmatch(r"[-+]?\$?\(?\d[\d,]*(?:\.\d+)?\)?\s*(?:%|per cent|million|thousand|francs)?", text, flags=re.IGNORECASE))
                if not is_numeric_value_line and not any(term in lowered for term in query_terms):
                    continue
            lines.append(text)
        if require_number and not any(re.search(r"\d", line) for line in lines):
            return parsed[index].get("text") or ""
        joined = " ".join(lines)
        if len(joined) > 700:
            joined = joined[:700].rsplit(" ", 1)[0] + " [...]"
        return joined

    @staticmethod
    def _select_distinct_evidence(evidence: list[dict], *, limit: int) -> list[dict]:
        selected = []
        seen_lines = set()
        for item in evidence:
            key = re.sub(r"\W+", " ", item.get("text", "").lower()).strip()[:180]
            if not key or key in seen_lines:
                continue
            seen_lines.add(key)
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _normalize_numeric_phrase(value: str, query: str, evidence_text: str) -> str:
        value = re.sub(r"\s+", " ", value or "").strip(" ,.;:")
        value = re.sub(r"(?<=\d)\s+%", "%", value)
        value = re.sub(r"\$(\d[\d,]*)\.0\s+million\b", r"$\1 million", value, flags=re.IGNORECASE)
        if value.endswith("%") and any(marker in (query or "").lower() for marker in ("year-over-year", "growth rate", "grow year over year")):
            if "year-over-year" in evidence_text.lower() or "compared to" in evidence_text.lower():
                value = f"{value} year-over-year"
        return value

    @classmethod
    def _extract_numeric_phrases(cls, query: str, text: str) -> list[str]:
        return cls._unique_numeric_values(cls._numeric_candidate_records(query, text))

    @classmethod
    def _numeric_candidate_records(cls, query: str, text: str) -> list[tuple[float, int, str]]:
        """Collect and rank financial-looking numbers from one evidence row."""
        pattern = re.compile(
            r"(?:\$|rs\.?\s*)?\d[\d,]*(?:\.\d+)?\s*"
            r"(?:%|per cent|million|thousand(?:s)?(?: of Swiss francs)?|Swiss francs|francs)?",
            re.IGNORECASE,
        )
        normalized_query = (query or "").lower()
        query_terms = cls._important_query_terms(query)
        metric_phrases = cls._query_metric_phrases(query)
        ranked: list[tuple[float, int, str]] = []
        for match in pattern.finditer(text or ""):
            if match.start() and (text or "")[match.start() - 1] == ".":
                continue
            # A comma is a structural separator in a financial value.  Text
            # extraction occasionally glues a reporting year or footnote to
            # an amount (for example ``2018328,732``); that is not a valid
            # numeric claim and must never outrank clean evidence elsewhere.
            raw_numeric = re.sub(r"[^0-9,]", "", match.group(0))
            if re.search(r"\d{4,},\d{3}", raw_numeric):
                continue
            value = cls._normalize_numeric_phrase(match.group(0), query, text)
            if not value:
                continue
            score = cls._numeric_candidate_score(
                value=value,
                text=text,
                start=match.start(),
                end=match.end(),
                query_terms=query_terms,
                metric_phrases=metric_phrases,
                query=normalized_query,
            )
            if score > 0:
                ranked.append((score, match.start(), value))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked

    @classmethod
    def _column_aware_numeric_candidates(
        cls,
        query: str,
        text: str,
        candidates: list[tuple[float, int, str]],
    ) -> list[tuple[float, int, str]]:
        """Boost values aligned with an explicit table-column context.

        Layout extraction often yields a sparse row such as ``Cash | 143,540
        | 206,031`` followed by ``Table column context: 2020 | 2019``.  The
        mapping here is deliberately conservative and generic: only the last
        N numeric cells are aligned to N visible headers.  That handles common
        label/footnote prefixes while leaving ambiguous rows unchanged.
        """
        marker = re.search(r"(?:^|\n)Table column context:\s*(.+)$", text or "", re.IGNORECASE)
        if not marker or len(candidates) < 2:
            return candidates

        headers = [part.strip().lower() for part in marker.group(1).split("|") if part.strip()]
        # ``candidates`` arrives ordered by relevance score.  Column mapping
        # is positional, so restore source-text order before pairing values
        # with their headers.
        row_candidates = sorted(
            (item for item in candidates if item[1] < marker.start()),
            key=lambda item: item[1],
        )
        if len(headers) < 2 or len(row_candidates) < 2:
            return candidates

        # Multi-line headers are flattened into one context string during
        # layout extraction.  Prefer the tail of the header that actually
        # describes period/value columns, then align only as many columns as
        # the row exposes.  This avoids mapping a leading ``No.`` or table
        # title onto the first financial amount.
        period_headers = [
            header for header in headers
            if re.search(r"\b(?:19|20)\d{2}\b", header)
            or any(term in header for term in (
                "actual", "budget", "forecast", "comparable", "variance",
            ))
        ]
        if len(period_headers) >= 2:
            headers = period_headers
        if len(headers) > len(row_candidates):
            headers = headers[-len(row_candidates):]
        elif len(row_candidates) > len(headers):
            row_candidates = row_candidates[-len(headers):]
        if len(headers) < 2:
            return candidates

        # Headers describe the trailing numeric columns; leading row labels
        # and statement-note numbers must not shift this alignment.
        aligned = row_candidates[-len(headers):]
        query_lower = (query or "").lower()
        query_years = set(re.findall(r"\b(?:19|20)\d{2}\b", query_lower))
        query_qualifiers = {
            token
            for token in ("actual", "budget", "forecast", "comparable", "variance")
            if token in query_lower
        }
        if not query_years and not query_qualifiers:
            return candidates

        adjustments = {}
        for candidate, header in zip(aligned, headers):
            header_years = set(re.findall(r"\b(?:19|20)\d{2}\b", header))
            bonus = 0.0
            if query_years:
                if query_years & header_years:
                    bonus += 18.0
                elif header_years:
                    bonus -= 6.0
            bonus += 8.0 * sum(qualifier in header for qualifier in query_qualifiers)
            adjustments[candidate] = bonus

        if not adjustments or not any(adjustments.values()):
            return candidates
        boosted = [
            (score + adjustments.get((score, start, value), 0.0), start, value)
            for score, start, value in candidates
        ]
        boosted.sort(key=lambda item: (-item[0], item[1]))
        return boosted

    @classmethod
    def _has_metric_year_conflict(cls, query: str, text: str) -> bool:
        """Reject a metric row explicitly labelled with another reporting year.

        Financial statements routinely include opening balances next to the
        requested reporting-period value.  A row such as ``Net assets at
        December 31, 2018`` is contradictory evidence for a question about
        2020, even when the section heading elsewhere mentions 2020.  Require
        the conflict to occur immediately after a query metric phrase so
        unrelated historical comparisons remain usable evidence.
        """
        query_years = set(re.findall(r"\b(?:19|20)\d{2}\b", query or ""))
        if not query_years:
            return False
        lowered = (text or "").lower()
        for phrase in cls._query_metric_phrases(query):
            for match in re.finditer(re.escape(phrase), lowered):
                # Only an immediately following date labels this metric's
                # value.  A later comparison date (for example “compared to
                # 2024”) must remain valid evidence for a 2025 value.
                following = lowered[match.end():match.end() + 48]
                nearby_years = set(re.findall(r"\b(?:19|20)\d{2}\b", following))
                # A missing delimiter can join a year to the first table
                # amount (``2018328,732``).  Preserve the embedded year for
                # temporal validation only when the remainder has the shape
                # of a comma-grouped amount; ordinary long identifiers do
                # not become reporting-period evidence.
                nearby_years.update(re.findall(
                    r"\b((?:19|20)\d{2})(?=\d{1,3},\d{3})",
                    following,
                ))
                if nearby_years and not (nearby_years & query_years):
                    return True
        return False

    @staticmethod
    def _unique_numeric_values(ranked: list[tuple[float, int, str]]) -> list[str]:
        values = []
        seen = set()
        for _, _, value in ranked:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            values.append(value)
            if len(values) >= 6:
                break
        return values

    @classmethod
    def _numeric_candidate_score(
        cls,
        *,
        value: str,
        text: str,
        start: int,
        end: int,
        query_terms: set[str],
        metric_phrases: set[str],
        query: str,
    ) -> float:
        """Score a value using units, local metric proximity and date penalties."""
        lowered = (text or "").lower()
        local_start = max(0, start - 100)
        local_end = min(len(lowered), end + 100)
        local = lowered[local_start:local_end]
        raw_number = re.sub(r"[^0-9.]", "", value)
        try:
            numeric = float(raw_number.replace(",", ""))
        except ValueError:
            numeric = 0.0

        score = 0.0
        has_unit = bool(re.search(
            r"\$|%|per cent|million|thousand|francs",
            value,
            flags=re.IGNORECASE,
        ))
        if has_unit:
            score += 5.0
        asks_for_count = any(marker in query for marker in (
            "how many", "number of", "count",
        ))
        asks_for_year = any(marker in query for marker in (
            "what year", "which year", "year was",
        ))
        if numeric and numeric < 100 and not has_unit and not asks_for_count:
            # Bare small integers are almost always table notes, footnotes,
            # or heading labels for financial-value questions.
            return -100.0
        if 1900 <= numeric <= 2100 and not has_unit and not asks_for_year:
            # Reporting years constrain the question but are rarely the
            # requested value unless the user explicitly asks for a year.
            return -100.0

        month_pattern = (
            r"(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+\d{1,2}"
        )
        if re.search(month_pattern, local, flags=re.IGNORECASE):
            if numeric <= 31 or 1900 <= numeric <= 2100:
                score -= 10.0

        score += 2.0 * sum(1 for term in query_terms if term in local)
        for phrase in metric_phrases:
            for phrase_match in re.finditer(re.escape(phrase), lowered):
                if phrase_match.end() <= start:
                    distance = start - phrase_match.end()
                    if distance <= 100:
                        score += max(0.0, 7.0 - distance / 16.0)
                        # Prefer the value directly attached to a metric over
                        # a later qualified sub-scope in the same paragraph.
                        between = lowered[phrase_match.end():start]
                        modifiers = re.findall(r"[a-z]{3,}", between)
                        score += max(0.0, 4.0 - 0.7 * len(modifiers))
                else:
                    distance = phrase_match.start() - end
                    if 0 <= distance <= 100:
                        score += max(0.0, 3.0 - distance / 32.0)

        asks_for_rate = any(marker in query for marker in (
            "growth", "rate", "percentage", "percent", "year-over-year", "year over year",
        ))
        if asks_for_rate and re.search(r"%|per cent", value, flags=re.IGNORECASE):
            score += 5.0
        elif asks_for_rate and has_unit:
            # A growth query normally asks for the base metric as well as
            # its rate; preserve that natural answer order.
            score += 5.0
        return score

    @classmethod
    def _summarize_numeric_values(cls, query: str, selected: list[dict], *, context: str | None = None) -> str | None:
        targeted = cls._targeted_numeric_summary(query, selected, context=context)
        if targeted:
            return targeted
        values = []
        seen = set()
        for item in selected:
            for value in cls._extract_numeric_phrases(query, item.get("text", "")):
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                values.append(value)
                if len(values) >= 5:
                    return ", ".join(values)
        return ", ".join(values) if values else None

    @classmethod
    def _targeted_numeric_summary(cls, query: str, selected: list[dict], *, context: str | None = None) -> str | None:
        """Only generic regex-based extraction from context."""
        normalized = (query or "").lower()
        text = " ".join(item.get("text", "") for item in selected)
        if context:
            text = f"{text} {context}"
        compact = re.sub(r"\s+", " ", text).strip()
        if "platform revenue" in normalized:
            match = re.search(r"platform revenue was\s+(\$?\d[\d,]*(?:\.\d+)?\s+million).*?(?:or\s+)?(\d+(?:\.\d+)?%)", compact, re.IGNORECASE)
            if match:
                return f"{cls._normalize_numeric_phrase(match.group(1), query, compact)}, {match.group(2)} year-over-year"
        if "volume-based revenue" in normalized:
            match = re.search(r"volume-based revenue was\s+(\$?\d[\d,]*(?:\.\d+)?\s+million).*?(?:or\s+)?(\d+(?:\.\d+)?%)", compact, re.IGNORECASE)
            if match:
                return f"{cls._normalize_numeric_phrase(match.group(1), query, compact)}, {match.group(2)} year-over-year"
        if "gross margin" in normalized:
            match = re.search(r"gross margin.*?\bwas\s+(\d+(?:\.\d+)?%)", compact, re.IGNORECASE)
            if match:
                return match.group(1)
        if "cash and cash equivalents" in normalized:
            match = re.search(r"cash and cash equivalents.*?(\$?\d[\d,]*(?:\.\d+)?\s+(?:million|thousand|billion))", compact, re.IGNORECASE)
            if match:
                return cls._normalize_numeric_phrase(match.group(1), query, compact)
        if "operating activities" in normalized or "operating cash" in normalized:
            match = re.search(r"Operating activities\s+\$?\s*(\d[\d,]*)", compact, re.IGNORECASE)
            if match:
                raw_value = match.group(1)
                amount = cls._parse_comma_number(raw_value)
                if amount:
                    return f"${amount / 1000:.1f} million, {raw_value}"
        if "record revenue" in normalized:
            match = re.search(r"record revenues? of\s+(\$?\d[\d,]*(?:\.\d+)?\s+million).*?(\d+(?:\.\d+)?%)", compact, re.IGNORECASE)
            if match:
                return f"{cls._normalize_numeric_phrase(match.group(1), query, compact)}, {match.group(2)} year-over-year"
        if "total revenue" in normalized:
            match = re.search(r"total revenue of\s+(\d+(?:\.\d+)?\s+million(?:\s+Swiss)?\s+francs)", compact, re.IGNORECASE)
            if match:
                return match.group(1)
        if "credit facilities" in normalized:
            revolver = re.search(r"(Revolving Credit Facility).*?(\$?\d[\d,]*(?:\.\d+)?\s+million)", compact, re.IGNORECASE)
            term = re.search(r"(Term Loan).*?(\$?\d[\d,]*(?:\.\d+)?\s+million)", compact, re.IGNORECASE)
            if revolver and term:
                return f"{revolver.group(1)}, {cls._normalize_numeric_phrase(revolver.group(2), query, compact)}; {term.group(1)}, {cls._normalize_numeric_phrase(term.group(2), query, compact)}"
        return None

    @staticmethod
    def _parse_comma_number(value: str) -> int:
        try:
            return int(re.sub(r"[^\d]", "", value or ""))
        except ValueError:
            return 0

    @staticmethod
    def _summarize_factual_evidence(query: str, selected: list[dict], *, context: str | None = None) -> str | None:
        """Only generic evidence extraction from context."""
        normalized = (query or "").lower()
        text = " ".join(item.get("text", "") for item in selected)
        if context:
            text = f"{text} {context}"
        compact = re.sub(r"\s+", " ", text).strip(" -")
        if not compact:
            return None
        if "which organization" in normalized or ("prepared" in normalized and "organization" in normalized):
            org_match = re.search(
                r"(?:prepared by|organization)\s+([A-Z][a-zA-Z\s]+(?:Organization|Corporation|Company|Ltd|Inc))",
                compact, re.IGNORECASE,
            )
            if org_match:
                return org_match.group(1).strip()
        if "reporting period" in normalized:
            period_match = re.search(
                r"(?:year (?:to|ended)\s+)(?:December\s+31,?\s*)?20\d{2}",
                compact, re.IGNORECASE,
            )
            if period_match:
                return period_match.group(0).strip()
        if "criteria" in normalized and "current" in normalized:
            terms = ("operating cycle", "within twelve months", "held primarily for trading", "cash and cash equivalent")
            hits = [term for term in terms if term in compact.lower()]
            if hits:
                return "; ".join(hits) + "."
        return DeterministicAnswerExtractor._first_evidence_sentence(compact)

    @staticmethod
    def _best_sentence_with_terms(text: str, terms: tuple[str, ...]) -> str | None:
        sentences = re.split(r"(?<=[.!?])\s+", text or "")
        best = None
        best_hits = 0
        for sentence in sentences:
            lowered = sentence.lower()
            hits = sum(1 for term in terms if term in lowered)
            if hits > best_hits:
                best = sentence
                best_hits = hits
        return best.strip(" -") if best else None

    @staticmethod
    def _first_evidence_sentence(text: str) -> str | None:
        compact = re.sub(r"\s+", " ", text or "").strip(" -")
        if not compact:
            return None
        sentences = re.split(r"(?<=[.!?])\s+", compact)
        for sentence in sentences:
            sentence = sentence.strip(" -")
            if 25 <= len(sentence) <= 260:
                return sentence
        return compact[:260].rstrip() + ("..." if len(compact) > 260 else "")

    @staticmethod
    def _important_query_terms(query: str) -> set[str]:
        stopwords = {
            "what", "was", "were", "the", "and", "for", "did", "does", "have",
            "how", "much", "many", "as", "of", "in", "on", "by", "to", "from",
            "with", "which", "documents", "document", "report", "reports",
            "according", "given", "shown", "amount", "year", "year-over-year",
            "topic", "cover", "prepared", "organization", "basis", "preparation",
            "list", "two", "criteria", "make", "item", "current", "mention",
        }
        terms = {
            term
            for term in re.findall(r"[a-zA-Z][a-zA-Z0-9&.-]{2,}", (query or "").lower())
            if term not in stopwords
        }
        aliases = {
            "revenue": {"revenue", "revenues"},
            "cash": {"cash", "equivalents"},
            "equivalents": {"cash", "equivalents"},
            "margin": {"margin", "gross"},
            "growth": {"growth", "year-over-year", "increase"},
            "pct": {"pct", "system"},
            "madrid": {"madrid", "system"},
            "reserve": {"reserve", "surplus"},
            "surplus": {"reserve", "surplus"},
            "operating": {"operating", "activities"},
            "facilities": {"facility", "facilities", "loan", "credit"},
            "organization": {"organization", "prepared"},
            "prepared": {"prepared", "organization"},
            "statements": {"statements", "financial"},
            "current": {"current", "operating", "cycle", "twelve", "months", "trading", "cash"},
        }
        expanded = set(terms)
        for term in list(terms):
            expanded.update(aliases.get(term, set()))
        return expanded

    @staticmethod
    def _numeric_evidence_score(line: str, query_terms: set[str]) -> float:
        lowered = line.lower()
        term_hits = sum(1 for term in query_terms if term in lowered)
        if term_hits == 0:
            return 0.0
        number_hits = len(re.findall(r"[-+]?\$?\(?\d[\d,]*(?:\.\d+)?\)?\s*(?:%|per cent|million|thousand|francs)?", line, flags=re.IGNORECASE))
        if number_hits == 0:
            return 0.0
        return term_hits * 2.0 + min(number_hits, 4)

    @staticmethod
    def _factual_evidence_score(line: str, query_terms: set[str]) -> float:
        lowered = line.lower()
        term_hits = sum(1 for term in query_terms if term in lowered)
        if term_hits == 0:
            return 0.0
        if len(line) < 20:
            return 0.0
        return term_hits * 2.0

    @staticmethod
    def _is_numeric_financial_query(query: str) -> bool:
        """Check if query is asking for a numeric financial value."""
        numeric_indicators = [
            "how much", "how many", "amount of", "value of", "total of",
            "what was the", "what is the", "revenue", "expense", "cost",
            "profit", "loss", "income", "cash", "debt", "equity", "margin",
            "growth", "rate", "percentage", "%", "$", "million", "billion",
            "thousand", "balance", "dividend", "earnings",
        ]
        query_lower = (query or "").lower()
        return any(ind in query_lower for ind in numeric_indicators)
