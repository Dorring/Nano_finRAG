from dataclasses import dataclass
import re

_STOP = {
    "what", "was", "were", "the", "and", "for", "with", "from", "does",
    "did", "have", "has", "how", "much", "many", "please", "calculate",
    "according", "based", "report", "document", "this", "that", "in", "on",
    "of", "to", "by", "as", "a", "an", "total", "came", "reported", "main",
    "each", "which", "documents", "services", "amount", "value", "shown",
}
_METRICS = ("revenue","cash","equivalents","margin","assets","liabilities","income","expense","profit","loss","fees","budget","growth","rate","percentage","percent")

@dataclass(frozen=True)
class QueryProfile:
    original_query: str
    entities: tuple[str, ...]
    metrics: tuple[str, ...]
    periods: tuple[str, ...]
    document_scope: tuple[str, ...]
    answer_type: str
    is_numeric: bool
    is_multi_document: bool

@dataclass(frozen=True)
class QueryVariant:
    name: str
    text: str
    weight: float

def profile_query(query: str, *, is_numeric: bool) -> QueryProfile:
    text = query or ""
    normalized = text.lower()
    periods = tuple(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", text)))
    metrics = tuple(metric for metric in _METRICS if metric in normalized)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9&.-]*", text)
    entities = tuple(dict.fromkeys(
        token for token in tokens
        if token.lower() not in _STOP
        and token.lower() not in _METRICS
        and not re.fullmatch(r"(?:19|20)\d{2}", token)
    ))
    multi = any(marker in normalized for marker in (
        "compare", "comparison", "which documents", "across documents",
        "each document", "both ", "respectively",
    ))
    if multi:
        answer_type = "multi_document"
    elif any(marker in normalized for marker in ("percentage","percent","share of")):
        answer_type = "percentage"
    elif any(marker in normalized for marker in ("compare","difference","versus","vs")):
        answer_type = "comparison"
    elif is_numeric:
        answer_type = "numeric"
    else:
        answer_type = "fact"
    return QueryProfile(text, entities, metrics, periods, (), answer_type, is_numeric, multi)

def should_use_multi_query(profile: QueryProfile) -> bool:
    return profile.is_numeric or profile.is_multi_document or profile.answer_type in {"percentage","comparison","aggregation"}

def build_compact_query(profile: QueryProfile) -> str | None:
    parts = []
    for value in (*profile.entities, *profile.metrics, *profile.periods):
        clean = re.sub(r"\s+", " ", value).strip()
        if clean and clean.lower() not in {item.lower() for item in parts}:
            parts.append(clean)
    compact = " ".join(parts)
    if not compact or re.sub(r"\W+", "", compact).lower() == re.sub(r"\W+", "", profile.original_query).lower():
        return None
    return compact

def build_query_variants(profile: QueryProfile) -> list[QueryVariant]:
    variants = [QueryVariant("original", profile.original_query, 1.0)]
    if should_use_multi_query(profile):
        compact = build_compact_query(profile)
        if compact:
            variants.append(QueryVariant("compact", compact, 0.85))
    return variants
