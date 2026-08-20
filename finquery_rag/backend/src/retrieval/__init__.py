"""Retrieval pipeline modules."""
from src.retrieval.query_processor import QueryProcessor as QueryProcessor
from src.retrieval.retrieval_pipeline import RetrievalPipeline as RetrievalPipeline
from src.retrieval.candidate_fusion import (
    normalize_scores as normalize_scores,
    dedupe_chunks as dedupe_chunks,
    chunk_doc_name as chunk_doc_name,
    ensure_multi_doc_coverage as ensure_multi_doc_coverage,
    boost_front_matter_chunks as boost_front_matter_chunks,
    summarize_retrieved_chunks as summarize_retrieved_chunks,
    source_from_chunk as source_from_chunk,
)
from src.retrieval.context_builder import (
    ContextBuilder as ContextBuilder,
    EvidenceSufficiencyEvaluator as EvidenceSufficiencyEvaluator,
    SufficiencyResult as SufficiencyResult,
)
from src.retrieval.metadata_scope import (
    FinancialDocumentMetadataV1, FilterStrength, MetadataAwareRetrieverV1,
    MetadataFilterPlannerV1, MetadataProvenance, PeriodSemantics,
    RetrievalScopeV1, ScopeConditionV1, apply_hard_scope,
    apply_soft_preferences, enforce_reranker_subset,
)
