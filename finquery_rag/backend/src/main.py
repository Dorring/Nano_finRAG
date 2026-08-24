from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
import json
import time

from .services.auth import create_access_token, get_current_user, get_password_hash, verify_password
from .services.ingest import get_ingest_lineage, process_pdf
from .services.vector_store import add_documents, list_all_documents, delete_document_collection, get_collection_stats
from .services.rag_engine import RAGEngine
from .services.document_registry import DocumentRegistry, VALID_TRANSITIONS
from .services.session_manager import SessionManager
from .services.memory_profile import UserMemoryStore
from .services.health import collect_health_snapshot
from .services.feedback import FeedbackStore
from .services.query_scope import resolve_query_document_names
from .services.streaming import make_stream_done_event, make_stream_error_event, safe_log_query_trace
from .services.retrieval_config import get_embedding_model_name, get_reranker_model, get_reranker_name
from .evaluation.evaluation import compare_reports, evaluate_payload, feedback_to_replay_case, trace_to_replay_case
from .models.schemas import (
    AnswerabilityResponse,
    DocumentInfo,
    DocumentsListResponse,
    EvalCompareRequest,
    EvalScoreRequest,
    FeedbackRequest,
    FeedbackResponse,
    MemoryProfileRequest,
    MemoryProfileResponse,
    QueryRequest,
    QueryResponse,
    RepairResponse,
    Token,
    UploadResponse,
    UserLogin,
    UserRegister,
    UserResponse,
    ValidationResponse,
)
from .models.user import User #User ORM 模型
from .database import get_db, engine, Base #SQLAlchemy 数据库连接和基础模型
from sqlalchemy.orm import Session #SQLAlchemy 会话管理
from .runtime import (
    FinancialQueryRequest,
    LegacyFinancialRuntimeAdapter,
    QueryExecutionService,
    to_legacy_query_dict,
)

from datetime import timedelta, datetime
import os
import uuid
import shutil
import tempfile
from dotenv import load_dotenv

# Create database tables (if relying on this instead of alembic for initial dev)
#创建数据库表（如果不使用 alembic 进行迁移的话）
Base.metadata.create_all(bind=engine)

# Disable tokenizer parallelism to avoid fork warnings
# 禁用分词器并行化以避免 fork 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize FastAPI
# 初始化 FastAPI 应用
app = FastAPI(
    title="FinQuery API",
    description="Multi-Document Financial Q&A System with User Management",
    version="3.0.0"
)

# Get allowed origins from environment variable
# 从环境变量获取允许的跨域请求来源
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

# Add CORS middleware
# 添加跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载环境变量（必须在读取环境变量之前）
load_dotenv()

# Initialize LLM clients
# 在线对话：通过 FRP 穿透访问学校 GPU 服务器上的 nanochat OpenAI 兼容适配层
llm_client = OpenAI(
    base_url=os.getenv("LLM_API_BASE_URL", "http://127.0.0.1:8500/v1"),
    api_key=os.getenv("LLM_API_KEY", "not-needed-for-local")
)
llm_model_name = os.getenv("LLM_MODEL_NAME", "nanochat")

rag_engine: RAGEngine | None = None
document_registry = DocumentRegistry()
session_manager = SessionManager()
memory_store = UserMemoryStore()
feedback_store = FeedbackStore()

def get_rag_engine():
    """
    延迟初始化 RAG 引擎。
    当需要使用时才进行实例化，以节省系统资源。

    Returns:
        RAGEngine: 初始化后的 RAGEngine 实例。
    """
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine(
            llm_client,
            model_name=llm_model_name,
            use_hybrid=True,
            # Default is os.getenv("RAG_RERANKER", "heuristic") via retrieval_config.
            reranker_name=get_reranker_name(),
            # Model path comes from os.getenv("RAG_RERANKER_MODEL") via retrieval_config.
            reranker_model=get_reranker_model(),
            retrieval_candidate_multiplier=int(os.getenv("RAG_CANDIDATE_MULTIPLIER", "4")),
        )
    return rag_engine

def api_error(status_code: int, error_code: str, message: str) -> HTTPException:
    """Build a stable business-error response envelope."""
    return HTTPException(
        status_code=status_code,
        detail={"error_code": error_code, "message": message},
    )


def _normalize_api_pagination(limit, offset, default_limit=50, max_limit=1000):
    """Normalize API pagination and reject ambiguous negative values."""
    try:
        normalized_limit = int(limit if limit is not None else default_limit)
        normalized_offset = int(offset or 0)
    except (TypeError, ValueError):
        raise api_error(400, "invalid_pagination", "limit and offset must be integers")
    if normalized_limit < 1:
        raise api_error(400, "invalid_pagination", "limit must be >= 1")
    if normalized_offset < 0:
        raise api_error(400, "invalid_pagination", "offset must be >= 0")
    return min(normalized_limit, max_limit), normalized_offset


def _financial_runtime_adapter_enabled() -> bool:
    """Return the I3 migration flag without creating a V1/V2 router."""
    value = os.getenv("FINANCIAL_RUNTIME_ADAPTER_ENABLED", "true")
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _public_registry_document(row: dict) -> dict:
    """Return document lifecycle fields without file/content hashes."""
    keys = [
        "document_id",
        "filename",
        "chunk_count",
        "page_count",
        "version",
        "status",
        "parser_version",
        "splitter_version",
        "embedding_version",
        "created_at",
        "updated_at",
        "error_message",
    ]
    return {key: row.get(key) for key in keys}


def _json_field(value):
    """Decode trace JSON columns while tolerating legacy/plain values."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _public_feedback(row: dict) -> dict:
    """Return feedback data without exposing tenant_id."""
    keys = ["feedback_id", "trace_id", "rating", "comment", "created_at"]
    return {key: row.get(key) for key in keys}



def _validate_session_id(session_id: str) -> str:
    """Validate path/body session IDs before touching session storage."""
    if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
        raise api_error(400, "invalid_session_id", "session_id must be 1-128 characters")
    return session_id


def _safe_document_filename(filename: str, *, require_pdf: bool = True) -> str:
    """Normalize document filenames before they touch local paths, URLs, or indexes."""
    if not isinstance(filename, str):
        raise api_error(400, "invalid_filename", "Invalid filename")
    safe_filename = os.path.basename(filename).strip()
    if (
        not safe_filename
        or safe_filename in {".", ".."}
        or len(safe_filename) > 180
        or any(ord(ch) < 32 for ch in safe_filename)
        or "/" in safe_filename
        or "\\" in safe_filename
    ):
        raise api_error(400, "invalid_filename", "Invalid filename")
    if require_pdf and not safe_filename.lower().endswith(".pdf"):
        raise api_error(400, "invalid_file_type", "Only PDF files are supported")
    return safe_filename


def _safe_upload_filename(filename: str) -> str:
    """Normalize uploaded PDF filenames before they touch local paths or indexes."""
    return _safe_document_filename(filename, require_pdf=True)


def _assistant_session_metadata(result=None, sources=None, trace_id=None, context_sufficient=None,
                                confidence=None, intent=None, intent_confidence=None):
    """Build UI-facing metadata for persisted assistant session messages."""
    result = result or {}
    return {
        "sources": sources if sources is not None else result.get("sources", []),
        "diagnostics": {
            "traceId": trace_id if trace_id is not None else result.get("trace_id"),
            "contextSufficient": (
                context_sufficient if context_sufficient is not None else result.get("context_sufficient")
            ),
            "retrievalConfidence": confidence if confidence is not None else result.get("confidence"),
            "intent": intent if intent is not None else result.get("intent"),
            "intentConfidence": intent_confidence if intent_confidence is not None else result.get("intent_confidence"),
        },
    }


def _resolve_query_document_names_for_user(user_id, requested_doc_names):
    """Return ready document names for query, rejecting stale/unready filters."""
    ready_names = [
        row.get("filename")
        for row in document_registry.list_documents(user_id)
        if row.get("filename")
    ]
    fallback_names = []
    if not ready_names:
        fallback_names = [
            row.get("name")
            for row in list_all_documents(user_id)
            if row.get("name")
        ]

    resolved, invalid = resolve_query_document_names(
        requested_doc_names,
        ready_names,
        fallback_names,
    )
    if invalid:
        raise api_error(
            400,
            "documents_not_ready",
            "Documents are not ready or not found: %s" % ", ".join(invalid),
        )
    return resolved


def _public_trace(row: dict) -> dict:
    """Return tenant-scoped trace data without exposing sensitive content.

    Phase 4 hotfix: ``final_context``, ``answer``, and ``error_message``
    are intentionally excluded from the public trace payload. Only
    availability flags, lengths, and hashes are exposed from diagnostics.
    """
    keys = [
        "trace_id",
        "query_original",
        "query_rewritten",
        "intent",
        "model_name",
        "prompt_version",
        "index_version",
        "latency_ms",
        "created_at",
    ]
    trace = {key: row.get(key) for key in keys}
    trace["filter_conditions"] = _json_field(row.get("filter_conditions"))
    trace["candidates"] = _json_field(row.get("candidates_json"))
    trace["sources"] = _json_field(row.get("sources_json")) or []
    diagnostics = _json_field(row.get("diagnostics_json")) or {}
    trace["diagnostics"] = diagnostics
    # Expose only safe content-availability diagnostics (no raw text).
    trace["context_available"] = diagnostics.get("context_length", 0) > 0
    trace["answer_available"] = diagnostics.get("answer_length", 0) > 0
    trace["context_length"] = diagnostics.get("context_length")
    trace["context_sha256"] = diagnostics.get("context_sha256")
    trace["answer_length"] = diagnostics.get("answer_length")
    trace["answer_sha256"] = diagnostics.get("answer_sha256")
    trace["error_code"] = diagnostics.get("error_code") or (
        diagnostics.get("calculation", {}) or {}
    ).get("error_code")
    return trace

def _tenant_ops_summary(user_id: int) -> dict:
    """Return compact tenant-scoped RAG operations diagnostics without content."""
    registry_summary = document_registry.status_summary(user_id)
    session_summary = session_manager.storage_summary(user_id)
    trace_summary = get_rag_engine().trace_logger.summary_for_tenant(user_id)
    feedback_summary = feedback_store.summary_for_tenant(user_id)
    return {
        "documents": registry_summary,
        "sessions": session_summary,
        "traces": trace_summary,
        "feedback": feedback_summary,
        "generated_at": time.time(),
    }

def _replay_cases_payload_from_traces(rows: list[dict]) -> dict:
    """Convert trace rows to replay cases while tolerating invalid legacy traces."""
    cases = []
    skipped = []
    for row in rows:
        trace_id = row.get("trace_id")
        try:
            cases.append(trace_to_replay_case(row).to_dict())
        except ValueError as exc:
            skipped.append({"trace_id": trace_id, "reason": str(exc)})
    return {
        "cases": cases,
        "total_returned": len(rows),
        "total_cases": len(cases),
        "skipped_traces": skipped,
    }


def _replay_cases_payload_from_feedback(feedback_rows: list[dict], trace_lookup) -> dict:
    """Convert feedback rows to replay cases while skipping missing/legacy traces."""
    cases = []
    skipped = []
    for feedback in feedback_rows:
        trace_id = feedback.get("trace_id")
        trace = trace_lookup(trace_id)
        if trace is None:
            skipped.append({"trace_id": trace_id, "reason": "trace_not_found"})
            continue
        try:
            cases.append(feedback_to_replay_case(feedback, trace).to_dict())
        except ValueError as exc:
            skipped.append({"trace_id": trace_id, "reason": str(exc)})
    return {
        "cases": cases,
        "total_returned": len(feedback_rows),
        "total_cases": len(cases),
        "skipped_feedback": skipped,
    }

def _eval_report_from_payload(cases_payload: list[dict], predictions_payload: list[dict]) -> dict:
    """Build an evaluation report from in-memory JSON payloads."""
    try:
        report = evaluate_payload(cases_payload, predictions_payload)
    except ValueError as exc:
        raise api_error(400, "invalid_eval_payload", str(exc)) from exc
    report["generated_at"] = time.time()
    return report


def _eval_comparison_from_payload(baseline: dict, candidate: dict, tolerance: float) -> dict:
    """Compare two in-memory evaluation reports."""
    try:
        comparison = compare_reports(baseline, candidate, regression_tolerance=tolerance)
    except ValueError as exc:
        raise api_error(400, "invalid_eval_report", str(exc)) from exc
    comparison["generated_at"] = time.time()
    return comparison


######################### API Endpoints #########################

# <---------------------- GET requests ---------------------->
@app.get("/")
async def root():
    """
    健康检查端点。
    用于验证服务是否正常运行。

    Returns:
        dict: 包含服务状态、名称和版本号的字典。
    """
    return {
        "status": "healthy",
        "service": "FinQuery Multi-Document API",
        "version": "2.0.0"
    }

@app.get("/healthz")
async def healthz():
    """Lightweight liveness probe."""
    return {
        "status": "healthy",
        "service": "FinQuery Multi-Document API",
        "version": app.version,
    }

@app.get("/readyz")
async def readyz():
    """Readiness probe with non-secret RAG dependency diagnostics."""
    snapshot = collect_health_snapshot(
        document_registry=document_registry,
        session_manager=session_manager,
        feedback_store=feedback_store,
    )
    status_code = 200 if snapshot["ready"] else 503
    return JSONResponse(status_code=status_code, content=snapshot)
@app.get("/ops/summary")
async def get_ops_summary(current_user: User = Depends(get_current_user)):
    """Return current user's compact RAG operations summary without document/query content."""
    return _tenant_ops_summary(current_user.id)

@app.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    获取当前登录用户的信息。
    通过 JWT 依赖注入直接获取 User 对象，无需额外查询数据库。

    Args:
        current_user (User): 当前登录用户，通过 JWT 依赖注入获取。

    Returns:
        dict: 包含用户邮箱和账户创建时间的字典。
    """
    return {
        "email": current_user.email,
        "created_at": current_user.created_at
    }

@app.get("/documents", response_model=DocumentsListResponse)
async def list_documents(current_user: User = Depends(get_current_user)):
    """
    列出当前用户上传的所有文档。
    如果向量数据库目录不存在，则返回空列表。

    Args:
        current_user (User): 当前登录用户，通过 JWT 依赖注入获取。

    Returns:
        DocumentsListResponse: 包含文档信息列表和文档总数的响应模型。
    """
    if not os.path.exists("./chroma_db"):
        return DocumentsListResponse(documents=[], total_documents=0)

    docs = list_all_documents(current_user.id)

    return DocumentsListResponse(
        documents=[DocumentInfo(**doc) for doc in docs],
        total_documents=len(docs)
    )

@app.get("/document-registry")
async def list_document_registry(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
):
    """List current user's document lifecycle registry rows."""
    if status is not None and status not in VALID_TRANSITIONS:
        raise api_error(400, "invalid_document_status", f"Invalid document status: {status}")
    normalized_limit, normalized_offset = _normalize_api_pagination(limit, offset, default_limit=50)

    rows = document_registry.list_all(
        current_user.id,
        status=status,
        limit=normalized_limit,
        offset=normalized_offset,
    )
    total_documents = document_registry.count_all(current_user.id, status=status)
    return {
        "documents": [_public_registry_document(row) for row in rows],
        "total_returned": len(rows),
        "total_documents": total_documents,
        "has_more": normalized_offset + len(rows) < total_documents,
        "limit": normalized_limit,
        "offset": normalized_offset,
        "status_counts": document_registry.status_counts(current_user.id),
        "status_summary": document_registry.status_summary(current_user.id),
    }


@app.get("/document-registry/quality")
async def document_registry_quality(current_user: User = Depends(get_current_user)):
    """Return non-content parsing/index quality signals for ready documents."""
    engine = get_rag_engine()
    quality_by_filename = {
        item["filename"]: item
        for item in engine.bm25_retriever.document_quality_report(current_user.id)
    }
    rows = document_registry.list_all(current_user.id, status="ready", limit=1000)
    latest_by_filename = {}
    for row in rows:
        filename = row.get("filename")
        if not filename:
            continue
        if int(row.get("version") or 0) >= int(latest_by_filename.get(filename, {}).get("version") or 0):
            latest_by_filename[filename] = row

    documents = []
    for filename, row in sorted(latest_by_filename.items()):
        quality = quality_by_filename.get(filename)
        warnings = list((quality or {}).get("warnings", []))
        if quality is None:
            warnings.append("missing_sparse_index")
        page_count = int(row.get("page_count") or 0)
        indexed_pages = int((quality or {}).get("indexed_page_count") or 0)
        coverage = (indexed_pages / page_count) if page_count > 0 else None
        if page_count > 0 and indexed_pages > page_count:
            warnings.append("indexed_page_count_exceeds_registry")
        documents.append({
            **_public_registry_document(row),
            "quality": {
                **(quality or {
                    "chunk_count": 0,
                    "text_chunk_count": 0,
                    "table_chunk_count": 0,
                    "missing_page_metadata": 0,
                    "indexed_page_count": 0,
                    "max_indexed_page": 0,
                    "page_1_chunk_count": 0,
                    "page_1_share": 0.0,
                }),
                "registry_page_count": page_count,
                "indexed_page_coverage": round(coverage, 4) if coverage is not None else None,
                "warnings": sorted(set(warnings)),
            },
        })
    return {
        "documents": documents,
        "document_count": len(documents),
        "warning_document_count": sum(bool(item["quality"]["warnings"]) for item in documents),
    }

@app.get("/traces")
async def list_query_traces(
    limit: int = 20,
    offset: int = 0,
    error_only: bool = False,
    created_after: float | None = None,
    created_before: float | None = None,
    current_user: User = Depends(get_current_user),
):
    """List current user's recent query traces for troubleshooting/replay."""
    normalized_limit, normalized_offset = _normalize_api_pagination(limit, offset, default_limit=20)
    if created_after is not None and created_before is not None and created_after > created_before:
        raise api_error(400, "invalid_time_range", "created_after must be <= created_before")
    rows = get_rag_engine().trace_logger.query_traces(
        tenant_id=current_user.id,
        limit=normalized_limit,
        offset=normalized_offset,
        created_after=created_after,
        created_before=created_before,
        error_only=error_only,
    )
    total_traces = get_rag_engine().trace_logger.count_traces(
        tenant_id=current_user.id,
        created_after=created_after,
        created_before=created_before,
        error_only=error_only,
    )
    return {
        "traces": [_public_trace(row) for row in rows],
        "total_returned": len(rows),
        "total_traces": total_traces,
        "has_more": normalized_offset + len(rows) < total_traces,
        "limit": normalized_limit,
        "offset": normalized_offset,
    }


@app.get("/traces/{trace_id}")
async def get_query_trace(trace_id: str, current_user: User = Depends(get_current_user)):
    """Return one query trace when it belongs to the current user."""
    if not trace_id or len(trace_id) > 128:
        raise api_error(400, "invalid_trace_id", "Invalid trace_id")

    row = get_rag_engine().trace_logger.get_trace_for_tenant(current_user.id, trace_id)
    if row is None:
        raise api_error(404, "trace_not_found", "Trace not found")
    return {"trace": _public_trace(row)}


@app.get("/replay/traces")
async def export_trace_replay_cases_api(
    limit: int = 100,
    offset: int = 0,
    error_only: bool = False,
    created_after: float | None = None,
    created_before: float | None = None,
    current_user: User = Depends(get_current_user),
):
    """Return replay cases generated from current user's query traces without server-side file writes."""
    normalized_limit, normalized_offset = _normalize_api_pagination(limit, offset, default_limit=100)
    if created_after is not None and created_before is not None and created_after > created_before:
        raise api_error(400, "invalid_time_range", "created_after must be <= created_before")
    logger = get_rag_engine().trace_logger
    rows = logger.query_traces(
        tenant_id=current_user.id,
        limit=normalized_limit,
        offset=normalized_offset,
        created_after=created_after,
        created_before=created_before,
        error_only=error_only,
    )
    total_traces = logger.count_traces(
        tenant_id=current_user.id,
        created_after=created_after,
        created_before=created_before,
        error_only=error_only,
    )
    payload = _replay_cases_payload_from_traces(rows)
    payload.update({
        "total_traces": total_traces,
        "has_more": normalized_offset + len(rows) < total_traces,
        "limit": normalized_limit,
        "offset": normalized_offset,
    })
    return payload


@app.get("/replay/feedback")
async def export_feedback_replay_cases_api(
    limit: int = 100,
    offset: int = 0,
    rating: str | None = "down",
    current_user: User = Depends(get_current_user),
):
    """Return replay cases generated from feedback-linked traces for the current user."""
    if rating is not None and rating not in FeedbackStore.VALID_RATINGS:
        raise api_error(400, "invalid_rating", "Invalid rating")
    normalized_limit, normalized_offset = _normalize_api_pagination(limit, offset, default_limit=100)
    rows = feedback_store.list_for_tenant(
        current_user.id,
        limit=normalized_limit,
        offset=normalized_offset,
        rating=rating,
    )
    total_feedback = feedback_store.count_for_tenant(current_user.id, rating=rating)
    logger = get_rag_engine().trace_logger
    payload = _replay_cases_payload_from_feedback(
        rows,
        lambda trace_id: logger.get_trace_for_tenant(current_user.id, trace_id),
    )
    payload.update({
        "total_feedback": total_feedback,
        "has_more": normalized_offset + len(rows) < total_feedback,
        "limit": normalized_limit,
        "offset": normalized_offset,
        "rating": rating,
    })
    return payload


@app.post("/eval/score")
async def score_eval_report(request: EvalScoreRequest, current_user: User = Depends(get_current_user)):
    """Score offline evaluation predictions for an authenticated user without server-side file writes."""
    _ = current_user
    return _eval_report_from_payload(request.cases, request.predictions)


@app.post("/eval/compare")
async def compare_eval_reports(request: EvalCompareRequest, current_user: User = Depends(get_current_user)):
    """Compare two evaluation reports as a lightweight authenticated quality gate."""
    _ = current_user
    return _eval_comparison_from_payload(
        request.baseline,
        request.candidate,
        request.regression_tolerance,
    )

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_answer_feedback(request: FeedbackRequest, current_user: User = Depends(get_current_user)):
    """Store current user's answer feedback after validating trace ownership."""
    trace = get_rag_engine().trace_logger.get_trace_for_tenant(current_user.id, request.trace_id)
    if trace is None:
        raise api_error(404, "trace_not_found", "Trace not found")

    row = feedback_store.submit(
        tenant_id=current_user.id,
        trace_id=request.trace_id,
        rating=request.rating,
        comment=request.comment,
    )
    if row is None:
        raise api_error(400, "invalid_feedback", "Invalid feedback")
    return _public_feedback(row)


@app.get("/feedback")
async def list_answer_feedback(
    limit: int = 50,
    offset: int = 0,
    rating: str | None = None,
    current_user: User = Depends(get_current_user),
):
    """List current user's answer feedback for review/export."""
    if rating is not None and rating not in FeedbackStore.VALID_RATINGS:
        raise api_error(400, "invalid_rating", "Invalid rating")
    normalized_limit, normalized_offset = _normalize_api_pagination(limit, offset, default_limit=50)
    rows = feedback_store.list_for_tenant(current_user.id, limit=normalized_limit, offset=normalized_offset, rating=rating)
    total_feedback = feedback_store.count_for_tenant(current_user.id, rating=rating)
    return {
        "feedback": [_public_feedback(row) for row in rows],
        "total_returned": len(rows),
        "total_feedback": total_feedback,
        "has_more": normalized_offset + len(rows) < total_feedback,
        "limit": normalized_limit,
        "offset": normalized_offset,
    }


@app.get("/documents/{doc_name}")
async def get_document_stats(doc_name: str, current_user: User = Depends(get_current_user)):
    """
    获取特定文档的统计信息。
    根据文档名称和用户 ID 查询对应向量库集合的统计信息。

    Args:
        doc_name (str): 文档名称。
        current_user (User): 当前登录用户，通过 JWT 依赖注入获取。

    Returns:
        dict: 包含文档统计信息的字典（如块数量等）。

    Raises:
        HTTPException: 如果指定文档不存在，抛出 404 异常。
    """
    stats = get_collection_stats(doc_name, current_user.id)

    if not stats["exists"]:
        raise api_error(404, "document_not_found", f"Document '{doc_name}' not found")

    return stats


# <---------------------- POST requests ---------------------->
@app.post("/register", response_model=Token)
async def register(user: UserRegister, db: Session = Depends(get_db)):
    """
    注册新用户。
    检查邮箱是否已被注册，若未注册则对密码进行哈希处理并存入数据库，随后生成并返回 JWT。

    Args:
        user (UserRegister): 用户注册请求体，包含邮箱和密码。
        db (Session): SQLAlchemy 数据库会话，通过依赖注入获取。

    Returns:
        dict: 包含访问令牌、令牌类型和用户邮箱的字典。

    Raises:
        HTTPException: 如果邮箱已被注册，抛出 400 异常。
    """
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(400, "Email already registered")

    hashed_password = get_password_hash(user.password)
    new_user = User(
        email=user.email,
        hashed_password=hashed_password,
        created_at=datetime.utcnow()
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(
        data={"sub": str(new_user.id)},
        expires_delta=timedelta(minutes=30)
    )

    print(f"✓ New user registered: {new_user.email} (ID: {new_user.id})")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": new_user.email
    }

@app.post("/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    """
    用户登录。
    验证邮箱和密码，如果验证成功则生成并返回 JWT。

    Args:
        user (UserLogin): 用户登录请求体，包含邮箱和密码。
        db (Session): SQLAlchemy 数据库会话，通过依赖注入获取。

    Returns:
        dict: 包含访问令牌、令牌类型和用户邮箱的字典。

    Raises:
        HTTPException: 如果邮箱不存在或密码错误，抛出 401 异常。
    """
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user:
        raise HTTPException(401, "Invalid email or password")

    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(401, "Invalid email or password")

    access_token = create_access_token(
        data={"sub": str(db_user.id)},
        expires_delta=timedelta(minutes=30)
    )

    print(f"✓ User logged in: {db_user.email} (ID: {db_user.id})")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": db_user.email
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """
    上传并处理 PDF 文档。
    仅支持 PDF 格式。文件会被临时保存，提取文本块后存入当前用户专属的向量集合中，最后删除临时文件并同步稀疏索引。

    Args:
        file (UploadFile): 上传的文件对象，默认通过表单数据获取。
        current_user (User): 当前登录用户，通过 JWT 依赖注入获取。

    Returns:
        UploadResponse: 包含文件名、处理消息、页数及向量化结果的响应模型。

    Raises:
        HTTPException: 如果文件非 PDF 格式抛出 400 异常；如果 PDF 未提取到内容抛出 400 异常；处理过程中发生其他错误抛出 500 异常。
    """
    # Validate and normalize file name before using it in temp paths or indexes.
    safe_filename = _safe_upload_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, safe_filename)

    # Save file temporarily and process it
    # 临时保存文件并进行处理
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Phase 1: compute file hash and check for duplicate
        with open(temp_path, "rb") as f:
            file_bytes = f.read()
        fh = DocumentRegistry.file_hash(file_bytes)
        parser_version, splitter_version = get_ingest_lineage(temp_path)
        embedding_version = get_embedding_model_name()
        existing = document_registry.find_by_file_hash(
            current_user.id,
            fh,
            parser_version=parser_version,
            splitter_version=splitter_version,
            embedding_version=embedding_version,
        )
        if existing:
            os.remove(temp_path)
            os.rmdir(temp_dir)
            return UploadResponse(
                filename=safe_filename,
                message="Duplicate file skipped (already indexed)",
                pages=existing["page_count"],
                collection_name="",
                total_docs=existing["chunk_count"],
            )

        # Register document in lifecycle registry
        doc_id = uuid.uuid4().hex
        document_registry.register(
            doc_id, current_user.id, safe_filename, fh, status="parsing",
            parser_version=parser_version,
            splitter_version=splitter_version,
            embedding_version=embedding_version,
        )

        # process pdf
        chunks, no_of_pages = process_pdf(temp_path, user_id=current_user.id)

        if not chunks:
            document_registry.mark_failed(doc_id, "No content extracted from PDF")
            raise api_error(400, "empty_document", "No content extracted from PDF")

        # Phase 1: content hash dedup
        ch = DocumentRegistry.content_hash(chunks)
        content_existing = document_registry.find_by_content_hash(
            current_user.id,
            ch,
            parser_version=parser_version,
            splitter_version=splitter_version,
            embedding_version=embedding_version,
        )
        if content_existing and content_existing["document_id"] != doc_id:
            document_registry.mark_failed(doc_id, "Duplicate content (same as %s)" % content_existing["filename"])
            os.remove(temp_path)
            os.rmdir(temp_dir)
            return UploadResponse(
                filename=safe_filename,
                message="Duplicate content skipped (matches %s)" % content_existing["filename"],
                pages=content_existing["page_count"],
                collection_name="",
                total_docs=content_existing["chunk_count"],
            )

        document_registry.mark_indexing(doc_id)

        # add to vector store
        result = add_documents(chunks, safe_filename, current_user.id, no_of_pages)

        # sync to BM25 (rollback dense on failure)
        try:
            engine = get_rag_engine()
            engine.bm25_retriever.add_chunks(chunks, current_user.id)
        except Exception as bm25_err:
            # Rollback ChromaDB dense data for this doc
            delete_document_collection(safe_filename, current_user.id)
            document_registry.mark_failed(doc_id, f"BM25 write failed, dense rolled back: {bm25_err}")
            raise api_error(500, "indexing_error", f"Indexing error: {bm25_err}")

        # Mark ready in registry
        document_registry.mark_ready(doc_id, len(chunks), ch, page_count=no_of_pages)

        os.remove(temp_path)
        os.rmdir(temp_dir)

        print("\u2713 Document uploaded: %s (user: %s, doc_id: %s)" % (safe_filename, current_user.id, doc_id))

        return UploadResponse(
            filename=safe_filename,
            message="Successfully processed %s" % safe_filename,
            pages=no_of_pages,
            **result,
        )

    except HTTPException:
        # Mark failed in registry if doc_id exists
        try:
            document_registry.mark_failed(doc_id, "Client error")
        except Exception:
            pass
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except Exception as e:
        try:
            document_registry.mark_failed(doc_id, str(e))
        except Exception:
            pass
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise api_error(500, "processing_error", "Processing error: %s" % str(e))

@app.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
async def query_documents(request: QueryRequest, current_user: User = Depends(get_current_user)):
    """
    对一个或多个文档进行提问。
    支持会话记忆：传入 session_id 将自动加载历史消息用于问题改写。
    历史回答不会作为金融事实进入检索上下文。
    """
    try:
        engine = get_rag_engine()

        # Phase 4: Load conversation history for query rewriting
        session_id = _validate_session_id(request.session_id) if request.session_id else None
        conversation_history = None
        if session_id:
            conversation_history = session_manager.get_recent_messages(
                session_id, current_user.id
            )
        memory_profile = memory_store.get_profile(current_user.id)
        # UserMemoryStore returns a dict. Keep test doubles or malformed
        # profiles from changing the legacy engine's existing empty-profile
        # behavior while preserving real production profiles unchanged.
        if not isinstance(memory_profile, dict):
            memory_profile = None

        resolved_doc_names = _resolve_query_document_names_for_user(
            current_user.id,
            request.document_names,
        )

        # I3 routes only /query through the contract-compatible V1 adapter.
        # The migration flag keeps an exact direct-call rollback path; both
        # branches use this same cached RAGEngine and unchanged V1 inputs.
        request_id = uuid.uuid4().hex
        runtime_request = FinancialQueryRequest(
            request_id=request_id,
            user_id=str(current_user.id),
            session_id=session_id or f"__stateless__:{request_id}",
            original_query=request.question,
            standalone_query=request.question,
            query_as_resolved=False,
            conversation_metadata={},
            request_metadata={
                "document_names": resolved_doc_names,
                "n_results": request.n_results,
                "conversation_history": conversation_history,
                "memory_profile": memory_profile,
            },
        )
        if _financial_runtime_adapter_enabled():
            runtime = LegacyFinancialRuntimeAdapter(engine)
            execution_service = QueryExecutionService(runtime)
            runtime_result = await execution_service.execute(runtime_request)
            result = to_legacy_query_dict(runtime_result)
        else:
            # Keep the original direct call as a deterministic migration
            # fallback and as the I3 parity reference path.
            result = await engine.query(
                question=request.question,
                doc_names=resolved_doc_names,
                n_results=request.n_results,
                user_id=current_user.id,
                conversation_history=conversation_history,
                memory_profile=memory_profile,
            )

        # Phase 4: Save messages to session
        rewritten = result.get("rewritten_question")
        if session_id:
            session_manager.add_message(session_id, current_user.id, "user", request.question)
            session_manager.add_message(
                session_id,
                current_user.id,
                "assistant",
                result["answer"],
                metadata=_assistant_session_metadata(result=result),
            )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            question=request.question,
            searched_docs=result["searched_docs"],
            session_id=session_id,
            rewritten_question=rewritten,
            confidence=result.get("confidence"),
            context_sufficient=result.get("context_sufficient"),
            intent=result.get("intent"),
            intent_confidence=result.get("intent_confidence"),
            trace_id=result.get("trace_id"),
            retrieved_chunks=result.get("retrieved_chunks", []),
            retrieval_debug=result.get("retrieval_debug", {}),
            calculations=result.get("calculations", []),
            answerability=(
                AnswerabilityResponse(**result["answerability"])
                if result.get("answerability") else None
            ),
            validation=(
                ValidationResponse(**result["validation"])
                if result.get("validation") else None
            ),
            repair=(
                RepairResponse(**result["repair"])
                if result.get("repair") else None
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise api_error(500, "query_error", f"Query error: {str(e)}")



@app.post("/query/stream")
async def query_documents_stream(request: QueryRequest, current_user: User = Depends(get_current_user)):
    """
    使用服务器发送事件（SSE）流式传输查询响应。

    Phase 3 热修复：统一调用 ``engine.query()``，使计算链路在生产
    流式接口中生效。当 ``calculations`` 非空时（EXECUTED/BLOCKED/
    FAILED），确定性答案作为单个 token 事件输出，不调用 LLM 流式
    生成；非计算请求的答案同样通过 ``engine.query()`` 获取并以
    SSE 协议输出。

    Args:
        request (QueryRequest): 查询请求体。
        current_user (User): 当前登录用户。
    """
    resolved_doc_names = _resolve_query_document_names_for_user(
        current_user.id,
        request.document_names,
    )

    async def generate():
        try:
            engine = get_rag_engine()
            session_id = _validate_session_id(request.session_id) if request.session_id else None

            # Load conversation history for query rewriting (shared with /query).
            conversation_history = None
            if session_id:
                conversation_history = session_manager.get_recent_messages(
                    session_id, current_user.id
                )
            memory_profile = memory_store.get_profile(current_user.id)

            # Unified: call engine.query() which runs the full orchestrator
            # including the Phase 3 calculation pipeline. This ensures the
            # streaming path exercises the same calculation logic as /query.
            result = await engine.query(
                question=request.question,
                doc_names=resolved_doc_names,
                n_results=request.n_results,
                user_id=current_user.id,
                conversation_history=conversation_history,
                memory_profile=memory_profile,
            )

            answer = result["answer"]
            sources = result.get("sources", [])
            confidence = result.get("confidence")
            context_sufficient = result.get("context_sufficient")
            intent = result.get("intent")
            intent_confidence = result.get("intent_confidence")
            trace_id = result.get("trace_id")
            calculations = result.get("calculations", [])
            # Phase 4: forward validation verdicts to the SSE done event.
            # Only included when the validation pipeline produced them, so
            # the legacy event shape is preserved when validation is disabled.
            answerability_data = result.get("answerability")
            validation_data = result.get("validation")
            repair_data = result.get("repair")

            # Save session messages (compatible with /query behavior).
            if session_id:
                session_manager.add_message(session_id, current_user.id, "user", request.question)
                session_manager.add_message(
                    session_id,
                    current_user.id,
                    "assistant",
                    answer,
                    metadata=_assistant_session_metadata(
                        result=result,
                        sources=sources,
                        trace_id=trace_id,
                        context_sufficient=context_sufficient,
                        confidence=confidence,
                        intent=intent,
                        intent_confidence=intent_confidence,
                    ),
                )

            # Emit the answer as a token event. For calculation results
            # (EXECUTED/BLOCKED/FAILED) the LLM stream is never invoked —
            # the answer is deterministic. For non-calculation results the
            # answer was already generated by engine.query() and is emitted
            # as a single token event.
            yield f"data: {json.dumps({'type': 'token', 'content': answer})}\n\n"

            # Emit the done event with all fields including calculations.
            done_kwargs = dict(
                sources=sources,
                confidence=confidence,
                context_sufficient=context_sufficient,
                intent=intent,
                intent_confidence=intent_confidence,
                trace_id=trace_id,
                calculations=calculations,
            )
            if answerability_data is not None:
                done_kwargs["answerability"] = answerability_data
            if validation_data is not None:
                done_kwargs["validation"] = validation_data
            if repair_data is not None:
                done_kwargs["repair"] = repair_data
            yield make_stream_done_event(**done_kwargs)

        except Exception as exc:
            error_message = "Streaming query failed. Please retry."
            try:
                engine = get_rag_engine()
                trace_payload = {
                    "tenant_id": current_user.id,
                    "query_original": request.question,
                    "model_name": llm_model_name,
                    "answer": None,
                    "error_message": None,
                    "diagnostics": {
                        "stream_error": True,
                        "error_code": "STREAM_INTERNAL_ERROR",
                        "exception_type": type(exc).__name__,
                    },
                    "latency_ms": 0,
                }
                trace_id = safe_log_query_trace(engine, trace_payload)
            except Exception:
                trace_id = None
            yield make_stream_error_event("stream_error", error_message, retryable=True, trace_id=trace_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# <---------------------- Session endpoints (Phase 4) ---------------------->

@app.get("/sessions")
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
):
    """List current user's conversation sessions without message content."""
    normalized_limit, normalized_offset = _normalize_api_pagination(limit, offset, default_limit=50)
    sessions = session_manager.list_sessions(
        current_user.id,
        limit=normalized_limit,
        offset=normalized_offset,
    )
    total_sessions = session_manager.count_sessions(current_user.id)
    return {
        "sessions": sessions,
        "total_returned": len(sessions),
        "total_sessions": total_sessions,
        "has_more": normalized_offset + len(sessions) < total_sessions,
        "limit": normalized_limit,
        "offset": normalized_offset,
        "summary": session_manager.storage_summary(current_user.id),
    }

@app.post("/sessions/clear")
async def clear_session(request: QueryRequest, current_user: User = Depends(get_current_user)):
    """
    清除指定会话的历史消息。
    """
    if not request.session_id:
        raise api_error(400, "session_id_required", "session_id is required")
    session_id = _validate_session_id(request.session_id)
    cleared = session_manager.clear_session(session_id, current_user.id)
    return {"message": "Session cleared", "session_id": session_id, "cleared": cleared}

@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str, current_user: User = Depends(get_current_user)):
    """
    获取指定会话的历史消息（用于前端恢复对话）。
    """
    session_id = _validate_session_id(session_id)
    messages = session_manager.get_recent_messages(session_id, current_user.id)
    count = session_manager.get_session_count(session_id, current_user.id)
    return {"session_id": session_id, "messages": messages, "total_messages": count}

@app.delete("/sessions")
async def clear_all_sessions(current_user: User = Depends(get_current_user)):
    """Clear all conversation sessions for the current user."""
    deleted_messages = session_manager.clear_all_for_user(current_user.id)
    return {
        "message": "All sessions cleared",
        "deleted_messages": deleted_messages,
        "summary": session_manager.storage_summary(current_user.id),
    }


@app.get("/memory/profile", response_model=MemoryProfileResponse)
async def get_memory_profile(current_user: User = Depends(get_current_user)):
    """Return the current user's editable preference memory profile."""
    profile = memory_store.get_profile(current_user.id)
    updated_at = profile.pop("updated_at", None) if profile else None
    return {"profile": profile, "updated_at": updated_at}


@app.put("/memory/profile", response_model=MemoryProfileResponse)
async def update_memory_profile(request: MemoryProfileRequest, current_user: User = Depends(get_current_user)):
    """Patch the current user's preference memory profile."""
    payload = request.model_dump(exclude_none=True)
    profile = memory_store.upsert_profile(current_user.id, payload)
    updated_at = profile.pop("updated_at", None) if profile else None
    return {"profile": profile, "updated_at": updated_at}


@app.delete("/memory/profile")
async def clear_memory_profile(current_user: User = Depends(get_current_user)):
    """Clear the current user's editable preference memory profile."""
    cleared = memory_store.clear_profile(current_user.id)
    return {"message": "Memory profile cleared", "cleared": cleared}


# <---------------------- DELETE requests ---------------------->
@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str, current_user: User = Depends(get_current_user)):
    """
    删除特定文档及其对应的向量集合。
    同时会清理该文档在 BM25 索引中的数据。

    Args:
        doc_name (str): 要删除的文档名称。
        current_user (User): 当前登录用户，通过 JWT 依赖注入获取。

    Returns:
        dict: 包含成功删除消息的字典。

    Raises:
        HTTPException: 如果指定文档不存在，抛出 404 异常。
    """
    doc_name = _safe_document_filename(doc_name, require_pdf=False)

    # Failed uploads may have a registry row but no Chroma/BM25 chunks. Treat
    # deletion as successful when any backing store had something to remove.
    vector_deleted = delete_document_collection(doc_name, current_user.id)

    engine = get_rag_engine()
    try:
        engine.bm25_retriever.delete_doc(doc_name, current_user.id)
        bm25_deleted = True
        print(f"✓ Deleted {doc_name} from BM25 index (user: {current_user.id})")
    except Exception as e:
        bm25_deleted = False
        print(f"BM25 delete skipped/failed for {doc_name} (user: {current_user.id}): {e}")

    registry_deleted = document_registry.delete(current_user.id, doc_name)
    if registry_deleted:
        print(f"✓ Deleted {doc_name} from document registry (user: {current_user.id})")

    if not vector_deleted and not registry_deleted:
        raise api_error(404, "document_not_found", f"Document '{doc_name}' not found")

    return {
        "message": f"Document '{doc_name}' deleted successfully",
        "deleted": {
            "vectors": bool(vector_deleted),
            "bm25_cleanup_attempted": bool(bm25_deleted),
            "registry_rows": int(registry_deleted or 0),
        },
    }

@app.delete("/documents")
async def clear_all_documents(current_user: User = Depends(get_current_user)):
    """
    清除当前登录用户的所有文档（不再清除全局数据）。
    分别从向量库和 BM25 索引中删除该用户的全部数据。

    Returns:
        dict: 包含成功清除消息的字典。

    Raises:
        HTTPException: 如果清除过程中发生错误，抛出 500 异常。
    """
    errors = []
    # Delete current user's vectors from ChromaDB
    # Note: delete_document_collection returns False when no data exists (by design).
    # That is NOT an error for clear-all — clearing nothing is idempotent.
    try:
        delete_document_collection(None, current_user.id)
    except ValueError:
        pass  # user_id is always provided here via Depends(get_current_user)

    # Delete current user's BM25 index entries
    try:
        engine = get_rag_engine()
        engine.bm25_retriever.delete_all_for_user(current_user.id)
    except Exception as e:
        errors.append(f"BM25 deletion failed: {e}")

    # Sync document registry (clear orphaned entries)
    try:
        document_registry.delete_all_for_tenant(current_user.id)
    except Exception as e:
        errors.append(f"Registry sync failed: {e}")

    if errors:
        raise api_error(500, "partial_failure", f"Partial failure: {'; '.join(errors)}")

    return {"message": f"All documents cleared for user {current_user.id}"}



@app.post("/ops/retrieval-candidates")
async def retrieval_candidates(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
):
    """Return privacy-safe per-document retrieval stages for offline evaluation.

    This diagnostic route intentionally does not call generation, create a
    session, or mutate indexes. It is authenticated and intended for the
    NF37 candidate-pool exporter.
    """
    document_names = list(request.document_names or [])
    if not document_names:
        document_names = [
            row.get("name") or row.get("filename")
            for row in list_all_documents(user_id=current_user.id)
            if row.get("name") or row.get("filename")
        ]
    engine = get_rag_engine()
    per_document = []
    for document_name in document_names:
        engine.retrieve_single_document(
            document_name,
            request.question,
            user_id=current_user.id,
            n_results=max(5, request.n_results),
        )
        debug = dict(engine._last_retrieval_debug)
        per_document.append({
            "document_name": document_name,
            "candidate_stages": debug.get("candidate_stages", {}),
        })
    return {"question": request.question, "documents": per_document}
