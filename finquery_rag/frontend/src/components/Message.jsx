import React, { useMemo, useState } from 'react';
import { getQueryTrace, submitAnswerFeedback } from '../api';

const EMPTY_VALUE = '-';

const asNumber = (value) => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const formatPercent = (value) => {
  const numeric = asNumber(value);
  if (numeric === null) return null;
  return `${Math.round(numeric * 100)}%`;
};

const shortText = (value, limit = 220) => {
  if (!value) return EMPTY_VALUE;
  const text = String(value);
  return text.length > limit ? `${text.slice(0, limit)}...` : text;
};

const getSourceChunkId = (source) => source.chunk_id || source.doc_id || null;

const sourceKey = (source) => [
  getSourceChunkId(source) || source.filename || source.doc_name || source.source || 'source',
  source.page ?? '',
  source.type || '',
].join('::');

const BENCHMARK_COMPANY_NAMES = {
  aapl_fy2025: 'Apple 2025财年 10-K',
  msft_fy2025: 'Microsoft 2025财年 10-K',
  nvda_fy2025: 'NVIDIA 2025财年 10-K',
  tsla_fy2025: 'Tesla 2025财年 10-K',
  v_fy2025: 'Visa 2025财年 10-K',
  pfe_fy2024: 'Pfizer 2024财年 10-K',
  ko_fy2025: 'Coca-Cola 2025财年 10-K',
  jpm_fy2025: 'JPMorgan 2025财年 10-K',
};

const formatSourceLabel = (source) => {
  const rawName = source.filename || source.doc_name || source.source || 'source';
  const name = BENCHMARK_COMPANY_NAMES[rawName] || rawName;
  const page = source.page !== undefined && source.page !== null ? ` 第 ${source.page} 页` : '';
  const type = source.type && source.type !== 'text' ? ` [${source.type}]` : '';
  return `${name}${page}${type}`;
};

const formatChunkLabel = (chunkId) => {
  if (!chunkId) return null;
  return '底稿引证';
};

const formatTraceScope = (filterConditions) => {
  const docNames = filterConditions?.doc_names || [];
  if (!Array.isArray(docNames) || docNames.length === 0) {
    return 'All ready documents';
  }
  if (docNames.length <= 2) {
    return docNames.join(', ');
  }
  return `${docNames.length} documents`;
};

const formatTraceTopK = (filterConditions) => {
  const nResults = filterConditions?.n_results;
  return typeof nResults === 'number' ? String(nResults) : EMPTY_VALUE;
};

const formatTraceContext = (diagnostics) => {
  if (typeof diagnostics?.context_sufficient !== 'boolean') return EMPTY_VALUE;
  return diagnostics.context_sufficient ? 'sufficient' : 'weak';
};

const uniqueSources = (sources = []) => {
  const seen = new Set();
  return sources.filter((source) => {
    const key = sourceKey(source);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

const getReliability = (diagnostics, citationCount) => {
  if (!diagnostics) return null;
  if (diagnostics.streamError) {
    return { label: 'Stream error', className: 'error', help: diagnostics.streamError };
  }
  if (diagnostics.contextSufficient === false) {
    return { label: 'Insufficient context', className: 'warn', help: 'The answer was generated as a refusal or should be reviewed.' };
  }
  const confidence = asNumber(diagnostics.retrievalConfidence);
  if (confidence !== null && confidence < 0.35) {
    return { label: 'Low retrieval confidence', className: 'warn', help: 'Retrieved context may be weak for this answer.' };
  }
  if (citationCount === 0 && diagnostics.intent !== 'conversation') {
    return { label: 'No citations', className: 'warn', help: 'No source chunks were returned with this answer.' };
  }
  return { label: 'Grounded', className: 'ok', help: 'Context passed the reliability checks available to the UI.' };
};

const Message = ({ message, onSendMessage }) => {
  const isUser = message.role === 'user';
  const diagnostics = message.diagnostics;
  const allCitations = useMemo(() => uniqueSources(message.sources), [message.sources]);
  const renderedContent = useMemo(() => {
    if (!message.content) return '';
    return message.content.replace(/\[citation:v2:[a-f0-9]+\]/gi, ' [📌 经形式化审计底稿核验]');
  }, [message.content]);
  const citations = allCitations.slice(0, 5);
  const extraCitationCount = Math.max(0, allCitations.length - citations.length);
  const intentConfidence = diagnostics ? formatPercent(diagnostics.intentConfidence) : null;
  const retrievalConfidence = diagnostics ? formatPercent(diagnostics.retrievalConfidence) : null;
  const reliability = getReliability(diagnostics, allCitations.length);
  const [copiedTraceId, setCopiedTraceId] = useState(false);
  const [copiedChunkId, setCopiedChunkId] = useState(null);
  const [selectedEvidence, setSelectedEvidence] = useState(null);
  const [traceDetails, setTraceDetails] = useState(null);
  const [isTraceOpen, setIsTraceOpen] = useState(false);
  const [isTraceLoading, setIsTraceLoading] = useState(false);
  const [traceError, setTraceError] = useState(null);
  const [feedbackRating, setFeedbackRating] = useState(null);
  const [feedbackComment, setFeedbackComment] = useState('');
  const [isFeedbackCommentOpen, setIsFeedbackCommentOpen] = useState(false);
  const [isFeedbackSaving, setIsFeedbackSaving] = useState(false);
  const [feedbackError, setFeedbackError] = useState(null);

  const clarificationData = useMemo(() => {
    if (message.clarification) return message.clarification;
    if (message.diagnostics?.clarification) return message.diagnostics.clarification;
    return null;
  }, [message.clarification, message.diagnostics]);

  const disambiguationOptions = useMemo(() => {
    if (clarificationData?.structured_options && clarificationData.structured_options.length > 0) {
      return clarificationData.structured_options;
    }
    if (clarificationData?.options && clarificationData.options.length > 0) {
      return clarificationData.options.map((opt, idx) => {
        const mVal = opt.match(/数值\s+([^\s(]+|\([^)]+\))/);
        const mPage = opt.match(/第\s*(\d+)\s*页/);
        return {
          index: idx + 1,
          value: mVal ? mVal[1] : opt,
          page: mPage ? mPage[1] : null,
          rawText: opt,
        };
      });
    }
    if (
      message.content &&
      message.content.includes('会计科目') &&
      (message.content.includes('检索到多项') || message.content.includes('披露数据'))
    ) {
      const optionMatches = [
        ...message.content.matchAll(/(\d+)\.\s*金额\/数值:\s*([^\s（]+|\([^）]+\))（来自\s*([^）]+)）/g),
      ];
      if (optionMatches.length > 0) {
        return optionMatches.map((m) => {
          const pageMatch = m[3].match(/第\s*(\d+)\s*页/);
          return {
            index: Number(m[1]),
            value: m[2],
            page: pageMatch ? pageMatch[1] : null,
            rawText: m[0],
            context: m[3],
          };
        });
      }
    }
    return null;
  }, [clarificationData, message.content]);

  const handleCopyTraceId = async () => {
    if (!diagnostics?.traceId) return;

    try {
      await navigator.clipboard.writeText(diagnostics.traceId);
      setCopiedTraceId(true);
      window.setTimeout(() => setCopiedTraceId(false), 1600);
    } catch (error) {
      console.error('Failed to copy trace ID:', error);
    }
  };

  const handleCopyChunkId = async (chunkId) => {
    if (!chunkId) return;

    try {
      await navigator.clipboard.writeText(chunkId);
      setCopiedChunkId(chunkId);
      window.setTimeout(() => setCopiedChunkId(null), 1600);
    } catch (error) {
      console.error('Failed to copy chunk ID:', error);
    }
  };

  const handleToggleTraceDetails = async () => {
    if (!diagnostics?.traceId) return;
    if (traceDetails || traceError) {
      setIsTraceOpen(!isTraceOpen);
      return;
    }

    setIsTraceLoading(true);
    setTraceError(null);
    setIsTraceOpen(true);

    try {
      const data = await getQueryTrace(diagnostics.traceId);
      setTraceDetails(data.trace);
    } catch (error) {
      console.error('Failed to load trace details:', error);
      setTraceError('Trace details unavailable');
    } finally {
      setIsTraceLoading(false);
    }
  };

  const handleSubmitFeedback = async (rating, comment = null) => {
    if (!diagnostics?.traceId || isFeedbackSaving) return;

    setIsFeedbackSaving(true);
    setFeedbackError(null);

    try {
      await submitAnswerFeedback(diagnostics.traceId, rating, comment);
      setFeedbackRating(rating);
      if (rating === 'down') {
        setIsFeedbackCommentOpen(false);
      }
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      setFeedbackError('Feedback failed');
    } finally {
      setIsFeedbackSaving(false);
    }
  };

  const handleDownFeedbackClick = () => {
    if (feedbackRating === 'down') return;
    setIsFeedbackCommentOpen(true);
    setFeedbackError(null);
  };

  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-content">
        {!isUser && (
          <div className="message-sources">
            <span>FinQuery</span>
            {reliability && (
              <span className={`answer-state ${reliability.className}`} title={reliability.help}>
                {reliability.label}
              </span>
            )}
          </div>
        )}
        <div style={{ whiteSpace: 'pre-wrap' }}>{renderedContent}</div>
        {/* Feature 5: Interactive Disambiguation Cards */}
        {disambiguationOptions && disambiguationOptions.length > 0 && (
          <div className="disambiguation-card">
            <div className="disambiguation-header">
              <span className="disambiguation-icon">🔀</span>
              <div>
                <div className="disambiguation-title">报表重名科目消歧卡片</div>
                <div className="disambiguation-subtitle">
                  检测到多项不同报表位置或会计内涵的披露，请直接点击卡片快速确定：
                </div>
              </div>
            </div>
            <div className="disambiguation-options-grid">
              {disambiguationOptions.map((opt) => (
                <button
                  key={opt.index}
                  type="button"
                  className="disambiguation-option-btn"
                  onClick={() => onSendMessage && onSendMessage(opt.value || String(opt.index))}
                  title={`点击选择: ${opt.value}`}
                >
                  <span className="option-badge">选项 {opt.index}</span>
                  <span className="option-val">{opt.value}</span>
                  {(opt.page || opt.context) && (
                    <div className="option-meta">
                      {opt.page && <span className="option-page">第 {opt.page} 页</span>}
                      {opt.context && <span className="option-ctx" title={opt.context}>{opt.context}</span>}
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>
        )}
        {!isUser && citations.length > 0 && (
          <div className="message-citations" aria-label="Answer sources">
            <span className="citations-label">Sources</span>
            {citations.map((source) => {
              const chunkId = getSourceChunkId(source);
              const label = formatSourceLabel(source);
              return (
                <button
                  key={sourceKey(source)}
                  type="button"
                  className="citation-chip citation-button"
                  title="点击查看审计底稿证据详情与坐标溯源"
                  onClick={() => setSelectedEvidence(source)}
                >
                  <span>{label}</span>
                  <code>{copiedChunkId === chunkId ? '已复制' : '底稿引证 🔍'}</code>
                </button>
              );
            })}
            {extraCitationCount > 0 && (
              <span className="citation-chip muted">+{extraCitationCount} more</span>
            )}
            <button
              type="button"
              className="citation-chip citation-button evidence-inspect-all-btn"
              title="查看当前回答完整的形式化审计证据底稿与物理切片坐标"
              onClick={() => setSelectedEvidence(citations[0])}
            >
              <span>🔍 证据溯源看板</span>
            </button>
          </div>
        )}
        {!isUser && diagnostics && (
          <>
            <div className="message-diagnostics" aria-label="Answer diagnostics">
              {diagnostics.traceId && (
                <>
                  <button
                    type="button"
                    className="diagnostic-chip trace-chip"
                    title={`Copy full trace ID: ${diagnostics.traceId}`}
                    onClick={handleCopyTraceId}
                  >
                    {copiedTraceId ? 'Trace copied' : `Trace ${diagnostics.traceId.slice(0, 8)}`}
                  </button>
                  <button
                    type="button"
                    className="diagnostic-chip trace-chip"
                    onClick={handleToggleTraceDetails}
                  >
                    {isTraceOpen ? 'Hide details' : 'Details'}
                  </button>
                </>
              )}
              {typeof diagnostics.contextSufficient === 'boolean' && (
                <span className={`diagnostic-chip ${diagnostics.contextSufficient ? 'ok' : 'warn'}`}>
                  Context {diagnostics.contextSufficient ? 'sufficient' : 'weak'}
                </span>
              )}
              {retrievalConfidence && (
                <span className="diagnostic-chip">
                  Retrieval {retrievalConfidence}
                </span>
              )}
              {diagnostics.intent && (
                <span className="diagnostic-chip">
                  Intent {diagnostics.intent}{intentConfidence ? ` - ${intentConfidence}` : ''}
                </span>
              )}
              {diagnostics.streamError && (
                <span className="diagnostic-chip error">
                  Stream error
                </span>
              )}
            </div>
            {diagnostics.traceId && (
              <div className="answer-feedback" aria-label="Answer feedback">
                <span>Helpful?</span>
                <button
                  type="button"
                  className={`feedback-btn ${feedbackRating === 'up' ? 'selected' : ''}`}
                  onClick={() => handleSubmitFeedback('up')}
                  disabled={isFeedbackSaving}
                  title="Mark as helpful"
                >
                  Up
                </button>
                <button
                  type="button"
                  className={`feedback-btn ${feedbackRating === 'down' ? 'selected' : ''}`}
                  onClick={handleDownFeedbackClick}
                  disabled={isFeedbackSaving}
                  title="Mark as not helpful"
                >
                  Down
                </button>
                {feedbackRating && <span className="feedback-status">Saved</span>}
                {feedbackError && <span className="feedback-error">{feedbackError}</span>}
              </div>
            )}
            {isFeedbackCommentOpen && (
              <div className="feedback-comment-box">
                <textarea
                  value={feedbackComment}
                  onChange={(event) => setFeedbackComment(event.target.value)}
                  maxLength={2000}
                  placeholder="Optional: what was wrong or missing?"
                  disabled={isFeedbackSaving}
                />
                <div className="feedback-comment-actions">
                  <button
                    type="button"
                    className="feedback-submit-btn"
                    onClick={() => handleSubmitFeedback('down', feedbackComment.trim() || null)}
                    disabled={isFeedbackSaving}
                  >
                    Submit feedback
                  </button>
                  <button
                    type="button"
                    className="feedback-cancel-btn"
                    onClick={() => setIsFeedbackCommentOpen(false)}
                    disabled={isFeedbackSaving}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
            {isTraceOpen && (
              <div className="trace-details">
                {isTraceLoading && <div className="trace-muted">Loading trace details...</div>}
                {traceError && <div className="trace-error">{traceError}</div>}
                {traceDetails && (
                  <>
                    <div className="trace-row">
                      <span>Question</span>
                      <p>{shortText(traceDetails.query_original)}</p>
                    </div>
                    {traceDetails.query_rewritten && (
                      <div className="trace-row">
                        <span>Rewritten</span>
                        <p>{shortText(traceDetails.query_rewritten)}</p>
                      </div>
                    )}
                    <div className="trace-row trace-grid">
                      <div>
                        <span>Intent</span>
                        <p>{traceDetails.intent || EMPTY_VALUE}</p>
                      </div>
                      <div>
                        <span>Latency</span>
                        <p>{typeof traceDetails.latency_ms === 'number' ? `${Math.round(traceDetails.latency_ms)} ms` : EMPTY_VALUE}</p>
                      </div>
                      <div>
                        <span>Scope</span>
                        <p title={(traceDetails.filter_conditions?.doc_names || []).join(', ')}>
                          {formatTraceScope(traceDetails.filter_conditions)}
                        </p>
                      </div>
                      <div>
                        <span>Top-K</span>
                        <p>{formatTraceTopK(traceDetails.filter_conditions)}</p>
                      </div>
                      <div>
                        <span>Retrieval</span>
                        <p>{formatPercent(traceDetails.diagnostics?.confidence) || (traceDetails.diagnostics?.release_status ? `${traceDetails.diagnostics.release_status}` : EMPTY_VALUE)}</p>
                      </div>
                      <div>
                        <span>Context</span>
                        <p>{formatTraceContext(traceDetails.diagnostics) !== EMPTY_VALUE ? formatTraceContext(traceDetails.diagnostics) : (traceDetails.diagnostics?.reason_codes?.[0] || (traceDetails.diagnostics?.release_status ? 'Grounded' : EMPTY_VALUE))}</p>
                      </div>
                    </div>
                    {traceDetails.sources?.length > 0 && (
                      <div className="trace-row">
                        <span>Sources</span>
                        <div className="trace-source-list">
                          {uniqueSources(traceDetails.sources).map((source) => {
                            const chunkId = getSourceChunkId(source);
                            return (
                              <div key={sourceKey(source)} className="trace-source-item">
                                <span>{formatSourceLabel(source)}</span>
                                {chunkId && (
                                  <button
                                    type="button"
                                    title={`Copy chunk ID: ${chunkId}`}
                                    onClick={() => handleCopyChunkId(chunkId)}
                                  >
                                    {copiedChunkId === chunkId ? 'copied' : formatChunkLabel(chunkId)}
                                  </button>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    {traceDetails.error_message && (
                      <div className="trace-row trace-error">
                        <span>Error</span>
                        <p>{traceDetails.error_message}</p>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </>
        )}
        {/* Feature 4: Visual Evidence Inspector Modal */}
        {selectedEvidence && (
          <div className="evidence-modal-backdrop" onClick={() => setSelectedEvidence(null)}>
            <div className="evidence-modal" onClick={(e) => e.stopPropagation()}>
              <div className="evidence-modal-header">
                <div className="evidence-modal-title">
                  <span className="evidence-badge-icon">🔍</span>
                  <div>
                    <h3>审计底稿证据溯源看板</h3>
                    <p className="evidence-modal-subtitle">权威披露切片、图表定位与物理页面坐标</p>
                  </div>
                </div>
                <button
                  type="button"
                  className="evidence-modal-close"
                  onClick={() => setSelectedEvidence(null)}
                  title="关闭"
                >
                  ×
                </button>
              </div>

              <div className="evidence-modal-body">
                {/* Document & Page Identification */}
                <div className="evidence-section">
                  <div className="evidence-section-label">权威底稿归属</div>
                  <div className="evidence-doc-row">
                    <span className="evidence-doc-pill">
                      📄 {formatSourceLabel(selectedEvidence)}
                    </span>
                    {selectedEvidence.type && (
                      <span className="evidence-type-tag">
                        类型: {selectedEvidence.type === 'table_cell' ? '表格单元格' : (selectedEvidence.type === 'table' ? '报表数据表' : '正文文本')}
                      </span>
                    )}
                    {selectedEvidence.page !== undefined && selectedEvidence.page !== null && (
                      <span className="evidence-page-tag">
                        第 {selectedEvidence.page} 页
                      </span>
                    )}
                  </div>
                </div>

                {/* Bounding Box Visualizer */}
                <div className="evidence-section">
                  <div className="evidence-section-label">页面坐标视窗 (Bounding Box)</div>
                  {Array.isArray(selectedEvidence.bbox) && selectedEvidence.bbox.length === 4 ? (
                    <div className="evidence-bbox-container">
                      <div className="bbox-metrics">
                        <code>[x0: {selectedEvidence.bbox[0]}, y0: {selectedEvidence.bbox[1]}, x1: {selectedEvidence.bbox[2]}, y1: {selectedEvidence.bbox[3]}]</code>
                        <button
                          type="button"
                          className="evidence-copy-btn"
                          onClick={() => handleCopyChunkId(JSON.stringify(selectedEvidence.bbox))}
                        >
                          {copiedChunkId === JSON.stringify(selectedEvidence.bbox) ? '已复制坐标' : '复制坐标'}
                        </button>
                      </div>
                      {/* Mini Visual Page Wireframe */}
                      <div className="bbox-wireframe-page">
                        <div className="wireframe-header-line"></div>
                        <div className="wireframe-text-line" style={{ width: '85%' }}></div>
                        <div className="wireframe-text-line" style={{ width: '90%' }}></div>
                        <div className="wireframe-text-line" style={{ width: '70%' }}></div>
                        {/* Visual Bounding Box Box */}
                        <div
                          className="wireframe-bbox-highlight"
                          style={{
                            left: `${Math.max(5, Math.min(85, (selectedEvidence.bbox[0] / 612) * 100))}%`,
                            top: `${Math.max(10, Math.min(80, (selectedEvidence.bbox[1] / 792) * 100))}%`,
                            width: `${Math.max(15, Math.min(70, ((selectedEvidence.bbox[2] - selectedEvidence.bbox[0]) / 612) * 100))}%`,
                            height: `${Math.max(8, Math.min(30, ((selectedEvidence.bbox[3] - selectedEvidence.bbox[1]) / 792) * 100))}%`,
                          }}
                        >
                          <span className="bbox-marker-pin">📌 证据切片</span>
                        </div>
                        <div className="wireframe-text-line" style={{ width: '80%' }}></div>
                        <div className="wireframe-text-line" style={{ width: '75%' }}></div>
                        <div className="wireframe-footer-line"></div>
                      </div>
                    </div>
                  ) : (
                    <div className="evidence-empty-box">
                      <span>坐标模式：该切片为结构化报表行映射（已由底层语义解析器与表格单元格对齐）</span>
                    </div>
                  )}
                </div>

                {/* Table Cell Coordinates */}
                {selectedEvidence.cell_coordinates && (
                  <div className="evidence-section">
                    <div className="evidence-section-label">财务报表网格坐标 (Table Cell Coordinates)</div>
                    <div className="evidence-table-grid">
                      {selectedEvidence.cell_coordinates.table_title && (
                        <div className="table-grid-item">
                          <span className="table-grid-k">所属报表/表格:</span>
                          <span className="table-grid-v">{selectedEvidence.cell_coordinates.table_title}</span>
                        </div>
                      )}
                      {selectedEvidence.cell_coordinates.row_label && (
                        <div className="table-grid-item">
                          <span className="table-grid-k">报表行 (Row):</span>
                          <span className="table-grid-v font-bold">{selectedEvidence.cell_coordinates.row_label} (Row {selectedEvidence.cell_coordinates.row_idx ?? '-'})</span>
                        </div>
                      )}
                      {selectedEvidence.cell_coordinates.col_label && (
                        <div className="table-grid-item">
                          <span className="table-grid-k">报告期列 (Column):</span>
                          <span className="table-grid-v font-bold">{selectedEvidence.cell_coordinates.col_label} (Col {selectedEvidence.cell_coordinates.col_idx ?? '-'})</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Footnote Annotation */}
                {selectedEvidence.footnote_annotation && (
                  <div className="evidence-section">
                    <div className="evidence-section-label">报表附注核验 (Footnote Annotation)</div>
                    <div className="evidence-footnote-badge">
                      📎 {selectedEvidence.footnote_annotation}
                    </div>
                  </div>
                )}

                {/* Snippet / Content Excerpt */}
                {(selectedEvidence.snippet || selectedEvidence.content) && (
                  <div className="evidence-section">
                    <div className="evidence-section-label">底稿原文核验摘录</div>
                    <div className="evidence-snippet-container">
                      <pre className="evidence-snippet-text">
                        {selectedEvidence.snippet || selectedEvidence.content}
                      </pre>
                      <button
                        type="button"
                        className="evidence-copy-btn"
                        onClick={() => handleCopyChunkId(selectedEvidence.snippet || selectedEvidence.content)}
                      >
                        {copiedChunkId === (selectedEvidence.snippet || selectedEvidence.content) ? '已复制原文' : '复制原文'}
                      </button>
                    </div>
                  </div>
                )}

                {/* Chunk & Evidence Identifiers */}
                <div className="evidence-section">
                  <div className="evidence-section-label">切片唯一标识符 (Chunk / Evidence ID)</div>
                  <div className="evidence-id-row">
                    <code>{getSourceChunkId(selectedEvidence) || selectedEvidence.evidence_id || '-'}</code>
                    <button
                      type="button"
                      className="evidence-copy-btn"
                      onClick={() => handleCopyChunkId(getSourceChunkId(selectedEvidence) || selectedEvidence.evidence_id)}
                    >
                      {copiedChunkId === (getSourceChunkId(selectedEvidence) || selectedEvidence.evidence_id) ? '已复制 ID' : '复制 ID'}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Message;
