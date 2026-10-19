import React, { useRef, useEffect } from 'react';
import Message from './Message';
import EvalWorkbench from './EvalWorkbench';

const formatSessionTime = (timestamp) => {
  if (!timestamp) return 'Unknown time';
  const date = new Date(Number(timestamp) * 1000);
  if (Number.isNaN(date.getTime())) return 'Unknown time';
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
};

const ChatArea = ({
  messages,
  isLoading,
  onExampleClick,
  sessionId,
  retrievalK,
  retrievalKOptions,
  onRetrievalKChange,
  onNewSession,
  sessions,
  sessionSummary,
  opsSummary,
  sessionsLoading,
  isSessionPanelOpen,
  onToggleSessionPanel,
  onRefreshSessions,
  onSelectSession,
  onClearAllSessions,
  onExportTraceReplay,
  onExportFeedbackReplay,
  queryDisabled,
  queryDisabledReason,
  readyDocumentCount = 0,
}) => {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Example questions
  const exampleQuestions = [
    "What was Apple's Total net sales in FY2025?",
    'What was NVIDIA FY2025 Revenue?',
    '比亚迪最近股价怎么样',
    'Calculate the difference between Apple Net Income and Microsoft Net Income in FY2025.',
  ];

  return (
    <div className="chat-area">
      <div className="chat-toolbar">
        <div className="toolbar-left">
          <div className="session-label">
            Session {sessionId ? sessionId.slice(0, 8) : 'starting'}
          </div>
          <label className="retrieval-control">
            <span>Top-K</span>
            <select
              value={retrievalK}
              onChange={(event) => onRetrievalKChange(event.target.value)}
              disabled={isLoading}
            >
              {retrievalKOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="toolbar-actions">
          <button
            type="button"
            className="session-history-btn"
            onClick={onToggleSessionPanel}
            disabled={isLoading}
            aria-expanded={isSessionPanelOpen}
          >
            Sessions
          </button>
          <button
            type="button"
            className="new-session-btn"
            onClick={onNewSession}
            disabled={isLoading}
          >
            New chat
          </button>
        </div>
      </div>

      {isSessionPanelOpen && (
        <div className="session-panel" role="region" aria-label="Conversation sessions">
          <div className="session-panel-header">
            <div>
              <div className="session-panel-title">Conversation memory</div>
              <div className="session-panel-subtitle">
                {sessionSummary
                  ? `${sessionSummary.sessions} sessions - ${sessionSummary.messages} messages stored`
                  : 'Stored on the server for this account'}
              </div>
              {opsSummary && (
                <div className="ops-summary" aria-label="RAG operations summary">
                  <span>{opsSummary.documents?.ready || 0} ready docs</span>
                  <span>{opsSummary.traces?.total || 0} traces</span>
                  <span>{opsSummary.traces?.errors || 0} errors</span>
                  <span>{opsSummary.feedback?.total || 0} feedback</span>
                </div>
              )}
            </div>
            <div className="session-panel-actions">
              <button type="button" onClick={onRefreshSessions} disabled={sessionsLoading}>
                {sessionsLoading ? 'Refreshing...' : 'Refresh'}
              </button>
              <button
                type="button"
                onClick={onExportTraceReplay}
                disabled={sessionsLoading || !opsSummary?.traces?.total}
                title="Download replay cases generated from recent traces"
              >
                Export traces
              </button>
              <button
                type="button"
                onClick={onExportFeedbackReplay}
                disabled={sessionsLoading || !opsSummary?.feedback?.down}
                title="Download replay cases generated from down-rated feedback"
              >
                Export feedback
              </button>
              <button
                type="button"
                className="danger"
                onClick={onClearAllSessions}
                disabled={sessionsLoading || sessions.length === 0}
              >
                Clear all
              </button>
            </div>
          </div>

          <EvalWorkbench />

          {sessions.length === 0 ? (
            <div className="session-empty">
              {sessionsLoading ? 'Loading sessions...' : 'No stored sessions yet'}
            </div>
          ) : (
            <div className="session-list">
              {sessions.map((session) => {
                const active = session.session_id === sessionId;
                return (
                  <button
                    type="button"
                    key={session.session_id}
                    className={`session-item ${active ? 'active' : ''}`}
                    onClick={() => onSelectSession(session.session_id)}
                    disabled={active || isLoading}
                    title={session.session_id}
                  >
                    <span className="session-item-main">
                      <span className="session-item-id">{session.session_id.slice(0, 12)}</span>
                      <span className="session-item-time">{formatSessionTime(session.updated_at)}</span>
                    </span>
                    <span className="session-item-count">
                      {session.message_count} msg{session.message_count === 1 ? '' : 's'}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}

      {messages.length === 0 ? (
        <div className="chat-empty">
          <div className="chat-empty-icon">
            <svg viewBox="0 0 24 24">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
          </div>
          <div className="chat-empty-text">Ready when you are.</div>
          <div className="chat-empty-subtext">
            {readyDocumentCount > 0
              ? "Follow-up questions will use this session's recent context."
              : "Global benchmark filings (Apple, NVIDIA, MSFT, Tesla...) & market quotes are ready."}
          </div>
          {queryDisabledReason && (
            <div className="query-disabled-notice" role="status">
              {queryDisabledReason}
            </div>
          )}

          {/* Example Questions */}
          <div className="example-questions">
            <div className="example-title">Try asking:</div>
            <div className="example-grid">
              {exampleQuestions.map((question) => (
                <button
                  key={question}
                  className="example-button"
                  onClick={() => onExampleClick(question)}
                  disabled={queryDisabled}
                  title={queryDisabledReason || question}
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <>
          {messages.map((message, index) => (
            <Message key={`${message.role}-${index}`} message={message} onSendMessage={onExampleClick} />
          ))}
          {isLoading && (
            <div className="loading-message">
              <div className="loading-content">
                Thinking<span className="loading-dots"></span>
              </div>
            </div>
          )}
        </>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatArea;
