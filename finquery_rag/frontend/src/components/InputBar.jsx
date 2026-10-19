import React, { useState } from 'react';

const InputBar = ({ selectedDocs, onRemoveDoc, onSendMessage, disabled, disabledReason, readyDocumentCount = 0 }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const selectedDocsLabel = selectedDocs.length <= 2
    ? selectedDocs.join(', ')
    : `${selectedDocs.length} selected documents`;
  const placeholder = selectedDocs.length === 0
    ? (readyDocumentCount === 0
        ? 'Ask about benchmark filings (Apple, NVIDIA, MSFT...) or market quotes...'
        : 'Ask a question (will search all ready documents)...')
    : `Ask about ${selectedDocsLabel}...`;

  return (
    <div className="input-area">
      <div className="input-container">
        {selectedDocs.length > 0 && (
          <div className="selected-docs-pills">
            {selectedDocs.map((docName) => (
              <div key={docName} className="doc-pill">
                <span>{docName}</span>
                <button
                  className="pill-remove"
                  onClick={() => onRemoveDoc(docName)}
                  title="Remove document"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {disabledReason && (
          <div id="query-disabled-reason" className="input-disabled-reason" role="status">
            {disabledReason}
          </div>
        )}

        <form onSubmit={handleSubmit} className="input-form">
          <div className="input-wrapper">
            <textarea
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={disabledReason || placeholder}
              disabled={disabled}
              aria-describedby={disabledReason ? 'query-disabled-reason' : undefined}
              rows={1}
            />
          </div>
          <button
            type="submit"
            className="send-button"
            disabled={disabled || !input.trim()}
            title={disabledReason || 'Send question'}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default InputBar;
