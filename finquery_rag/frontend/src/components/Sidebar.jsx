import React, { useRef, useState } from 'react';
import toast from 'react-hot-toast';

const Sidebar = ({
  documents,
  selectedDocs,
  onSelectDoc,
  onUpload,
  onDelete,
  onSelectAllReadyDocs,
  onClearSelectedDocs,
  isUploading,
  user,
  onLogout,
}) => {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const statusLabel = {
    pending: 'Pending',
    parsing: 'Parsing',
    indexing: 'Indexing',
    ready: 'Ready',
    failed: 'Failed',
  };
  const readyDocuments = documents.filter((doc) => (doc.status || 'ready') === 'ready');
  const processingDocuments = documents.filter((doc) => ['pending', 'parsing', 'indexing'].includes(doc.status));
  const failedDocuments = documents.filter((doc) => doc.status === 'failed');
  const allReadySelected = readyDocuments.length > 0
    && readyDocuments.every((doc) => selectedDocs.includes(doc.name));

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onUpload(file);
      e.target.value = '';
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file && file.name.endsWith('.pdf')) {
      onUpload(file);
    } else {
      alert('Please upload a PDF file');
    }
  };

  const handleDelete = (e, docName) => {
    e.stopPropagation();

    toast((t) => (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <div style={{ fontWeight: 500 }}>Delete {docName}?</div>
        <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>
          This action cannot be undone.
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
          <button
            onClick={() => {
              onDelete(docName);
              toast.dismiss(t.id);
            }}
            style={{
              flex: 1,
              padding: '0.5rem',
              backgroundColor: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 500,
            }}
          >
            Delete
          </button>
          <button
            onClick={() => toast.dismiss(t.id)}
            style={{
              flex: 1,
              padding: '0.5rem',
              backgroundColor: '#f3f4f6',
              color: '#1f2937',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 500,
            }}
          >
            Cancel
          </button>
        </div>
      </div>
    ), {
      duration: Infinity,
      style: { maxWidth: '400px' }
    });
  };

  return (
    <div className="sidebar">
      {/* Header with user info */}
      <div className="sidebar-header">
        <div className="sidebar-logo">FinQuery</div>
        <div className="sidebar-tagline">Financial Document Q&A</div>
        {user && (
          <div className="user-info">
            <span className="user-email">{user.email}</span>
            <button className="logout-btn" onClick={onLogout}>
              Logout
            </button>
          </div>
        )}
      </div>

      {/* Documents List */}
      <div className="sidebar-content">
        <div className="documents-section-header">
          <div>
            <div className="documents-section-title">Documents</div>
            <div className="documents-section-subtitle">
              {selectedDocs.length === 0
                ? 'Searching all ready documents'
                : `${selectedDocs.length} selected`}
            </div>
            {documents.length > 0 && (
              <div className="documents-status-summary" aria-label="Document status summary">
                <span>{readyDocuments.length} ready</span>
                {processingDocuments.length > 0 && <span>{processingDocuments.length} processing</span>}
                {failedDocuments.length > 0 && <span>{failedDocuments.length} failed</span>}
              </div>
            )}
          </div>
          {documents.length > 0 && (
            <div className="documents-actions">
              <button
                type="button"
                onClick={allReadySelected ? onClearSelectedDocs : onSelectAllReadyDocs}
                disabled={readyDocuments.length === 0}
              >
                {allReadySelected ? 'Clear' : 'Select ready'}
              </button>
            </div>
          )}
        </div>

        {documents.length === 0 ? (
          <div className="empty-state">
            <div style={{ marginBottom: '0.25rem' }}>No private documents yet</div>
            <div style={{ fontSize: '0.75rem', color: '#9ca3af', lineHeight: 1.4 }}>
              Global 10-K filings & market data are enabled. Upload a PDF anytime to analyze your own documents.
            </div>
          </div>
        ) : (
          <div className="document-list">
            {documents.map((doc) => {
              const isSelected = selectedDocs.includes(doc.name);
              const status = doc.status || 'ready';
              const isSelectable = status === 'ready';
              return (
                <div
                  key={doc.name}
                  onClick={() => {
                    if (isSelectable) onSelectDoc(doc.name);
                  }}
                  className={`document-item ${isSelected ? 'selected' : ''} ${!isSelectable ? 'disabled' : ''}`}
                  aria-disabled={!isSelectable}
                  title={isSelectable ? `Select ${doc.name}` : `${doc.name} is ${status}`}
                >
                  <div className="document-title-row">
                    <div className="document-name">{doc.name}</div>
                    <span className={`document-status status-${status}`}>
                      {statusLabel[status] || status}
                    </span>
                  </div>
                  <div className="document-meta">
                    <div className="document-stats">
                      <span>{doc.pages || 0} pages</span>
                      <span aria-hidden="true">•</span>
                      <span>{doc.count || 0} chunks</span>
                    </div>
                    {doc.error_message && (
                      <div className="document-error" title={doc.error_message}>
                        {doc.error_message}
                      </div>
                    )}
                    <button
                      className="delete-btn"
                      onClick={(e) => handleDelete(e, doc.name)}
                      title="Delete document"
                    >
                      ×
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Upload Area - EXACTLY like PDFtoChat */}
      <div className="upload-section">
        <div
          className={`upload-area ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleUploadClick}
        >
          <div className="upload-content">
            <button
              className="upload-button"
              disabled={isUploading}
              onClick={(e) => {
                e.stopPropagation();
                handleUploadClick();
              }}
            >
              {isUploading ? 'Uploading...' : 'Upload a File'}
            </button>
            <div className="upload-subtext">...or drag and drop a file.</div>
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
      </div>
    </div>
  );
};

export default Sidebar;
