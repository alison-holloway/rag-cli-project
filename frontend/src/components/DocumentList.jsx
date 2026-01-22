import { useState } from 'react';
import { deleteDocument } from '../services/api';
import './DocumentList.css';

const FILE_TYPE_ICONS = {
  pdf: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  ),
  md: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <path d="M9 15l2-2 2 2" />
      <path d="M13 13v4" />
    </svg>
  ),
  html: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  ),
  default: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  ),
};

/**
 * Display list of indexed documents with delete functionality
 */
function DocumentList({ documents, totalChunks, onDelete, onError }) {
  const [deletingId, setDeletingId] = useState(null);

  const handleDelete = async (documentId) => {
    if (!window.confirm(`Delete "${documentId}" from the knowledge base?`)) {
      return;
    }

    setDeletingId(documentId);
    try {
      await deleteDocument(documentId);
      onDelete?.(documentId);
    } catch (err) {
      onError?.(err.message);
    } finally {
      setDeletingId(null);
    }
  };

  const getFileIcon = (fileType) => {
    return FILE_TYPE_ICONS[fileType] || FILE_TYPE_ICONS.default;
  };

  if (!documents || documents.length === 0) {
    return (
      <div className="document-list empty">
        <div className="empty-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
          </svg>
          <p>No documents indexed yet</p>
          <span>Upload documents to start querying</span>
        </div>
      </div>
    );
  }

  return (
    <div className="document-list">
      <div className="document-list-header">
        <h3>Indexed Documents ({documents.length})</h3>
        <span className="total-chunks">{totalChunks} chunks total</span>
      </div>

      <ul className="documents">
        {documents.map((doc) => (
          <li key={doc.id} className={deletingId === doc.id ? 'deleting' : ''}>
            <div className="doc-icon" data-type={doc.file_type}>
              {getFileIcon(doc.file_type)}
            </div>
            <div className="doc-info">
              <span className="doc-name" title={doc.filename}>
                {doc.filename}
              </span>
              <span className="doc-meta">
                {doc.file_type.toUpperCase()} &middot; {doc.chunk_count} chunks
              </span>
            </div>
            <button
              className="delete-btn"
              onClick={() => handleDelete(doc.id)}
              disabled={deletingId === doc.id}
              title="Delete document"
            >
              {deletingId === doc.id ? (
                <span className="delete-spinner"></span>
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
              )}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default DocumentList;
