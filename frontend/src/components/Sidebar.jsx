import { useState, useEffect, useCallback } from 'react';
import DocumentUpload from './DocumentUpload';
import DocumentList from './DocumentList';
import { listDocuments } from '../services/api';
import './Sidebar.css';

/**
 * Sidebar component containing document upload and list
 */
function Sidebar({ onNotification, isOpen, onToggle }) {
  const [documents, setDocuments] = useState([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  const fetchDocuments = useCallback(async () => {
    try {
      const result = await listDocuments();
      setDocuments(result.documents);
      setTotalChunks(result.total_chunks);
    } catch (err) {
      onNotification?.({
        type: 'error',
        message: 'Failed to load documents: ' + err.message,
      });
    } finally {
      setIsLoading(false);
    }
  }, [onNotification]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleUploadSuccess = (result) => {
    onNotification?.({
      type: 'success',
      message: `Uploaded "${result.filename}" (${result.chunks_created} chunks)`,
    });
    fetchDocuments();
  };

  const handleUploadError = (message) => {
    onNotification?.({
      type: 'error',
      message: message,
    });
  };

  const handleDelete = (documentId) => {
    onNotification?.({
      type: 'success',
      message: `Deleted "${documentId}"`,
    });
    fetchDocuments();
  };

  const handleDeleteError = (message) => {
    onNotification?.({
      type: 'error',
      message: message,
    });
  };

  return (
    <>
      {/* Toggle button for mobile */}
      <button className="sidebar-toggle" onClick={onToggle} title="Toggle documents panel">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
        <span className="doc-count">{documents.length}</span>
      </button>

      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h2>Documents</h2>
          <button className="close-btn" onClick={onToggle}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="sidebar-content">
          <DocumentUpload
            onUploadSuccess={handleUploadSuccess}
            onUploadError={handleUploadError}
          />

          <div className="divider"></div>

          {isLoading ? (
            <div className="loading-state">
              <div className="spinner"></div>
              <span>Loading documents...</span>
            </div>
          ) : (
            <DocumentList
              documents={documents}
              totalChunks={totalChunks}
              onDelete={handleDelete}
              onError={handleDeleteError}
            />
          )}
        </div>
      </aside>

      {/* Overlay for mobile */}
      {isOpen && <div className="sidebar-overlay" onClick={onToggle}></div>}
    </>
  );
}

export default Sidebar;
