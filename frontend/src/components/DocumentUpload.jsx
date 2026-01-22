import { useState, useRef, useCallback } from 'react';
import { uploadDocument } from '../services/api';
import './DocumentUpload.css';

const SUPPORTED_EXTENSIONS = ['.pdf', '.md', '.markdown', '.html', '.htm'];
const SUPPORTED_TYPES = {
  'application/pdf': '.pdf',
  'text/markdown': '.md',
  'text/x-markdown': '.md',
  'text/html': '.html',
};

/**
 * Document upload component with drag-and-drop support
 */
function DocumentUpload({ onUploadSuccess, onUploadError }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const fileInputRef = useRef(null);

  const validateFile = (file) => {
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(extension)) {
      return `Unsupported file type: ${extension}. Supported formats: ${SUPPORTED_EXTENSIONS.join(', ')}`;
    }
    return null;
  };

  const handleUpload = useCallback(async (file) => {
    const error = validateFile(file);
    if (error) {
      onUploadError?.(error);
      return;
    }

    setIsUploading(true);
    setUploadProgress({ filename: file.name, status: 'uploading' });

    try {
      const result = await uploadDocument(file);
      setUploadProgress({ filename: file.name, status: 'success', chunks: result.chunks_created });
      onUploadSuccess?.(result);

      // Clear progress after a delay
      setTimeout(() => setUploadProgress(null), 3000);
    } catch (err) {
      setUploadProgress({ filename: file.name, status: 'error', message: err.message });
      onUploadError?.(err.message);

      // Clear progress after a delay
      setTimeout(() => setUploadProgress(null), 5000);
    } finally {
      setIsUploading(false);
    }
  }, [onUploadSuccess, onUploadError]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleUpload(files[0]);
    }
  }, [handleUpload]);

  const handleFileSelect = useCallback((e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      handleUpload(files[0]);
    }
    // Reset input so same file can be selected again
    e.target.value = '';
  }, [handleUpload]);

  const handleClick = () => {
    if (!isUploading) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="document-upload">
      <div
        className={`upload-zone ${isDragging ? 'dragging' : ''} ${isUploading ? 'uploading' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={SUPPORTED_EXTENSIONS.join(',')}
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />

        {uploadProgress ? (
          <div className={`upload-progress ${uploadProgress.status}`}>
            {uploadProgress.status === 'uploading' && (
              <>
                <div className="spinner"></div>
                <p>Uploading {uploadProgress.filename}...</p>
              </>
            )}
            {uploadProgress.status === 'success' && (
              <>
                <span className="success-icon">&#10003;</span>
                <p>Uploaded {uploadProgress.filename}</p>
                <span className="chunks">{uploadProgress.chunks} chunks indexed</span>
              </>
            )}
            {uploadProgress.status === 'error' && (
              <>
                <span className="error-icon">&#10007;</span>
                <p>Failed to upload {uploadProgress.filename}</p>
                <span className="error-message">{uploadProgress.message}</span>
              </>
            )}
          </div>
        ) : (
          <>
            <div className="upload-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="upload-text">
              <strong>Drop a file here</strong> or click to browse
            </p>
            <p className="upload-hint">
              Supports: PDF, Markdown, HTML
            </p>
          </>
        )}
      </div>
    </div>
  );
}

export default DocumentUpload;
